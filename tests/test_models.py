import pandas as pd

from risk_stratification_engine.models import (
    GRAPH_SNAPSHOT_FEATURE_COLUMNS,
    train_discrete_time_risk_model,
)


def _labeled_snapshot_frame() -> pd.DataFrame:
    rows = []
    for athlete_index, athlete_id in enumerate(["a1", "a2", "a3", "a4", "a5"]):
        for time_index in range(3):
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "season_id": "2026",
                    "snapshot_date": pd.Timestamp(f"2026-01-0{time_index + 1}"),
                    "time_index": time_index,
                    "node_count": 4,
                    "edge_count": athlete_index + time_index,
                    "mean_abs_correlation": 0.1 * (athlete_index + time_index),
                    "days_to_event": 3 if athlete_id in {"a1", "a2"} else 30,
                    "event_observed": athlete_id in {"a1", "a2"},
                    "event_within_7d": athlete_id in {"a1", "a2"},
                    "event_within_14d": athlete_id in {"a1", "a2"},
                    "event_within_30d": athlete_id in {"a1", "a2", "a3"},
                    "primary_model_event": athlete_id != "a3",
                }
            )
    return pd.DataFrame(rows)


def test_train_discrete_time_risk_model_predicts_horizon_risks_without_leakage():
    result = train_discrete_time_risk_model(_labeled_snapshot_frame())

    risk_columns = ["risk_7d", "risk_14d", "risk_30d"]
    assert risk_columns == [f"risk_{horizon}d" for horizon in result.summary["horizons"]]
    assert set(risk_columns).issubset(result.timeline.columns)
    assert result.timeline[risk_columns].ge(0.0).all().all()
    assert result.timeline[risk_columns].le(1.0).all().all()

    leakage_columns = {
        "days_to_event",
        "event_observed",
        "event_within_7d",
        "event_within_14d",
        "event_within_30d",
        "primary_model_event",
    }
    assert result.summary["feature_columns"] == list(GRAPH_SNAPSHOT_FEATURE_COLUMNS)
    assert leakage_columns.isdisjoint(result.summary["feature_columns"])
    assert set(result.summary["train_athlete_ids"]).isdisjoint(
        result.summary["test_athlete_ids"]
    )
    assert result.summary["event_policy"] == "primary_model_event"


def test_train_discrete_time_risk_model_falls_back_when_training_labels_one_class():
    frame = _labeled_snapshot_frame()
    frame["event_within_7d"] = False
    frame["event_within_14d"] = False
    frame["event_within_30d"] = False

    result = train_discrete_time_risk_model(frame)

    for horizon in (7, 14, 30):
        horizon_summary = result.summary["horizon_models"][str(horizon)]
        assert horizon_summary["model_kind"] == "prevalence_fallback"
        assert horizon_summary["train_positive_rate"] == 0.0
        assert result.timeline[f"risk_{horizon}d"].eq(0.0).all()
