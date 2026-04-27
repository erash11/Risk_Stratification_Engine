import pandas as pd

from risk_stratification_engine.evaluation import evaluate_risk_model


def _timeline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "athlete_id": ["train1", "train2", "test1", "test1", "test2", "test2"],
            "season_id": ["2026"] * 6,
            "event_within_7d": [False, True, True, False, True, False],
            "event_within_14d": [False, True, True, False, True, False],
            "event_within_30d": [False, True, True, False, True, False],
            "primary_model_event": [False, True, True, False, True, False],
            "risk_7d": [0.1, 0.7, 0.9, 0.2, 0.8, 0.1],
            "risk_14d": [0.1, 0.7, 0.9, 0.2, 0.8, 0.1],
            "risk_30d": [0.1, 0.7, 0.9, 0.2, 0.8, 0.1],
        }
    )


def _model_summary() -> dict[str, object]:
    return {
        "model_type": "discrete_time_logistic_baseline",
        "horizons": [7, 14, 30],
        "event_policy": "primary_model_event",
        "split_policy": "athlete_level_sorted_holdout_20pct",
        "train_athlete_ids": ["train1", "train2"],
        "test_athlete_ids": ["test1", "test2"],
        "horizon_models": {
            "7": {"train_positive_rate": 0.5},
            "14": {"train_positive_rate": 0.5},
            "30": {"train_positive_rate": 0.5},
        },
    }


def test_evaluate_risk_model_compares_holdout_predictions_to_prevalence_baseline():
    evaluation = evaluate_risk_model(_timeline(), _model_summary())

    assert evaluation["model_type"] == "discrete_time_logistic_baseline"
    assert evaluation["split_policy"] == "athlete_level_sorted_holdout_20pct"
    assert evaluation["event_policy"] == "primary_model_event"
    assert evaluation["test_athlete_count"] == 2
    assert evaluation["test_snapshot_count"] == 4

    horizon = evaluation["horizons"]["7"]
    assert horizon["test_positive_count"] == 2
    assert horizon["test_positive_rate"] == 0.5
    assert horizon["prevalence_baseline_risk"] == 0.5
    assert horizon["model_brier_score"] < horizon["prevalence_brier_score"]
    assert horizon["brier_skill_score"] > 0.0
    assert horizon["beats_prevalence_baseline"] is True
    assert horizon["roc_auc"] == 1.0
    assert horizon["average_precision"] == 1.0
    assert horizon["top_decile_lift"] == 2.0


def test_evaluate_risk_model_omits_discrimination_metrics_for_one_class_holdout():
    timeline = _timeline()
    timeline.loc[timeline["athlete_id"].str.startswith("test"), "event_within_7d"] = False

    evaluation = evaluate_risk_model(timeline, _model_summary())

    horizon = evaluation["horizons"]["7"]
    assert horizon["test_positive_count"] == 0
    assert horizon["roc_auc"] is None
    assert horizon["average_precision"] is None
    assert horizon["top_decile_lift"] is None
