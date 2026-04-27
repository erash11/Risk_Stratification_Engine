from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from risk_stratification_engine.events import DEFAULT_HORIZONS


MODEL_TYPE = "discrete_time_logistic_baseline"
GRAPH_SNAPSHOT_FEATURE_COLUMNS = (
    "time_index",
    "node_count",
    "edge_count",
    "mean_abs_correlation",
    "edge_density",
    "delta_edge_count",
    "delta_mean_abs_correlation",
    "delta_edge_density",
    "graph_instability",
)


@dataclass(frozen=True)
class RiskModelResult:
    timeline: pd.DataFrame
    summary: dict[str, Any]


def train_discrete_time_risk_model(
    labeled: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> RiskModelResult:
    _require_columns(labeled, GRAPH_SNAPSHOT_FEATURE_COLUMNS)
    timeline = labeled.copy()
    event_policy = _event_policy(timeline)
    train_ids, test_ids = _athlete_holdout_ids(timeline["athlete_id"])
    train_mask = timeline["athlete_id"].isin(train_ids)
    test_mask = timeline["athlete_id"].isin(test_ids)
    features = _feature_frame(timeline)

    horizon_models: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        label_column = f"event_within_{horizon}d"
        _require_columns(timeline, (label_column,))
        labels = _training_labels(timeline, label_column, event_policy)
        train_labels = labels.loc[train_mask]
        train_features = features.loc[train_mask]

        if train_labels.nunique(dropna=False) < 2:
            probability = float(train_labels.mean()) if not train_labels.empty else 0.0
            probabilities = pd.Series(probability, index=timeline.index)
            model_kind = "prevalence_fallback"
        else:
            model = LogisticRegression(max_iter=1000, random_state=0)
            model.fit(train_features, train_labels)
            probabilities = pd.Series(
                model.predict_proba(features)[:, 1],
                index=timeline.index,
            )
            model_kind = "logistic_regression"

        risk_column = f"risk_{horizon}d"
        timeline[risk_column] = probabilities.clip(lower=0.0, upper=1.0).round(6)
        horizon_models[str(horizon)] = _horizon_summary(
            labels=labels,
            predictions=timeline[risk_column],
            train_mask=train_mask,
            test_mask=test_mask,
            model_kind=model_kind,
        )

    return RiskModelResult(
        timeline=timeline,
        summary={
            "model_type": MODEL_TYPE,
            "horizons": list(horizons),
            "feature_columns": list(GRAPH_SNAPSHOT_FEATURE_COLUMNS),
            "event_policy": event_policy,
            "split_policy": "athlete_level_sorted_holdout_20pct",
            "train_athlete_count": len(train_ids),
            "test_athlete_count": len(test_ids),
            "train_athlete_ids": train_ids,
            "test_athlete_ids": test_ids,
            "horizon_models": horizon_models,
        },
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"model input missing required columns: {', '.join(missing)}")


def _event_policy(frame: pd.DataFrame) -> str:
    if "primary_model_event" in frame.columns:
        return "primary_model_event"
    return "event_observed"


def _athlete_holdout_ids(athlete_ids: pd.Series) -> tuple[list[str], list[str]]:
    unique_ids = sorted(str(athlete_id) for athlete_id in athlete_ids.dropna().unique())
    if len(unique_ids) <= 1:
        return unique_ids, []
    test_count = max(1, ceil(len(unique_ids) * 0.2))
    test_ids = unique_ids[-test_count:]
    train_ids = unique_ids[:-test_count]
    return train_ids, test_ids


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[:, GRAPH_SNAPSHOT_FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def _training_labels(
    frame: pd.DataFrame,
    label_column: str,
    event_policy: str,
) -> pd.Series:
    labels = frame[label_column].astype(bool)
    if event_policy == "primary_model_event":
        labels = labels & frame["primary_model_event"].astype(bool)
    return labels


def _horizon_summary(
    labels: pd.Series,
    predictions: pd.Series,
    train_mask: pd.Series,
    test_mask: pd.Series,
    model_kind: str,
) -> dict[str, Any]:
    train_labels = labels.loc[train_mask]
    test_labels = labels.loc[test_mask]
    test_predictions = predictions.loc[test_mask]
    summary: dict[str, Any] = {
        "model_kind": model_kind,
        "train_snapshot_count": int(train_mask.sum()),
        "test_snapshot_count": int(test_mask.sum()),
        "train_positive_count": int(train_labels.sum()),
        "test_positive_count": int(test_labels.sum()),
        "train_positive_rate": float(train_labels.mean()) if len(train_labels) else 0.0,
        "test_positive_rate": float(test_labels.mean()) if len(test_labels) else None,
    }
    if len(test_labels) > 0:
        summary["test_brier_score"] = float(
            brier_score_loss(test_labels.astype(int), test_predictions)
        )
    else:
        summary["test_brier_score"] = None
    return summary
