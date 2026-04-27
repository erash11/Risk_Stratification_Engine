from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate_risk_model(
    timeline: pd.DataFrame,
    model_summary: dict[str, Any],
) -> dict[str, Any]:
    test_ids = [str(athlete_id) for athlete_id in model_summary["test_athlete_ids"]]
    test_mask = timeline["athlete_id"].astype(str).isin(test_ids)
    horizons = [int(horizon) for horizon in model_summary["horizons"]]
    event_policy = str(model_summary["event_policy"])

    return {
        "model_type": model_summary["model_type"],
        "event_policy": event_policy,
        "split_policy": model_summary["split_policy"],
        "test_athlete_count": len(test_ids),
        "test_snapshot_count": int(test_mask.sum()),
        "horizons": {
            str(horizon): _evaluate_horizon(
                timeline=timeline,
                test_mask=test_mask,
                horizon=horizon,
                event_policy=event_policy,
                train_positive_rate=_train_positive_rate(model_summary, horizon),
            )
            for horizon in horizons
        },
    }


def _evaluate_horizon(
    timeline: pd.DataFrame,
    test_mask: pd.Series,
    horizon: int,
    event_policy: str,
    train_positive_rate: float,
) -> dict[str, Any]:
    labels = _labels(timeline, horizon, event_policy).loc[test_mask]
    predictions = timeline.loc[test_mask, f"risk_{horizon}d"].astype(float)
    baseline_predictions = pd.Series(train_positive_rate, index=labels.index)

    model_brier = _brier_score(labels, predictions)
    prevalence_brier = _brier_score(labels, baseline_predictions)
    positive_rate = float(labels.mean()) if len(labels) else None
    brier_skill_score = _brier_skill_score(model_brier, prevalence_brier)

    return {
        "test_snapshot_count": int(len(labels)),
        "test_positive_count": int(labels.sum()),
        "test_positive_rate": positive_rate,
        "mean_predicted_risk": float(predictions.mean()) if len(predictions) else None,
        "prevalence_baseline_risk": train_positive_rate,
        "model_brier_score": model_brier,
        "prevalence_brier_score": prevalence_brier,
        "brier_skill_score": brier_skill_score,
        "beats_prevalence_baseline": (
            None
            if model_brier is None or prevalence_brier is None
            else model_brier < prevalence_brier
        ),
        "roc_auc": _roc_auc(labels, predictions),
        "average_precision": _average_precision(labels, predictions),
        "top_decile_lift": _top_decile_lift(labels, predictions),
    }


def _train_positive_rate(model_summary: dict[str, Any], horizon: int) -> float:
    return float(model_summary["horizon_models"][str(horizon)]["train_positive_rate"])


def _labels(
    timeline: pd.DataFrame,
    horizon: int,
    event_policy: str,
) -> pd.Series:
    labels = timeline[f"event_within_{horizon}d"].astype(bool)
    if event_policy == "primary_model_event":
        labels = labels & timeline["primary_model_event"].astype(bool)
    return labels


def _brier_score(labels: pd.Series, predictions: pd.Series) -> float | None:
    if labels.empty:
        return None
    return float(brier_score_loss(labels.astype(int), predictions))


def _brier_skill_score(
    model_brier: float | None,
    prevalence_brier: float | None,
) -> float | None:
    if model_brier is None or prevalence_brier in {None, 0.0}:
        return None
    return float(1.0 - (model_brier / prevalence_brier))


def _roc_auc(labels: pd.Series, predictions: pd.Series) -> float | None:
    if labels.nunique(dropna=False) < 2:
        return None
    return float(roc_auc_score(labels.astype(int), predictions))


def _average_precision(labels: pd.Series, predictions: pd.Series) -> float | None:
    if labels.nunique(dropna=False) < 2:
        return None
    return float(average_precision_score(labels.astype(int), predictions))


def _top_decile_lift(labels: pd.Series, predictions: pd.Series) -> float | None:
    positive_rate = float(labels.mean()) if len(labels) else 0.0
    if positive_rate <= 0.0:
        return None
    top_count = max(1, int(len(labels) * 0.1))
    top_indices = predictions.sort_values(ascending=False).head(top_count).index
    top_positive_rate = float(labels.loc[top_indices].mean())
    return float(top_positive_rate / positive_rate)
