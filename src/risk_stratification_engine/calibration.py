from __future__ import annotations

import pandas as pd


def build_calibration_bins(
    predictions: pd.Series,
    labels: pd.Series,
    n_bins: int = 10,
) -> list[dict]:
    """Equal-count quantile binning of OOF predictions vs. observed labels.

    Returns n_bins rows ordered from lowest to highest predicted risk. Bins
    that receive no snapshots have snapshot_count=0 and None for rate/mean.
    """
    n = len(predictions)
    if n == 0:
        return [_empty_bin(i) for i in range(n_bins)]

    # Rank-based equal-count assignment: bin = floor(rank * n_bins / n)
    # rank method='first' breaks ties by position so every row is uniquely ranked
    ranks = predictions.rank(method="first") - 1  # 0-indexed
    bin_assignments = (ranks * n_bins // n).astype(int).clip(0, n_bins - 1)

    rows = []
    for bin_index in range(n_bins):
        mask = bin_assignments == bin_index
        bin_preds = predictions[mask]
        bin_labels = labels[mask]
        if len(bin_preds) == 0:
            rows.append(_empty_bin(bin_index))
        else:
            rows.append(
                {
                    "bin_index": bin_index,
                    "predicted_risk_mean": float(bin_preds.mean()),
                    "observed_event_rate": float(bin_labels.mean()),
                    "snapshot_count": int(len(bin_preds)),
                    "positive_count": int(bin_labels.sum()),
                }
            )
    return rows


def build_threshold_table(
    predictions: pd.Series,
    labels: pd.Series,
    percentile_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
    probability_thresholds: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50),
) -> list[dict]:
    """Build a threshold table combining top-N% (percentile) and fixed-risk
    (probability) thresholds. Returns one row per threshold.

    Columns: threshold_kind, threshold_value, alert_count, event_capture,
    precision, lift.
    """
    overall_positive_rate = float(labels.mean()) if len(labels) else 0.0
    total_positives = int(labels.sum())
    rows = []

    for pct in sorted(percentile_thresholds):
        top_n = max(0, round(len(predictions) * pct))
        if top_n > 0:
            alert_indices = predictions.nlargest(top_n).index
        else:
            alert_indices = pd.Index([])
        rows.append(
            _threshold_row(
                kind="percentile",
                value=pct,
                alert_indices=alert_indices,
                labels=labels,
                total_positives=total_positives,
                overall_positive_rate=overall_positive_rate,
            )
        )

    for prob in sorted(probability_thresholds):
        alert_indices = predictions[predictions >= prob].index
        rows.append(
            _threshold_row(
                kind="probability",
                value=prob,
                alert_indices=alert_indices,
                labels=labels,
                total_positives=total_positives,
                overall_positive_rate=overall_positive_rate,
            )
        )

    return rows


def _threshold_row(
    kind: str,
    value: float,
    alert_indices: pd.Index,
    labels: pd.Series,
    total_positives: int,
    overall_positive_rate: float,
) -> dict:
    alert_count = int(len(alert_indices))
    if alert_count == 0:
        return {
            "threshold_kind": kind,
            "threshold_value": float(value),
            "alert_count": 0,
            "event_capture": 0.0,
            "precision": None,
            "lift": None,
        }
    captured = int(labels.loc[alert_indices].sum())
    event_capture = (
        float(captured / total_positives) if total_positives > 0 else None
    )
    precision = float(captured / alert_count)
    lift = (
        float(precision / overall_positive_rate)
        if overall_positive_rate > 0
        else None
    )
    return {
        "threshold_kind": kind,
        "threshold_value": float(value),
        "alert_count": alert_count,
        "event_capture": event_capture,
        "precision": precision,
        "lift": lift,
    }


def _empty_bin(bin_index: int) -> dict:
    return {
        "bin_index": bin_index,
        "predicted_risk_mean": None,
        "observed_event_rate": None,
        "snapshot_count": 0,
        "positive_count": 0,
    }
