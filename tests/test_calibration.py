from __future__ import annotations

import pandas as pd
import pytest

from risk_stratification_engine.calibration import (
    build_calibration_bins,
    build_threshold_table,
)


def _predictions_and_labels() -> tuple[pd.Series, pd.Series]:
    """20 snapshots: 4 positives, 16 negatives. Predictions are
    stratified so higher predictions correspond to higher labels."""
    predictions = pd.Series(
        [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28,
         0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.85, 0.92],
        name="risk",
    )
    # Top 4 predictions are positive
    labels = pd.Series(
        [False] * 16 + [True, True, True, True],
        dtype=bool,
        name="label",
    )
    return predictions, labels


# ---------------------------------------------------------------------------
# build_calibration_bins
# ---------------------------------------------------------------------------


def test_build_calibration_bins_returns_one_row_per_bin():
    predictions, labels = _predictions_and_labels()
    bins = build_calibration_bins(predictions, labels, n_bins=5)

    assert isinstance(bins, list)
    assert len(bins) == 5


def test_build_calibration_bins_each_row_has_required_keys():
    predictions, labels = _predictions_and_labels()
    bins = build_calibration_bins(predictions, labels, n_bins=5)

    required = {
        "bin_index",
        "predicted_risk_mean",
        "observed_event_rate",
        "snapshot_count",
        "positive_count",
    }
    for row in bins:
        assert required.issubset(row.keys()), f"missing keys in bin row: {row}"


def test_build_calibration_bins_snapshot_counts_sum_to_total():
    predictions, labels = _predictions_and_labels()
    bins = build_calibration_bins(predictions, labels, n_bins=5)

    total = sum(row["snapshot_count"] for row in bins)
    assert total == len(predictions)


def test_build_calibration_bins_positive_counts_sum_to_total():
    predictions, labels = _predictions_and_labels()
    bins = build_calibration_bins(predictions, labels, n_bins=5)

    total = sum(row["positive_count"] for row in bins)
    assert total == int(labels.sum())


def test_build_calibration_bins_returns_none_for_empty_bin():
    predictions = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = pd.Series([False, False, False, False, True], dtype=bool)
    # 5 bins for 5 points → each bin has 1 point; bin with 0 positives
    # observed_event_rate is defined (0.0), not None — but if bin is empty
    # (0 snapshots) it returns None.
    bins = build_calibration_bins(predictions, labels, n_bins=10)

    # Some bins will be empty because only 5 points → 10 bins
    empty_bins = [row for row in bins if row["snapshot_count"] == 0]
    for row in empty_bins:
        assert row["observed_event_rate"] is None
        assert row["predicted_risk_mean"] is None


def test_build_calibration_bins_observed_rate_is_correct():
    predictions, labels = _predictions_and_labels()
    bins = build_calibration_bins(predictions, labels, n_bins=5)

    # Top bin (bin_index 4) should contain 4 positives out of 4 → rate 1.0
    top_bin = next(row for row in bins if row["bin_index"] == 4)
    assert top_bin["positive_count"] == 4
    assert top_bin["observed_event_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_threshold_table
# ---------------------------------------------------------------------------


def test_build_threshold_table_returns_list_of_rows():
    predictions, labels = _predictions_and_labels()
    rows = build_threshold_table(predictions, labels)

    assert isinstance(rows, list)
    assert len(rows) > 0


def test_build_threshold_table_each_row_has_required_keys():
    predictions, labels = _predictions_and_labels()
    rows = build_threshold_table(predictions, labels)

    required = {
        "threshold_kind",
        "threshold_value",
        "alert_count",
        "event_capture",
        "precision",
        "lift",
    }
    for row in rows:
        assert required.issubset(row.keys()), f"missing keys in row: {row}"


def test_build_threshold_table_includes_percentile_and_probability_kinds():
    predictions, labels = _predictions_and_labels()
    rows = build_threshold_table(predictions, labels)

    kinds = {row["threshold_kind"] for row in rows}
    assert "percentile" in kinds
    assert "probability" in kinds


def test_build_threshold_table_top_decile_captures_all_positives():
    """All 4 positives are in the top 20% (4/20) by construction."""
    predictions, labels = _predictions_and_labels()
    rows = build_threshold_table(predictions, labels)

    top20 = next(
        row for row in rows
        if row["threshold_kind"] == "percentile" and row["threshold_value"] == pytest.approx(0.20)
    )
    assert top20["alert_count"] == 4
    assert top20["event_capture"] == pytest.approx(1.0)
    assert top20["precision"] == pytest.approx(1.0)


def test_build_threshold_table_alert_count_is_non_negative():
    predictions, labels = _predictions_and_labels()
    rows = build_threshold_table(predictions, labels)

    for row in rows:
        assert row["alert_count"] >= 0


def test_build_threshold_table_event_capture_between_zero_and_one():
    predictions, labels = _predictions_and_labels()
    rows = build_threshold_table(predictions, labels)

    for row in rows:
        if row["event_capture"] is not None:
            assert 0.0 <= row["event_capture"] <= 1.0


def test_build_threshold_table_precision_none_when_no_alerts():
    """A threshold so high that no alerts are issued → precision is None."""
    predictions, labels = _predictions_and_labels()
    rows = build_threshold_table(predictions, labels, probability_thresholds=(0.999,))

    # Alert count for probability threshold 0.999 should be 0
    high_thresh = next(
        row for row in rows
        if row["threshold_kind"] == "probability" and row["threshold_value"] == pytest.approx(0.999)
    )
    assert high_thresh["alert_count"] == 0
    assert high_thresh["precision"] is None
    assert high_thresh["lift"] is None


def test_build_threshold_table_lift_matches_manual_calculation():
    """Verify lift = precision / overall_positive_rate."""
    predictions, labels = _predictions_and_labels()
    overall_positive_rate = float(labels.mean())  # 4/20 = 0.2

    rows = build_threshold_table(predictions, labels)
    for row in rows:
        if row["precision"] is not None and row["alert_count"] > 0:
            expected_lift = row["precision"] / overall_positive_rate
            assert row["lift"] == pytest.approx(expected_lift, rel=1e-5)
