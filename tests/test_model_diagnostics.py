from __future__ import annotations

import pandas as pd
import pytest

from risk_stratification_engine.model_diagnostics import (
    build_model_improvement_diagnostics,
)


def _alert_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-01-01",
                "risk_30d": 0.90,
                "top_feature_30d": "mean_abs_correlation",
                "elevated_z_features": ["z_mean_abs_correlation"],
                "event_observed": True,
                "event_date": "2026-01-10",
                "days_to_event": 9,
                "injury_type": "hamstring",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 2,
                "primary_model_event": True,
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 1,
                "snapshot_date": "2026-01-02",
                "risk_30d": 0.82,
                "top_feature_30d": "mean_abs_correlation",
                "elevated_z_features": [],
                "event_observed": True,
                "event_date": "2026-01-10",
                "days_to_event": 8,
                "injury_type": "hamstring",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 2,
                "primary_model_event": True,
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-02-01",
                "risk_30d": 0.12,
                "top_feature_30d": "edge_density",
                "elevated_z_features": [],
                "event_observed": True,
                "event_date": "2026-02-20 00:00:00",
                "days_to_event": 19,
                "injury_type": "ankle",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 3,
                "primary_model_event": True,
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "time_index": 1,
                "snapshot_date": "2026-02-15",
                "risk_30d": 0.18,
                "top_feature_30d": "graph_instability",
                "elevated_z_features": ["z_graph_instability"],
                "event_observed": True,
                "event_date": "2026-02-20 00:00:00",
                "days_to_event": 5,
                "injury_type": "ankle",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 3,
                "primary_model_event": True,
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "time_index": 2,
                "snapshot_date": "2026-02-21",
                "risk_30d": 0.99,
                "top_feature_30d": "mean_abs_correlation",
                "elevated_z_features": ["z_mean_abs_correlation"],
                "event_observed": True,
                "event_date": "2026-02-20 00:00:00",
                "days_to_event": -1,
                "injury_type": "ankle",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 3,
                "primary_model_event": True,
            },
            {
                "athlete_id": "a3",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-03-01",
                "risk_30d": 0.80,
                "top_feature_30d": "edge_density",
                "elevated_z_features": [],
                "event_observed": False,
                "event_date": "2026-03-30",
                "days_to_event": 29,
                "injury_type": "censored",
                "event_window_quality": "censored",
                "nearest_measurement_gap_days": None,
                "primary_model_event": False,
            },
            {
                "athlete_id": "a4",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-04-01",
                "risk_30d": 0.72,
                "top_feature_30d": "mean_abs_correlation",
                "elevated_z_features": ["z_edge_count"],
                "event_observed": True,
                "event_date": "2026-04-20",
                "days_to_event": 19,
                "injury_type": "fracture",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 4,
                "primary_model_event": True,
            },
        ]
    )


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "horizon_days": 30,
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
                "start_time_index": 0,
                "end_time_index": 1,
                "peak_risk": 0.90,
                "mean_risk": 0.86,
                "snapshot_count": 2,
                "duration_days": 1,
                "injury_type": "hamstring",
                "days_from_start_to_event": 9,
                "event_within_horizon_after_start": True,
                "top_model_features": [
                    {"feature": "mean_abs_correlation", "mean_abs_contribution": 0.7}
                ],
                "elevated_z_features": ["z_mean_abs_correlation"],
            },
            {
                "athlete_id": "a3",
                "season_id": "2026",
                "horizon_days": 30,
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
                "start_time_index": 0,
                "end_time_index": 0,
                "peak_risk": 0.80,
                "mean_risk": 0.80,
                "snapshot_count": 1,
                "duration_days": 0,
                "injury_type": "censored",
                "days_from_start_to_event": None,
                "event_within_horizon_after_start": False,
                "top_model_features": [
                    {"feature": "edge_density", "mean_abs_contribution": 0.5}
                ],
                "elevated_z_features": [],
            },
        ]
    )


def _quality() -> dict[str, object]:
    return {
        "quality_rows": [
            {
                "horizon_days": 30,
                "threshold": "percentile:0.05",
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
            },
            {
                "horizon_days": 14,
                "threshold": "percentile:0.05",
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
            },
        ]
    }


def _rows_by_group(diagnostics: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["comparison_group"]: row for row in diagnostics["diagnostic_rows"]}


def test_build_model_improvement_diagnostics_compares_alert_and_missed_groups():
    diagnostics = build_model_improvement_diagnostics(
        episodes=_episodes(),
        alert_timeline=_alert_timeline(),
        quality=_quality(),
    )

    rows = _rows_by_group(diagnostics)

    assert diagnostics["diagnostic_row_count"] == 6
    assert rows["true_positive_episode"]["row_count"] == 1
    assert rows["true_positive_episode"]["median_peak_risk"] == pytest.approx(0.90)
    assert rows["true_positive_episode"]["elevated_z_rate"] == pytest.approx(1.0)
    assert rows["true_positive_episode"]["top_feature_counts"] == {
        "mean_abs_correlation": 1
    }
    assert rows["true_positive_episode"]["recommended_next_action"] == (
        "retain_policy_signal"
    )

    assert rows["false_positive_episode"]["row_count"] == 1
    assert rows["false_positive_episode"]["median_peak_risk"] == pytest.approx(0.80)
    assert rows["false_positive_episode"]["recommended_next_action"] == (
        "add_context_features"
    )

    assert rows["missed_event"]["row_count"] == 2
    assert rows["missed_event"]["max_pre_event_risk"] == pytest.approx(0.72)
    assert rows["missed_event"]["median_pre_event_snapshot_count"] == pytest.approx(1.5)
    assert rows["missed_event"]["event_window_quality_counts"] == {"modelable": 2}
    assert rows["missed_event"]["recommended_next_action"] == (
        "review_threshold_policy"
    )


def test_build_model_improvement_diagnostics_uses_only_pre_event_lead_window():
    diagnostics = build_model_improvement_diagnostics(
        episodes=_episodes(),
        alert_timeline=_alert_timeline(),
        quality=_quality(),
    )

    missed = _rows_by_group(diagnostics)["missed_event"]

    assert missed["max_pre_event_risk"] == pytest.approx(0.72)
    assert missed["top_feature_counts"] == {
        "edge_density": 1,
        "graph_instability": 1,
        "mean_abs_correlation": 1,
    }


def test_build_model_improvement_diagnostics_marks_low_risk_modelable_misses():
    timeline = _alert_timeline()
    timeline = timeline[timeline["athlete_id"] != "a4"]
    diagnostics = build_model_improvement_diagnostics(
        episodes=_episodes(),
        alert_timeline=timeline,
        quality={"quality_rows": [_quality()["quality_rows"][0]]},
    )

    missed = _rows_by_group(diagnostics)["missed_event"]

    assert missed["row_count"] == 1
    assert missed["max_pre_event_risk"] == pytest.approx(0.18)
    assert missed["recommended_next_action"] == "add_event_specific_features"


def test_build_model_improvement_diagnostics_writes_stable_empty_group_rows():
    diagnostics = build_model_improvement_diagnostics(
        episodes=_episodes(),
        alert_timeline=_alert_timeline(),
        quality={"quality_rows": [_quality()["quality_rows"][1]]},
    )

    rows = _rows_by_group(diagnostics)

    assert rows["true_positive_episode"]["row_count"] == 0
    assert rows["true_positive_episode"]["median_peak_risk"] is None
    assert rows["false_positive_episode"]["row_count"] == 0
    assert rows["missed_event"]["row_count"] == 3
