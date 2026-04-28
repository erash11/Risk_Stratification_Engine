from __future__ import annotations

import pytest
import pandas as pd

from risk_stratification_engine.episode_quality import build_alert_episode_quality


def _timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-01-01",
                "event_observed": True,
                "event_date": "2026-01-10",
                "injury_type": "hamstring",
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 1,
                "snapshot_date": "2026-01-02",
                "event_observed": True,
                "event_date": "2026-01-10",
                "injury_type": "hamstring",
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-02-01",
                "event_observed": True,
                "event_date": "2026-02-20",
                "injury_type": "ankle",
            },
            {
                "athlete_id": "a3",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-03-01",
                "event_observed": False,
                "event_date": "2026-03-30",
                "injury_type": "censored",
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
                "peak_time_index": 0,
                "start_date": "2026-01-01",
                "end_date": "2026-01-02",
                "peak_date": "2026-01-01",
                "snapshot_count": 2,
                "duration_days": 1,
                "peak_risk": 0.90,
                "mean_risk": 0.85,
                "event_observed": True,
                "injury_type": "hamstring",
                "days_from_start_to_event": 9,
                "days_from_peak_to_event": 9,
                "days_from_end_to_event": 8,
                "event_within_horizon_after_start": True,
                "event_within_horizon_after_peak": True,
                "event_within_horizon_after_end": True,
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
                "peak_time_index": 0,
                "start_date": "2026-03-01",
                "end_date": "2026-03-01",
                "peak_date": "2026-03-01",
                "snapshot_count": 1,
                "duration_days": 0,
                "peak_risk": 0.80,
                "mean_risk": 0.80,
                "event_observed": False,
                "injury_type": "censored",
                "days_from_start_to_event": None,
                "days_from_peak_to_event": None,
                "days_from_end_to_event": None,
                "event_within_horizon_after_start": False,
                "event_within_horizon_after_peak": False,
                "event_within_horizon_after_end": False,
                "top_model_features": [
                    {"feature": "edge_density", "mean_abs_contribution": 0.5}
                ],
                "elevated_z_features": [],
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "horizon_days": 30,
                "threshold_kind": "percentile",
                "threshold_value": 0.10,
                "start_time_index": 0,
                "end_time_index": 1,
                "peak_time_index": 0,
                "start_date": "2026-01-01",
                "end_date": "2026-01-02",
                "peak_date": "2026-01-01",
                "snapshot_count": 2,
                "duration_days": 1,
                "peak_risk": 0.90,
                "mean_risk": 0.85,
                "event_observed": True,
                "injury_type": "hamstring",
                "days_from_start_to_event": 9,
                "days_from_peak_to_event": 9,
                "days_from_end_to_event": 8,
                "event_within_horizon_after_start": True,
                "event_within_horizon_after_peak": True,
                "event_within_horizon_after_end": True,
                "top_model_features": [
                    {"feature": "mean_abs_correlation", "mean_abs_contribution": 0.7}
                ],
                "elevated_z_features": ["z_mean_abs_correlation"],
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "horizon_days": 30,
                "threshold_kind": "percentile",
                "threshold_value": 0.10,
                "start_time_index": 0,
                "end_time_index": 0,
                "peak_time_index": 0,
                "start_date": "2026-02-01",
                "end_date": "2026-02-01",
                "peak_date": "2026-02-01",
                "snapshot_count": 1,
                "duration_days": 0,
                "peak_risk": 0.60,
                "mean_risk": 0.60,
                "event_observed": True,
                "injury_type": "ankle",
                "days_from_start_to_event": 19,
                "days_from_peak_to_event": 19,
                "days_from_end_to_event": 19,
                "event_within_horizon_after_start": True,
                "event_within_horizon_after_peak": True,
                "event_within_horizon_after_end": True,
                "top_model_features": [
                    {"feature": "graph_instability", "mean_abs_contribution": 0.3}
                ],
                "elevated_z_features": [],
            },
        ]
    )


def test_build_alert_episode_quality_reports_capture_burden_and_false_positive_metrics():
    quality = build_alert_episode_quality(_episodes(), _timeline())

    row = next(
        row
        for row in quality["quality_rows"]
        if row["horizon_days"] == 30 and row["threshold_value"] == 0.05
    )

    assert row["episode_count"] == 2
    assert row["true_positive_episode_count"] == 1
    assert row["true_positive_episode_rate"] == pytest.approx(0.5)
    assert row["false_positive_episode_count"] == 1
    assert row["false_positive_episode_rate"] == pytest.approx(0.5)
    assert row["unique_observed_event_count"] == 2
    assert row["unique_captured_event_count"] == 1
    assert row["unique_event_capture_rate"] == pytest.approx(0.5)
    assert row["missed_event_count"] == 1
    assert row["episodes_per_athlete_season"] == pytest.approx(2 / 3)
    assert row["median_start_lead_days"] == pytest.approx(9.0)
    assert row["median_peak_lead_days"] == pytest.approx(9.0)
    assert row["median_end_lead_days"] == pytest.approx(8.0)
    assert row["true_positive_median_peak_risk"] == pytest.approx(0.90)
    assert row["false_positive_median_peak_risk"] == pytest.approx(0.80)
    assert row["true_positive_elevated_z_episode_rate"] == pytest.approx(1.0)
    assert row["false_positive_elevated_z_episode_rate"] == pytest.approx(0.0)
    assert row["true_positive_top_model_feature_counts"] == {"mean_abs_correlation": 1}
    assert row["false_positive_top_model_feature_counts"] == {"edge_density": 1}


def test_build_alert_episode_quality_reports_threshold_overlap():
    quality = build_alert_episode_quality(_episodes(), _timeline())

    overlap = quality["threshold_overlaps"][0]

    assert overlap == {
        "horizon_days": 30,
        "threshold_a": "percentile:0.05",
        "threshold_b": "percentile:0.1",
        "threshold_a_episode_count": 2,
        "threshold_b_episode_count": 2,
        "overlap_episode_count": 1,
        "threshold_a_overlap_rate": 0.5,
        "threshold_b_overlap_rate": 0.5,
    }


def test_build_alert_episode_quality_reports_representative_cases():
    quality = build_alert_episode_quality(_episodes(), _timeline())

    cases = quality["representative_cases"]["30"]["percentile:0.05"]

    assert cases["true_positive_episode"]["athlete_id"] == "a1"
    assert cases["true_positive_episode"]["days_from_start_to_event"] == 9
    assert cases["false_positive_episode"]["athlete_id"] == "a3"
    assert cases["missed_injury"] == {
        "athlete_id": "a2",
        "season_id": "2026",
        "event_date": "2026-02-20",
        "injury_type": "ankle",
    }
    assert cases["high_intra_individual_deviation_episode"]["athlete_id"] == "a1"
    assert cases["high_intra_individual_deviation_episode"]["elevated_z_features"] == [
        "z_mean_abs_correlation"
    ]
