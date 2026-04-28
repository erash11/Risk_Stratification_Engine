from __future__ import annotations

import pandas as pd
import pytest

from risk_stratification_engine.alert_episodes import (
    build_alert_episode_summary,
    build_alert_episodes,
)


def _timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-01-01",
                "risk_7d": 0.95,
                "risk_14d": 0.90,
                "risk_30d": 0.80,
                "days_to_event": 8,
                "event_observed": True,
                "injury_type": "hamstring",
                "top_feature_7d": "mean_abs_correlation",
                "top_contribution_7d": 0.70,
                "top_feature_14d": "mean_abs_correlation",
                "top_contribution_14d": 0.60,
                "top_feature_30d": "edge_density",
                "top_contribution_30d": -0.30,
                "elevated_z_features": ["z_mean_abs_correlation"],
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 1,
                "snapshot_date": "2026-01-02",
                "risk_7d": 0.90,
                "risk_14d": 0.85,
                "risk_30d": 0.75,
                "days_to_event": 7,
                "event_observed": True,
                "injury_type": "hamstring",
                "top_feature_7d": "edge_density",
                "top_contribution_7d": -0.50,
                "top_feature_14d": "mean_abs_correlation",
                "top_contribution_14d": 0.50,
                "top_feature_30d": "edge_density",
                "top_contribution_30d": -0.40,
                "elevated_z_features": ["z_edge_density"],
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 2,
                "snapshot_date": "2026-01-03",
                "risk_7d": 0.10,
                "risk_14d": 0.20,
                "risk_30d": 0.30,
                "days_to_event": 6,
                "event_observed": True,
                "injury_type": "hamstring",
                "top_feature_7d": "edge_count",
                "top_contribution_7d": 0.10,
                "top_feature_14d": "edge_count",
                "top_contribution_14d": 0.10,
                "top_feature_30d": "edge_count",
                "top_contribution_30d": 0.10,
                "elevated_z_features": [],
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 3,
                "snapshot_date": "2026-01-04",
                "risk_7d": 0.88,
                "risk_14d": 0.82,
                "risk_30d": 0.70,
                "days_to_event": 5,
                "event_observed": True,
                "injury_type": "hamstring",
                "top_feature_7d": "mean_abs_correlation",
                "top_contribution_7d": 0.40,
                "top_feature_14d": "edge_density",
                "top_contribution_14d": -0.30,
                "top_feature_30d": "edge_density",
                "top_contribution_30d": -0.20,
                "elevated_z_features": ["z_graph_instability"],
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-01-01",
                "risk_7d": 0.05,
                "risk_14d": 0.10,
                "risk_30d": 0.20,
                "days_to_event": 30,
                "event_observed": False,
                "injury_type": "none",
                "top_feature_7d": "edge_count",
                "top_contribution_7d": 0.10,
                "top_feature_14d": "edge_count",
                "top_contribution_14d": 0.10,
                "top_feature_30d": "edge_count",
                "top_contribution_30d": 0.10,
                "elevated_z_features": [],
            },
        ]
    )


def test_build_alert_episodes_collapses_contiguous_top_percentile_snapshots():
    episodes = build_alert_episodes(
        _timeline(),
        horizons=(7,),
        percentile_thresholds=(0.60,),
    )

    assert len(episodes) == 2
    first = episodes.iloc[0]
    assert first["athlete_id"] == "a1"
    assert first["horizon_days"] == 7
    assert first["threshold_kind"] == "percentile"
    assert first["threshold_value"] == pytest.approx(0.60)
    assert first["start_time_index"] == 0
    assert first["end_time_index"] == 1
    assert first["snapshot_count"] == 2
    assert first["peak_time_index"] == 0
    assert first["peak_risk"] == pytest.approx(0.95)

    second = episodes.iloc[1]
    assert second["start_time_index"] == 3
    assert second["end_time_index"] == 3
    assert second["snapshot_count"] == 1


def test_build_alert_episodes_marks_temporal_event_capture_without_future_leakage():
    episodes = build_alert_episodes(
        _timeline(),
        horizons=(7,),
        percentile_thresholds=(0.60,),
    )

    first = episodes.iloc[0]
    assert bool(first["event_within_horizon_after_start"]) is False
    assert bool(first["event_within_horizon_after_peak"]) is False
    assert bool(first["event_within_horizon_after_end"]) is True
    assert first["days_from_start_to_event"] == 8
    assert first["days_from_peak_to_event"] == 8
    assert first["days_from_end_to_event"] == 7
    assert first["injury_type"] == "hamstring"


def test_build_alert_episodes_leaves_event_timing_empty_for_censored_episodes():
    episodes = build_alert_episodes(
        _timeline(),
        horizons=(7,),
        percentile_thresholds=(1.0,),
    )

    censored = episodes.loc[episodes["athlete_id"] == "a2"].iloc[0]

    assert bool(censored["event_observed"]) is False
    assert pd.isna(censored["days_from_start_to_event"])
    assert pd.isna(censored["days_from_peak_to_event"])
    assert pd.isna(censored["days_from_end_to_event"])
    assert bool(censored["event_within_horizon_after_start"]) is False


def test_build_alert_episodes_rolls_up_model_and_intra_individual_explanations():
    episodes = build_alert_episodes(
        _timeline(),
        horizons=(7,),
        percentile_thresholds=(0.60,),
    )

    first = episodes.iloc[0]
    assert first["top_model_features"] == [
        {"feature": "mean_abs_correlation", "mean_abs_contribution": 0.7},
        {"feature": "edge_density", "mean_abs_contribution": 0.5},
    ]
    assert first["elevated_z_features"] == [
        "z_mean_abs_correlation",
        "z_edge_density",
    ]


def test_build_alert_episode_summary_counts_episodes_and_capture_rates():
    episodes = build_alert_episodes(
        _timeline(),
        horizons=(7,),
        percentile_thresholds=(0.60,),
    )

    summary = build_alert_episode_summary(episodes)

    assert summary["episode_count"] == 2
    assert summary["horizons"]["7"]["thresholds"]["percentile:0.6"] == {
        "episode_count": 2,
        "episode_with_event_after_start_count": 1,
        "episode_with_event_after_peak_count": 1,
        "episode_with_event_after_end_count": 2,
        "event_capture_after_start_rate": 0.5,
        "event_capture_after_peak_rate": 0.5,
        "event_capture_after_end_rate": 1.0,
        "median_snapshot_count": 1.5,
        "median_duration_days": 0.5,
    }
