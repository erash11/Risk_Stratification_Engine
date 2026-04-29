import pandas as pd

from risk_stratification_engine.shadow_mode import (
    DEFAULT_SHADOW_MODE_CHANNELS,
    build_shadow_mode_stability_audit,
)


def test_default_shadow_mode_channels_encode_current_policy_package():
    channels = {channel["channel_name"]: channel for channel in DEFAULT_SHADOW_MODE_CHANNELS}

    assert channels["broad_30d"]["policy_name"] == "exclude_concussion"
    assert channels["broad_30d"]["graph_window_size"] == 4
    assert channels["broad_30d"]["horizon_days"] == 30
    assert channels["broad_30d"]["threshold_value"] == 0.05
    assert channels["severity_7d"]["policy_name"] == "model_safe_time_loss"
    assert channels["severity_14d"]["policy_name"] == "model_safe_time_loss"
    assert channels["subtype_lower_extremity_soft_tissue_30d"]["graph_window_size"] == 2


def test_build_shadow_mode_stability_audit_summarizes_channel_variability():
    rows = pd.DataFrame(
        [
            _row("broad_30d", "2024", capture=0.20, burden=0.40, captured=4),
            _row("broad_30d", "2025", capture=0.18, burden=0.45, captured=5),
            _row("severity_7d", "2024", capture=0.05, burden=1.20, captured=1),
            _row("severity_7d", "2025", capture=0.25, burden=1.10, captured=6),
        ]
    )

    audit = build_shadow_mode_stability_audit(rows)
    summary = pd.DataFrame(audit["channel_summaries"]).set_index("channel_name")

    assert audit["experiment_type"] == "shadow_mode_policy_stability"
    assert audit["slice_count"] == 2
    assert summary.loc["broad_30d", "stability_status"] == "stable"
    assert summary.loc["broad_30d", "capture_rate_range"] == 0.02
    assert summary.loc["severity_7d", "stability_status"] == "unstable"
    assert audit["overall_recommendation"] == "review_before_shadow_pilot"


def _row(channel, season, *, capture, burden, captured):
    return {
        "channel_name": channel,
        "slice_type": "season",
        "slice_id": season,
        "policy_name": "fixture_policy",
        "graph_window_size": 4,
        "horizon_days": 30,
        "threshold": "percentile:0.05",
        "unique_observed_event_count": 20,
        "unique_captured_event_count": captured,
        "unique_event_capture_rate": capture,
        "episode_count": 10,
        "episodes_per_athlete_season": burden,
        "median_start_lead_days": 7,
    }
