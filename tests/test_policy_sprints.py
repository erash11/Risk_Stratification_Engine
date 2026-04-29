import pandas as pd

from risk_stratification_engine.policy_sprints import (
    build_operational_policy_package,
    build_policy_window_sensitivity,
    build_two_channel_alert_policy,
)


def test_build_two_channel_alert_policy_selects_broad_and_severity_channels():
    rows = pd.DataFrame(
        [
            _row("any_injury", 30, "percentile:0.05", brier=0.012, capture=0.18),
            _row("exclude_concussion", 30, "percentile:0.05", brier=0.017, capture=0.17),
            _row("model_safe_time_loss", 7, "percentile:0.05", lift=3.1, capture=0.14),
            _row("model_safe_time_loss", 7, "percentile:0.1", lift=3.1, capture=0.19),
            _row("model_safe_time_loss", 14, "percentile:0.05", lift=2.3, capture=0.15),
            _row("model_safe_time_loss", 14, "percentile:0.1", lift=2.3, capture=0.24),
            _row("lower_extremity_soft_tissue", 30, "percentile:0.05", capture=0.20),
        ]
    )

    policy = build_two_channel_alert_policy(rows)

    assert policy["experiment_type"] == "two_channel_alert_policy"
    assert policy["channels"]["broad_early_warning"]["policy_name"] == (
        "exclude_concussion"
    )
    assert policy["channels"]["broad_early_warning"]["horizon_days"] == 30
    assert policy["channels"]["severity_short_horizon"]["policy_name"] == (
        "model_safe_time_loss"
    )
    assert policy["channels"]["severity_short_horizon"]["views"] == [
        {"horizon_days": 7, "threshold": "percentile:0.1"},
        {"horizon_days": 14, "threshold": "percentile:0.1"},
    ]
    assert policy["channels"]["subtype_review"]["policy_name"] == (
        "lower_extremity_soft_tissue"
    )


def test_build_policy_window_sensitivity_picks_window_per_channel():
    rows = pd.DataFrame(
        [
            _row("exclude_concussion", 30, "percentile:0.05", window=2, brier=0.010),
            _row("exclude_concussion", 30, "percentile:0.05", window=4, brier=0.017),
            _row(
                "model_safe_time_loss",
                7,
                "percentile:0.1",
                window=2,
                capture=0.21,
            ),
            _row(
                "model_safe_time_loss",
                7,
                "percentile:0.1",
                window=4,
                capture=0.18,
            ),
            _row(
                "lower_extremity_soft_tissue",
                30,
                "percentile:0.05",
                window=7,
                capture=0.24,
            ),
        ]
    )

    sensitivity = build_policy_window_sensitivity(rows)

    assert sensitivity["experiment_type"] == "policy_window_sensitivity"
    assert sensitivity["recommendations"]["broad_early_warning"]["graph_window_size"] == 4
    assert sensitivity["recommendations"]["severity_7d"]["graph_window_size"] == 2
    assert sensitivity["recommendations"]["subtype_30d"]["graph_window_size"] == 7


def test_build_operational_policy_package_marks_shadow_mode_and_next_sprint():
    two_channel = {
        "channels": {
            "broad_early_warning": {
                "policy_name": "exclude_concussion",
                "horizon_days": 30,
                "threshold": "percentile:0.05",
            },
            "severity_short_horizon": {
                "policy_name": "model_safe_time_loss",
                "views": [{"horizon_days": 14, "threshold": "percentile:0.1"}],
            },
        }
    }
    sensitivity = {
        "recommendations": {
            "broad_early_warning": {"graph_window_size": 4},
            "severity_14d": {"graph_window_size": 2},
        }
    }

    package = build_operational_policy_package(two_channel, sensitivity)

    assert package["experiment_type"] == "operational_policy_package"
    assert package["status"] == "research_shadow_mode"
    assert package["recommended_policy"]["broad_early_warning"]["policy_name"] == (
        "exclude_concussion"
    )
    assert "severe_time_loss" in package["not_recommended_primary_targets"]
    assert package["next_sprint"] == "shadow-mode policy stability audit"


def _row(
    policy_name,
    horizon,
    threshold,
    *,
    window=4,
    brier=0.0,
    lift=1.0,
    capture=0.0,
):
    return {
        "graph_window_size": window,
        "policy_name": policy_name,
        "policy_event_count": 100,
        "horizon_days": horizon,
        "threshold": threshold,
        "roc_auc": 0.6,
        "brier_skill_score": brier,
        "top_decile_lift": lift,
        "episode_count": 10,
        "unique_observed_event_count": 20,
        "unique_captured_event_count": int(capture * 20),
        "unique_event_capture_rate": capture,
        "missed_event_count": 20 - int(capture * 20),
        "episodes_per_athlete_season": 0.5,
        "median_start_lead_days": 7,
    }
