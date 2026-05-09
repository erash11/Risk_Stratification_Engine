import json

from risk_stratification_engine.exposure_load_guardrail_policy import (
    build_exposure_load_guardrail_policy,
    write_exposure_load_guardrail_policy_report,
)


def test_guardrail_policy_blocks_probability_use_after_over_sharpening(tmp_path):
    policy = build_exposure_load_guardrail_policy(
        _failure_mode_summary(),
        _diagnostic_rows(),
    )

    assert policy["experiment_type"] == "exposure_load_guardrail_policy_sprint"
    assert policy["overall_recommendation"] == "use_exposure_load_for_shadow_ranking_only"
    assert policy["production_readiness"] == "not_ready_for_probability_or_pilot"

    decisions = {
        row["guardrail_domain"]: row["decision"]
        for row in policy["guardrail_rows"]
    }
    assert decisions["probability_calibration"] == "blocked_until_failure_mode_resolved"
    assert decisions["ranking_triage"] == "allowed_for_shadow_research"
    assert decisions["minute_load_expansion"] == "defer"
    assert decisions["feature_domain_review"] == "required_before_next_model_expansion"

    report_path = tmp_path / "guardrail.md"
    write_exposure_load_guardrail_policy_report(report_path, policy)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Guardrail Policy Sprint" in report
    assert "shadow ranking" in report
    assert "not production or pilot clearance" in report

    json.dumps(policy, allow_nan=False)


def _failure_mode_summary() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_failure_mode_sprint",
        "overall_recommendation": "inspect_exposure_feature_shift_drivers",
        "failure_seasons": ["2024-2025"],
        "top_driver_features": [
            {
                "feature_name": "exposure_game_events_28d",
                "feature_domain": "game_exposure",
                "shift_direction": "elevated_in_failure",
                "failure_mean": 3.5,
                "comparator_mean": 1.0,
            }
        ],
        "domain_shift_summary": [
            {
                "feature_domain": "game_exposure",
                "shifted_feature_count": 1,
                "mean_abs_shift": 2.5,
            }
        ],
    }


def _diagnostic_rows() -> list[dict[str, object]]:
    return [
        {
            "test_season_id": "2024-2025",
            "horizon_days": 30,
            "diagnostic_label": "ranking_triage_gain_calibration_loss",
            "target_reason": "over_sharpened_probability_slice",
            "delta_roc_auc": 0.05,
            "delta_brier_skill_score": -0.54,
            "delta_top_decile_lift": 0.95,
        },
        {
            "test_season_id": "2025-2026",
            "horizon_days": 30,
            "diagnostic_label": "calibration_supported",
            "target_reason": "forward_calibration_comparator",
            "delta_roc_auc": 0.08,
            "delta_brier_skill_score": 0.04,
            "delta_top_decile_lift": 0.74,
        },
    ]
