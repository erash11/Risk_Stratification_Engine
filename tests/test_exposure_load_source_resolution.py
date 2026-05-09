import json

from risk_stratification_engine.exposure_load_source_resolution import (
    build_exposure_load_source_resolution_policy,
    clean_source_resolution_rows,
    write_exposure_load_source_resolution_report,
)


def test_source_resolution_excludes_failed_season_until_source_is_resolved(tmp_path):
    summary = build_exposure_load_source_resolution_policy(
        source_context_classification=_source_context_classification()
    )

    assert summary["experiment_type"] == "exposure_load_source_resolution_sprint"
    assert summary["overall_recommendation"] == (
        "exclude_failed_season_from_probability_calibration_until_source_resolved"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert summary["failure_seasons"] == ["2024-2025"]

    policy = {row["policy_domain"]: row for row in summary["policy_rows"]}
    assert policy["season_eligibility"]["policy_decision"] == (
        "exclude_failed_season_from_probability_calibration"
    )
    assert policy["probability_calibration"]["policy_decision"] == (
        "blocked_pending_source_resolution"
    )
    assert policy["shadow_ranking"]["policy_decision"] == (
        "allowed_shadow_only_with_season_monitoring"
    )
    assert policy["model_expansion"]["policy_decision"] == (
        "blocked_pending_source_resolution"
    )
    assert policy["minute_load_expansion"]["policy_decision"] == (
        "deferred_until_source_resolution"
    )

    actions = {row["action_domain"]: row for row in summary["resolution_actions"]}
    assert actions["source_resolution"]["priority"] == "high"
    assert actions["availability_capture_repair"]["blocks"] == "model_expansion"
    assert actions["probability_dataset"]["blocks"] == "probability_calibration"

    report_path = tmp_path / "source_resolution.md"
    write_exposure_load_source_resolution_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Source Resolution Sprint" in report
    assert "complete athlete-season trajectories" in report
    assert "not pilot clearance" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(clean_source_resolution_rows(summary["policy_rows"]), allow_nan=False)


def _source_context_classification() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_source_context_classification_sprint",
        "overall_recommendation": (
            "treat_failed_season_as_schedule_roster_plus_capture_shift"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "failure_seasons": ["2024-2025"],
        "comparator_seasons": ["2023-2024", "2025-2026"],
        "source_context_classification_rows": [
            {
                "classification_domain": "managed_risk_support",
                "classification": "not_supported_by_source_flags",
                "confidence": "high",
                "evidence": (
                    "reduced_modified_availability_flagging; issue_linkage_absent"
                ),
                "required_next_step": (
                    "do not interpret lower availability flagging as managed risk"
                ),
            },
            {
                "classification_domain": "schedule_roster_context",
                "classification": "supported_schedule_roster_shift",
                "confidence": "high",
                "evidence": "review_failed_season_schedule_roster_shift",
                "required_next_step": (
                    "review game, training, lift, and participation-density changes"
                ),
            },
            {
                "classification_domain": "availability_capture_context",
                "classification": "supported_capture_or_documentation_shift",
                "confidence": "high",
                "evidence": "review_failed_season_availability_capture",
                "required_next_step": (
                    "review modified/no-participation documentation and source linkage"
                ),
            },
            {
                "classification_domain": "next_model_action",
                "classification": "do_not_expand_model_features",
                "confidence": "high",
                "evidence": (
                    "keep_shadow_ranking_and_resolve_context_before_model_expansion"
                ),
                "required_next_step": (
                    "resolve context classification before minute-load or probability work"
                ),
            },
        ],
        "source_evidence_rows": [
            {
                "evidence_domain": "source_schedule",
                "key_failure_value": 13,
                "key_comparator_value": 12,
                "review_signal": "elevated_game_schedule; reduced_lift_schedule",
            },
            {
                "evidence_domain": "source_participation_flags",
                "key_failure_value": 0.014,
                "key_comparator_value": 0.021,
                "review_signal": (
                    "reduced_modified_availability_flagging; issue_linkage_absent"
                ),
            },
        ],
    }
