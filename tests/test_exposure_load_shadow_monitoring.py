import json

from risk_stratification_engine.exposure_load_shadow_monitoring import (
    build_exposure_load_shadow_monitoring_plan,
    clean_shadow_monitoring_rows,
    write_exposure_load_shadow_monitoring_plan_report,
)


def test_shadow_monitoring_plan_keeps_retained_channels_and_blocks_product(
    tmp_path,
):
    plan = build_exposure_load_shadow_monitoring_plan(_decision_package())

    assert plan["experiment_type"] == "exposure_load_shadow_monitoring_plan_sprint"
    assert plan["overall_recommendation"] == "launch_retained_channel_shadow_monitoring"
    assert plan["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert plan["retained_channels"] == ["broad_30d", "severity_14d"]
    assert plan["paused_or_revision_channels"] == ["severity_7d"]

    retained = {row["channel_name"]: row for row in plan["retained_channel_rows"]}
    assert retained["broad_30d"]["collection_unit"] == (
        "complete source-eligible athlete-season"
    )
    assert retained["broad_30d"]["minimum_new_review_packets"] == 4
    assert retained["broad_30d"]["evidence_gate"] == (
        "prospective_shadow_review_before_calibration"
    )

    paused = plan["paused_channel_rows"]
    assert paused == [
        {
            "channel_name": "severity_7d",
            "monitoring_status": "pause_or_revise",
            "required_action": "revise_threshold_or_channel_definition",
            "reason": "completed packets did not show useful, source-trustworthy, actionable evidence",
        }
    ]

    gates = {row["gate_name"]: row for row in plan["evidence_gate_rows"]}
    assert gates["prospective_shadow_review"]["gate_status"] == "required"
    assert gates["probability_calibration"]["gate_status"] == "blocked"
    assert gates["pilot_dashboard_readiness"]["gate_status"] == "blocked"

    report_path = tmp_path / "monitoring.md"
    write_exposure_load_shadow_monitoring_plan_report(report_path, plan)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Monitoring Plan Sprint" in report
    assert "launch_retained_channel_shadow_monitoring" in report
    assert "not probability calibration or dashboard clearance" in report

    json.dumps(plan, allow_nan=False)
    json.dumps(clean_shadow_monitoring_rows(plan["retained_channel_rows"]), allow_nan=False)


def _decision_package() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_shadow_adjudication_decision_sprint",
        "overall_recommendation": "continue_shadow_monitoring_with_channel_revisions",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "continued_shadow_channels": ["broad_30d", "severity_14d"],
        "paused_or_revision_channels": ["severity_7d"],
        "channel_decision_rows": [
            {
                "channel_name": "broad_30d",
                "complete_valid_rows": 4,
                "useful_source_ok_actionable_rows": 2,
                "channel_decision": "continue_shadow_monitoring",
                "decision_rationale": "multiple completed packets were useful, source-trustworthy, and actionable",
            },
            {
                "channel_name": "severity_14d",
                "complete_valid_rows": 4,
                "useful_source_ok_actionable_rows": 2,
                "channel_decision": "continue_shadow_monitoring",
                "decision_rationale": "multiple completed packets were useful, source-trustworthy, and actionable",
            },
            {
                "channel_name": "severity_7d",
                "complete_valid_rows": 4,
                "useful_source_ok_actionable_rows": 0,
                "channel_decision": "pause_or_revise_before_more_collection",
                "decision_rationale": "completed packets did not show useful, source-trustworthy, actionable evidence",
            },
        ],
    }
