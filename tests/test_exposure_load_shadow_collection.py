import json

from risk_stratification_engine.exposure_load_shadow_collection import (
    build_exposure_load_shadow_collection_template,
    clean_shadow_collection_rows,
    write_exposure_load_shadow_collection_template_report,
)


def test_shadow_collection_template_creates_required_packet_rows(tmp_path):
    template = build_exposure_load_shadow_collection_template(_monitoring_plan())

    assert template["experiment_type"] == (
        "exposure_load_shadow_collection_template_sprint"
    )
    assert template["overall_recommendation"] == (
        "collect_retained_channel_shadow_packets"
    )
    assert template["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert template["retained_channels"] == ["broad_30d", "severity_14d"]
    assert template["paused_or_revision_channels"] == ["severity_7d"]

    collection_rows = template["collection_template_rows"]
    assert len(collection_rows) == 8
    assert collection_rows[0]["collection_packet_id"] == "broad_30d__prospective_001"
    assert collection_rows[0]["channel_name"] == "broad_30d"
    assert collection_rows[0]["collection_status"] == "pending_collection"
    assert collection_rows[0]["packet_sequence"] == 1
    assert collection_rows[0]["alert_usefulness"] == ""
    assert collection_rows[-1]["collection_packet_id"] == (
        "severity_14d__prospective_004"
    )

    completion_rows = template["completion_check_rows"]
    assert completion_rows[0]["completion_status"] == "pending_required_fields"
    assert completion_rows[0]["missing_required_field_count"] == 10
    assert "reviewer_id" in completion_rows[0]["missing_required_fields"]

    schema_fields = {row["field_name"] for row in template["schema_rows"]}
    assert {
        "collection_packet_id",
        "channel_name",
        "collection_season_id",
        "source_eligible",
        "episode_count",
        "unique_observed_event_count",
        "unique_captured_event_count",
        "alert_usefulness",
        "source_context_ok",
        "action_taken",
    }.issubset(schema_fields)

    report_path = tmp_path / "collection.md"
    write_exposure_load_shadow_collection_template_report(report_path, template)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Collection Template Sprint" in report
    assert "collect_retained_channel_shadow_packets" in report
    assert "not probability calibration or dashboard clearance" in report

    json.dumps(template, allow_nan=False)
    json.dumps(clean_shadow_collection_rows(template["collection_template_rows"]), allow_nan=False)


def _monitoring_plan() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_shadow_monitoring_plan_sprint",
        "overall_recommendation": "launch_retained_channel_shadow_monitoring",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "retained_channels": ["broad_30d", "severity_14d"],
        "paused_or_revision_channels": ["severity_7d"],
        "retained_channel_rows": [
            {
                "channel_name": "broad_30d",
                "monitoring_status": "continue_shadow_monitoring",
                "collection_unit": "complete source-eligible athlete-season",
                "minimum_new_review_packets": 4,
                "review_cadence": "review after each complete source-eligible season",
                "evidence_gate": "prospective_shadow_review_before_calibration",
                "source_rule": "stop if source eligibility fails or alert burden exceeds policy cap",
            },
            {
                "channel_name": "severity_14d",
                "monitoring_status": "continue_shadow_monitoring",
                "collection_unit": "complete source-eligible athlete-season",
                "minimum_new_review_packets": 4,
                "review_cadence": "review after each complete source-eligible season",
                "evidence_gate": "prospective_shadow_review_before_calibration",
                "source_rule": "stop if source eligibility fails or alert burden exceeds policy cap",
            },
        ],
        "paused_channel_rows": [
            {
                "channel_name": "severity_7d",
                "monitoring_status": "pause_or_revise",
                "required_action": "revise_threshold_or_channel_definition",
                "reason": "completed packets did not show useful, source-trustworthy, actionable evidence",
            }
        ],
    }
