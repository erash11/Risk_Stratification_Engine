import json

from risk_stratification_engine.exposure_load_shadow_collection import (
    build_exposure_load_shadow_collection_evidence_prefill,
    build_exposure_load_shadow_collection_packet_workflow,
    build_exposure_load_shadow_collection_template,
    build_exposure_load_shadow_collection_summary,
    clean_shadow_collection_rows,
    write_exposure_load_shadow_collection_evidence_prefill_report,
    write_exposure_load_shadow_collection_packet_workflow_report,
    write_exposure_load_shadow_collection_reviewer_instructions,
    write_exposure_load_shadow_collection_summary_report,
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


def test_shadow_collection_summary_validates_prospective_rows_and_gates_calibration_review(
    tmp_path,
):
    summary = build_exposure_load_shadow_collection_summary(
        [
            {
                "collection_packet_id": "broad_30d__prospective_001",
                "channel_name": "broad_30d",
                "packet_sequence": 1,
                "collection_season_id": "2026-2027",
                "packet_start_date": "2026-08-01",
                "packet_end_date": "2026-12-01",
                "source_eligible": "true",
                "episode_count": "3",
                "unique_observed_event_count": "1",
                "unique_captured_event_count": "1",
                "alert_usefulness": "useful",
                "outcome_confirmed": "true",
                "source_context_ok": "true",
                "action_taken": "monitor",
                "reviewer_id": "ER1",
                "review_date": "2026-12-15",
                "notes": "Prospective retained-channel packet.",
                "collection_status": "complete",
            },
            {
                "collection_packet_id": "broad_30d__prospective_002",
                "channel_name": "broad_30d",
                "packet_sequence": 2,
                "collection_season_id": "2026-2027",
                "packet_start_date": "2026-12-02",
                "packet_end_date": "2027-02-01",
                "source_eligible": "false",
                "episode_count": "0",
                "unique_observed_event_count": "0",
                "unique_captured_event_count": "0",
                "alert_usefulness": "unclear",
                "outcome_confirmed": "false",
                "source_context_ok": "false",
                "action_taken": "none",
                "reviewer_id": "ER1",
                "review_date": "2027-02-15",
                "notes": "Not source eligible.",
                "collection_status": "complete",
            },
            {
                "collection_packet_id": "severity_14d__prospective_001",
                "channel_name": "severity_14d",
                "packet_sequence": 1,
                "collection_season_id": "",
                "packet_start_date": "bad-date",
                "packet_end_date": "2026-12-01",
                "source_eligible": "maybe",
                "episode_count": "-1",
                "unique_observed_event_count": "0",
                "unique_captured_event_count": "1",
                "alert_usefulness": "maybe",
                "outcome_confirmed": "",
                "source_context_ok": "true",
                "action_taken": "none",
                "reviewer_id": "",
                "review_date": "",
                "notes": "",
                "collection_status": "pending_collection",
            },
        ]
    )

    assert summary["experiment_type"] == "exposure_load_shadow_collection_summary"
    assert summary["overall_recommendation"] == (
        "complete_shadow_collection_before_calibration_readiness_review"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert summary["calibration_readiness"] == "not_ready_for_calibration_claims"
    assert summary["total_rows"] == 3
    assert summary["complete_valid_rows"] == 2
    assert summary["pending_or_invalid_rows"] == 1
    assert summary["complete_source_eligible_rows"] == 1
    assert summary["useful_source_ok_actionable_rows"] == 1

    validation_rows = summary["validation_rows"]
    assert validation_rows[0]["completion_status"] == "complete_valid"
    assert validation_rows[2]["completion_status"] == "pending_or_invalid"
    assert validation_rows[2]["missing_required_fields"] == (
        "collection_season_id,reviewer_id,review_date"
    )
    assert validation_rows[2]["invalid_fields"] == (
        "packet_start_date,source_eligible,episode_count,"
        "unique_captured_event_count,alert_usefulness"
    )

    channel_rows = {
        row["channel_name"]: row
        for row in summary["channel_summary_rows"]
    }
    assert channel_rows["broad_30d"]["complete_valid_rows"] == 2
    assert channel_rows["broad_30d"]["complete_source_eligible_rows"] == 1
    assert channel_rows["broad_30d"]["minimum_required_packets"] == 2
    assert channel_rows["broad_30d"]["calibration_review_gate"] == (
        "continue_collection"
    )

    report_path = tmp_path / "collection_summary.md"
    write_exposure_load_shadow_collection_summary_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Collection Summary Sprint" in report
    assert "Complete valid rows: 2" in report
    assert "not probability calibration or dashboard clearance" in report

    json.dumps(summary, allow_nan=False)


def test_shadow_collection_packet_workflow_creates_reviewer_materials_without_claiming_readiness(
    tmp_path,
):
    workflow = build_exposure_load_shadow_collection_packet_workflow(
        [
            {
                "collection_packet_id": "broad_30d__prospective_001",
                "channel_name": "broad_30d",
                "packet_sequence": 1,
                "collection_unit": "complete source-eligible athlete-season",
                "evidence_gate": "prospective_shadow_review_before_calibration",
                "source_rule": "stop if source eligibility fails or alert burden exceeds policy cap",
                "collection_status": "pending_collection",
            },
            {
                "collection_packet_id": "severity_14d__prospective_001",
                "channel_name": "severity_14d",
                "packet_sequence": 1,
                "collection_unit": "complete source-eligible athlete-season",
                "evidence_gate": "prospective_shadow_review_before_calibration",
                "source_rule": "stop if source eligibility fails or alert burden exceeds policy cap",
                "collection_status": "pending_collection",
            },
        ]
    )

    assert workflow["experiment_type"] == (
        "exposure_load_shadow_collection_packet_workflow_sprint"
    )
    assert workflow["overall_recommendation"] == (
        "prepare_retained_channel_reviewer_packets"
    )
    assert workflow["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert workflow["calibration_readiness"] == "not_ready_for_calibration_claims"
    assert workflow["packet_count"] == 2

    manifest_rows = workflow["packet_manifest_rows"]
    assert manifest_rows[0]["collection_packet_id"] == "broad_30d__prospective_001"
    assert manifest_rows[0]["packet_filename"] == (
        "review_packets/broad_30d__prospective_001.md"
    )
    assert manifest_rows[0]["packet_status"] == (
        "ready_for_reviewer_evidence_collection"
    )

    checklist_items = {
        row["checklist_item"]
        for row in workflow["packet_checklist_rows"]
        if row["collection_packet_id"] == "broad_30d__prospective_001"
    }
    assert {
        "confirm_source_eligibility",
        "record_alert_usefulness",
        "record_action_taken",
        "preserve_deidentified_notes",
    }.issubset(checklist_items)

    audit_rows = workflow["packet_audit_trail_rows"]
    assert audit_rows[0]["audit_event"] == "packet_created_for_review"
    assert audit_rows[0]["evidence_status"] == "not_collected"

    packet_documents = workflow["packet_documents"]
    assert packet_documents[0]["packet_filename"] == (
        "review_packets/broad_30d__prospective_001.md"
    )
    assert "Collection Packet: broad_30d__prospective_001" in packet_documents[0]["content"]
    assert "probability calibration or dashboard clearance" in packet_documents[0]["content"]
    assert "source_eligible" in packet_documents[0]["content"]

    instructions_path = tmp_path / "instructions.md"
    write_exposure_load_shadow_collection_reviewer_instructions(
        instructions_path,
        workflow,
    )
    instructions = instructions_path.read_text(encoding="utf-8")
    assert "Reviewer Instructions" in instructions
    assert "Do not enter identifiable athlete information" in instructions

    report_path = tmp_path / "packet_workflow.md"
    write_exposure_load_shadow_collection_packet_workflow_report(
        report_path,
        workflow,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Collection Packet Workflow Sprint" in report
    assert "Reviewer packet count: 2" in report

    json.dumps(workflow, allow_nan=False)


def test_shadow_collection_evidence_prefill_uses_replay_fields_and_leaves_judgment_blank(
    tmp_path,
):
    prefill = build_exposure_load_shadow_collection_evidence_prefill(
        [
            {
                "review_packet_id": "broad_30d__2023-2024",
                "channel_name": "broad_30d",
                "test_season_id": "2023-2024",
                "source_eligible": True,
                "episode_count": 109,
                "unique_observed_event_count": 85,
                "unique_captured_event_count": 20,
                "replay_status": "ready_for_research_adjudication",
                "minimum_review_unit": "complete source-eligible athlete-season",
            },
            {
                "review_packet_id": "broad_30d__2024-2025",
                "channel_name": "broad_30d",
                "test_season_id": "2024-2025",
                "source_eligible": False,
                "episode_count": 98,
                "unique_observed_event_count": 64,
                "unique_captured_event_count": 13,
                "replay_status": "source_ineligible_stop",
                "minimum_review_unit": "complete source-eligible athlete-season",
            },
            {
                "review_packet_id": "severity_14d__2025-2026",
                "channel_name": "severity_14d",
                "test_season_id": "2025-2026",
                "episode_count": 161,
                "unique_observed_event_count": 47,
                "unique_captured_event_count": 7,
                "review_packet_status": "ready_for_research_adjudication",
                "minimum_review_unit": "complete source-eligible athlete-season",
            },
            {
                "review_packet_id": "severity_7d__2025-2026",
                "channel_name": "severity_7d",
                "test_season_id": "2025-2026",
                "source_eligible": True,
                "episode_count": 124,
                "unique_observed_event_count": 47,
                "unique_captured_event_count": 6,
                "replay_status": "ready_for_research_adjudication",
                "minimum_review_unit": "complete source-eligible athlete-season",
            },
        ]
    )

    assert prefill["experiment_type"] == (
        "exposure_load_shadow_collection_evidence_prefill_sprint"
    )
    assert prefill["overall_recommendation"] == (
        "review_prefilled_retained_channel_shadow_collection_rows"
    )
    assert prefill["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert prefill["calibration_readiness"] == "not_ready_for_calibration_claims"
    assert prefill["prefilled_row_count"] == 2
    assert prefill["excluded_row_count"] == 2
    assert prefill["reviewer_required_field_count"] == 6

    rows = {
        row["collection_packet_id"]: row
        for row in prefill["prefilled_collection_rows"]
    }
    broad = rows["broad_30d__2023-2024"]
    assert broad["collection_season_id"] == "2023-2024"
    assert broad["packet_start_date"] == "2023-07-01"
    assert broad["packet_end_date"] == "2024-06-30"
    assert broad["source_eligible"] is True
    assert broad["episode_count"] == 109
    assert broad["unique_observed_event_count"] == 85
    assert broad["unique_captured_event_count"] == 20
    assert broad["alert_usefulness"] == ""
    assert broad["outcome_confirmed"] == ""
    assert broad["source_context_ok"] == ""
    assert broad["action_taken"] == ""
    assert broad["collection_status"] == "pending_reviewer_judgment"

    assert "broad_30d__2024-2025" in {
        row["review_packet_id"] for row in prefill["excluded_rows"]
    }
    assert "severity_7d__2025-2026" in {
        row["review_packet_id"] for row in prefill["excluded_rows"]
    }

    report_path = tmp_path / "prefill.md"
    write_exposure_load_shadow_collection_evidence_prefill_report(
        report_path,
        prefill,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Evidence Prefill Sprint" in report
    assert "Prefilled retained-channel rows: 2" in report
    assert "Reviewer fields still required: 6" in report

    json.dumps(prefill, allow_nan=False)


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
