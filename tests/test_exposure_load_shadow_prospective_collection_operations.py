import json

from risk_stratification_engine.exposure_load_shadow_prospective_collection_operations import (
    build_exposure_load_shadow_prospective_collection_operations,
    clean_shadow_prospective_collection_operation_rows,
    write_exposure_load_shadow_prospective_collection_operations_report,
    write_exposure_load_shadow_prospective_collection_reviewer_instructions,
)


def test_prospective_collection_operations_builds_reviewer_ready_packet_package(
    tmp_path,
):
    operations = build_exposure_load_shadow_prospective_collection_operations(
        _gate_payload()
    )

    assert operations["experiment_type"] == (
        "exposure_load_shadow_prospective_collection_operations_sprint"
    )
    assert operations["overall_recommendation"] == (
        "prepare_prospective_collection_operations_before_retest"
    )
    assert operations["milestone_status"] == (
        "reviewer_ready_prospective_packet_operations_defined"
    )
    assert operations["retest_readiness"] == (
        "pending_required_prospective_collection"
    )
    assert operations["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert operations["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert operations["pilot_dashboard_readiness"] == "blocked"
    assert operations["load_modification_readiness"] == "blocked"

    manifest = operations["packet_manifest_rows"]
    assert len(manifest) == 8
    assert manifest[0]["collection_packet_id"] == (
        "broad_30d__prospective_collection_001"
    )
    assert manifest[0]["target_type"] == "monitoring_context_packet"
    assert manifest[0]["packet_status"] == (
        "ready_for_prospective_practitioner_collection"
    )
    assert manifest[3]["target_type"] == "outcome_context_packet"
    assert manifest[-1]["collection_packet_id"] == (
        "severity_14d__prospective_collection_004"
    )

    worksheet = operations["collection_worksheet_rows"]
    assert worksheet[0]["collection_status"] == "pending_prospective_collection"
    assert worksheet[0]["reviewer_id"] == ""
    assert worksheet[0]["unique_captured_event_count"] == ""
    assert worksheet[0]["target_captured_events_needed"] == 8
    assert worksheet[0]["maximum_allowed_missed_event_rate"] == 0.75

    checklist_items = {
        row["checklist_item"]
        for row in operations["packet_checklist_rows"]
        if row["collection_packet_id"] == "broad_30d__prospective_collection_001"
    }
    assert {
        "confirm_source_eligibility",
        "record_prospective_window",
        "record_captured_and_missed_events",
        "complete_practitioner_adjudication",
        "preserve_no_claim_boundary",
    }.issubset(checklist_items)

    channel_plan = {
        row["channel_name"]: row for row in operations["channel_operation_rows"]
    }
    assert channel_plan["broad_30d"]["required_packet_count"] == 4
    assert channel_plan["broad_30d"]["required_captured_events"] == 8
    assert channel_plan["broad_30d"]["operation_status"] == (
        "ready_for_prospective_collection_not_retest"
    )

    audit_rows = operations["audit_trail_rows"]
    assert audit_rows[0]["audit_event"] == "prospective_packet_created"
    assert audit_rows[0]["evidence_status"] == "not_collected"

    instructions_path = tmp_path / "instructions.md"
    write_exposure_load_shadow_prospective_collection_reviewer_instructions(
        instructions_path,
        operations,
    )
    instructions = instructions_path.read_text(encoding="utf-8")
    assert "Prospective Collection Reviewer Instructions" in instructions
    assert "Do not enter identifiable athlete information" in instructions
    assert "Do not make calibration claims" in instructions

    report_path = tmp_path / "operations.md"
    write_exposure_load_shadow_prospective_collection_operations_report(
        report_path,
        operations,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Prospective Collection Operations Sprint" in report
    assert "Reviewer-ready packet count: 8" in report
    assert "not probability-facing output" in report

    json.dumps(operations, allow_nan=False)
    json.dumps(
        clean_shadow_prospective_collection_operation_rows(
            operations["collection_worksheet_rows"]
        ),
        allow_nan=False,
    )


def test_prospective_collection_operations_closes_when_gate_has_no_targets():
    payload = _gate_payload()
    payload["collection_target_rows"] = []
    payload["packet_target_rows"] = []

    operations = build_exposure_load_shadow_prospective_collection_operations(
        payload
    )

    assert operations["overall_recommendation"] == (
        "no_prospective_collection_operations_required"
    )
    assert operations["milestone_status"] == "no_retained_collection_targets"
    assert operations["packet_manifest_rows"] == []
    assert operations["collection_worksheet_rows"] == []


def _gate_payload():
    return {
        "experiment_type": "exposure_load_shadow_prospective_evidence_gate_sprint",
        "overall_recommendation": (
            "collect_prospective_retained_channel_evidence_before_retesting"
        ),
        "milestone_status": "prospective_evidence_collection_gate_defined",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "pilot_dashboard_readiness": "blocked",
        "load_modification_readiness": "blocked",
        "gate_definition": {
            "minimum_new_prospective_packets_per_channel": 4,
            "minimum_captured_events_per_channel": 8,
            "maximum_allowed_missed_event_rate": 0.75,
            "requires_practitioner_review": True,
        },
        "collection_target_rows": [
            _collection_target("broad_30d"),
            _collection_target("severity_14d"),
        ],
        "packet_target_rows": [
            _packet_target("broad_30d", "monitoring_context_packet", 2),
            _packet_target("broad_30d", "missed_only_error_packet", 1),
            _packet_target("broad_30d", "outcome_context_packet", 1),
            _packet_target("severity_14d", "monitoring_context_packet", 2),
            _packet_target("severity_14d", "missed_only_error_packet", 1),
            _packet_target("severity_14d", "outcome_context_packet", 1),
        ],
    }


def _collection_target(channel_name):
    return {
        "channel_name": channel_name,
        "prior_observed_event_count": 100,
        "prior_captured_event_count": 15,
        "prior_missed_event_count": 85,
        "minimum_new_prospective_packets": 4,
        "minimum_captured_events_needed": 8,
        "maximum_allowed_missed_event_rate": 0.75,
        "target_decision": "collect_prospective_evidence_then_retest",
    }


def _packet_target(channel_name, target_type, count):
    return {
        "channel_name": channel_name,
        "target_type": target_type,
        "minimum_packet_count": count,
        "required_action": f"review_{target_type}",
    }
