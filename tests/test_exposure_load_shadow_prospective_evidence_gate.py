import json

from risk_stratification_engine.exposure_load_shadow_prospective_evidence_gate import (
    build_exposure_load_shadow_prospective_evidence_gate,
    clean_shadow_prospective_evidence_gate_rows,
    write_exposure_load_shadow_prospective_evidence_gate_report,
)


def test_prospective_evidence_gate_sets_collection_targets_and_blocks_claims(
    tmp_path,
):
    gate = build_exposure_load_shadow_prospective_evidence_gate(
        _stress_test_payload()
    )

    assert gate["experiment_type"] == (
        "exposure_load_shadow_prospective_evidence_gate_sprint"
    )
    assert gate["overall_recommendation"] == (
        "collect_prospective_retained_channel_evidence_before_retesting"
    )
    assert gate["milestone_status"] == (
        "prospective_evidence_collection_gate_defined"
    )
    assert gate["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert gate["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert gate["pilot_dashboard_readiness"] == "blocked"
    assert gate["load_modification_readiness"] == "blocked"

    targets = {row["channel_name"]: row for row in gate["collection_target_rows"]}
    assert targets["broad_30d"]["minimum_new_prospective_packets"] == 4
    assert targets["broad_30d"]["minimum_captured_events_needed"] == 8
    assert targets["broad_30d"]["maximum_allowed_missed_event_rate"] == 0.75
    assert targets["broad_30d"]["target_decision"] == (
        "collect_prospective_evidence_then_retest"
    )
    assert targets["severity_14d"]["minimum_captured_events_needed"] == 8

    packet_targets = {
        (row["channel_name"], row["target_type"]): row
        for row in gate["packet_target_rows"]
    }
    assert packet_targets[("broad_30d", "monitoring_context_packet")][
        "minimum_packet_count"
    ] == 2
    assert packet_targets[("broad_30d", "missed_only_error_packet")][
        "minimum_packet_count"
    ] == 1
    assert packet_targets[("severity_14d", "outcome_context_packet")][
        "required_action"
    ] == "capture_outcome_context_or_mark_unavailable"

    gates = {
        (row["channel_name"], row["gate_name"]): row
        for row in gate["evidence_gate_rows"]
    }
    assert gates[("broad_30d", "prospective_collection_required")][
        "gate_status"
    ] == "required"
    assert gates[("broad_30d", "calibration_claim_boundary")][
        "gate_status"
    ] == "blocked"
    assert gates[("severity_14d", "retest_after_collection")][
        "gate_status"
    ] == "pending"

    report_path = tmp_path / "gate.md"
    write_exposure_load_shadow_prospective_evidence_gate_report(report_path, gate)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Prospective Evidence Gate Sprint" in report
    assert "collect_prospective_retained_channel_evidence_before_retesting" in report
    assert "not calibration claims" in report
    assert "load modification" in report

    json.dumps(gate, allow_nan=False)
    json.dumps(
        clean_shadow_prospective_evidence_gate_rows(
            gate["collection_target_rows"]
        ),
        allow_nan=False,
    )


def test_prospective_evidence_gate_closes_when_no_retained_channels_need_collection():
    payload = _stress_test_payload()
    payload["channel_stress_rows"] = []

    gate = build_exposure_load_shadow_prospective_evidence_gate(payload)

    assert gate["overall_recommendation"] == (
        "close_limited_finding_no_retained_channels_for_collection"
    )
    assert gate["milestone_status"] == "limited_finding_closeout_ready"
    assert gate["collection_target_rows"] == []


def _stress_test_payload():
    return {
        "experiment_type": (
            "exposure_load_shadow_bounded_calibration_stress_test_sprint"
        ),
        "overall_recommendation": (
            "preserve_limited_calibration_finding_and_collect_more_prospective_evidence"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "pilot_dashboard_readiness": "blocked",
        "load_modification_readiness": "blocked",
        "stress_test_status": "completed_descriptive_stress_test_not_calibration_claim",
        "channel_stress_rows": [
            _channel_stress_row("broad_30d", 234, 37, 197, 0.15812, 0.84188),
            _channel_stress_row("severity_14d", 129, 18, 111, 0.139535, 0.860465),
        ],
        "stress_scenario_rows": [
            _scenario_row("broad_30d", "monitoring_context_only", 2, 163, 37, 126),
            _scenario_row("broad_30d", "missed_only_error_bound", 1, 71, 0, 71),
            _scenario_row("severity_14d", "monitoring_context_only", 2, 93, 18, 75),
            _scenario_row("severity_14d", "missed_only_error_bound", 1, 36, 0, 36),
        ],
        "stress_gate_rows": [],
    }


def _channel_stress_row(
    channel_name,
    observed,
    captured,
    missed,
    capture_rate,
    missed_rate,
):
    return {
        "channel_name": channel_name,
        "observed_event_count": observed,
        "captured_event_count": captured,
        "missed_event_count": missed,
        "capture_rate": capture_rate,
        "missed_event_rate": missed_rate,
        "stress_classification": "high_miss_limited_calibration_signal",
        "stress_decision": "preserve_limited_monitoring_value_collect_more_evidence",
        "calibration_claim_status": "blocked",
        "probability_output_status": "blocked",
        "load_modification_status": "blocked",
    }


def _scenario_row(channel_name, scenario_name, packets, observed, captured, missed):
    return {
        "channel_name": channel_name,
        "scenario_name": scenario_name,
        "packet_count": packets,
        "observed_event_count": observed,
        "captured_event_count": captured,
        "missed_event_count": missed,
        "capture_rate": 0 if observed == 0 else round(captured / observed, 6),
        "missed_event_rate": 0 if observed == 0 else round(missed / observed, 6),
    }
