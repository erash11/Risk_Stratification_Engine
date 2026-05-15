import json

from risk_stratification_engine.exposure_load_shadow_bounded_calibration_protocol import (
    build_exposure_load_shadow_bounded_calibration_protocol,
    clean_shadow_bounded_calibration_protocol_rows,
    write_exposure_load_shadow_bounded_calibration_protocol_report,
)


def test_bounded_calibration_protocol_defines_research_only_analysis(tmp_path):
    protocol = build_exposure_load_shadow_bounded_calibration_protocol(
        _error_control_policy()
    )

    assert protocol["experiment_type"] == (
        "exposure_load_shadow_bounded_calibration_protocol_sprint"
    )
    assert protocol["overall_recommendation"] == (
        "run_bounded_calibration_stress_test_without_claims"
    )
    assert protocol["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert protocol["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert protocol["pilot_dashboard_readiness"] == "blocked"
    assert protocol["load_modification_readiness"] == "blocked"
    assert protocol["bounded_protocol_status"] == (
        "ready_for_research_only_stress_test_protocol"
    )
    assert protocol["protocol_definition"] == {
        "allowed_analysis": "descriptive_shadow_calibration_stress_test",
        "required_controls": [
            "miss_fraction_control",
            "monitoring_prediction_boundary",
            "packet_review_control",
        ],
        "monitoring_context_is_not_calibration_evidence": True,
        "probability_or_load_modification_allowed": False,
    }

    channels = {row["channel_name"]: row for row in protocol["channel_protocol_rows"]}
    assert channels["broad_30d"]["protocol_status"] == (
        "eligible_for_bounded_stress_test_not_claims"
    )
    assert channels["broad_30d"]["analysis_scope"] == (
        "monitoring_context_error_controlled_only"
    )
    assert channels["broad_30d"]["required_control_count"] == 3
    assert channels["broad_30d"]["next_analysis"] == (
        "bounded_calibration_stress_test_without_claims"
    )
    assert channels["severity_14d"]["protocol_status"] == (
        "eligible_for_bounded_stress_test_not_claims"
    )

    evidence = {
        row["collection_packet_id"]: row
        for row in protocol["evidence_use_rows"]
    }
    assert evidence["broad_30d__2021-2022"]["protocol_evidence_role"] == (
        "outcome_context_gap_excluded_from_calibration_signal"
    )
    assert evidence["broad_30d__2022-2023"]["protocol_evidence_role"] == (
        "missed_only_error_case_for_sensitivity_bounds"
    )
    assert evidence["broad_30d__2023-2024"]["protocol_evidence_role"] == (
        "monitoring_context_only_not_calibration_claim"
    )

    gates = {
        (row["channel_name"], row["gate_name"]): row
        for row in protocol["protocol_gate_rows"]
    }
    assert gates[("broad_30d", "controls_complete")]["gate_status"] == "pass"
    assert gates[("broad_30d", "claim_boundary")]["gate_status"] == "blocked"
    assert gates[("severity_14d", "load_modification_boundary")][
        "gate_status"
    ] == "blocked"

    report_path = tmp_path / "protocol.md"
    write_exposure_load_shadow_bounded_calibration_protocol_report(
        report_path,
        protocol,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Bounded Calibration Protocol Sprint" in report
    assert "descriptive shadow calibration stress test" in report
    assert "not calibration claims" in report
    assert "load modification" in report

    json.dumps(protocol, allow_nan=False)
    json.dumps(
        clean_shadow_bounded_calibration_protocol_rows(
            protocol["channel_protocol_rows"]
        ),
        allow_nan=False,
    )


def test_bounded_calibration_protocol_collects_more_rows_when_no_retained_channel():
    policy = _error_control_policy()
    policy["decision_rows"][0]["retained_research_candidate"] = False
    policy["decision_rows"][0]["next_gate_decision"] = (
        "collect_more_practitioner_reviewed_rows"
    )
    policy["decision_rows"][1]["retained_research_candidate"] = False
    policy["decision_rows"][1]["next_gate_decision"] = (
        "collect_more_practitioner_reviewed_rows"
    )

    protocol = build_exposure_load_shadow_bounded_calibration_protocol(policy)

    assert protocol["overall_recommendation"] == (
        "collect_more_practitioner_reviewed_rows_before_stress_test"
    )
    assert protocol["bounded_protocol_status"] == (
        "not_ready_for_bounded_calibration_protocol"
    )
    assert protocol["channel_protocol_rows"][0]["protocol_status"] == (
        "not_eligible_collect_more_practitioner_rows"
    )


def _error_control_policy():
    return {
        "experiment_type": "exposure_load_shadow_error_control_sprint",
        "overall_recommendation": (
            "continue_bounded_calibration_research_with_error_controls_not_claims"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "pilot_dashboard_readiness": "blocked",
        "load_modification_readiness": "blocked",
        "bounded_error_control_status": (
            "ready_for_error_controlled_research_decision_not_claims"
        ),
        "decision_rows": [
            _decision_row("broad_30d", 2),
            _decision_row("severity_14d", 3),
        ],
        "refined_evidence_dossier_rows": [
            _dossier_row("broad_30d", "2021-2022", 0, 0, 0, "none"),
            _dossier_row("broad_30d", "2022-2023", 71, 0, 71, "none"),
            _dossier_row("broad_30d", "2023-2024", 85, 20, 65, "monitor"),
            _dossier_row("severity_14d", "2022-2023", 36, 0, 36, "monitor"),
            _dossier_row("severity_14d", "2023-2024", 46, 11, 35, "monitor"),
        ],
        "error_control_rows": [
            _control_row("broad_30d", "miss_fraction_control"),
            _control_row("broad_30d", "monitoring_prediction_boundary"),
            _control_row("broad_30d", "packet_review_control"),
            _control_row("severity_14d", "miss_fraction_control"),
            _control_row("severity_14d", "monitoring_prediction_boundary"),
            _control_row("severity_14d", "packet_review_control"),
        ],
    }


def _decision_row(channel_name, useful_rows):
    return {
        "channel_name": channel_name,
        "retained_research_candidate": True,
        "useful_actionable_rows": useful_rows,
        "capture_rate": 0.15,
        "missed_event_rate": 0.85,
        "monitoring_usefulness_status": "supported_for_monitoring_context_only",
        "prediction_calibration_usefulness_status": (
            "not_established_high_miss_fraction"
        ),
        "error_control_status": "requires_high_miss_fraction_controls",
        "next_gate_decision": (
            "continue_bounded_calibration_research_with_error_controls"
        ),
        "allowed_use": "shadow_monitoring_review_only",
        "blocked_use": (
            "probability_output,calibration_claims,pilot_dashboard,"
            "autonomous_intervention,load_modification"
        ),
    }


def _dossier_row(channel_name, season_id, observed, captured, missed, action):
    return {
        "collection_packet_id": f"{channel_name}__{season_id}",
        "channel_name": channel_name,
        "collection_season_id": season_id,
        "alert_usefulness": "useful" if action == "monitor" else "unclear",
        "action_taken": action,
        "observed_event_count": observed,
        "captured_event_count": captured,
        "missed_event_count": missed,
        "dossier_refinement_action": (
            "collect_outcome_context_before_calibration_weight"
            if observed == 0
            else "review_missed_only_packet_before_channel_escalation"
            if captured == 0
            else "preserve_as_monitoring_useful_not_prediction_evidence"
        ),
        "monitoring_prediction_boundary": (
            "monitoring_useful_not_prediction_evidence"
            if action == "monitor"
            else "not_monitoring_supported"
        ),
    }


def _control_row(channel_name, control_name):
    return {
        "channel_name": channel_name,
        "control_name": control_name,
        "required": True,
        "control_action": f"apply_{control_name}",
    }
