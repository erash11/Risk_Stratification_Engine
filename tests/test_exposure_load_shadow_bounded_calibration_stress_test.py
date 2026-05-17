import json

from risk_stratification_engine.exposure_load_shadow_bounded_calibration_stress_test import (
    build_exposure_load_shadow_bounded_calibration_stress_test,
    clean_shadow_bounded_calibration_stress_test_rows,
    write_exposure_load_shadow_bounded_calibration_stress_test_report,
)


def test_bounded_calibration_stress_test_preserves_limited_finding(tmp_path):
    stress = build_exposure_load_shadow_bounded_calibration_stress_test(
        _protocol_payload()
    )

    assert stress["experiment_type"] == (
        "exposure_load_shadow_bounded_calibration_stress_test_sprint"
    )
    assert stress["overall_recommendation"] == (
        "preserve_limited_calibration_finding_and_collect_more_prospective_evidence"
    )
    assert stress["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert stress["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert stress["pilot_dashboard_readiness"] == "blocked"
    assert stress["load_modification_readiness"] == "blocked"
    assert stress["stress_test_status"] == (
        "completed_descriptive_stress_test_not_calibration_claim"
    )

    channels = {row["channel_name"]: row for row in stress["channel_stress_rows"]}
    assert channels["broad_30d"]["observed_event_count"] == 234
    assert channels["broad_30d"]["captured_event_count"] == 37
    assert channels["broad_30d"]["missed_event_count"] == 197
    assert channels["broad_30d"]["capture_rate"] == 0.15812
    assert channels["broad_30d"]["missed_event_rate"] == 0.84188
    assert channels["broad_30d"]["monitoring_context_capture_rate"] == 0.226994
    assert channels["broad_30d"]["stress_classification"] == (
        "high_miss_limited_calibration_signal"
    )
    assert channels["broad_30d"]["stress_decision"] == (
        "preserve_limited_monitoring_value_collect_more_evidence"
    )
    assert channels["severity_14d"]["capture_rate"] == 0.139535
    assert channels["severity_14d"]["stress_decision"] == (
        "preserve_limited_monitoring_value_collect_more_evidence"
    )

    scenarios = {
        (row["channel_name"], row["scenario_name"]): row
        for row in stress["stress_scenario_rows"]
    }
    assert scenarios[("broad_30d", "observed_replay_all")]["capture_rate"] == 0.15812
    assert scenarios[("broad_30d", "monitoring_context_only")][
        "packet_count"
    ] == 2
    assert scenarios[("severity_14d", "missed_only_error_bound")][
        "missed_event_count"
    ] == 36

    gates = {
        (row["channel_name"], row["gate_name"]): row
        for row in stress["stress_gate_rows"]
    }
    assert gates[("broad_30d", "stress_test_complete")]["gate_status"] == "pass"
    assert gates[("broad_30d", "calibration_claim_boundary")][
        "gate_status"
    ] == "blocked"
    assert gates[("severity_14d", "prospective_evidence_needed")][
        "gate_status"
    ] == "required"

    report_path = tmp_path / "stress_test.md"
    write_exposure_load_shadow_bounded_calibration_stress_test_report(
        report_path,
        stress,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Bounded Calibration Stress Test Sprint" in report
    assert "preserve_limited_calibration_finding" in report
    assert "not calibration claims" in report
    assert "load modification" in report

    json.dumps(stress, allow_nan=False)
    json.dumps(
        clean_shadow_bounded_calibration_stress_test_rows(
            stress["channel_stress_rows"]
        ),
        allow_nan=False,
    )


def test_bounded_calibration_stress_test_collects_protocol_when_not_eligible():
    payload = _protocol_payload()
    payload["channel_protocol_rows"][0]["protocol_status"] = (
        "not_eligible_collect_more_practitioner_rows"
    )
    payload["channel_protocol_rows"][1]["protocol_status"] = (
        "not_eligible_collect_more_practitioner_rows"
    )

    stress = build_exposure_load_shadow_bounded_calibration_stress_test(payload)

    assert stress["overall_recommendation"] == (
        "complete_bounded_protocol_before_stress_test"
    )
    assert stress["stress_test_status"] == "not_ready_for_stress_test"
    assert stress["channel_stress_rows"][0]["stress_decision"] == (
        "complete_protocol_before_stress_test"
    )


def _protocol_payload():
    return {
        "experiment_type": (
            "exposure_load_shadow_bounded_calibration_protocol_sprint"
        ),
        "overall_recommendation": "run_bounded_calibration_stress_test_without_claims",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "pilot_dashboard_readiness": "blocked",
        "load_modification_readiness": "blocked",
        "bounded_protocol_status": "ready_for_research_only_stress_test_protocol",
        "channel_protocol_rows": [
            _protocol_row("broad_30d"),
            _protocol_row("severity_14d"),
        ],
        "evidence_use_rows": [
            _evidence("broad_30d", "2021-2022", 0, 0, 0, "gap"),
            _evidence("broad_30d", "2022-2023", 71, 0, 71, "missed"),
            _evidence("broad_30d", "2023-2024", 85, 20, 65, "monitor"),
            _evidence("broad_30d", "2025-2026", 78, 17, 61, "monitor"),
            _evidence("severity_14d", "2021-2022", 0, 0, 0, "gap"),
            _evidence("severity_14d", "2022-2023", 36, 0, 36, "missed"),
            _evidence("severity_14d", "2023-2024", 46, 11, 35, "monitor"),
            _evidence("severity_14d", "2025-2026", 47, 7, 40, "monitor"),
        ],
        "protocol_gate_rows": [],
    }


def _protocol_row(channel_name):
    return {
        "channel_name": channel_name,
        "protocol_status": "eligible_for_bounded_stress_test_not_claims",
        "analysis_scope": "monitoring_context_error_controlled_only",
        "required_control_count": 3,
        "next_analysis": "bounded_calibration_stress_test_without_claims",
        "calibration_claim_status": "blocked",
        "probability_output_status": "blocked",
        "load_modification_status": "blocked",
    }


def _evidence(channel_name, season_id, observed, captured, missed, role):
    role_map = {
        "gap": "outcome_context_gap_excluded_from_calibration_signal",
        "missed": "missed_only_error_case_for_sensitivity_bounds",
        "monitor": "monitoring_context_only_not_calibration_claim",
    }
    return {
        "collection_packet_id": f"{channel_name}__{season_id}",
        "channel_name": channel_name,
        "collection_season_id": season_id,
        "observed_event_count": observed,
        "captured_event_count": captured,
        "missed_event_count": missed,
        "protocol_evidence_role": role_map[role],
    }
