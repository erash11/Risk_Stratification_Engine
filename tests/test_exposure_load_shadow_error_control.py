import json

from risk_stratification_engine.exposure_load_shadow_error_control import (
    build_exposure_load_shadow_error_control_review,
    clean_shadow_error_control_rows,
    write_exposure_load_shadow_error_control_report,
)


def test_shadow_error_control_keeps_retained_channels_bounded_and_not_claims(
    tmp_path,
):
    review = build_exposure_load_shadow_error_control_review(
        _sensitivity_payload()
    )

    assert review["experiment_type"] == "exposure_load_shadow_error_control_sprint"
    assert review["overall_recommendation"] == (
        "continue_bounded_calibration_research_with_error_controls_not_claims"
    )
    assert review["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert review["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert review["pilot_dashboard_readiness"] == "blocked"
    assert review["load_modification_readiness"] == "blocked"
    assert review["bounded_error_control_status"] == (
        "ready_for_error_controlled_research_decision_not_claims"
    )
    assert review["error_control_definition"] == {
        "minimum_useful_actionable_rows": 2,
        "high_miss_fraction_requires_control": True,
        "monitoring_useful_is_not_prediction_useful": True,
        "empty_or_missed_only_packets_require_packet_review": True,
        "probability_or_load_modification_allowed": False,
    }

    decisions = {row["channel_name"]: row for row in review["decision_rows"]}
    assert decisions["broad_30d"]["retained_research_candidate"] is True
    assert decisions["broad_30d"]["monitoring_usefulness_status"] == (
        "supported_for_monitoring_context_only"
    )
    assert decisions["broad_30d"]["prediction_calibration_usefulness_status"] == (
        "not_established_high_miss_fraction"
    )
    assert decisions["broad_30d"]["error_control_status"] == (
        "requires_high_miss_fraction_controls"
    )
    assert decisions["broad_30d"]["next_gate_decision"] == (
        "continue_bounded_calibration_research_with_error_controls"
    )
    assert decisions["broad_30d"]["allowed_use"] == "shadow_monitoring_review_only"
    assert decisions["broad_30d"]["blocked_use"] == (
        "probability_output,calibration_claims,pilot_dashboard,"
        "autonomous_intervention,load_modification"
    )
    assert decisions["severity_14d"]["retained_research_candidate"] is True
    assert decisions["severity_14d"]["next_gate_decision"] == (
        "continue_bounded_calibration_research_with_error_controls"
    )

    refined = {
        row["collection_packet_id"]: row
        for row in review["refined_evidence_dossier_rows"]
    }
    assert refined["broad_30d__2021-2022"]["dossier_refinement_action"] == (
        "collect_outcome_context_before_calibration_weight"
    )
    assert refined["broad_30d__2022-2023"]["dossier_refinement_action"] == (
        "review_missed_only_packet_before_channel_escalation"
    )
    assert refined["broad_30d__2023-2024"]["dossier_refinement_action"] == (
        "preserve_as_monitoring_useful_not_prediction_evidence"
    )

    controls = {
        (row["channel_name"], row["control_name"]): row
        for row in review["error_control_rows"]
    }
    assert controls[("broad_30d", "miss_fraction_control")]["required"] is True
    assert controls[("severity_14d", "monitoring_prediction_boundary")][
        "control_action"
    ] == "separate_monitoring_value_from_calibration_evidence"

    report_path = tmp_path / "error_control.md"
    write_exposure_load_shadow_error_control_report(report_path, review)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Error-Control Decision Sprint" in report
    assert "monitoring context only" in report
    assert "not calibration claims" in report
    assert "load modification" in report

    json.dumps(review, allow_nan=False)
    json.dumps(
        clean_shadow_error_control_rows(review["decision_rows"]),
        allow_nan=False,
    )


def test_shadow_error_control_collects_more_rows_when_usefulness_floor_fails():
    payload = _sensitivity_payload()
    payload["sensitivity_rows"][0]["useful_actionable_rows"] = 1
    payload["error_mode_rows"] = [
        row for row in payload["error_mode_rows"] if row["channel_name"] == "broad_30d"
    ]

    review = build_exposure_load_shadow_error_control_review(payload)

    assert review["overall_recommendation"] == (
        "collect_more_practitioner_reviewed_rows_before_calibration_research"
    )
    assert review["decision_rows"][0]["retained_research_candidate"] is False
    assert review["decision_rows"][0]["next_gate_decision"] == (
        "collect_more_practitioner_reviewed_rows"
    )


def _sensitivity_payload():
    return {
        "experiment_type": "exposure_load_shadow_calibration_sensitivity_sprint",
        "overall_recommendation": (
            "continue_bounded_calibration_research_with_error_mode_controls"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "pilot_dashboard_readiness": "blocked",
        "bounded_research_status": "ready_for_bounded_sensitivity_review_not_claims",
        "sensitivity_rows": [
            _channel_row("broad_30d", 2, 0.15812, 0.84188),
            _channel_row("severity_14d", 3, 0.139535, 0.860465),
        ],
        "evidence_dossier_rows": [
            _dossier_row("broad_30d", "2021-2022", 0, 0, 0, "unclear", "none"),
            _dossier_row("broad_30d", "2022-2023", 71, 0, 71, "noisy", "none"),
            _dossier_row("broad_30d", "2023-2024", 85, 20, 65, "useful", "monitor"),
            _dossier_row("broad_30d", "2025-2026", 78, 17, 61, "useful", "monitor"),
            _dossier_row("severity_14d", "2021-2022", 0, 0, 0, "unclear", "none"),
            _dossier_row("severity_14d", "2022-2023", 36, 0, 36, "useful", "monitor"),
            _dossier_row("severity_14d", "2023-2024", 46, 11, 35, "useful", "monitor"),
            _dossier_row("severity_14d", "2025-2026", 47, 7, 40, "useful", "monitor"),
        ],
        "error_mode_rows": [
            _error_row("broad_30d", "empty_outcome_packet", "medium"),
            _error_row("broad_30d", "missed_only_packet", "high"),
            _error_row("broad_30d", "high_miss_fraction", "high"),
            _error_row("broad_30d", "monitor_only_action_boundary", "medium"),
            _error_row("severity_14d", "empty_outcome_packet", "medium"),
            _error_row("severity_14d", "missed_only_packet", "high"),
            _error_row("severity_14d", "high_miss_fraction", "high"),
            _error_row("severity_14d", "monitor_only_action_boundary", "medium"),
        ],
    }


def _channel_row(channel_name, useful_actionable, capture_rate, missed_rate):
    return {
        "channel_name": channel_name,
        "complete_practitioner_rows": 4,
        "source_eligible_rows": 4,
        "useful_actionable_rows": useful_actionable,
        "captured_event_count": 18,
        "missed_event_count": 111,
        "observed_event_count": 129,
        "capture_rate": capture_rate,
        "missed_event_rate": missed_rate,
        "no_observed_packet_count": 1,
        "practitioner_adjudication_gate": "pass",
        "usefulness_floor_gate": "pass",
        "miss_rate_gate": "caution",
        "calibration_claim_status": "blocked",
        "required_next_action": (
            "bounded_calibration_research_with_error_mode_controls"
        ),
    }


def _dossier_row(
    channel_name,
    season_id,
    observed,
    captured,
    missed,
    usefulness,
    action,
):
    return {
        "collection_packet_id": f"{channel_name}__{season_id}",
        "channel_name": channel_name,
        "collection_season_id": season_id,
        "alert_usefulness": usefulness,
        "outcome_confirmed": usefulness == "useful",
        "source_context_ok": True,
        "action_taken": action,
        "episode_count": 8,
        "observed_event_count": observed,
        "captured_event_count": captured,
        "missed_event_count": missed,
        "capture_rate": 0 if observed == 0 else round(captured / observed, 6),
        "evidence_label": "fixture",
        "notes": "Reviewed.",
    }


def _error_row(channel_name, mode, severity):
    return {
        "channel_name": channel_name,
        "error_mode": mode,
        "severity": severity,
        "packet_count": 1,
        "affected_packet_ids": f"{channel_name}__2023-2024",
        "interpretation": "Fixture error mode.",
    }
