import json

from risk_stratification_engine.exposure_load_shadow_prospective_collection_completion import (
    build_exposure_load_shadow_prospective_collection_completion,
    clean_shadow_prospective_collection_completion_rows,
    write_exposure_load_shadow_prospective_collection_completion_report,
)


def test_prospective_collection_completion_blocks_retest_for_pending_packets(
    tmp_path,
):
    completion = build_exposure_load_shadow_prospective_collection_completion(
        _operations_payload(completed=False)
    )

    assert completion["experiment_type"] == (
        "exposure_load_shadow_prospective_collection_completion_sprint"
    )
    assert completion["overall_recommendation"] == (
        "continue_prospective_collection_before_bounded_retest"
    )
    assert completion["milestone_status"] == "prospective_collection_incomplete"
    assert completion["bounded_retest_readiness"] == "blocked_pending_collection"
    assert completion["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert completion["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert completion["pilot_dashboard_readiness"] == "blocked"
    assert completion["load_modification_readiness"] == "blocked"

    validation_rows = completion["packet_validation_rows"]
    assert len(validation_rows) == 8
    assert validation_rows[0]["completion_status"] == "pending_or_invalid"
    assert "collection_season_id" in validation_rows[0]["missing_required_fields"]
    assert "source_eligible" in validation_rows[0]["missing_required_fields"]

    channel_rows = {
        row["channel_name"]: row for row in completion["channel_completion_rows"]
    }
    assert channel_rows["broad_30d"]["required_packet_count"] == 4
    assert channel_rows["broad_30d"]["complete_practitioner_packet_count"] == 0
    assert channel_rows["broad_30d"]["captured_event_count"] == 0
    assert channel_rows["broad_30d"]["completion_gate"] == (
        "blocked_pending_required_packets"
    )

    gates = {
        (row["channel_name"], row["gate_name"]): row
        for row in completion["completion_gate_rows"]
    }
    assert gates[("broad_30d", "bounded_retest_gate")]["gate_status"] == "blocked"
    assert gates[("severity_14d", "calibration_claim_boundary")][
        "gate_status"
    ] == "blocked"

    report_path = tmp_path / "completion.md"
    write_exposure_load_shadow_prospective_collection_completion_report(
        report_path,
        completion,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Prospective Collection Completion Sprint" in report
    assert "blocked_pending_collection" in report
    assert "not calibration claims" in report

    json.dumps(completion, allow_nan=False)
    json.dumps(
        clean_shadow_prospective_collection_completion_rows(validation_rows),
        allow_nan=False,
    )


def test_prospective_collection_completion_clears_only_bounded_retest_gate_when_targets_met():
    completion = build_exposure_load_shadow_prospective_collection_completion(
        _operations_payload(completed=True)
    )

    assert completion["overall_recommendation"] == (
        "run_bounded_retest_after_completed_prospective_collection"
    )
    assert completion["milestone_status"] == "prospective_collection_complete_for_retest"
    assert completion["bounded_retest_readiness"] == (
        "ready_for_bounded_retest_not_claims"
    )
    assert completion["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )

    channel_rows = {
        row["channel_name"]: row for row in completion["channel_completion_rows"]
    }
    assert channel_rows["broad_30d"]["complete_practitioner_packet_count"] == 4
    assert channel_rows["broad_30d"]["captured_event_count"] == 8
    assert channel_rows["broad_30d"]["missed_event_rate"] == 0.2
    assert channel_rows["broad_30d"]["completion_gate"] == (
        "ready_for_bounded_retest_not_claims"
    )
    assert channel_rows["severity_14d"]["completion_gate"] == (
        "ready_for_bounded_retest_not_claims"
    )


def _operations_payload(completed):
    rows = []
    for channel_name in ("broad_30d", "severity_14d"):
        for sequence in range(1, 5):
            rows.append(_worksheet_row(channel_name, sequence, completed))
    return {
        "experiment_type": (
            "exposure_load_shadow_prospective_collection_operations_sprint"
        ),
        "overall_recommendation": (
            "prepare_prospective_collection_operations_before_retest"
        ),
        "milestone_status": (
            "reviewer_ready_prospective_packet_operations_defined"
        ),
        "retest_readiness": "pending_required_prospective_collection",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "pilot_dashboard_readiness": "blocked",
        "load_modification_readiness": "blocked",
        "channel_operation_rows": [
            _channel_row("broad_30d"),
            _channel_row("severity_14d"),
        ],
        "collection_worksheet_rows": rows,
    }


def _channel_row(channel_name):
    return {
        "channel_name": channel_name,
        "required_packet_count": 4,
        "required_captured_events": 8,
        "maximum_allowed_missed_event_rate": 0.75,
        "operation_status": "ready_for_prospective_collection_not_retest",
    }


def _worksheet_row(channel_name, sequence, completed):
    base = {
        "collection_packet_id": f"{channel_name}__prospective_collection_{sequence:03d}",
        "channel_name": channel_name,
        "packet_sequence": sequence,
        "target_type": "monitoring_context_packet",
        "target_captured_events_needed": 8,
        "maximum_allowed_missed_event_rate": 0.75,
    }
    if not completed:
        base.update(
            {
                "collection_season_id": "",
                "packet_start_date": "",
                "packet_end_date": "",
                "source_eligible": "",
                "episode_count": "",
                "unique_observed_event_count": "",
                "unique_captured_event_count": "",
                "unique_missed_event_count": "",
                "missed_event_rate": "",
                "alert_usefulness": "",
                "outcome_confirmed": "",
                "source_context_ok": "",
                "action_taken": "",
                "reviewer_id": "",
                "review_date": "",
                "collection_status": "pending_prospective_collection",
            }
        )
        return base
    base.update(
        {
            "collection_season_id": "2026-2027",
            "packet_start_date": "2026-08-01",
            "packet_end_date": "2026-12-01",
            "source_eligible": "true",
            "episode_count": "2",
            "unique_observed_event_count": "10",
            "unique_captured_event_count": "2",
            "unique_missed_event_count": "2",
            "missed_event_rate": "0.2",
            "alert_usefulness": "useful",
            "outcome_confirmed": "true",
            "source_context_ok": "true",
            "action_taken": "monitor",
            "reviewer_id": "ER1",
            "review_date": "2027-01-15",
            "collection_status": "complete_practitioner_adjudication",
        }
    )
    return base
