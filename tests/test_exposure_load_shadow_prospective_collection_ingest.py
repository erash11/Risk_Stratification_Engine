import json

from risk_stratification_engine.exposure_load_shadow_prospective_collection_ingest import (
    build_exposure_load_shadow_prospective_collection_ingest,
    clean_shadow_prospective_collection_ingest_rows,
    write_exposure_load_shadow_prospective_collection_ingest_report,
)


def test_prospective_collection_ingest_updates_known_deidentified_completed_rows(
    tmp_path,
):
    ingest = build_exposure_load_shadow_prospective_collection_ingest(
        _operations_payload(),
        [
            _completed_row("broad_30d__prospective_collection_001"),
            _completed_row("broad_30d__prospective_collection_999"),
            _completed_row("broad_30d__prospective_collection_001"),
            {
                **_completed_row("broad_30d__prospective_collection_002"),
                "athlete_name": "Identifiable Athlete",
            },
        ],
    )

    assert ingest["experiment_type"] == (
        "exposure_load_shadow_prospective_collection_ingest_sprint"
    )
    assert ingest["overall_recommendation"] == (
        "repair_completed_collection_worksheet_before_ingest"
    )
    assert ingest["milestone_status"] == "completed_collection_ingest_path_ready"
    assert ingest["bounded_retest_readiness"] == "pending_completion_validation"
    assert ingest["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert ingest["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert ingest["pilot_dashboard_readiness"] == "blocked"
    assert ingest["load_modification_readiness"] == "blocked"
    assert ingest["ingested_completed_rows"] == 1
    assert ingest["ingest_error_rows"] == 3

    errors = {row["error_type"] for row in ingest["ingest_validation_rows"]}
    assert {
        "unknown_packet_id",
        "duplicate_packet_id",
        "deidentification_violation",
    }.issubset(errors)

    worksheet = {
        row["collection_packet_id"]: row
        for row in ingest["updated_collection_worksheet_rows"]
    }
    updated = worksheet["broad_30d__prospective_collection_001"]
    assert updated["collection_status"] == "complete_practitioner_adjudication"
    assert updated["collection_season_id"] == "2026-2027"
    assert updated["reviewer_id"] == "PRAC1"
    assert updated["unique_captured_event_count"] == "2"
    assert worksheet["broad_30d__prospective_collection_002"][
        "collection_status"
    ] == "pending_prospective_collection"

    summary = {row["channel_name"]: row for row in ingest["ingest_summary_rows"]}
    assert summary["broad_30d"]["known_packet_count"] == 2
    assert summary["broad_30d"]["ingested_completed_rows"] == 1
    assert summary["broad_30d"]["ingest_status"] == (
        "repair_completed_collection_worksheet_before_validation"
    )

    report_path = tmp_path / "ingest.md"
    write_exposure_load_shadow_prospective_collection_ingest_report(
        report_path,
        ingest,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Prospective Collection Ingest Sprint" in report
    assert "not calibration claims" in report
    assert "de-identified" in report

    json.dumps(ingest, allow_nan=False)
    json.dumps(
        clean_shadow_prospective_collection_ingest_rows(
            ingest["updated_collection_worksheet_rows"]
        ),
        allow_nan=False,
    )


def test_prospective_collection_ingest_waits_when_completed_rows_are_absent():
    ingest = build_exposure_load_shadow_prospective_collection_ingest(
        _operations_payload(),
        [],
    )

    assert ingest["overall_recommendation"] == (
        "await_completed_practitioner_collection_before_ingest"
    )
    assert ingest["milestone_status"] == "completed_collection_ingest_path_ready"
    assert ingest["ingested_completed_rows"] == 0
    assert ingest["ingest_error_rows"] == 0
    assert ingest["pending_input_rows"] == 0
    assert ingest["bounded_retest_readiness"] == "pending_completed_practitioner_rows"
    assert ingest["updated_collection_worksheet_rows"] == (
        _operations_payload()["collection_worksheet_rows"]
    )


def _operations_payload():
    return {
        "experiment_type": (
            "exposure_load_shadow_prospective_collection_operations_sprint"
        ),
        "overall_recommendation": (
            "prepare_prospective_collection_operations_before_retest"
        ),
        "retest_readiness": "pending_required_prospective_collection",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "pilot_dashboard_readiness": "blocked",
        "load_modification_readiness": "blocked",
        "channel_operation_rows": [
            {
                "channel_name": "broad_30d",
                "required_packet_count": 2,
                "required_captured_events": 4,
                "maximum_allowed_missed_event_rate": 0.75,
                "operation_status": "ready_for_prospective_collection_not_retest",
            }
        ],
        "collection_worksheet_rows": [
            _pending_row("broad_30d__prospective_collection_001", 1),
            _pending_row("broad_30d__prospective_collection_002", 2),
        ],
    }


def _pending_row(packet_id, sequence):
    return {
        "collection_packet_id": packet_id,
        "channel_name": "broad_30d",
        "packet_sequence": sequence,
        "target_type": "monitoring_context_packet",
        "required_action": "review_monitoring_context_packet",
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
        "notes": "",
        "target_captured_events_needed": 4,
        "maximum_allowed_missed_event_rate": 0.75,
        "collection_status": "pending_prospective_collection",
    }


def _completed_row(packet_id):
    return {
        "collection_packet_id": packet_id,
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
        "reviewer_id": "PRAC1",
        "review_date": "2027-01-15",
        "notes": "De-identified practitioner note.",
        "collection_status": "complete_practitioner_adjudication",
    }
