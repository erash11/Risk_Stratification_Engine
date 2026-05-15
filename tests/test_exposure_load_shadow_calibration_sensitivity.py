import json

from risk_stratification_engine.exposure_load_shadow_calibration_sensitivity import (
    build_exposure_load_shadow_calibration_sensitivity_review,
    clean_shadow_calibration_sensitivity_rows,
    write_exposure_load_shadow_calibration_sensitivity_report,
)


def test_shadow_calibration_sensitivity_summarizes_retained_channel_limits(tmp_path):
    review = build_exposure_load_shadow_calibration_sensitivity_review(
        calibration_readiness=_readiness_payload(),
        collection_rows=_collection_rows(),
        event_crosswalk_rows=_event_crosswalk_rows(),
    )

    assert review["experiment_type"] == (
        "exposure_load_shadow_calibration_sensitivity_sprint"
    )
    assert review["overall_recommendation"] == (
        "continue_bounded_calibration_research_with_error_mode_controls"
    )
    assert review["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert review["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert review["pilot_dashboard_readiness"] == "blocked"
    assert review["bounded_research_status"] == (
        "ready_for_bounded_sensitivity_review_not_claims"
    )

    channel_rows = {
        row["channel_name"]: row for row in review["sensitivity_rows"]
    }
    assert channel_rows["broad_30d"]["practitioner_adjudication_gate"] == "pass"
    assert channel_rows["broad_30d"]["usefulness_floor_gate"] == "pass"
    assert channel_rows["broad_30d"]["miss_rate_gate"] == "caution"
    assert channel_rows["broad_30d"]["captured_event_count"] == 2
    assert channel_rows["broad_30d"]["missed_event_count"] == 4
    assert channel_rows["broad_30d"]["capture_rate"] == 0.333333
    assert channel_rows["broad_30d"]["required_next_action"] == (
        "bounded_calibration_research_with_error_mode_controls"
    )
    assert channel_rows["severity_14d"]["useful_actionable_rows"] == 3

    dossier_rows = {
        row["collection_packet_id"]: row for row in review["evidence_dossier_rows"]
    }
    assert dossier_rows["broad_30d__2021-2022"]["evidence_label"] == (
        "no_observed_event_evidence_gap"
    )
    assert dossier_rows["broad_30d__2022-2023"]["evidence_label"] == (
        "missed_only_practitioner_nonuseful"
    )
    assert dossier_rows["broad_30d__2023-2024"]["evidence_label"] == (
        "captured_practitioner_useful_monitoring"
    )

    error_modes = {
        (row["channel_name"], row["error_mode"]): row
        for row in review["error_mode_rows"]
    }
    assert error_modes[("broad_30d", "high_miss_fraction")]["severity"] == "high"
    assert error_modes[("severity_14d", "monitor_only_action_boundary")][
        "packet_count"
    ] == 3
    assert error_modes[("severity_14d", "empty_outcome_packet")]["packet_count"] == 2

    report_path = tmp_path / "sensitivity.md"
    write_exposure_load_shadow_calibration_sensitivity_report(report_path, review)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Calibration Sensitivity Sprint" in report
    assert "not calibration claims" in report
    assert "probability-facing output" in report

    json.dumps(review, allow_nan=False)
    json.dumps(
        clean_shadow_calibration_sensitivity_rows(review["sensitivity_rows"]),
        allow_nan=False,
    )


def test_shadow_calibration_sensitivity_blocks_without_practitioner_readiness():
    review = build_exposure_load_shadow_calibration_sensitivity_review(
        calibration_readiness={
            "calibration_research_status": (
                "research_candidate_pending_independent_practitioner_adjudication"
            ),
            "independent_adjudication_required": True,
            "channel_readiness_rows": [],
        },
        collection_rows=_collection_rows(),
        event_crosswalk_rows=_event_crosswalk_rows(),
    )

    assert review["overall_recommendation"] == (
        "complete_practitioner_adjudication_before_sensitivity_review"
    )
    assert review["bounded_research_status"] == (
        "not_ready_practitioner_adjudication_incomplete"
    )
    assert review["sensitivity_rows"][0]["practitioner_adjudication_gate"] == "fail"


def _readiness_payload():
    return {
        "overall_recommendation": "advance_to_bounded_calibration_research_not_claims",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "calibration_claim_readiness": "not_ready_for_calibration_claims",
        "calibration_research_status": (
            "ready_for_bounded_calibration_research_not_claims"
        ),
        "independent_adjudication_required": False,
        "evidence_basis": "independent_practitioner_adjudicated_shadow_collection",
        "channel_readiness_rows": [
            {
                "channel_name": "broad_30d",
                "complete_valid_rows": 4,
                "useful_source_ok_actionable_rows": 2,
                "readiness_status": (
                    "calibration_research_candidate_practitioner_adjudicated"
                ),
            },
            {
                "channel_name": "severity_14d",
                "complete_valid_rows": 4,
                "useful_source_ok_actionable_rows": 3,
                "readiness_status": (
                    "calibration_research_candidate_practitioner_adjudicated"
                ),
            },
        ],
    }


def _collection_rows():
    return [
        _collection_row("broad_30d", "2021-2022", 8, 0, 0, "unclear", "none"),
        _collection_row("broad_30d", "2022-2023", 9, 4, 0, "noisy", "none"),
        _collection_row("broad_30d", "2023-2024", 10, 2, 2, "useful", "monitor"),
        _collection_row("broad_30d", "2025-2026", 11, 0, 0, "useful", "monitor"),
        _collection_row("severity_14d", "2021-2022", 13, 0, 0, "unclear", "none"),
        _collection_row("severity_14d", "2022-2023", 15, 2, 0, "useful", "monitor"),
        _collection_row("severity_14d", "2023-2024", 18, 1, 1, "useful", "monitor"),
        _collection_row("severity_14d", "2025-2026", 20, 0, 0, "useful", "monitor"),
    ]


def _collection_row(
    channel_name,
    season_id,
    episode_count,
    observed_count,
    captured_count,
    usefulness,
    action,
):
    return {
        "collection_packet_id": f"{channel_name}__{season_id}",
        "channel_name": channel_name,
        "collection_season_id": season_id,
        "source_eligible": "true",
        "episode_count": episode_count,
        "unique_observed_event_count": observed_count,
        "unique_captured_event_count": captured_count,
        "alert_usefulness": usefulness,
        "outcome_confirmed": "true" if usefulness == "useful" else "false",
        "source_context_ok": "true",
        "action_taken": action,
        "reviewer_id": "ER1",
        "review_date": "2026-05-15",
        "collection_status": "complete_practitioner_adjudication",
        "notes": "Practitioner reviewed.",
    }


def _event_crosswalk_rows():
    rows = []
    rows.extend(_events("broad_30d__2022-2023", "broad_30d", "missed", 4))
    rows.extend(_events("broad_30d__2023-2024", "broad_30d", "captured", 2))
    rows.extend(_events("severity_14d__2022-2023", "severity_14d", "missed", 2))
    rows.extend(_events("severity_14d__2023-2024", "severity_14d", "captured", 1))
    return rows


def _events(packet_id, channel_name, capture_status, count):
    return [
        {
            "review_packet_id": packet_id,
            "channel_name": channel_name,
            "capture_status": capture_status,
            "event_date": f"2024-01-{index + 1:02d}",
            "injury_type": "lower_extremity",
            "body_area": "knee",
            "classification": "time_loss",
            "nearest_days_from_start_to_event": index,
        }
        for index in range(count)
    ]
