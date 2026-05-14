import json

from risk_stratification_engine.exposure_load_shadow_calibration_readiness import (
    build_exposure_load_shadow_calibration_readiness_review,
    clean_shadow_calibration_readiness_rows,
    write_exposure_load_shadow_calibration_readiness_report,
)


def test_shadow_calibration_readiness_marks_channels_as_candidates_with_adjudication_caveat(
    tmp_path,
):
    review = build_exposure_load_shadow_calibration_readiness_review(
        {
            "experiment_type": "exposure_load_shadow_collection_summary",
            "overall_recommendation": (
                "revisit_calibration_readiness_with_prospective_shadow_evidence"
            ),
            "production_readiness": "not_ready_for_probability_or_pilot",
            "calibration_readiness": (
                "ready_for_calibration_readiness_review_not_calibration_claim"
            ),
            "total_rows": 8,
            "complete_valid_rows": 8,
            "pending_or_invalid_rows": 0,
            "complete_source_eligible_rows": 8,
            "useful_source_ok_actionable_rows": 4,
            "channel_summary_rows": [
                {
                    "channel_name": "broad_30d",
                    "minimum_required_packets": 4,
                    "planned_rows": 4,
                    "complete_valid_rows": 4,
                    "complete_source_eligible_rows": 4,
                    "useful_rows": 2,
                    "source_context_ok_rows": 4,
                    "actionable_rows": 2,
                    "useful_source_ok_actionable_rows": 2,
                    "calibration_review_gate": (
                        "ready_for_calibration_readiness_review"
                    ),
                },
                {
                    "channel_name": "severity_14d",
                    "minimum_required_packets": 4,
                    "planned_rows": 4,
                    "complete_valid_rows": 4,
                    "complete_source_eligible_rows": 4,
                    "useful_rows": 2,
                    "source_context_ok_rows": 4,
                    "actionable_rows": 2,
                    "useful_source_ok_actionable_rows": 2,
                    "calibration_review_gate": (
                        "ready_for_calibration_readiness_review"
                    ),
                },
            ],
        }
    )

    assert review["experiment_type"] == (
        "exposure_load_shadow_calibration_readiness_sprint"
    )
    assert review["overall_recommendation"] == (
        "defer_calibration_claims_pending_independent_practitioner_adjudication"
    )
    assert review["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert review["calibration_claim_readiness"] == (
        "not_ready_for_calibration_claims"
    )
    assert review["calibration_research_status"] == (
        "research_candidate_pending_independent_practitioner_adjudication"
    )
    assert review["independent_adjudication_required"] is True
    assert review["collection_summary_complete_valid_rows"] == 8
    assert review["collection_summary_pending_or_invalid_rows"] == 0

    channel_rows = {
        row["channel_name"]: row
        for row in review["channel_readiness_rows"]
    }
    assert channel_rows["broad_30d"]["readiness_status"] == (
        "candidate_pending_independent_practitioner_adjudication"
    )
    assert channel_rows["broad_30d"]["required_next_action"] == (
        "independent_practitioner_adjudication"
    )
    assert channel_rows["broad_30d"]["calibration_claim_status"] == "blocked"
    assert channel_rows["severity_14d"]["useful_source_ok_actionable_rows"] == 2

    gap_rows = {row["gate_name"]: row for row in review["evidence_gap_rows"]}
    assert gap_rows["independent_practitioner_adjudication"]["gate_status"] == (
        "required"
    )
    assert gap_rows["probability_facing_outputs"]["gate_status"] == "blocked"
    assert gap_rows["pilot_dashboard_readiness"]["gate_status"] == "blocked"

    report_path = tmp_path / "readiness.md"
    write_exposure_load_shadow_calibration_readiness_report(report_path, review)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Calibration Readiness Sprint" in report
    assert "not calibration claims" in report
    assert "independent practitioner adjudication" in report

    json.dumps(review, allow_nan=False)
    json.dumps(
        clean_shadow_calibration_readiness_rows(review["channel_readiness_rows"]),
        allow_nan=False,
    )


def test_shadow_calibration_readiness_blocks_when_collection_summary_is_incomplete():
    review = build_exposure_load_shadow_calibration_readiness_review(
        {
            "pending_or_invalid_rows": 1,
            "complete_valid_rows": 7,
            "channel_summary_rows": [
                {
                    "channel_name": "broad_30d",
                    "minimum_required_packets": 4,
                    "complete_valid_rows": 3,
                    "complete_source_eligible_rows": 3,
                    "useful_source_ok_actionable_rows": 1,
                    "calibration_review_gate": "continue_collection",
                }
            ],
        }
    )

    assert review["overall_recommendation"] == (
        "complete_shadow_collection_before_calibration_readiness_review"
    )
    assert review["calibration_research_status"] == "not_ready_collection_incomplete"
    assert review["independent_adjudication_required"] is True
    assert review["channel_readiness_rows"][0]["readiness_status"] == (
        "not_ready_collection_incomplete"
    )
