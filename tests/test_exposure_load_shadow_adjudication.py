import json

from risk_stratification_engine.exposure_load_shadow_adjudication import (
    build_exposure_load_shadow_adjudication_decision_package,
    build_exposure_load_shadow_adjudication_package,
    build_exposure_load_shadow_adjudication_summary,
    clean_shadow_adjudication_rows,
    write_exposure_load_shadow_adjudication_decision_report,
    write_exposure_load_shadow_adjudication_report,
    write_exposure_load_shadow_adjudication_summary_report,
)


def test_shadow_adjudication_package_creates_schema_template_and_checks(tmp_path):
    summary = build_exposure_load_shadow_adjudication_package(_shadow_replay())

    assert summary["experiment_type"] == "exposure_load_shadow_adjudication_sprint"
    assert summary["overall_recommendation"] == (
        "adjudication_template_ready_for_prospective_collection"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"

    schema_fields = {row["field_name"] for row in summary["schema_rows"]}
    assert {
        "reviewer_id",
        "review_date",
        "alert_usefulness",
        "outcome_confirmed",
        "source_context_ok",
        "action_taken",
        "notes",
    }.issubset(schema_fields)

    template_rows = summary["adjudication_template_rows"]
    assert len(template_rows) == 2
    assert template_rows[0]["review_packet_id"] == "broad_30d__2023-2024"
    assert template_rows[0]["alert_usefulness"] == ""
    assert template_rows[0]["adjudication_status"] == "pending_review"

    completion_rows = summary["completion_check_rows"]
    assert completion_rows[0]["missing_required_field_count"] == 6
    assert completion_rows[0]["completion_status"] == "pending_required_fields"

    report_path = tmp_path / "adjudication.md"
    write_exposure_load_shadow_adjudication_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Adjudication Package Sprint" in report
    assert "prospective collection" in report
    assert "not pilot or dashboard clearance" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(clean_shadow_adjudication_rows(summary["schema_rows"]), allow_nan=False)


def test_shadow_adjudication_summary_validates_completion_and_actionability(
    tmp_path,
):
    summary = build_exposure_load_shadow_adjudication_summary(
        [
            {
                "review_packet_id": "broad_30d__2023-2024",
                "channel_name": "broad_30d",
                "test_season_id": "2023-2024",
                "reviewer_id": "ER1",
                "review_date": "2026-05-13",
                "alert_usefulness": "useful",
                "outcome_confirmed": "true",
                "source_context_ok": "true",
                "action_taken": "monitor",
                "notes": "De-identified useful managed-risk signal.",
            },
            {
                "review_packet_id": "severity_14d__2023-2024",
                "channel_name": "severity_14d",
                "test_season_id": "2023-2024",
                "reviewer_id": "ER1",
                "review_date": "2026-05-13",
                "alert_usefulness": "noisy",
                "outcome_confirmed": "false",
                "source_context_ok": "true",
                "action_taken": "none",
                "notes": "De-identified noisy packet.",
            },
            {
                "review_packet_id": "severity_7d__2023-2024",
                "channel_name": "severity_7d",
                "test_season_id": "2023-2024",
                "reviewer_id": "",
                "review_date": "bad-date",
                "alert_usefulness": "maybe",
                "outcome_confirmed": "",
                "source_context_ok": "true",
                "action_taken": "none",
                "notes": "",
            },
        ]
    )

    assert summary["experiment_type"] == "exposure_load_shadow_adjudication_summary"
    assert summary["overall_recommendation"] == (
        "complete_adjudication_required_before_operational_summary"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert summary["total_rows"] == 3
    assert summary["complete_valid_rows"] == 2
    assert summary["pending_or_invalid_rows"] == 1
    assert summary["useful_source_ok_actionable_rows"] == 1

    validation_rows = summary["validation_rows"]
    assert validation_rows[0]["completion_status"] == "complete_valid"
    assert validation_rows[2]["completion_status"] == "pending_or_invalid"
    assert validation_rows[2]["missing_required_fields"] == (
        "reviewer_id,outcome_confirmed"
    )
    assert validation_rows[2]["invalid_fields"] == "review_date,alert_usefulness"

    channel_rows = summary["channel_summary_rows"]
    assert {
        row["channel_name"]: row["useful_source_ok_actionable_rows"]
        for row in channel_rows
    } == {"broad_30d": 1, "severity_14d": 0}

    report_path = tmp_path / "adjudication_summary.md"
    write_exposure_load_shadow_adjudication_summary_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Adjudication Summary Sprint" in report
    assert "Complete valid rows: 2" in report
    assert "not probability calibration or dashboard clearance" in report

    json.dumps(summary, allow_nan=False)


def test_shadow_adjudication_decision_package_selects_channels_and_blocks_product(
    tmp_path,
):
    decision = build_exposure_load_shadow_adjudication_decision_package(
        {
            "experiment_type": "exposure_load_shadow_adjudication_summary",
            "production_readiness": "not_ready_for_probability_or_pilot",
            "total_rows": 12,
            "complete_valid_rows": 12,
            "pending_or_invalid_rows": 0,
            "useful_source_ok_actionable_rows": 4,
            "channel_summary_rows": [
                {
                    "channel_name": "broad_30d",
                    "complete_valid_rows": 4,
                    "useful_rows": 2,
                    "source_context_ok_rows": 4,
                    "actionable_rows": 2,
                    "useful_source_ok_actionable_rows": 2,
                },
                {
                    "channel_name": "severity_14d",
                    "complete_valid_rows": 4,
                    "useful_rows": 2,
                    "source_context_ok_rows": 4,
                    "actionable_rows": 2,
                    "useful_source_ok_actionable_rows": 2,
                },
                {
                    "channel_name": "severity_7d",
                    "complete_valid_rows": 4,
                    "useful_rows": 0,
                    "source_context_ok_rows": 4,
                    "actionable_rows": 0,
                    "useful_source_ok_actionable_rows": 0,
                },
            ],
        }
    )

    assert decision["experiment_type"] == (
        "exposure_load_shadow_adjudication_decision_sprint"
    )
    assert decision["overall_recommendation"] == (
        "continue_shadow_monitoring_with_channel_revisions"
    )
    assert decision["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert decision["continued_shadow_channels"] == ["broad_30d", "severity_14d"]
    assert decision["paused_or_revision_channels"] == ["severity_7d"]

    channel_decisions = {
        row["channel_name"]: row["channel_decision"]
        for row in decision["channel_decision_rows"]
    }
    assert channel_decisions == {
        "broad_30d": "continue_shadow_monitoring",
        "severity_14d": "continue_shadow_monitoring",
        "severity_7d": "pause_or_revise_before_more_collection",
    }

    report_path = tmp_path / "decision.md"
    write_exposure_load_shadow_adjudication_decision_report(report_path, decision)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Adjudication Decision Sprint" in report
    assert "continue_shadow_monitoring_with_channel_revisions" in report
    assert "not probability calibration or dashboard clearance" in report

    json.dumps(decision, allow_nan=False)


def _shadow_replay() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_shadow_replay_sprint",
        "overall_recommendation": (
            "historical_shadow_replay_ready_for_prospective_collection"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "review_packet_rows": [
            {
                "review_packet_id": "broad_30d__2023-2024",
                "channel_name": "broad_30d",
                "test_season_id": "2023-2024",
                "minimum_review_unit": "complete source-eligible athlete-season",
                "required_evidence": (
                    "frozen alert episodes, source eligibility, exposure capture "
                    "status, outcome adjudication, and alert burden"
                ),
                "episode_count": 109,
                "unique_observed_event_count": 85,
                "unique_captured_event_count": 20,
                "missed_event_count": 65,
                "episodes_per_athlete_season": 0.685535,
                "review_packet_status": "ready_for_research_adjudication",
            },
            {
                "review_packet_id": "severity_14d__2023-2024",
                "channel_name": "severity_14d",
                "test_season_id": "2023-2024",
                "minimum_review_unit": "complete source-eligible athlete-season",
                "required_evidence": (
                    "frozen alert episodes, source eligibility, exposure capture "
                    "status, outcome adjudication, and alert burden"
                ),
                "episode_count": 148,
                "unique_observed_event_count": 46,
                "unique_captured_event_count": 11,
                "missed_event_count": 35,
                "episodes_per_athlete_season": 0.919255,
                "review_packet_status": "ready_for_research_adjudication",
            },
        ],
    }
