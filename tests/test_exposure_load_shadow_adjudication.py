import json

from risk_stratification_engine.exposure_load_shadow_adjudication import (
    build_exposure_load_shadow_adjudication_package,
    clean_shadow_adjudication_rows,
    write_exposure_load_shadow_adjudication_report,
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
