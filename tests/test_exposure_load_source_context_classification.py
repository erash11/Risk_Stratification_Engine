import json

from risk_stratification_engine.exposure_load_source_context_classification import (
    build_exposure_load_source_context_classification_summary,
    clean_source_context_rows,
    write_exposure_load_source_context_classification_report,
)


def test_source_context_classification_separates_schedule_shift_from_managed_risk(
    tmp_path,
):
    summary = build_exposure_load_source_context_classification_summary(
        exposure_events=_event_rows(),
        exposure_participations=_participation_rows(),
        exposure_load_shift_context=_shift_context(),
        exposure_load_schedule_roster=_schedule_roster(),
        exposure_load_availability_capture=_availability_capture(),
        exposure_load_context_decision=_context_decision(),
    )

    assert summary["experiment_type"] == (
        "exposure_load_source_context_classification_sprint"
    )
    assert summary["overall_recommendation"] == (
        "treat_failed_season_as_schedule_roster_plus_capture_shift"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert summary["failure_seasons"] == ["2024-2025"]

    classifications = {
        row["classification_domain"]: row
        for row in summary["source_context_classification_rows"]
    }
    assert classifications["managed_risk_support"]["classification"] == (
        "not_supported_by_source_flags"
    )
    assert classifications["schedule_roster_context"]["classification"] == (
        "supported_schedule_roster_shift"
    )
    assert classifications["availability_capture_context"]["classification"] == (
        "supported_capture_or_documentation_shift"
    )
    assert classifications["next_model_action"]["classification"] == (
        "do_not_expand_model_features"
    )

    evidence = {row["evidence_domain"]: row for row in summary["source_evidence_rows"]}
    assert evidence["source_participation_flags"]["failure_modified_rate"] == 0.0
    assert evidence["source_participation_flags"]["failure_linked_issue_rate"] == 0.0
    assert evidence["source_schedule"]["failure_game_event_count"] == 2

    report_path = tmp_path / "classification.md"
    write_exposure_load_source_context_classification_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Source Context Classification Sprint" in report
    assert "true managed-risk context" in report
    assert "not pilot clearance" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(
        clean_source_context_rows(summary["source_context_classification_rows"]),
        allow_nan=False,
    )


def _shift_context() -> dict[str, object]:
    return {
        "overall_recommendation": "review_schedule_roster_availability_context",
        "failure_seasons": ["2024-2025"],
        "comparator_seasons": ["2023-2024"],
    }


def _schedule_roster() -> dict[str, object]:
    return {
        "overall_recommendation": "review_failed_season_schedule_roster_shift",
        "schedule_roster_drivers": [
            {"metric_name": "game_event_count", "review_signal": "elevated_game_schedule"},
            {"metric_name": "lift_event_count", "review_signal": "reduced_lift_schedule"},
        ],
    }


def _availability_capture() -> dict[str, object]:
    return {
        "overall_recommendation": "review_failed_season_availability_capture",
        "availability_capture_drivers": [
            {
                "metric_name": "modified_participation_rate",
                "review_signal": "reduced_modified_availability_flagging",
            },
            {
                "metric_name": "no_participation_rate",
                "review_signal": "reduced_no_participation_flagging",
            },
        ],
    }


def _context_decision() -> dict[str, object]:
    return {
        "overall_recommendation": (
            "keep_shadow_ranking_and_resolve_context_before_model_expansion"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
    }


def _event_rows() -> list[dict[str, object]]:
    return [
        {
            "event_id": "fg1",
            "event_type": "game",
            "season_id": "2024-2025",
            "date": "2024-09-01",
            "exposure_category": "game",
        },
        {
            "event_id": "fg2",
            "event_type": "game",
            "season_id": "2024-2025",
            "date": "2024-09-08",
            "exposure_category": "game",
        },
        {
            "event_id": "cg1",
            "event_type": "game",
            "season_id": "2023-2024",
            "date": "2023-09-01",
            "exposure_category": "game",
        },
    ]


def _participation_rows() -> list[dict[str, object]]:
    return [
        {
            "event_id": "fg1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2024-2025",
            "participation_category": "full",
            "related_external_issue_id": "",
        },
        {
            "event_id": "fg2",
            "athlete_id": "a2",
            "athlete_match_status": "matched",
            "season_id": "2024-2025",
            "participation_category": "full",
            "related_external_issue_id": "",
        },
        {
            "event_id": "cg1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2023-2024",
            "participation_category": "modified",
            "related_external_issue_id": "issue-1",
        },
    ]
