import json

from risk_stratification_engine.exposure_load_context_review import (
    build_exposure_load_availability_capture_summary,
    build_exposure_load_context_decision_summary,
    build_exposure_load_schedule_roster_summary,
    write_exposure_load_availability_capture_report,
    write_exposure_load_context_decision_report,
    write_exposure_load_schedule_roster_report,
)


def test_schedule_roster_summary_flags_failed_season_schedule_shift(tmp_path):
    summary = build_exposure_load_schedule_roster_summary(
        exposure_events=_event_rows(),
        exposure_participations=_participation_rows(),
        exposure_load_shift_context=_shift_context(),
    )

    assert summary["experiment_type"] == "exposure_load_schedule_roster_sprint"
    assert summary["overall_recommendation"] == "review_failed_season_schedule_roster_shift"
    assert summary["failure_seasons"] == ["2024-2025"]
    assert summary["comparator_seasons"] == ["2023-2024"]

    failure = {
        row["season_id"]: row for row in summary["schedule_roster_rows"]
    }["2024-2025"]
    assert failure["game_event_count"] == 2
    assert failure["active_athlete_count"] == 2
    assert failure["participations_per_athlete"] > 1.0

    drivers = {row["metric_name"]: row for row in summary["schedule_roster_drivers"]}
    assert drivers["game_event_count"]["review_signal"] == "elevated_game_schedule"
    assert drivers["lift_event_count"]["review_signal"] == "reduced_lift_schedule"

    report_path = tmp_path / "schedule.md"
    write_exposure_load_schedule_roster_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Schedule Roster Sprint" in report
    assert "complete athlete-season trajectories" in report
    json.dumps(summary, allow_nan=False)


def test_availability_capture_summary_flags_lower_failed_season_flagging(tmp_path):
    summary = build_exposure_load_availability_capture_summary(
        exposure_participations=_participation_rows(),
        exposure_load_shift_context=_shift_context(),
    )

    assert summary["experiment_type"] == "exposure_load_availability_capture_sprint"
    assert summary["overall_recommendation"] == "review_failed_season_availability_capture"
    assert summary["failure_seasons"] == ["2024-2025"]

    failure = {
        row["season_id"]: row for row in summary["availability_capture_rows"]
    }["2024-2025"]
    assert failure["modified_participation_rate"] == 0.0
    assert failure["linked_issue_participation_count"] == 0

    drivers = {row["metric_name"]: row for row in summary["availability_capture_drivers"]}
    assert drivers["modified_participation_rate"]["review_signal"] == (
        "reduced_modified_availability_flagging"
    )
    assert drivers["linked_issue_participation_count"]["review_signal"] == (
        "issue_linkage_absent_or_reduced"
    )

    report_path = tmp_path / "availability.md"
    write_exposure_load_availability_capture_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Availability Capture Sprint" in report
    assert "managed-risk documentation" in report
    json.dumps(summary, allow_nan=False)


def test_context_decision_summary_keeps_exposure_load_blocked_for_expansion(tmp_path):
    summary = build_exposure_load_context_decision_summary(
        exposure_load_shift_context=_shift_context(),
        schedule_roster_summary={
            "overall_recommendation": "review_failed_season_schedule_roster_shift",
            "schedule_roster_drivers": [
                {"metric_name": "game_event_count", "review_signal": "elevated_game_schedule"}
            ],
        },
        availability_capture_summary={
            "overall_recommendation": "review_failed_season_availability_capture",
            "availability_capture_drivers": [
                {
                    "metric_name": "modified_participation_rate",
                    "review_signal": "reduced_modified_availability_flagging",
                }
            ],
        },
        guardrail_policy={
            "overall_recommendation": "use_exposure_load_for_shadow_ranking_only",
            "production_readiness": "not_ready_for_probability_or_pilot",
        },
    )

    assert summary["experiment_type"] == "exposure_load_context_decision_sprint"
    assert summary["overall_recommendation"] == (
        "keep_shadow_ranking_and_resolve_context_before_model_expansion"
    )
    decisions = {row["decision_domain"]: row for row in summary["decision_rows"]}
    assert decisions["probability_calibration"]["decision"] == "blocked"
    assert decisions["minute_load_expansion"]["decision"] == "blocked"
    assert decisions["shadow_ranking"]["decision"] == "allowed_with_monitoring"

    report_path = tmp_path / "decision.md"
    write_exposure_load_context_decision_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Context Decision Sprint" in report
    assert "not pilot clearance" in report
    json.dumps(summary, allow_nan=False)


def _shift_context() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_shift_context_sprint",
        "overall_recommendation": "review_schedule_roster_availability_context",
        "failure_seasons": ["2024-2025"],
        "comparator_seasons": ["2023-2024"],
        "driver_context_rows": [
            {
                "feature_name": "exposure_games_prior_count",
                "context_domain": "game_exposure",
                "context_signal": "elevated_game_exposure_in_failed_season",
            },
            {
                "feature_name": "exposure_lift_sessions_28d",
                "context_domain": "category_specific_load",
                "context_signal": "reduced_lift_exposure_in_failed_season",
            },
            {
                "feature_name": "exposure_modified_participations_28d",
                "context_domain": "participation_status",
                "context_signal": "reduced_availability_flagging_in_failed_season",
            },
        ],
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
            "event_id": "fl1",
            "event_type": "training",
            "season_id": "2024-2025",
            "date": "2024-09-03",
            "exposure_category": "weight_room",
        },
        {
            "event_id": "cg1",
            "event_type": "game",
            "season_id": "2023-2024",
            "date": "2023-09-01",
            "exposure_category": "game",
        },
        {
            "event_id": "cl1",
            "event_type": "training",
            "season_id": "2023-2024",
            "date": "2023-09-03",
            "exposure_category": "weight_room",
        },
        {
            "event_id": "cl2",
            "event_type": "training",
            "season_id": "2023-2024",
            "date": "2023-09-10",
            "exposure_category": "weight_room",
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
            "participation_level_reason": "",
            "related_external_issue_id": "",
            "duration_minutes": 180,
            "rpe": "",
            "workload_unit_amount": "",
        },
        {
            "event_id": "fg2",
            "athlete_id": "a2",
            "athlete_match_status": "matched",
            "season_id": "2024-2025",
            "participation_category": "full",
            "participation_level_reason": "",
            "related_external_issue_id": "",
            "duration_minutes": 180,
            "rpe": "",
            "workload_unit_amount": "",
        },
        {
            "event_id": "fl1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2024-2025",
            "participation_category": "full",
            "participation_level_reason": "",
            "related_external_issue_id": "",
            "duration_minutes": 50,
            "rpe": "",
            "workload_unit_amount": "",
        },
        {
            "event_id": "cg1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2023-2024",
            "participation_category": "modified",
            "participation_level_reason": "injury",
            "related_external_issue_id": "issue-1",
            "duration_minutes": 120,
            "rpe": 5,
            "workload_unit_amount": 600,
        },
        {
            "event_id": "cl1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2023-2024",
            "participation_category": "modified",
            "participation_level_reason": "injury",
            "related_external_issue_id": "issue-1",
            "duration_minutes": 45,
            "rpe": 4,
            "workload_unit_amount": 180,
        },
    ]
