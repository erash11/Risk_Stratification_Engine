import json

from risk_stratification_engine.exposure_load_shift_context import (
    build_exposure_load_shift_context_summary,
    clean_shift_context_rows,
    write_exposure_load_shift_context_report,
)


def test_shift_context_summary_links_failure_season_to_schedule_availability_context(
    tmp_path,
):
    summary = build_exposure_load_shift_context_summary(
        exposure_events=_event_rows(),
        exposure_participations=_participation_rows(),
        exposure_load_features=_feature_rows(),
        exposure_load_diagnostics=_diagnostic_rows(),
        exposure_load_failure_modes=_failure_modes(),
    )

    assert summary["experiment_type"] == "exposure_load_shift_context_sprint"
    assert summary["overall_recommendation"] == "review_schedule_roster_availability_context"
    assert summary["failure_seasons"] == ["2024-2025"]
    assert summary["comparator_seasons"] == ["2023-2024"]

    failure_context = {
        row["context_domain"]: row
        for row in summary["shift_context_rows"]
        if row["season_id"] == "2024-2025"
    }
    assert failure_context["game_exposure"]["game_event_count"] == 2
    assert failure_context["game_exposure"]["games_prior_mean"] > 1.0
    assert failure_context["participation_status"]["modified_participation_rate"] == 0.0
    assert failure_context["category_specific_load"]["lift_event_count"] == 1
    assert (
        failure_context["category_specific_load"]["lift_sessions_28d_mean"]
        < summary["comparator_context"]["category_specific_load"]["lift_sessions_28d_mean"]
    )

    drivers = {
        row["feature_name"]: row for row in summary["driver_context_rows"]
    }
    assert drivers["exposure_games_prior_count"]["context_domain"] == "game_exposure"
    assert drivers["exposure_lift_sessions_28d"]["context_signal"] == (
        "reduced_lift_exposure_in_failed_season"
    )
    assert drivers["exposure_modified_participations_28d"]["context_domain"] == (
        "participation_status"
    )

    cases = summary["shift_context_cases"]
    assert cases[0]["test_season_id"] == "2024-2025"
    assert cases[0]["primary_context_signal"] == "elevated_game_exposure"

    report_path = tmp_path / "shift_context.md"
    write_exposure_load_shift_context_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Shift Context Sprint" in report
    assert "complete athlete-season trajectories" in report
    assert "schedule, roster, availability, and managed-risk context" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(clean_shift_context_rows(summary["shift_context_rows"]), allow_nan=False)


def _failure_modes() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_failure_mode_sprint",
        "overall_recommendation": "inspect_exposure_feature_shift_drivers",
        "failure_seasons": ["2024-2025"],
        "comparator_seasons": ["2023-2024"],
        "top_driver_features": [
            {
                "feature_name": "exposure_lift_sessions_28d",
                "feature_domain": "category_specific_load",
                "shift_direction": "reduced_in_failure",
                "driver_score": 0.32,
            },
            {
                "feature_name": "exposure_games_prior_count",
                "feature_domain": "game_exposure",
                "shift_direction": "elevated_in_failure",
                "driver_score": 0.24,
            },
            {
                "feature_name": "exposure_modified_participations_28d",
                "feature_domain": "participation_status",
                "shift_direction": "reduced_in_failure",
                "driver_score": 0.21,
            },
        ],
    }


def _event_rows() -> list[dict[str, object]]:
    return [
        {
            "event_id": "f_game_1",
            "event_type": "game",
            "season_id": "2024-2025",
            "date": "2024-09-01",
            "exposure_category": "game",
            "duration_minutes": 180,
            "rtp_flag": False,
        },
        {
            "event_id": "f_game_2",
            "event_type": "game",
            "season_id": "2024-2025",
            "date": "2024-09-08",
            "exposure_category": "game",
            "duration_minutes": 180,
            "rtp_flag": False,
        },
        {
            "event_id": "f_lift_1",
            "event_type": "training",
            "season_id": "2024-2025",
            "date": "2024-09-10",
            "exposure_category": "weight_room",
            "duration_minutes": 50,
            "rtp_flag": False,
        },
        {
            "event_id": "c_game_1",
            "event_type": "game",
            "season_id": "2023-2024",
            "date": "2023-09-01",
            "exposure_category": "game",
            "duration_minutes": 180,
            "rtp_flag": False,
        },
        {
            "event_id": "c_lift_1",
            "event_type": "training",
            "season_id": "2023-2024",
            "date": "2023-09-03",
            "exposure_category": "weight_room",
            "duration_minutes": 55,
            "rtp_flag": False,
        },
        {
            "event_id": "c_lift_2",
            "event_type": "training",
            "season_id": "2023-2024",
            "date": "2023-09-10",
            "exposure_category": "weight_room",
            "duration_minutes": 55,
            "rtp_flag": False,
        },
    ]


def _participation_rows() -> list[dict[str, object]]:
    return [
        {
            "event_id": "f_game_1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2024-2025",
            "date": "2024-09-01",
            "exposure_category": "game",
            "participation_category": "full",
            "related_external_issue_id": "",
            "duration_minutes": 180,
            "rpe": "",
            "workload_unit_amount": "",
        },
        {
            "event_id": "f_game_2",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2024-2025",
            "date": "2024-09-08",
            "exposure_category": "game",
            "participation_category": "full",
            "related_external_issue_id": "",
            "duration_minutes": 180,
            "rpe": "",
            "workload_unit_amount": "",
        },
        {
            "event_id": "f_lift_1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2024-2025",
            "date": "2024-09-10",
            "exposure_category": "weight_room",
            "participation_category": "full",
            "related_external_issue_id": "",
            "duration_minutes": 50,
            "rpe": "",
            "workload_unit_amount": "",
        },
        {
            "event_id": "c_game_1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2023-2024",
            "date": "2023-09-01",
            "exposure_category": "game",
            "participation_category": "modified",
            "related_external_issue_id": "issue-1",
            "duration_minutes": 150,
            "rpe": 6,
            "workload_unit_amount": 900,
        },
        {
            "event_id": "c_lift_1",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2023-2024",
            "date": "2023-09-03",
            "exposure_category": "weight_room",
            "participation_category": "modified",
            "related_external_issue_id": "issue-1",
            "duration_minutes": 45,
            "rpe": 5,
            "workload_unit_amount": 225,
        },
        {
            "event_id": "c_lift_2",
            "athlete_id": "a1",
            "athlete_match_status": "matched",
            "season_id": "2023-2024",
            "date": "2023-09-10",
            "exposure_category": "weight_room",
            "participation_category": "full",
            "related_external_issue_id": "",
            "duration_minutes": 55,
            "rpe": 5,
            "workload_unit_amount": 275,
        },
    ]


def _feature_rows() -> list[dict[str, object]]:
    return [
        {
            "season_id": "2024-2025",
            "athlete_id": "a1",
            "exposure_games_prior_count": 3,
            "exposure_game_events_28d": 2,
            "exposure_lift_sessions_28d": 1,
            "exposure_modified_participations_28d": 0,
            "exposure_days_since_last_modified_or_no_participation": 30,
        },
        {
            "season_id": "2023-2024",
            "athlete_id": "a1",
            "exposure_games_prior_count": 1,
            "exposure_game_events_28d": 1,
            "exposure_lift_sessions_28d": 2,
            "exposure_modified_participations_28d": 2,
            "exposure_days_since_last_modified_or_no_participation": 4,
        },
    ]


def _diagnostic_rows() -> list[dict[str, object]]:
    return [
        {
            "test_season_id": "2024-2025",
            "horizon_days": 30,
            "diagnostic_label": "ranking_triage_gain_calibration_loss",
            "target_reason": "over_sharpened_probability_slice",
            "delta_roc_auc": 0.058,
            "delta_brier_skill_score": -0.544,
            "delta_top_decile_lift": 0.949,
            "delta_prediction_to_observed_gap": 0.097,
        }
    ]
