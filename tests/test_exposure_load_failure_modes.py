import json

from risk_stratification_engine.exposure_load_failure_modes import (
    build_exposure_load_failure_mode_summary,
    write_exposure_load_failure_mode_report,
)


def test_failure_mode_summary_flags_shifted_exposure_domains(tmp_path):
    summary = build_exposure_load_failure_mode_summary(
        _feature_rows(),
        _diagnostic_rows(),
    )

    assert summary["experiment_type"] == "exposure_load_failure_mode_sprint"
    assert summary["overall_recommendation"] == "inspect_exposure_feature_shift_drivers"
    assert summary["failure_seasons"] == ["2024-2025"]
    assert summary["comparator_seasons"] == ["2023-2024", "2025-2026"]

    top_feature = summary["top_driver_features"][0]
    assert top_feature["feature_name"] == "exposure_game_events_28d"
    assert top_feature["feature_domain"] == "game_exposure"
    assert top_feature["shift_direction"] == "elevated_in_failure"
    assert top_feature["failure_mean"] > top_feature["comparator_mean"]

    domain_names = {
        row["feature_domain"] for row in summary["domain_shift_summary"]
    }
    assert {"game_exposure", "training_session_load"}.issubset(domain_names)

    report_path = tmp_path / "report.md"
    write_exposure_load_failure_mode_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Failure Mode Sprint" in report
    assert "complete athlete-season trajectories" in report
    assert "game_exposure" in report

    json.dumps(summary, allow_nan=False)


def _feature_rows() -> list[dict[str, object]]:
    return [
        {
            "athlete_id": "a1",
            "season_id": "2024-2025",
            "snapshot_date": "2024-09-01",
            "exposure_training_sessions_28d": 18,
            "exposure_game_events_28d": 4,
            "exposure_modified_participations_28d": 1,
            "exposure_no_participations_28d": 0,
        },
        {
            "athlete_id": "a2",
            "season_id": "2024-2025",
            "snapshot_date": "2024-09-02",
            "exposure_training_sessions_28d": 16,
            "exposure_game_events_28d": 3,
            "exposure_modified_participations_28d": 0,
            "exposure_no_participations_28d": 1,
        },
        {
            "athlete_id": "a1",
            "season_id": "2023-2024",
            "snapshot_date": "2023-09-01",
            "exposure_training_sessions_28d": 10,
            "exposure_game_events_28d": 1,
            "exposure_modified_participations_28d": 1,
            "exposure_no_participations_28d": 0,
        },
        {
            "athlete_id": "a2",
            "season_id": "2025-2026",
            "snapshot_date": "2025-09-01",
            "exposure_training_sessions_28d": 8,
            "exposure_game_events_28d": 1,
            "exposure_modified_participations_28d": 0,
            "exposure_no_participations_28d": 0,
        },
    ]


def _diagnostic_rows() -> list[dict[str, object]]:
    return [
        {
            "test_season_id": "2024-2025",
            "horizon_days": 30,
            "diagnostic_label": "ranking_triage_gain_calibration_loss",
            "priority_tier": "high",
        },
        {
            "test_season_id": "2023-2024",
            "horizon_days": 30,
            "diagnostic_label": "calibration_supported",
            "priority_tier": "medium",
        },
        {
            "test_season_id": "2025-2026",
            "horizon_days": 30,
            "diagnostic_label": "calibration_supported",
            "priority_tier": "medium",
        },
    ]
