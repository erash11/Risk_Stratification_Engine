import json

from risk_stratification_engine.exposure_load_source_eligible_policy import (
    build_exposure_load_source_eligible_policy_package,
    clean_source_eligible_policy_rows,
    write_exposure_load_source_eligible_policy_report,
)


def test_source_eligible_policy_selects_burden_tolerable_shadow_thresholds(tmp_path):
    summary = build_exposure_load_source_eligible_policy_package(
        validation_rows=_validation_rows(),
        source_eligible_calibration=_source_eligible_calibration(),
        burden_cap_episodes_per_athlete_season=1.0,
    )

    assert summary["experiment_type"] == "exposure_load_source_eligible_policy_sprint"
    assert summary["overall_recommendation"] == (
        "advance_source_eligible_shadow_mode_threshold_research"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert summary["excluded_test_seasons"] == ["2024-2025"]

    channels = {row["channel_name"]: row for row in summary["policy_rows"]}
    assert channels["broad_30d"]["recommended_threshold_policy"] == (
        "burden_capped_percentile"
    )
    assert channels["broad_30d"]["recommended_shadow_mode_status"] == (
        "shadow_research_candidate"
    )
    assert channels["broad_30d"]["mean_capture_rate"] == 0.20
    assert channels["broad_30d"]["mean_episodes_per_athlete_season"] == 0.80
    assert summary["policy_package"]["status"] == "source_eligible_research_shadow_mode"
    assert summary["policy_package"]["deployment_boundary"] == (
        "research shadow-mode only; not pilot, dashboard, or autonomous intervention"
    )

    report_path = tmp_path / "source_eligible_policy.md"
    write_exposure_load_source_eligible_policy_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Source-Eligible Policy Sprint" in report
    assert "complete athlete-season trajectories" in report
    assert "not pilot clearance" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(clean_source_eligible_policy_rows(summary["policy_rows"]), allow_nan=False)


def _source_eligible_calibration() -> dict[str, object]:
    return {
        "overall_recommendation": (
            "probability_research_can_resume_on_source_eligible_seasons"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "excluded_test_seasons": ["2024-2025"],
        "calibration_rows": [
            {
                "calibration_scope": "source_eligible",
                "ranking_triage_gain_calibration_loss_count": 0,
                "calibration_supported_count": 2,
                "mean_delta_brier_skill_score": 0.05,
            }
        ],
    }


def _validation_rows() -> list[dict[str, object]]:
    return [
        _alert_row(
            test_season_id="2023-2024",
            threshold_policy="season_local_percentile",
            capture=0.25,
            burden=1.20,
            threshold=0.05,
        ),
        _alert_row(
            test_season_id="2023-2024",
            threshold_policy="burden_capped_percentile",
            capture=0.20,
            burden=0.80,
            threshold=0.025,
        ),
        _alert_row(
            test_season_id="2024-2025",
            threshold_policy="burden_capped_percentile",
            capture=0.40,
            burden=0.70,
            threshold=0.05,
        ),
        _alert_row(
            test_season_id="2025-2026",
            threshold_policy="season_local_percentile",
            capture=0.30,
            burden=1.30,
            threshold=0.05,
        ),
        _alert_row(
            test_season_id="2025-2026",
            threshold_policy="burden_capped_percentile",
            capture=0.20,
            burden=0.80,
            threshold=0.025,
        ),
    ]


def _alert_row(
    test_season_id: str,
    threshold_policy: str,
    capture: float,
    burden: float,
    threshold: float,
) -> dict[str, object]:
    return {
        "row_type": "alert_policy",
        "test_season_id": test_season_id,
        "feature_set": "graph_plus_coverage_exposure_load",
        "threshold_policy": threshold_policy,
        "channel_name": "broad_30d",
        "policy_name": "exclude_concussion",
        "graph_window_size": 4,
        "horizon_days": 30,
        "role": "broad 30d early warning",
        "selected_threshold_value": threshold,
        "burden_cap_episodes_per_athlete_season": 1.0
        if threshold_policy == "burden_capped_percentile"
        else None,
        "unique_event_capture_rate": capture,
        "episodes_per_athlete_season": burden,
        "unique_captured_event_count": 3,
        "unique_observed_event_count": 12,
    }
