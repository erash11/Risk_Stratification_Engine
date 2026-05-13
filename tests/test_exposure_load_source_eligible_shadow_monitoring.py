import json

from risk_stratification_engine.exposure_load_source_eligible_shadow_monitoring import (
    build_exposure_load_source_eligible_shadow_monitoring_review,
    clean_source_eligible_shadow_monitoring_rows,
    write_exposure_load_source_eligible_shadow_monitoring_report,
)


def test_shadow_monitoring_reviews_source_eligible_frozen_policy(tmp_path):
    summary = build_exposure_load_source_eligible_shadow_monitoring_review(
        validation_rows=_validation_rows(),
        source_eligible_policy=_source_eligible_policy(),
    )

    assert (
        summary["experiment_type"]
        == "exposure_load_source_eligible_shadow_monitoring_sprint"
    )
    assert summary["overall_recommendation"] == (
        "proceed_with_prospective_source_eligible_shadow_monitoring"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert summary["excluded_test_seasons"] == ["2024-2025"]

    channels = {row["channel_name"]: row for row in summary["monitoring_rows"]}
    assert channels["broad_30d"]["source_eligible_season_count"] == 2
    assert channels["broad_30d"]["mean_capture_rate"] == 0.20
    assert channels["broad_30d"]["max_episodes_per_athlete_season"] == 0.90
    assert channels["broad_30d"]["monitoring_status"] == (
        "ready_for_prospective_shadow_review"
    )

    season_ids = {
        row["test_season_id"] for row in summary["monitoring_season_rows"]
    }
    assert season_ids == {"2023-2024", "2025-2026"}
    assert all(
        row["threshold_policy"] == "burden_capped_percentile"
        for row in summary["monitoring_season_rows"]
    )

    report_path = tmp_path / "shadow_monitoring.md"
    write_exposure_load_source_eligible_shadow_monitoring_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Source-Eligible Shadow Monitoring Sprint" in report
    assert "prospective shadow review" in report
    assert "not pilot or dashboard clearance" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(
        clean_source_eligible_shadow_monitoring_rows(summary["monitoring_rows"]),
        allow_nan=False,
    )


def _source_eligible_policy() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_source_eligible_policy_sprint",
        "overall_recommendation": (
            "advance_source_eligible_shadow_mode_threshold_research"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "excluded_test_seasons": ["2024-2025"],
        "burden_cap_episodes_per_athlete_season": 1.0,
        "policy_package": {
            "status": "source_eligible_research_shadow_mode",
            "recommended_channels": [
                {
                    "channel_name": "broad_30d",
                    "policy_name": "exclude_concussion",
                    "horizon_days": 30,
                    "threshold_policy": "burden_capped_percentile",
                    "mean_selected_threshold_value": 0.04,
                }
            ],
        },
    }


def _validation_rows() -> list[dict[str, object]]:
    return [
        _alert_row(
            test_season_id="2023-2024",
            threshold_policy="season_local_percentile",
            capture=0.30,
            burden=1.20,
            threshold=0.05,
        ),
        _alert_row(
            test_season_id="2023-2024",
            threshold_policy="burden_capped_percentile",
            capture=0.15,
            burden=0.70,
            threshold=0.03,
        ),
        _alert_row(
            test_season_id="2024-2025",
            threshold_policy="burden_capped_percentile",
            capture=0.45,
            burden=0.60,
            threshold=0.04,
        ),
        _alert_row(
            test_season_id="2025-2026",
            threshold_policy="burden_capped_percentile",
            capture=0.25,
            burden=0.90,
            threshold=0.05,
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
        "unique_event_capture_rate": capture,
        "episodes_per_athlete_season": burden,
        "unique_captured_event_count": 3,
        "unique_observed_event_count": 12,
    }
