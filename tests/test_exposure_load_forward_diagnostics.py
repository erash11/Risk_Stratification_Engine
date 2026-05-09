import json

from risk_stratification_engine.exposure_load_forward_diagnostics import (
    build_exposure_load_calibration_diagnostics,
    build_exposure_load_forward_diagnostic_cases,
    build_exposure_load_forward_diagnostic_summary,
    write_exposure_load_forward_diagnostic_report,
)


def test_exposure_load_diagnostics_identify_ranking_gain_calibration_loss():
    diagnostics = build_exposure_load_calibration_diagnostics(
        _validation_rows_for_diagnostics()
    )

    assert len(diagnostics) == 2
    high_priority = diagnostics[0]
    assert high_priority["test_season_id"] == "2024-2025"
    assert high_priority["horizon_days"] == 7
    assert (
        high_priority["diagnostic_label"]
        == "ranking_triage_gain_calibration_loss"
    )
    assert high_priority["priority_tier"] == "high"
    assert high_priority["delta_roc_auc"] == 0.05
    assert high_priority["delta_brier_skill_score"] == -0.66
    assert high_priority["delta_model_brier_score"] == 0.04
    assert high_priority["delta_top_decile_lift"] == 0.3
    assert high_priority["test_positive_rate"] == 0.03
    assert high_priority["source_mean_predicted_risk"] == 0.04
    assert high_priority["exposure_load_mean_predicted_risk"] == 0.12
    assert high_priority["delta_prediction_to_observed_gap"] == 0.08


def test_exposure_load_diagnostic_cases_and_summary_are_actionable(tmp_path):
    diagnostics = build_exposure_load_calibration_diagnostics(
        _validation_rows_for_diagnostics()
    )
    cases = build_exposure_load_forward_diagnostic_cases(diagnostics)
    summary = build_exposure_load_forward_diagnostic_summary(
        _validation_rows_for_diagnostics(),
        cases,
    )

    assert {case["feature_set"] for case in cases} == {
        "graph_plus_coverage_exposure_load"
    }
    high_priority_case = cases[0]
    assert high_priority_case["target_reason"] == "over_sharpened_probability_slice"
    assert summary["experiment_type"] == "exposure_load_forward_diagnostic_sprint"
    assert (
        summary["overall_recommendation"]
        == "inspect_exposure_load_forward_failure_modes"
    )
    assert summary["targeted_test_seasons"] == ["2024-2025", "2025-2026"]

    report_path = tmp_path / "report.md"
    write_exposure_load_forward_diagnostic_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Forward Diagnostic Sprint" in report
    assert "complete athlete-season trajectories" in report
    assert "over-sharpening" in report

    json.dumps(summary, allow_nan=False)


def _validation_rows_for_diagnostics() -> list[dict[str, object]]:
    return [
        {
            "row_type": "model_metric",
            "test_season_id": "2024-2025",
            "horizon_days": 7,
            "feature_set": "graph_plus_coverage_source",
            "roc_auc": 0.58,
            "brier_skill_score": -0.02,
            "model_brier_score": 0.04,
            "top_decile_lift": 1.0,
            "test_positive_rate": 0.03,
            "mean_predicted_risk": 0.04,
        },
        {
            "row_type": "model_metric",
            "test_season_id": "2024-2025",
            "horizon_days": 7,
            "feature_set": "graph_plus_coverage_exposure_load",
            "roc_auc": 0.63,
            "brier_skill_score": -0.68,
            "model_brier_score": 0.08,
            "top_decile_lift": 1.3,
            "test_positive_rate": 0.03,
            "mean_predicted_risk": 0.12,
        },
        {
            "row_type": "model_metric",
            "test_season_id": "2025-2026",
            "horizon_days": 7,
            "feature_set": "graph_plus_coverage_source",
            "roc_auc": 0.61,
            "brier_skill_score": 0.01,
            "model_brier_score": 0.05,
            "top_decile_lift": 1.1,
            "test_positive_rate": 0.04,
            "mean_predicted_risk": 0.05,
        },
        {
            "row_type": "model_metric",
            "test_season_id": "2025-2026",
            "horizon_days": 7,
            "feature_set": "graph_plus_coverage_exposure_load",
            "roc_auc": 0.6,
            "brier_skill_score": 0.04,
            "model_brier_score": 0.04,
            "top_decile_lift": 1.0,
            "test_positive_rate": 0.04,
            "mean_predicted_risk": 0.04,
        },
    ]
