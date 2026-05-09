import json

from risk_stratification_engine.exposure_load_source_eligible_calibration import (
    build_exposure_load_source_eligible_calibration_summary,
    clean_source_eligible_calibration_rows,
    write_exposure_load_source_eligible_calibration_report,
)


def test_source_eligible_calibration_recovers_after_excluding_failed_season(tmp_path):
    summary = build_exposure_load_source_eligible_calibration_summary(
        validation_rows=_validation_rows(),
        source_resolution_policy=_source_resolution_policy(),
    )

    assert summary["experiment_type"] == (
        "exposure_load_source_eligible_calibration_sprint"
    )
    assert summary["overall_recommendation"] == (
        "probability_research_can_resume_on_source_eligible_seasons"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert summary["excluded_test_seasons"] == ["2024-2025"]

    groups = {row["calibration_scope"]: row for row in summary["calibration_rows"]}
    assert groups["all_seasons"]["ranking_triage_gain_calibration_loss_count"] == 1
    assert groups["source_eligible"]["ranking_triage_gain_calibration_loss_count"] == 0
    assert groups["source_eligible"]["calibration_supported_count"] == 1
    assert groups["source_eligible"]["mean_delta_brier_skill_score"] == 0.04

    report_path = tmp_path / "source_eligible_calibration.md"
    write_exposure_load_source_eligible_calibration_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Load Source-Eligible Calibration Sprint" in report
    assert "complete athlete-season trajectories" in report
    assert "not pilot clearance" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(
        clean_source_eligible_calibration_rows(summary["calibration_rows"]),
        allow_nan=False,
    )


def _source_resolution_policy() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_source_resolution_sprint",
        "overall_recommendation": (
            "exclude_failed_season_from_probability_calibration_until_source_resolved"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "failure_seasons": ["2024-2025"],
        "policy_rows": [
            {
                "policy_domain": "season_eligibility",
                "policy_decision": (
                    "exclude_failed_season_from_probability_calibration"
                ),
            }
        ],
    }


def _validation_rows() -> list[dict[str, object]]:
    return [
        _metric_row(
            test_season_id="2024-2025",
            feature_set="graph_plus_coverage_source",
            brier_skill_score=0.10,
            roc_auc=0.70,
            top_decile_lift=1.20,
            mean_predicted_risk=0.08,
            test_positive_rate=0.04,
        ),
        _metric_row(
            test_season_id="2024-2025",
            feature_set="graph_plus_coverage_exposure_load",
            brier_skill_score=-0.20,
            roc_auc=0.76,
            top_decile_lift=1.80,
            mean_predicted_risk=0.18,
            test_positive_rate=0.04,
        ),
        _metric_row(
            test_season_id="2025-2026",
            feature_set="graph_plus_coverage_source",
            brier_skill_score=0.02,
            roc_auc=0.65,
            top_decile_lift=1.30,
            mean_predicted_risk=0.05,
            test_positive_rate=0.06,
        ),
        _metric_row(
            test_season_id="2025-2026",
            feature_set="graph_plus_coverage_exposure_load",
            brier_skill_score=0.06,
            roc_auc=0.69,
            top_decile_lift=1.50,
            mean_predicted_risk=0.06,
            test_positive_rate=0.06,
        ),
    ]


def _metric_row(
    test_season_id: str,
    feature_set: str,
    brier_skill_score: float,
    roc_auc: float,
    top_decile_lift: float,
    mean_predicted_risk: float,
    test_positive_rate: float,
) -> dict[str, object]:
    return {
        "row_type": "model_metric",
        "test_season_id": test_season_id,
        "horizon_days": 30,
        "feature_set": feature_set,
        "brier_skill_score": brier_skill_score,
        "model_brier_score": 0.10 - brier_skill_score,
        "roc_auc": roc_auc,
        "top_decile_lift": top_decile_lift,
        "mean_predicted_risk": mean_predicted_risk,
        "test_positive_rate": test_positive_rate,
    }
