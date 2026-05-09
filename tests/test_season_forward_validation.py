from pathlib import Path

from risk_stratification_engine.season_forward_validation import (
    build_season_forward_validation_summary,
    write_season_forward_validation_report,
)


def test_build_season_forward_validation_summary_keeps_model_and_alert_results():
    rows = [
        {
            "row_type": "model_metric",
            "train_season_ids": "2024",
            "test_season_id": "2025",
            "feature_set": "graph_trajectory",
            "horizon_days": 14,
            "roc_auc": 0.61,
            "brier_skill_score": 0.02,
            "top_decile_lift": 1.3,
        },
        {
            "row_type": "model_metric",
            "train_season_ids": "2024",
            "test_season_id": "2025",
            "feature_set": "graph_plus_coverage_source",
            "horizon_days": 14,
            "roc_auc": 0.68,
            "brier_skill_score": 0.03,
            "top_decile_lift": 1.1,
        },
        {
            "row_type": "alert_policy",
            "test_season_id": "2025",
            "channel_name": "severity_14d",
            "threshold_policy": "burden_capped_percentile",
            "unique_event_capture_rate": 0.12,
            "episodes_per_athlete_season": 0.8,
        },
    ]

    summary = build_season_forward_validation_summary(rows)

    assert summary["experiment_type"] == "season_forward_validation_sprint"
    assert summary["overall_recommendation"] == "continue_season_forward_research"
    assert summary["evaluated_test_seasons"] == ["2025"]
    assert summary["best_by_horizon"]["14"]["ranking"]["feature_set"] == (
        "graph_plus_coverage_source"
    )
    assert summary["alert_policy_summary"]["severity_14d"][
        "recommended_threshold_policy"
    ] == "burden_capped_percentile"


def test_write_season_forward_validation_report_names_temporal_guardrail(tmp_path):
    summary = build_season_forward_validation_summary(
        [
            {
                "row_type": "model_metric",
                "train_season_ids": "2024",
                "test_season_id": "2025",
                "feature_set": "graph_plus_coverage_source",
                "horizon_days": 30,
                "roc_auc": 0.7,
                "brier_skill_score": 0.04,
                "top_decile_lift": 1.5,
            }
        ]
    )
    output = tmp_path / "report.md"

    write_season_forward_validation_report(output, summary)

    report = output.read_text()
    assert "Season-Forward Validation Sprint" in report
    assert "train on earlier seasons" in report
    assert "complete athlete-season trajectories" in report


def test_build_season_forward_validation_summary_keeps_custom_experiment_type():
    rows = [
        {
            "row_type": "model_metric",
            "test_season_id": "2025",
            "feature_set": "graph_plus_coverage_injury_history",
            "horizon_days": 30,
            "roc_auc": 0.72,
            "brier_skill_score": 0.08,
            "top_decile_lift": 2.4,
        }
    ]

    summary = build_season_forward_validation_summary(
        rows,
        experiment_type="injury_history_season_forward_validation_sprint",
    )

    assert (
        summary["experiment_type"]
        == "injury_history_season_forward_validation_sprint"
    )
    assert summary["best_by_horizon"]["30"]["ranking"]["feature_set"] == (
        "graph_plus_coverage_injury_history"
    )
