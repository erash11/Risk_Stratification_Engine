from pathlib import Path

from risk_stratification_engine.injury_history_modeling import (
    build_injury_history_model_comparison_summary,
    write_injury_history_model_comparison_report,
)


def test_build_injury_history_model_comparison_summary_identifies_injury_history_wins():
    rows = [
        {
            "feature_set": "graph_plus_coverage_source",
            "horizon_days": 30,
            "roc_auc": 0.61,
            "brier_skill_score": 0.01,
            "model_brier_score": 0.20,
            "top_decile_lift": 1.5,
        },
        {
            "feature_set": "graph_plus_coverage_injury_history",
            "horizon_days": 30,
            "roc_auc": 0.66,
            "brier_skill_score": 0.03,
            "model_brier_score": 0.18,
            "top_decile_lift": 1.7,
        },
    ]

    summary = build_injury_history_model_comparison_summary(rows)

    assert summary["experiment_type"] == "injury_history_feature_sprint"
    assert summary["overall_recommendation"] == "continue_injury_history_research"
    assert summary["best_by_horizon"]["30"]["ranking"]["feature_set"] == (
        "graph_plus_coverage_injury_history"
    )
    assert summary["best_by_horizon"]["30"]["calibration"]["feature_set"] == (
        "graph_plus_coverage_injury_history"
    )


def test_write_injury_history_model_comparison_report_names_time_safe_prior_injury_context(
    tmp_path: Path,
):
    summary = {
        "experiment_type": "injury_history_feature_sprint",
        "overall_recommendation": "continue_injury_history_research",
        "feature_sets": [
            "graph_plus_coverage_source",
            "graph_plus_coverage_injury_history",
        ],
        "injury_history_feature_columns": [
            "injury_history_prior_injury_count",
        ],
        "comparison_rows": [
            {
                "feature_set": "graph_plus_coverage_injury_history",
                "horizon_days": 30,
                "roc_auc": 0.66,
                "brier_skill_score": 0.03,
                "model_brier_score": 0.18,
                "top_decile_lift": 1.7,
            }
        ],
        "best_by_horizon": {},
    }
    path = tmp_path / "injury_history_model_comparison_report.md"

    write_injury_history_model_comparison_report(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "Injury History Feature Sprint" in text
    assert "time-safe prior injury context" in text
    assert "injury_history_prior_injury_count" in text
