from pathlib import Path

import json
import pandas as pd

from risk_stratification_engine.coverage_source_modeling import (
    build_coverage_source_model_comparison_summary,
    write_coverage_source_model_comparison_report,
)


def test_build_coverage_source_model_comparison_summary_prefers_candidate_with_better_brier_skill():
    rows = [
        {
            "feature_set": "graph_trajectory",
            "horizon_days": 14,
            "roc_auc": 0.61,
            "brier_skill_score": 0.01,
            "model_brier_score": 0.20,
            "top_decile_lift": 1.5,
        },
        {
            "feature_set": "graph_plus_coverage_source",
            "horizon_days": 14,
            "roc_auc": 0.60,
            "brier_skill_score": 0.03,
            "model_brier_score": 0.18,
            "top_decile_lift": 1.4,
        },
    ]

    summary = build_coverage_source_model_comparison_summary(rows)

    assert summary["experiment_type"] == "coverage_source_aware_model_sprint"
    assert summary["best_by_horizon"]["14"]["calibration"]["feature_set"] == (
        "graph_plus_coverage_source"
    )
    assert summary["overall_recommendation"] == "continue_coverage_source_research"


def test_write_coverage_source_model_comparison_report_names_peterson_guardrail(tmp_path: Path):
    summary = {
        "experiment_type": "coverage_source_aware_model_sprint",
        "overall_recommendation": "continue_coverage_source_research",
        "feature_sets": ["graph_trajectory", "graph_plus_coverage_source"],
        "coverage_source_feature_columns": ["coverage_measurement_days_to_date"],
        "comparison_rows": [
            {
                "feature_set": "graph_plus_coverage_source",
                "horizon_days": 14,
                "roc_auc": 0.6,
                "brier_skill_score": 0.03,
                "model_brier_score": 0.18,
                "top_decile_lift": 1.4,
            }
        ],
        "best_by_horizon": {},
    }
    path = tmp_path / "coverage_source_model_comparison_report.md"

    write_coverage_source_model_comparison_report(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "Coverage/Source-Aware Model Sprint" in text
    assert "dynamic graph trajectory features remain the core signal" in text
    assert "coverage_measurement_days_to_date" in text
