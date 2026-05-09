from pathlib import Path

from risk_stratification_engine.injury_history_forward_diagnostics import (
    build_injury_history_forward_diagnostic_summary,
    write_injury_history_forward_diagnostic_report,
)


def test_build_injury_history_forward_diagnostic_summary_flags_ranking_gain_calibration_loss():
    rows = [
        {
            "row_type": "model_metric",
            "test_season_id": "2024-2025",
            "feature_set": "graph_plus_coverage_source",
            "horizon_days": 30,
            "roc_auc": 0.61,
            "brier_skill_score": 0.04,
            "model_brier_score": 0.12,
            "top_decile_lift": 1.4,
        },
        {
            "row_type": "model_metric",
            "test_season_id": "2024-2025",
            "feature_set": "graph_plus_coverage_injury_history",
            "horizon_days": 30,
            "roc_auc": 0.68,
            "brier_skill_score": -0.01,
            "model_brier_score": 0.15,
            "top_decile_lift": 3.0,
        },
        {
            "row_type": "model_metric",
            "test_season_id": "2025-2026",
            "feature_set": "graph_plus_coverage_source",
            "horizon_days": 14,
            "roc_auc": 0.62,
            "brier_skill_score": 0.02,
            "model_brier_score": 0.10,
            "top_decile_lift": 1.5,
        },
        {
            "row_type": "model_metric",
            "test_season_id": "2025-2026",
            "feature_set": "graph_plus_coverage_injury_history",
            "horizon_days": 14,
            "roc_auc": 0.64,
            "brier_skill_score": 0.05,
            "model_brier_score": 0.08,
            "top_decile_lift": 1.6,
        },
    ]
    cases = [
        {
            "channel_name": "broad_30d",
            "test_season_id": "2024-2025",
            "case_type": "false_positive_episode",
            "diagnostic_label": "missing_context_or_managed_risk",
        },
        {
            "channel_name": "severity_14d",
            "test_season_id": "2025-2026",
            "case_type": "true_positive_episode",
            "diagnostic_label": "model_signal_supported",
        },
    ]

    summary = build_injury_history_forward_diagnostic_summary(rows, cases)

    assert summary["experiment_type"] == "injury_history_forward_diagnostic_sprint"
    assert summary["overall_recommendation"] == "inspect_injury_history_forward_failure_modes"
    assert summary["calibration_diagnostic_summary"] == {
        "calibration_supported": 1,
        "ranking_triage_gain_calibration_loss": 1,
    }
    assert summary["case_diagnostic_summary"] == {
        "missing_context_or_managed_risk": 1,
        "model_signal_supported": 1,
    }
    high_priority = summary["calibration_diagnostics"][0]
    assert high_priority["test_season_id"] == "2024-2025"
    assert high_priority["horizon_days"] == 30
    assert high_priority["diagnostic_label"] == "ranking_triage_gain_calibration_loss"
    assert high_priority["priority_tier"] == "high"


def test_write_injury_history_forward_diagnostic_report_names_forward_guardrail(
    tmp_path: Path,
):
    summary = build_injury_history_forward_diagnostic_summary(
        [
            {
                "row_type": "model_metric",
                "test_season_id": "2024-2025",
                "feature_set": "graph_plus_coverage_source",
                "horizon_days": 30,
                "roc_auc": 0.61,
                "brier_skill_score": 0.04,
                "model_brier_score": 0.12,
                "top_decile_lift": 1.4,
            },
            {
                "row_type": "model_metric",
                "test_season_id": "2024-2025",
                "feature_set": "graph_plus_coverage_injury_history",
                "horizon_days": 30,
                "roc_auc": 0.68,
                "brier_skill_score": -0.01,
                "model_brier_score": 0.15,
                "top_decile_lift": 3.0,
            },
        ],
        [],
    )
    path = tmp_path / "injury_history_forward_diagnostic_report.md"

    write_injury_history_forward_diagnostic_report(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "Injury History Forward Diagnostic Sprint" in text
    assert "complete athlete-season trajectories" in text
    assert "2024-2025 high-lift calibration failures" in text
