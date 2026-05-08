from pathlib import Path

from risk_stratification_engine.forward_case_review import (
    build_forward_case_review_summary,
    write_forward_case_review_report,
)


def test_build_forward_case_review_summary_counts_targeted_diagnostics():
    cases = [
        {
            "target_reason": "forward_ranking_survivor",
            "channel_name": "severity_14d",
            "test_season_id": "2025-2026",
            "threshold_policy": "season_local_percentile",
            "case_type": "true_positive_episode",
            "diagnostic_label": "model_signal_supported",
        },
        {
            "target_reason": "forward_ranking_survivor",
            "channel_name": "severity_14d",
            "test_season_id": "2025-2026",
            "threshold_policy": "season_local_percentile",
            "case_type": "false_positive_episode",
            "diagnostic_label": "missing_context_or_managed_risk",
        },
        {
            "target_reason": "forward_calibration_survivor",
            "channel_name": "subtype_lower_extremity_soft_tissue_30d",
            "test_season_id": "2026-2027",
            "threshold_policy": "burden_capped_percentile",
            "case_type": "missed_injury",
            "diagnostic_label": "model_miss",
        },
    ]

    summary = build_forward_case_review_summary(cases)

    assert summary["experiment_type"] == "forward_case_review_sprint"
    assert summary["overall_recommendation"] == "continue_targeted_case_review"
    assert summary["case_count"] == 3
    assert summary["targeted_channel_count"] == 2
    assert summary["diagnostic_summary"] == {
        "missing_context_or_managed_risk": 1,
        "model_miss": 1,
        "model_signal_supported": 1,
    }
    assert summary["channel_summary"]["severity_14d"]["case_count"] == 2


def test_write_forward_case_review_report_names_viability_questions(tmp_path):
    summary = build_forward_case_review_summary(
        [
            {
                "target_reason": "forward_ranking_survivor",
                "channel_name": "severity_14d",
                "test_season_id": "2025-2026",
                "threshold_policy": "season_local_percentile",
                "case_type": "true_positive_episode",
                "diagnostic_label": "model_signal_supported",
            }
        ]
    )
    output = tmp_path / "report.md"

    write_forward_case_review_report(output, summary)

    report = output.read_text()
    assert "Forward Case Review Sprint" in report
    assert "forward-surviving windows and channels" in report
    assert "missing exposure, intervention, baseline, or mechanism context" in report
