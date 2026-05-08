import pandas as pd

from risk_stratification_engine.coverage_policy import (
    COVERAGE_ELIGIBILITY_SCOPES,
    build_coverage_normalized_policy_summary,
    write_coverage_normalized_policy_report,
)


def test_build_coverage_normalized_policy_summary_marks_ready_channel_when_scopes_are_stable():
    audits = {
        "all": {
            "channel_summaries": [
                {"channel_name": "severity_14d", "stability_status": "stable"},
                {"channel_name": "broad_30d", "stability_status": "unstable"},
            ]
        },
        "medium_high": {
            "channel_summaries": [
                {"channel_name": "severity_14d", "stability_status": "stable"},
                {"channel_name": "broad_30d", "stability_status": "unstable"},
            ]
        },
        "high_only": {
            "channel_summaries": [
                {"channel_name": "severity_14d", "stability_status": "stable"},
                {"channel_name": "broad_30d", "stability_status": "unstable"},
            ]
        },
    }

    summary = build_coverage_normalized_policy_summary(audits)

    assert summary["experiment_type"] == "coverage_normalized_policy_sprint"
    assert summary["eligibility_scopes"] == list(COVERAGE_ELIGIBILITY_SCOPES)
    assert (
        summary["channel_recommendations"]["severity_14d"]["recommendation"]
        == "candidate_after_coverage_control"
    )
    assert (
        summary["channel_recommendations"]["broad_30d"]["recommendation"]
        == "continue_research_review"
    )


def test_build_coverage_normalized_policy_summary_handles_missing_scope_evidence():
    audits = {
        "all": {
            "channel_summaries": [
                {"channel_name": "severity_14d", "stability_status": "stable"},
            ]
        },
        "medium_high": {"channel_summaries": []},
        "high_only": {"channel_summaries": []},
    }

    summary = build_coverage_normalized_policy_summary(audits)

    row = summary["channel_recommendations"]["severity_14d"]
    assert row["recommendation"] == "continue_research_review"
    assert row["stable_scope_count"] == 1
    assert row["evaluated_scope_count"] == 1


def test_write_coverage_normalized_policy_report_names_peterson_guardrail(tmp_path):
    summary = {
        "experiment_type": "coverage_normalized_policy_sprint",
        "overall_recommendation": "continue_research_shadow_mode",
        "eligibility_scopes": ["all"],
        "channel_recommendations": {
            "severity_14d": {
                "recommendation": "candidate_after_coverage_control",
                "stable_scope_count": 1,
                "evaluated_scope_count": 1,
                "scope_statuses": {"all": "stable"},
            }
        },
    }
    path = tmp_path / "coverage_normalized_policy_report.md"

    write_coverage_normalized_policy_report(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "Coverage-Normalized Policy Sprint" in text
    assert "athlete-season trajectories" in text
    assert "severity_14d" in text
