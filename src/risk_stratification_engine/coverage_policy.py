from __future__ import annotations

from pathlib import Path


COVERAGE_ELIGIBILITY_SCOPES = ("all", "medium_high", "high_only")

COVERAGE_SCOPE_TIERS = {
    "all": ("low", "medium", "high"),
    "medium_high": ("medium", "high"),
    "high_only": ("high",),
}


def build_coverage_normalized_policy_summary(
    scope_audits: dict[str, dict[str, object]],
) -> dict[str, object]:
    channel_names = sorted(
        {
            str(row["channel_name"])
            for audit in scope_audits.values()
            for row in audit.get("channel_summaries", [])
        }
    )
    recommendations = {}
    for channel_name in channel_names:
        scope_statuses = {}
        for scope in COVERAGE_ELIGIBILITY_SCOPES:
            status = _channel_status(scope_audits.get(scope, {}), channel_name)
            if status is not None:
                scope_statuses[scope] = status
        evaluated = len(scope_statuses)
        stable = sum(1 for status in scope_statuses.values() if status == "stable")
        recommendations[channel_name] = {
            "channel_name": channel_name,
            "evaluated_scope_count": evaluated,
            "stable_scope_count": stable,
            "scope_statuses": scope_statuses,
            "recommendation": (
                "candidate_after_coverage_control"
                if evaluated == len(COVERAGE_ELIGIBILITY_SCOPES)
                and stable == evaluated
                else "continue_research_review"
            ),
        }
    overall = (
        "candidate_channels_after_coverage_control"
        if any(
            row["recommendation"] == "candidate_after_coverage_control"
            for row in recommendations.values()
        )
        else "continue_research_shadow_mode"
    )
    return {
        "experiment_type": "coverage_normalized_policy_sprint",
        "overall_recommendation": overall,
        "eligibility_scopes": list(COVERAGE_ELIGIBILITY_SCOPES),
        "channel_recommendations": recommendations,
    }


def write_coverage_normalized_policy_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Coverage-Normalized Policy Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "Coverage controls are applied to complete athlete-season "
            "trajectories before season-local alert episodes are rebuilt. "
            "The sprint does not treat daily rows as independent examples."
        ),
        "",
        "## Channel Recommendations",
        "",
        "| Channel | Recommendation | Stable scopes | Scope statuses |",
        "|---|---|---:|---|",
    ]
    for channel_name, row in summary["channel_recommendations"].items():
        lines.append(
            "| "
            f"{channel_name} | "
            f"{row['recommendation']} | "
            f"{row['stable_scope_count']}/{row['evaluated_scope_count']} | "
            f"{_scope_status_text(row['scope_statuses'])} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _channel_status(audit: dict[str, object], channel_name: str) -> str | None:
    for row in audit.get("channel_summaries", []):
        if str(row.get("channel_name")) == channel_name:
            return str(row.get("stability_status"))
    return None


def _scope_status_text(statuses: dict[str, str]) -> str:
    if not statuses:
        return "n/a"
    return ", ".join(f"{scope}: {status}" for scope, status in statuses.items())


def _interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"] == "candidate_channels_after_coverage_control":
        return (
            "At least one fixed channel remained stable across all coverage "
            "eligibility scopes. Treat this as research shadow-mode evidence, "
            "not dashboard clearance."
        )
    return (
        "No fixed channel remained stable across all coverage eligibility scopes. "
        "Continue research review before dashboard or intervention work."
    )
