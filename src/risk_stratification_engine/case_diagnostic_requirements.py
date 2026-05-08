from __future__ import annotations

from collections import Counter
from pathlib import Path


DOMAIN_SPECS = (
    {
        "requirement_domain": "exposure_load",
        "diagnostic_labels": ("model_miss", "missing_context_or_managed_risk"),
        "case_types": ("missed_injury", "false_positive_episode"),
        "missing_data_fields": (
            "session_participation",
            "minutes_exposed",
            "practice_intensity",
            "acute_chronic_load",
            "game_exposure",
        ),
        "modeling_action": "add exposure_load_features before threshold escalation",
        "production_relevance": "separates true physiological risk from unobserved workload",
    },
    {
        "requirement_domain": "intervention_availability",
        "diagnostic_labels": ("missing_context_or_managed_risk",),
        "case_types": ("false_positive_episode",),
        "missing_data_fields": (
            "availability_status",
            "modified_training_status",
            "treatment_or_rehab_flag",
            "return_to_play_stage",
        ),
        "modeling_action": "add intervention_availability_features to distinguish managed alerts",
        "production_relevance": "prevents alerts that are already being managed from looking like misses",
    },
    {
        "requirement_domain": "baseline_frailty",
        "diagnostic_labels": ("model_miss", "explanation_gap"),
        "case_types": ("missed_injury", "high_intra_individual_deviation_episode"),
        "missing_data_fields": (
            "prior_injury_count",
            "chronic_condition_flag",
            "athlete_baseline_state",
            "recent_availability_history",
        ),
        "modeling_action": "add athlete_frailty_terms and athlete-relative baseline features",
        "production_relevance": "captures persistent athlete-specific susceptibility beyond graph movement",
    },
    {
        "requirement_domain": "injury_mechanism",
        "diagnostic_labels": ("model_miss",),
        "case_types": ("missed_injury",),
        "channel_tokens": ("subtype", "lower_extremity", "soft_tissue"),
        "missing_data_fields": (
            "injury_mechanism",
            "contact_non_contact",
            "activity_context",
            "body_area_detail",
            "tissue_specific_diagnosis",
        ),
        "modeling_action": "add mechanism_context_features and mechanism-specific targets",
        "production_relevance": "tests whether subtype misses are predictable or mostly mechanism-driven",
    },
    {
        "requirement_domain": "explanation_fidelity",
        "diagnostic_labels": ("explanation_gap",),
        "case_types": ("high_intra_individual_deviation_episode",),
        "missing_data_fields": (
            "edge_change_trace",
            "node_change_trace",
            "baseline_deviation_context",
            "top_feature_trajectory",
        ),
        "modeling_action": "upgrade graph explanation artifacts before operator-facing review",
        "production_relevance": "makes athlete-specific alerts auditable enough for human review",
    },
)


def build_case_diagnostic_requirements(
    cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    case_count = len(cases)
    rows: list[dict[str, object]] = []
    for spec in DOMAIN_SPECS:
        evidence_cases = [
            case for case in cases if _case_matches_requirement(case, spec)
        ]
        diagnostic_labels = sorted(
            {
                str(case.get("diagnostic_label"))
                for case in evidence_cases
                if case.get("diagnostic_label")
            }
        )
        channels = sorted(
            {
                str(case.get("channel_name"))
                for case in evidence_cases
                if case.get("channel_name")
            }
        )
        evidence_count = len(evidence_cases)
        rows.append(
            {
                "requirement_domain": str(spec["requirement_domain"]),
                "priority_tier": _priority_tier(evidence_count, case_count),
                "evidence_case_count": evidence_count,
                "evidence_case_share": (
                    round(evidence_count / case_count, 4) if case_count else 0.0
                ),
                "triggering_diagnostic_labels": diagnostic_labels,
                "affected_channels": channels,
                "missing_data_fields": list(spec["missing_data_fields"]),
                "modeling_action": str(spec["modeling_action"]),
                "production_relevance": str(spec["production_relevance"]),
                "example_case_ids": _example_case_ids(evidence_cases),
            }
        )
    return rows


def build_case_diagnostic_requirements_summary(
    requirements: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    priority_counts = Counter(str(row["priority_tier"]) for row in requirements)
    diagnostic_counts = Counter(
        str(case.get("diagnostic_label")) for case in cases if case.get("diagnostic_label")
    )
    return {
        "experiment_type": "case_diagnostic_requirements_sprint",
        "overall_recommendation": _overall_recommendation(requirements),
        "production_readiness": "not_ready_missing_context",
        "case_count": len(cases),
        "requirement_count": len(requirements),
        "critical_requirement_count": int(priority_counts.get("critical", 0)),
        "priority_summary": dict(sorted(priority_counts.items())),
        "diagnostic_summary": dict(sorted(diagnostic_counts.items())),
        "requirements": requirements,
    }


def write_case_diagnostic_requirements_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Case Diagnostic Requirements Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Cases reviewed: {summary['case_count']}",
        "",
        "## Production Blockers",
        "",
        (
            "These production blockers identify the exposure, intervention, "
            "baseline/frailty, and injury mechanism context needed before the "
            "model is worth pilot escalation."
        ),
        "",
        "| Requirement domain | Priority | Evidence cases | Modeling action |",
        "|---|---|---:|---|",
    ]
    for row in summary["requirements"]:
        lines.append(
            "| "
            f"{row['requirement_domain']} | "
            f"{row['priority_tier']} | "
            f"{row['evidence_case_count']} | "
            f"{row['modeling_action']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Do not treat these as dashboard requirements yet. They are "
                "model-viability requirements: collect or derive the fields, "
                "then rerun forward validation before any shadow pilot."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _case_matches_requirement(
    case: dict[str, object],
    spec: dict[str, object],
) -> bool:
    diagnostic_label = str(case.get("diagnostic_label", ""))
    case_type = str(case.get("case_type", ""))
    channel_name = str(case.get("channel_name", ""))
    if diagnostic_label in spec.get("diagnostic_labels", ()):
        return True
    if case_type in spec.get("case_types", ()):
        return True
    return any(token in channel_name for token in spec.get("channel_tokens", ()))


def _priority_tier(evidence_count: int, case_count: int) -> str:
    if evidence_count == 0:
        return "watch"
    critical_threshold = max(2, int(case_count * 0.25))
    if evidence_count >= critical_threshold:
        return "critical"
    return "high"


def _overall_recommendation(requirements: list[dict[str, object]]) -> str:
    if any(row["priority_tier"] == "critical" for row in requirements):
        return "prioritize_data_acquisition_before_production"
    return "continue_targeted_requirements_review"


def _example_case_ids(cases: list[dict[str, object]]) -> list[str]:
    examples: list[str] = []
    for index, case in enumerate(cases[:3], start=1):
        channel = str(case.get("channel_name", "unknown_channel"))
        case_type = str(case.get("case_type", "unknown_case"))
        examples.append(f"{index}:{channel}:{case_type}")
    return examples
