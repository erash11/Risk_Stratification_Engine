from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


def build_exposure_load_guardrail_policy(
    failure_mode_summary: dict[str, object],
    diagnostic_rows: list[dict[str, object]],
) -> dict[str, object]:
    high_failure_rows = [
        row
        for row in diagnostic_rows
        if row.get("diagnostic_label") == "ranking_triage_gain_calibration_loss"
    ]
    over_sharpened_rows = [
        row
        for row in diagnostic_rows
        if row.get("target_reason") == "over_sharpened_probability_slice"
        or (
            row.get("diagnostic_label") == "ranking_triage_gain_calibration_loss"
            and _positive(row.get("delta_prediction_to_observed_gap"))
        )
    ]
    calibration_supported_rows = [
        row
        for row in diagnostic_rows
        if row.get("diagnostic_label") == "calibration_supported"
    ]
    top_domains = _top_domains(failure_mode_summary)
    guardrail_rows = _guardrail_rows(
        high_failure_rows=high_failure_rows,
        over_sharpened_rows=over_sharpened_rows,
        calibration_supported_rows=calibration_supported_rows,
        top_domains=top_domains,
    )
    return {
        "experiment_type": "exposure_load_guardrail_policy_sprint",
        "overall_recommendation": _overall_recommendation(
            high_failure_rows,
            over_sharpened_rows,
        ),
        "production_readiness": _production_readiness(
            high_failure_rows,
            over_sharpened_rows,
        ),
        "failure_seasons": failure_mode_summary.get("failure_seasons", []),
        "top_shifted_domains": top_domains,
        "diagnostic_label_summary": _counter(diagnostic_rows, "diagnostic_label"),
        "target_reason_summary": _counter(diagnostic_rows, "target_reason"),
        "guardrail_rows": guardrail_rows,
    }


def write_exposure_load_guardrail_policy_report(
    path: Path,
    policy: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Guardrail Policy Sprint",
        "",
        f"Recommendation: {policy['overall_recommendation']}",
        f"Production readiness: {policy['production_readiness']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint converts the exposure-load diagnostics into research "
            "operating guardrails. It preserves athlete-season trajectory "
            "framing and is not production or pilot clearance."
        ),
        "",
        "## Guardrail Decisions",
        "",
        "| Domain | Decision | Priority | Required next step |",
        "|---|---|---|---|",
    ]
    for row in policy.get("guardrail_rows", []):
        lines.append(
            "| "
            f"{row['guardrail_domain']} | "
            f"{row['decision']} | "
            f"{row['priority_tier']} | "
            f"{row['required_next_step']} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(policy)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _guardrail_rows(
    high_failure_rows: list[dict[str, object]],
    over_sharpened_rows: list[dict[str, object]],
    calibration_supported_rows: list[dict[str, object]],
    top_domains: list[str],
) -> list[dict[str, object]]:
    has_failure = bool(high_failure_rows or over_sharpened_rows)
    shifted_domain_text = ", ".join(top_domains[:3]) if top_domains else "none"
    return [
        {
            "guardrail_domain": "probability_calibration",
            "decision": "blocked_until_failure_mode_resolved"
            if has_failure
            else "continue_research_validation",
            "priority_tier": "critical" if has_failure else "watch",
            "evidence": (
                f"{len(over_sharpened_rows)} over-sharpened failure rows and "
                f"{len(high_failure_rows)} high-priority calibration-loss rows"
            ),
            "required_next_step": (
                "resolve high-priority forward calibration failure before "
                "probability-facing use"
            ),
        },
        {
            "guardrail_domain": "ranking_triage",
            "decision": "allowed_for_shadow_research",
            "priority_tier": "medium",
            "evidence": (
                f"{len(calibration_supported_rows)} calibration-supported "
                "comparator rows plus ranking/triage gains in failed slices"
            ),
            "required_next_step": (
                "keep exposure-load as shadow ranking signal with explicit "
                "calibration monitoring"
            ),
        },
        {
            "guardrail_domain": "minute_load_expansion",
            "decision": "defer" if has_failure else "research_candidate",
            "priority_tier": "critical" if has_failure else "medium",
            "evidence": (
                "count/status exposure features already over-sharpen one "
                "forward season"
            ),
            "required_next_step": (
                "review duration semantics and resolve count-based failure "
                "mode before adding minute-load terms"
            ),
        },
        {
            "guardrail_domain": "feature_domain_review",
            "decision": "required_before_next_model_expansion"
            if top_domains
            else "continue_monitoring",
            "priority_tier": "high" if top_domains else "watch",
            "evidence": f"top shifted domains: {shifted_domain_text}",
            "required_next_step": (
                "inspect shifted exposure domains against schedule, roster, "
                "availability, and managed-risk context"
            ),
        },
    ]


def _overall_recommendation(
    high_failure_rows: list[dict[str, object]],
    over_sharpened_rows: list[dict[str, object]],
) -> str:
    if high_failure_rows or over_sharpened_rows:
        return "use_exposure_load_for_shadow_ranking_only"
    return "continue_exposure_load_guardrail_research"


def _production_readiness(
    high_failure_rows: list[dict[str, object]],
    over_sharpened_rows: list[dict[str, object]],
) -> str:
    if high_failure_rows or over_sharpened_rows:
        return "not_ready_for_probability_or_pilot"
    return "not_ready_research_validation_required"


def _top_domains(failure_mode_summary: dict[str, object]) -> list[str]:
    domains = []
    for row in failure_mode_summary.get("domain_shift_summary", []):
        if not isinstance(row, dict):
            continue
        domain = row.get("feature_domain")
        if domain:
            domains.append(str(domain))
    return domains[:5]


def _interpretation(policy: dict[str, object]) -> str:
    if policy["overall_recommendation"] == "use_exposure_load_for_shadow_ranking_only":
        return (
            "Exposure-load can remain useful for shadow ranking and triage "
            "research, but probability calibration, pilot escalation, and "
            "minute-load expansion should stay blocked until the forward "
            "over-sharpening failure mode is resolved."
        )
    return (
        "Use this policy as a research guardrail while continuing to evaluate "
        "exposure-load behavior across complete athlete-season trajectories."
    )


def _counter(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row.get(field)) for row in rows if row.get(field)).items()
        )
    )


def _positive(value: object) -> bool:
    if value is None:
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def clean_guardrail_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _clean_row(row: dict[str, object]) -> dict[str, object]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if pd.isna(number):
            return None
        return int(number) if number.is_integer() else round(number, 6)
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value
