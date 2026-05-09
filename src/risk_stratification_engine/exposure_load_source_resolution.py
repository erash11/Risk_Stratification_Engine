from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


def build_exposure_load_source_resolution_policy(
    source_context_classification: dict[str, object],
) -> dict[str, object]:
    classification_rows = list(
        source_context_classification.get("source_context_classification_rows", [])
    )
    evidence_rows = list(source_context_classification.get("source_evidence_rows", []))
    classifications = {
        str(row.get("classification_domain")): str(row.get("classification"))
        for row in classification_rows
    }
    failure_seasons = [
        str(season)
        for season in source_context_classification.get("failure_seasons", [])
    ]
    comparator_seasons = [
        str(season)
        for season in source_context_classification.get("comparator_seasons", [])
    ]
    source_unresolved = _source_context_is_unresolved(classifications)
    policy_rows = _policy_rows(
        source_context_classification=source_context_classification,
        classifications=classifications,
        evidence_rows=evidence_rows,
        source_unresolved=source_unresolved,
    )
    actions = _resolution_actions(
        failure_seasons=failure_seasons,
        source_unresolved=source_unresolved,
    )
    return {
        "experiment_type": "exposure_load_source_resolution_sprint",
        "overall_recommendation": _overall_recommendation(source_unresolved),
        "production_readiness": source_context_classification.get(
            "production_readiness",
            "not_ready_for_probability_or_pilot",
        ),
        "source_context_recommendation": source_context_classification.get(
            "overall_recommendation",
            "",
        ),
        "failure_seasons": failure_seasons,
        "comparator_seasons": comparator_seasons,
        "policy_rows": policy_rows,
        "resolution_actions": actions,
    }


def write_exposure_load_source_resolution_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Source Resolution Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint converts source-context classification into an explicit "
            "eligibility policy for complete athlete-season trajectories. It does "
            "not reinterpret source-shift evidence as independent daily-row labels, "
            "and it is not pilot clearance."
        ),
        "",
        "## Policy",
        "",
        "| Domain | Decision | Eligibility | Evidence | Required action |",
        "|---|---|---|---|---|",
    ]
    for row in summary.get("policy_rows", []):
        lines.append(
            "| "
            f"{row['policy_domain']} | "
            f"{row['policy_decision']} | "
            f"{row['eligibility']} | "
            f"{row['evidence']} | "
            f"{row['required_action']} |"
        )
    lines.extend(
        [
            "",
            "## Resolution Actions",
            "",
            "| Domain | Priority | Action | Blocks |",
            "|---|---|---|---|",
        ]
    )
    for row in summary.get("resolution_actions", []):
        lines.append(
            "| "
            f"{row['action_domain']} | "
            f"{row['priority']} | "
            f"{row['action']} | "
            f"{row['blocks']} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _source_context_is_unresolved(classifications: dict[str, str]) -> bool:
    schedule_shift = (
        classifications.get("schedule_roster_context")
        == "supported_schedule_roster_shift"
    )
    capture_shift = (
        classifications.get("availability_capture_context")
        == "supported_capture_or_documentation_shift"
    )
    managed_risk_not_supported = (
        classifications.get("managed_risk_support") == "not_supported_by_source_flags"
    )
    no_expansion = (
        classifications.get("next_model_action") == "do_not_expand_model_features"
    )
    return (
        (schedule_shift or capture_shift)
        and managed_risk_not_supported
        and no_expansion
    )


def _policy_rows(
    source_context_classification: dict[str, object],
    classifications: dict[str, str],
    evidence_rows: list[object],
    source_unresolved: bool,
) -> list[dict[str, object]]:
    evidence = _evidence_text(evidence_rows)
    source_recommendation = str(
        source_context_classification.get("overall_recommendation", "")
    )
    if not source_unresolved:
        return [
            _clean_row(
                {
                    "policy_domain": "source_resolution",
                    "policy_decision": "continue_source_review",
                    "eligibility": "research_review_only",
                    "evidence": source_recommendation,
                    "required_action": "complete source review before policy changes",
                }
            )
        ]
    return [
        _clean_row(
            {
                "policy_domain": "season_eligibility",
                "policy_decision": (
                    "exclude_failed_season_from_probability_calibration"
                ),
                "eligibility": "failed_season_excluded_until_resolved",
                "evidence": source_recommendation,
                "required_action": (
                    "repair or document schedule, roster, and capture source shift"
                ),
            }
        ),
        _clean_row(
            {
                "policy_domain": "probability_calibration",
                "policy_decision": "blocked_pending_source_resolution",
                "eligibility": "not_probability_facing",
                "evidence": evidence,
                "required_action": "rerun calibration only after source eligibility passes",
            }
        ),
        _clean_row(
            {
                "policy_domain": "shadow_ranking",
                "policy_decision": "allowed_shadow_only_with_season_monitoring",
                "eligibility": "shadow_research_only",
                "evidence": "forward ranking signal survived but calibration did not",
                "required_action": "monitor season-local calibration and alert burden",
            }
        ),
        _clean_row(
            {
                "policy_domain": "model_expansion",
                "policy_decision": "blocked_pending_source_resolution",
                "eligibility": "no_new_feature_domains",
                "evidence": classifications.get("next_model_action", ""),
                "required_action": "resolve source context before adding model domains",
            }
        ),
        _clean_row(
            {
                "policy_domain": "minute_load_expansion",
                "policy_decision": "deferred_until_source_resolution",
                "eligibility": "deferred",
                "evidence": "capture shift can distort duration or minute-load terms",
                "required_action": "prove capture consistency before duration features",
            }
        ),
    ]


def _resolution_actions(
    failure_seasons: list[str],
    source_unresolved: bool,
) -> list[dict[str, object]]:
    failed = ", ".join(failure_seasons) if failure_seasons else "failed season"
    if not source_unresolved:
        return [
            _clean_row(
                {
                    "action_domain": "source_review",
                    "priority": "medium",
                    "action": "continue source review before changing eligibility",
                    "owner_question": "Does source review support eligibility changes?",
                    "blocks": "probability_calibration",
                }
            )
        ]
    return [
        _clean_row(
            {
                "action_domain": "source_resolution",
                "priority": "high",
                "action": (
                    f"resolve {failed} schedule-roster and capture shift against "
                    "source-system context"
                ),
                "owner_question": (
                    "Were game, lift, roster, and participation-documentation changes "
                    "real operating context or source artifacts?"
                ),
                "blocks": "probability_calibration",
            }
        ),
        _clean_row(
            {
                "action_domain": "availability_capture_repair",
                "priority": "high",
                "action": (
                    "repair or explicitly document modified, no-participation, "
                    "and issue-linkage capture"
                ),
                "owner_question": (
                    "Can availability capture be made comparable to comparator seasons?"
                ),
                "blocks": "model_expansion",
            }
        ),
        _clean_row(
            {
                "action_domain": "probability_dataset",
                "priority": "high",
                "action": (
                    f"exclude {failed} from probability calibration datasets until "
                    "eligibility is resolved"
                ),
                "owner_question": (
                    "Does the calibration dataset include only source-eligible seasons?"
                ),
                "blocks": "probability_calibration",
            }
        ),
        _clean_row(
            {
                "action_domain": "shadow_ranking_monitoring",
                "priority": "medium",
                "action": (
                    "retain exposure-load shadow ranking with season-level calibration "
                    "and burden monitoring"
                ),
                "owner_question": (
                    "Does ranking remain useful without probability-facing claims?"
                ),
                "blocks": "pilot_escalation",
            }
        ),
    ]


def _overall_recommendation(source_unresolved: bool) -> str:
    if source_unresolved:
        return "exclude_failed_season_from_probability_calibration_until_source_resolved"
    return "continue_source_review_before_probability_policy_change"


def _evidence_text(evidence_rows: list[object]) -> str:
    signals = []
    for row in evidence_rows:
        if isinstance(row, dict):
            signal = str(row.get("review_signal", "")).strip()
            if signal:
                signals.append(signal)
    return "; ".join(dict.fromkeys(signals)) if signals else "source context unresolved"


def _interpretation(summary: dict[str, object]) -> str:
    if (
        summary["overall_recommendation"]
        == "exclude_failed_season_from_probability_calibration_until_source_resolved"
    ):
        return (
            "Exposure-load remains useful for shadow ranking research, but the failed "
            "season must be held out of probability calibration until schedule, roster, "
            "and availability-capture source context is resolved."
        )
    return (
        "Source review remains unresolved. Keep exposure-load out of probability-facing "
        "and pilot workflows until eligibility is explicitly controlled."
    )


def clean_source_resolution_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
