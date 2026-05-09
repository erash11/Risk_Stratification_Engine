from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


def build_exposure_load_source_context_classification_summary(
    exposure_events: list[dict[str, object]],
    exposure_participations: list[dict[str, object]],
    exposure_load_shift_context: dict[str, object],
    exposure_load_schedule_roster: dict[str, object],
    exposure_load_availability_capture: dict[str, object],
    exposure_load_context_decision: dict[str, object],
) -> dict[str, object]:
    events = _normalized_events(pd.DataFrame(exposure_events))
    participations = _normalized_participations(pd.DataFrame(exposure_participations))
    failure_seasons = [
        str(season)
        for season in exposure_load_shift_context.get("failure_seasons", [])
    ]
    comparator_seasons = [
        str(season)
        for season in exposure_load_shift_context.get("comparator_seasons", [])
    ]
    evidence_rows = _source_evidence_rows(
        events,
        participations,
        failure_seasons,
        comparator_seasons,
    )
    evidence = {row["evidence_domain"]: row for row in evidence_rows}
    classification_rows = _classification_rows(
        evidence=evidence,
        schedule_roster_summary=exposure_load_schedule_roster,
        availability_capture_summary=exposure_load_availability_capture,
        context_decision_summary=exposure_load_context_decision,
    )
    return {
        "experiment_type": "exposure_load_source_context_classification_sprint",
        "overall_recommendation": _overall_recommendation(classification_rows),
        "production_readiness": exposure_load_context_decision.get(
            "production_readiness",
            "not_ready_for_probability_or_pilot",
        ),
        "failure_seasons": failure_seasons,
        "comparator_seasons": comparator_seasons,
        "source_evidence_rows": evidence_rows,
        "source_context_classification_rows": classification_rows,
    }


def write_exposure_load_source_context_classification_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Source Context Classification Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint classifies the failed exposure-load season using "
            "source-level schedule, roster, availability, and capture evidence. "
            "It separates true managed-risk context from schedule/roster shifts "
            "and documentation artifacts. This is not pilot clearance."
        ),
        "",
        "## Classification",
        "",
        "| Domain | Classification | Confidence | Evidence |",
        "|---|---|---|---|",
    ]
    for row in summary.get("source_context_classification_rows", []):
        lines.append(
            "| "
            f"{row['classification_domain']} | "
            f"{row['classification']} | "
            f"{row['confidence']} | "
            f"{row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Source Evidence",
            "",
            "| Domain | Key failure value | Key comparator value | Review signal |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary.get("source_evidence_rows", []):
        lines.append(
            "| "
            f"{row['evidence_domain']} | "
            f"{_fmt(row.get('key_failure_value'))} | "
            f"{_fmt(row.get('key_comparator_value'))} | "
            f"{row['review_signal']} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _source_evidence_rows(
    events: pd.DataFrame,
    participations: pd.DataFrame,
    failure_seasons: list[str],
    comparator_seasons: list[str],
) -> list[dict[str, object]]:
    failure_events = events[events["season_id"].isin(failure_seasons)]
    comparator_events = events[events["season_id"].isin(comparator_seasons)]
    failure_parts = participations[participations["season_id"].isin(failure_seasons)]
    comparator_parts = participations[
        participations["season_id"].isin(comparator_seasons)
    ]
    failure_schedule = _schedule_metrics(failure_events)
    comparator_schedule = _schedule_metrics(comparator_events, len(comparator_seasons))
    failure_flags = _participation_flag_metrics(failure_parts)
    comparator_flags = _participation_flag_metrics(
        comparator_parts,
        max(len(comparator_seasons), 1),
    )
    failure_capture = _capture_metrics(failure_parts)
    comparator_capture = _capture_metrics(comparator_parts)
    return [
        _clean_row(
            {
                "evidence_domain": "source_schedule",
                "failure_event_count": failure_schedule["event_count"],
                "comparator_event_count": comparator_schedule["event_count"],
                "failure_game_event_count": failure_schedule["game_event_count"],
                "comparator_game_event_count": comparator_schedule["game_event_count"],
                "failure_lift_event_count": failure_schedule["lift_event_count"],
                "comparator_lift_event_count": comparator_schedule["lift_event_count"],
                "key_failure_value": failure_schedule["game_event_count"],
                "key_comparator_value": comparator_schedule["game_event_count"],
                "review_signal": _schedule_signal(failure_schedule, comparator_schedule),
            }
        ),
        _clean_row(
            {
                "evidence_domain": "source_participation_flags",
                "failure_modified_rate": failure_flags["modified_rate"],
                "comparator_modified_rate": comparator_flags["modified_rate"],
                "failure_no_participation_rate": failure_flags["no_participation_rate"],
                "comparator_no_participation_rate": comparator_flags[
                    "no_participation_rate"
                ],
                "failure_linked_issue_rate": failure_flags["linked_issue_rate"],
                "comparator_linked_issue_rate": comparator_flags["linked_issue_rate"],
                "key_failure_value": failure_flags["modified_rate"],
                "key_comparator_value": comparator_flags["modified_rate"],
                "review_signal": _availability_signal(failure_flags, comparator_flags),
            }
        ),
        _clean_row(
            {
                "evidence_domain": "source_capture_completeness",
                "failure_duration_recorded_rate": failure_capture["duration_rate"],
                "comparator_duration_recorded_rate": comparator_capture[
                    "duration_rate"
                ],
                "failure_rpe_recorded_rate": failure_capture["rpe_rate"],
                "comparator_rpe_recorded_rate": comparator_capture["rpe_rate"],
                "failure_workload_recorded_rate": failure_capture["workload_rate"],
                "comparator_workload_recorded_rate": comparator_capture[
                    "workload_rate"
                ],
                "key_failure_value": failure_capture["rpe_rate"],
                "key_comparator_value": comparator_capture["rpe_rate"],
                "review_signal": _capture_signal(failure_capture, comparator_capture),
            }
        ),
    ]


def _classification_rows(
    evidence: dict[str, dict[str, object]],
    schedule_roster_summary: dict[str, object],
    availability_capture_summary: dict[str, object],
    context_decision_summary: dict[str, object],
) -> list[dict[str, object]]:
    flag_evidence = evidence.get("source_participation_flags", {})
    schedule_supported = bool(schedule_roster_summary.get("schedule_roster_drivers"))
    capture_supported = bool(
        availability_capture_summary.get("availability_capture_drivers")
    )
    managed_supported = _managed_risk_supported(flag_evidence)
    return [
        _clean_row(
            {
                "classification_domain": "managed_risk_support",
                "classification": "supported_true_managed_risk_context"
                if managed_supported
                else "not_supported_by_source_flags",
                "confidence": "medium" if managed_supported else "high",
                "evidence": flag_evidence.get("review_signal", ""),
                "required_next_step": (
                    "review issue-linked participation context"
                    if managed_supported
                    else "do not interpret lower availability flagging as managed risk"
                ),
            }
        ),
        _clean_row(
            {
                "classification_domain": "schedule_roster_context",
                "classification": "supported_schedule_roster_shift"
                if schedule_supported
                else "not_supported",
                "confidence": "high" if schedule_supported else "low",
                "evidence": schedule_roster_summary.get("overall_recommendation", ""),
                "required_next_step": (
                    "review game, training, lift, and participation-density changes"
                ),
            }
        ),
        _clean_row(
            {
                "classification_domain": "availability_capture_context",
                "classification": "supported_capture_or_documentation_shift"
                if capture_supported
                else "not_supported",
                "confidence": "high" if capture_supported else "low",
                "evidence": availability_capture_summary.get(
                    "overall_recommendation",
                    "",
                ),
                "required_next_step": (
                    "review modified/no-participation documentation and source linkage"
                ),
            }
        ),
        _clean_row(
            {
                "classification_domain": "next_model_action",
                "classification": "do_not_expand_model_features"
                if context_decision_summary.get("production_readiness")
                == "not_ready_for_probability_or_pilot"
                else "continue_research_validation",
                "confidence": "high",
                "evidence": context_decision_summary.get("overall_recommendation", ""),
                "required_next_step": (
                    "resolve context classification before minute-load or probability work"
                ),
            }
        ),
    ]


def _overall_recommendation(rows: list[dict[str, object]]) -> str:
    classifications = {
        row["classification_domain"]: row["classification"] for row in rows
    }
    if classifications.get("managed_risk_support") == "supported_true_managed_risk_context":
        return "treat_failed_season_as_managed_risk_candidate"
    if (
        classifications.get("schedule_roster_context")
        == "supported_schedule_roster_shift"
        and classifications.get("availability_capture_context")
        == "supported_capture_or_documentation_shift"
    ):
        return "treat_failed_season_as_schedule_roster_plus_capture_shift"
    if classifications.get("schedule_roster_context") == "supported_schedule_roster_shift":
        return "treat_failed_season_as_schedule_roster_shift"
    return "keep_failed_season_context_under_source_review"


def _managed_risk_supported(flag_evidence: dict[str, object]) -> bool:
    modified = float(flag_evidence.get("failure_modified_rate") or 0.0)
    comparator_modified = float(flag_evidence.get("comparator_modified_rate") or 0.0)
    linked = float(flag_evidence.get("failure_linked_issue_rate") or 0.0)
    return linked > 0.0 and modified >= comparator_modified


def _schedule_metrics(events: pd.DataFrame, divisor: int = 1) -> dict[str, float]:
    divisor = max(divisor, 1)
    event_type = events["event_type"].astype(str).str.lower()
    category = events["exposure_category"].astype(str).str.lower()
    return {
        "event_count": len(events) / divisor,
        "game_event_count": (
            (event_type.eq("game") | category.eq("game")).sum() / divisor
        ),
        "lift_event_count": category.str.contains("weight_room").sum() / divisor,
    }


def _participation_flag_metrics(
    participations: pd.DataFrame,
    divisor: int = 1,
) -> dict[str, float]:
    if participations.empty:
        return {
            "modified_rate": 0.0,
            "no_participation_rate": 0.0,
            "linked_issue_rate": 0.0,
        }
    category = participations["participation_category"].astype(str).str.lower()
    linked = (
        participations.get("related_external_issue_id", pd.Series([], dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    total = len(participations) / max(divisor, 1)
    return {
        "modified_rate": float(category.eq("modified").sum()) / len(participations),
        "no_participation_rate": float(category.eq("no_participation").sum())
        / len(participations),
        "linked_issue_rate": float(linked.sum()) / len(participations),
        "season_normalized_participations": total,
    }


def _capture_metrics(participations: pd.DataFrame) -> dict[str, float]:
    return {
        "duration_rate": _recorded_rate(participations, "duration_minutes"),
        "rpe_rate": _recorded_rate(participations, "rpe"),
        "workload_rate": _recorded_rate(participations, "workload_unit_amount"),
    }


def _schedule_signal(
    failure: dict[str, float],
    comparator: dict[str, float],
) -> str:
    signals = []
    if failure["game_event_count"] > comparator["game_event_count"]:
        signals.append("elevated_game_schedule")
    if failure["lift_event_count"] < comparator["lift_event_count"]:
        signals.append("reduced_lift_schedule")
    return "; ".join(signals) if signals else "no_major_schedule_shift"


def _availability_signal(
    failure: dict[str, float],
    comparator: dict[str, float],
) -> str:
    signals = []
    if failure["modified_rate"] < comparator["modified_rate"]:
        signals.append("reduced_modified_availability_flagging")
    if failure["no_participation_rate"] < comparator["no_participation_rate"]:
        signals.append("reduced_no_participation_flagging")
    if failure["linked_issue_rate"] <= 0:
        signals.append("issue_linkage_absent")
    return "; ".join(signals) if signals else "no_major_availability_shift"


def _capture_signal(
    failure: dict[str, float],
    comparator: dict[str, float],
) -> str:
    signals = []
    if failure["duration_rate"] > comparator["duration_rate"]:
        signals.append("duration_capture_elevated")
    if failure["rpe_rate"] > comparator["rpe_rate"]:
        signals.append("rpe_capture_elevated")
    if failure["workload_rate"] < comparator["workload_rate"]:
        signals.append("workload_capture_reduced")
    return "; ".join(signals) if signals else "no_major_capture_shift"


def _normalized_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    for column in ("season_id", "event_type", "exposure_category"):
        if column not in frame:
            frame[column] = ""
    frame["season_id"] = frame["season_id"].fillna("").astype(str)
    return frame


def _normalized_participations(participations: pd.DataFrame) -> pd.DataFrame:
    frame = participations.copy()
    for column in ("season_id", "participation_category"):
        if column not in frame:
            frame[column] = ""
    frame["season_id"] = frame["season_id"].fillna("").astype(str)
    if "athlete_match_status" in frame:
        frame = frame[frame["athlete_match_status"].astype(str).eq("matched")]
    return frame


def _recorded_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.notna().sum()) / len(frame)


def _interpretation(summary: dict[str, object]) -> str:
    if (
        summary["overall_recommendation"]
        == "treat_failed_season_as_schedule_roster_plus_capture_shift"
    ):
        return (
            "The failed season is better classified as a schedule/roster plus "
            "exposure-capture or documentation shift than as a true managed-risk "
            "context. Keep exposure-load shadow-only until the source context is "
            "resolved."
        )
    return (
        "The failed season still needs source-level review before exposure-load "
        "model expansion or probability-facing use."
    )


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "n/a"
    return f"{number:.3f}"


def clean_source_context_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
