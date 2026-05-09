from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


def build_exposure_load_schedule_roster_summary(
    exposure_events: list[dict[str, object]],
    exposure_participations: list[dict[str, object]],
    exposure_load_shift_context: dict[str, object],
) -> dict[str, object]:
    events = _normalized_events(pd.DataFrame(exposure_events))
    participations = _normalized_participations(pd.DataFrame(exposure_participations))
    failure_seasons, comparator_seasons = _target_seasons(exposure_load_shift_context)
    seasons = sorted(set(failure_seasons + comparator_seasons))
    rows = [
        _schedule_roster_row(
            season,
            "failure" if season in failure_seasons else "comparator",
            events[events["season_id"].eq(season)],
            participations[participations["season_id"].eq(season)],
        )
        for season in seasons
    ]
    drivers = _comparison_drivers(
        rows,
        failure_seasons,
        comparator_seasons,
        metrics={
            "event_count": ("elevated_event_volume", "reduced_event_volume"),
            "game_event_count": ("elevated_game_schedule", "reduced_game_schedule"),
            "training_event_count": (
                "elevated_training_schedule",
                "reduced_training_schedule",
            ),
            "lift_event_count": ("elevated_lift_schedule", "reduced_lift_schedule"),
            "active_athlete_count": (
                "larger_active_roster",
                "smaller_active_roster",
            ),
            "participations_per_athlete": (
                "elevated_participation_density",
                "reduced_participation_density",
            ),
            "median_event_gap_days": (
                "wider_event_spacing",
                "denser_event_spacing",
            ),
        },
    )
    return {
        "experiment_type": "exposure_load_schedule_roster_sprint",
        "overall_recommendation": _review_recommendation(
            drivers,
            "review_failed_season_schedule_roster_shift",
            "keep_schedule_roster_context_under_review",
        ),
        "failure_seasons": failure_seasons,
        "comparator_seasons": comparator_seasons,
        "source_shift_recommendation": exposure_load_shift_context.get(
            "overall_recommendation"
        ),
        "schedule_roster_rows": rows,
        "schedule_roster_drivers": drivers,
    }


def build_exposure_load_availability_capture_summary(
    exposure_participations: list[dict[str, object]],
    exposure_load_shift_context: dict[str, object],
) -> dict[str, object]:
    participations = _normalized_participations(pd.DataFrame(exposure_participations))
    failure_seasons, comparator_seasons = _target_seasons(exposure_load_shift_context)
    seasons = sorted(set(failure_seasons + comparator_seasons))
    rows = [
        _availability_capture_row(
            season,
            "failure" if season in failure_seasons else "comparator",
            participations[participations["season_id"].eq(season)],
        )
        for season in seasons
    ]
    drivers = _comparison_drivers(
        rows,
        failure_seasons,
        comparator_seasons,
        metrics={
            "modified_participation_rate": (
                "elevated_modified_availability_flagging",
                "reduced_modified_availability_flagging",
            ),
            "no_participation_rate": (
                "elevated_no_participation_flagging",
                "reduced_no_participation_flagging",
            ),
            "linked_issue_participation_count": (
                "issue_linkage_elevated",
                "issue_linkage_absent_or_reduced",
            ),
            "duration_recorded_rate": (
                "duration_capture_elevated",
                "duration_capture_reduced",
            ),
            "rpe_recorded_rate": ("rpe_capture_elevated", "rpe_capture_reduced"),
            "workload_recorded_rate": (
                "workload_capture_elevated",
                "workload_capture_reduced",
            ),
        },
    )
    return {
        "experiment_type": "exposure_load_availability_capture_sprint",
        "overall_recommendation": _review_recommendation(
            drivers,
            "review_failed_season_availability_capture",
            "keep_availability_capture_under_review",
        ),
        "failure_seasons": failure_seasons,
        "comparator_seasons": comparator_seasons,
        "source_shift_recommendation": exposure_load_shift_context.get(
            "overall_recommendation"
        ),
        "availability_capture_rows": rows,
        "availability_capture_drivers": drivers,
    }


def build_exposure_load_context_decision_summary(
    exposure_load_shift_context: dict[str, object],
    schedule_roster_summary: dict[str, object],
    availability_capture_summary: dict[str, object],
    guardrail_policy: dict[str, object],
) -> dict[str, object]:
    schedule_needs_review = bool(
        schedule_roster_summary.get("schedule_roster_drivers")
    ) or str(schedule_roster_summary.get("overall_recommendation", "")).startswith(
        "review_"
    )
    availability_needs_review = bool(
        availability_capture_summary.get("availability_capture_drivers")
    ) or str(availability_capture_summary.get("overall_recommendation", "")).startswith(
        "review_"
    )
    guardrail_blocked = guardrail_policy.get(
        "production_readiness"
    ) == "not_ready_for_probability_or_pilot"
    context_unresolved = schedule_needs_review or availability_needs_review
    decision_rows = [
        {
            "decision_domain": "probability_calibration",
            "decision": "blocked" if guardrail_blocked or context_unresolved else "research_candidate",
            "priority": "critical" if guardrail_blocked or context_unresolved else "watch",
            "evidence": _join_evidence(
                [
                    guardrail_policy.get("overall_recommendation"),
                    schedule_roster_summary.get("overall_recommendation"),
                    availability_capture_summary.get("overall_recommendation"),
                ]
            ),
            "required_next_step": (
                "resolve schedule, roster, availability, and managed-risk context "
                "before probability-facing outputs"
            ),
        },
        {
            "decision_domain": "minute_load_expansion",
            "decision": "blocked" if context_unresolved else "research_candidate",
            "priority": "critical" if context_unresolved else "medium",
            "evidence": _join_evidence(_driver_signals(schedule_roster_summary, availability_capture_summary)),
            "required_next_step": (
                "do not add duration or minute-load terms until count/status "
                "context drivers are reviewed"
            ),
        },
        {
            "decision_domain": "shadow_ranking",
            "decision": "allowed_with_monitoring",
            "priority": "medium",
            "evidence": str(guardrail_policy.get("overall_recommendation", "")),
            "required_next_step": (
                "keep exposure-load ranking in shadow research with calibration "
                "and context monitoring"
            ),
        },
        {
            "decision_domain": "model_expansion",
            "decision": "blocked_pending_context_review"
            if context_unresolved
            else "research_candidate",
            "priority": "high" if context_unresolved else "medium",
            "evidence": str(exposure_load_shift_context.get("overall_recommendation", "")),
            "required_next_step": (
                "classify the failed season as true managed-risk context, "
                "schedule/roster shift, or exposure-capture change"
            ),
        },
    ]
    return {
        "experiment_type": "exposure_load_context_decision_sprint",
        "overall_recommendation": _decision_recommendation(
            guardrail_blocked,
            context_unresolved,
        ),
        "production_readiness": "not_ready_for_probability_or_pilot"
        if guardrail_blocked or context_unresolved
        else "not_ready_research_validation_required",
        "failure_seasons": exposure_load_shift_context.get("failure_seasons", []),
        "schedule_roster_recommendation": schedule_roster_summary.get(
            "overall_recommendation"
        ),
        "availability_capture_recommendation": availability_capture_summary.get(
            "overall_recommendation"
        ),
        "guardrail_recommendation": guardrail_policy.get("overall_recommendation"),
        "decision_rows": [_clean_row(row) for row in decision_rows],
    }


def write_exposure_load_schedule_roster_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Schedule Roster Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint inspects schedule and roster context after complete "
            "athlete-season trajectories have already been scored. It does not "
            "create independent daily injury-classification rows."
        ),
        "",
        "## Schedule/Roster Drivers",
        "",
        "| Metric | Failure mean | Comparator mean | Direction | Review signal |",
        "|---|---:|---:|---|---|",
    ]
    for row in summary.get("schedule_roster_drivers", []):
        lines.append(_driver_markdown_row(row))
    lines.extend(
        [
            "",
            "## Season Rows",
            "",
            "| Season | Group | Events | Games | Training | Active athletes | Participations/athlete |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("schedule_roster_rows", []):
        lines.append(
            "| "
            f"{row['season_id']} | {row['context_group']} | "
            f"{_fmt(row.get('event_count'))} | "
            f"{_fmt(row.get('game_event_count'))} | "
            f"{_fmt(row.get('training_event_count'))} | "
            f"{_fmt(row.get('active_athlete_count'))} | "
            f"{_fmt(row.get('participations_per_athlete'))} |"
        )
    lines.extend(["", "## Interpretation", "", _schedule_interpretation(summary)])
    _write_report(path, lines)


def write_exposure_load_availability_capture_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Availability Capture Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint inspects availability flagging and managed-risk "
            "documentation after trajectory-level exposure-load diagnostics. "
            "It is a research audit, not pilot clearance."
        ),
        "",
        "## Availability Drivers",
        "",
        "| Metric | Failure mean | Comparator mean | Direction | Review signal |",
        "|---|---:|---:|---|---|",
    ]
    for row in summary.get("availability_capture_drivers", []):
        lines.append(_driver_markdown_row(row))
    lines.extend(
        [
            "",
            "## Season Rows",
            "",
            "| Season | Group | Modified rate | No-participation rate | Linked issue rows | RPE recorded |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("availability_capture_rows", []):
        lines.append(
            "| "
            f"{row['season_id']} | {row['context_group']} | "
            f"{_fmt(row.get('modified_participation_rate'))} | "
            f"{_fmt(row.get('no_participation_rate'))} | "
            f"{_fmt(row.get('linked_issue_participation_count'))} | "
            f"{_fmt(row.get('rpe_recorded_rate'))} |"
        )
    lines.extend(["", "## Interpretation", "", _availability_interpretation(summary)])
    _write_report(path, lines)


def write_exposure_load_context_decision_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Context Decision Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint synthesizes exposure-load context diagnostics into "
            "research operating decisions. It is not pilot clearance and does "
            "not authorize probability-facing outputs."
        ),
        "",
        "## Decisions",
        "",
        "| Domain | Decision | Priority | Required next step |",
        "|---|---|---|---|",
    ]
    for row in summary.get("decision_rows", []):
        lines.append(
            "| "
            f"{row['decision_domain']} | {row['decision']} | "
            f"{row['priority']} | {row['required_next_step']} |"
        )
    lines.extend(["", "## Interpretation", "", _decision_interpretation(summary)])
    _write_report(path, lines)


def _schedule_roster_row(
    season: str,
    group: str,
    events: pd.DataFrame,
    participations: pd.DataFrame,
) -> dict[str, object]:
    event_type = events["event_type"].astype(str).str.lower()
    category = events["exposure_category"].astype(str).str.lower()
    event_count = int(events["event_id"].nunique()) if "event_id" in events else len(events)
    training_count = int(event_type.eq("training").sum())
    game_count = int((event_type.eq("game") | category.eq("game")).sum())
    active_athletes = (
        int(participations["athlete_id"].nunique()) if "athlete_id" in participations else 0
    )
    return _clean_row(
        {
            "season_id": season,
            "context_group": group,
            "event_count": event_count,
            "game_event_count": game_count,
            "training_event_count": training_count,
            "practice_event_count": int(
                (category.str.contains("practice") | category.eq("scrimmage")).sum()
            ),
            "lift_event_count": int(category.str.contains("weight_room").sum()),
            "conditioning_event_count": int(
                (
                    category.str.contains("conditioning")
                    | category.str.contains("speed_power")
                ).sum()
            ),
            "game_to_training_ratio": _safe_rate(game_count, training_count),
            "active_athlete_count": active_athletes,
            "matched_participation_count": int(len(participations)),
            "participations_per_athlete": _safe_rate(len(participations), active_athletes),
            "median_event_gap_days": _median_event_gap(events),
        }
    )


def _availability_capture_row(
    season: str,
    group: str,
    participations: pd.DataFrame,
) -> dict[str, object]:
    total = len(participations)
    category = participations["participation_category"].astype(str).str.lower()
    linked_issue = (
        participations.get("related_external_issue_id", pd.Series([], dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    reasons = _top_values(participations, "participation_level_reason")
    return _clean_row(
        {
            "season_id": season,
            "context_group": group,
            "matched_participation_count": int(total),
            "full_participation_count": int(category.eq("full").sum()),
            "modified_participation_count": int(category.eq("modified").sum()),
            "no_participation_count": int(category.eq("no_participation").sum()),
            "modified_participation_rate": _safe_rate(
                int(category.eq("modified").sum()),
                total,
            ),
            "no_participation_rate": _safe_rate(
                int(category.eq("no_participation").sum()),
                total,
            ),
            "linked_issue_participation_count": int(linked_issue.sum()),
            "linked_issue_participation_rate": _safe_rate(int(linked_issue.sum()), total),
            "duration_recorded_rate": _recorded_rate(
                _numeric_column(participations, "duration_minutes")
            ),
            "rpe_recorded_rate": _recorded_rate(_numeric_column(participations, "rpe")),
            "workload_recorded_rate": _recorded_rate(
                _numeric_column(participations, "workload_unit_amount")
            ),
            "top_participation_reasons": reasons,
        }
    )


def _comparison_drivers(
    rows: list[dict[str, object]],
    failure_seasons: list[str],
    comparator_seasons: list[str],
    metrics: dict[str, tuple[str, str]],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    failure = frame[frame["season_id"].astype(str).isin(failure_seasons)]
    comparator = frame[frame["season_id"].astype(str).isin(comparator_seasons)]
    drivers = []
    for metric, (elevated_signal, reduced_signal) in metrics.items():
        if metric not in frame:
            continue
        failure_mean = _mean(pd.to_numeric(failure[metric], errors="coerce"))
        comparator_mean = _mean(pd.to_numeric(comparator[metric], errors="coerce"))
        if failure_mean is None or comparator_mean is None:
            continue
        delta = failure_mean - comparator_mean
        if abs(delta) < 1e-12:
            continue
        drivers.append(
            _clean_row(
                {
                    "metric_name": metric,
                    "failure_mean": failure_mean,
                    "comparator_mean": comparator_mean,
                    "mean_delta": delta,
                    "relative_shift": _relative_shift(failure_mean, comparator_mean),
                    "shift_direction": "elevated_in_failure"
                    if delta > 0
                    else "reduced_in_failure",
                    "review_signal": elevated_signal if delta > 0 else reduced_signal,
                }
            )
        )
    return sorted(
        drivers,
        key=lambda row: (
            -abs(float(row.get("relative_shift") or 0.0) - 1.0),
            str(row["metric_name"]),
        ),
    )


def _target_seasons(summary: dict[str, object]) -> tuple[list[str], list[str]]:
    failure = summary.get("failure_seasons", [])
    comparator = summary.get("comparator_seasons", [])
    return (
        sorted(str(season) for season in failure if str(season)),
        sorted(str(season) for season in comparator if str(season)),
    )


def _normalized_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    for column in ("season_id", "event_id", "event_type", "exposure_category", "date"):
        if column not in frame:
            frame[column] = ""
    frame["season_id"] = frame["season_id"].fillna("").astype(str)
    return frame


def _normalized_participations(participations: pd.DataFrame) -> pd.DataFrame:
    frame = participations.copy()
    for column in ("season_id", "athlete_id", "participation_category"):
        if column not in frame:
            frame[column] = ""
    frame["season_id"] = frame["season_id"].fillna("").astype(str)
    if "athlete_match_status" in frame:
        frame = frame[frame["athlete_match_status"].astype(str).eq("matched")]
    return frame


def _median_event_gap(events: pd.DataFrame) -> float | None:
    if "date" not in events:
        return None
    dates = pd.to_datetime(events["date"], errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return None
    gaps = dates.diff().dt.days.dropna()
    if gaps.empty:
        return None
    return float(gaps.median())


def _top_values(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    if values.empty:
        return ""
    counts = Counter(values)
    return "; ".join(f"{value}:{count}" for value, count in counts.most_common(3))


def _driver_signals(
    schedule_roster_summary: dict[str, object],
    availability_capture_summary: dict[str, object],
) -> list[object]:
    signals: list[object] = []
    for row in schedule_roster_summary.get("schedule_roster_drivers", []):
        if isinstance(row, dict):
            signals.append(row.get("review_signal"))
    for row in availability_capture_summary.get("availability_capture_drivers", []):
        if isinstance(row, dict):
            signals.append(row.get("review_signal"))
    return signals


def _decision_recommendation(
    guardrail_blocked: bool,
    context_unresolved: bool,
) -> str:
    if guardrail_blocked or context_unresolved:
        return "keep_shadow_ranking_and_resolve_context_before_model_expansion"
    return "continue_exposure_load_research_validation"


def _review_recommendation(
    drivers: list[dict[str, object]],
    review_label: str,
    fallback: str,
) -> str:
    return review_label if drivers else fallback


def _schedule_interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"].startswith("review_"):
        return (
            "The failed exposure-load season has schedule or roster differences "
            "large enough to review before treating the exposure-load probability "
            "signal as calibrated."
        )
    return "No schedule or roster shift was large enough to change the current guardrail."


def _availability_interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"].startswith("review_"):
        return (
            "Availability flagging or managed-risk documentation changed enough "
            "to keep exposure-load in research validation before minute-load or "
            "probability-facing expansion."
        )
    return "No availability-capture shift was large enough to change the current guardrail."


def _decision_interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"].startswith("keep_shadow"):
        return (
            "Exposure-load can remain a shadow ranking signal, but context "
            "drivers are still unresolved. This is not pilot clearance."
        )
    return "Exposure-load remains in research validation while operating decisions are monitored."


def _driver_markdown_row(row: dict[str, object]) -> str:
    return (
        "| "
        f"{row['metric_name']} | "
        f"{_fmt(row.get('failure_mean'))} | "
        f"{_fmt(row.get('comparator_mean'))} | "
        f"{row['shift_direction']} | "
        f"{row['review_signal']} |"
    )


def _write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _join_evidence(values: list[object]) -> str:
    parts = [str(value) for value in values if value]
    return "; ".join(parts)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _recorded_rate(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return _safe_rate(int(values.notna().sum()), len(values))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _relative_shift(failure_mean: float, comparator_mean: float) -> float | None:
    if comparator_mean == 0:
        return None
    return failure_mean / comparator_mean


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


def clean_context_review_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
