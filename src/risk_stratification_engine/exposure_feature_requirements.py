from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


COUNT_FEATURES = (
    "prior_7_day_training_session_count",
    "prior_14_day_training_session_count",
    "prior_28_day_training_session_count",
    "prior_game_count",
    "days_since_last_game",
)
DURATION_FEATURES = (
    "prior_7_day_training_duration_minutes",
    "prior_14_day_training_duration_minutes",
    "prior_28_day_training_duration_minutes",
    "prior_game_participation_minutes",
)
PARTICIPATION_STATUS_FEATURES = (
    "prior_7_day_full_modified_no_participation_counts",
    "prior_14_day_full_modified_no_participation_counts",
    "prior_28_day_full_modified_no_participation_counts",
    "days_since_last_modified_or_no_participation_session",
)
CATEGORY_FEATURES = ("coarse_practice_type_exposure_counts",)


def build_exposure_feature_requirements(
    events: pd.DataFrame,
    participations: pd.DataFrame,
    audit: dict[str, Any],
) -> list[dict[str, object]]:
    duration = _duration_summary(participations)
    category_summary = _category_summary(events)
    match_summary = _match_summary(audit)
    duplicate_count = _duplicate_count(audit)
    unclassified_count = sum(
        _safe_dict(
            audit.get("event_counts", {})
            .get("training", {})
            .get("unclassified_session_types", {})
        ).values()
    )
    training_events = int(
        audit.get("event_counts", {}).get("training", {}).get("retained_events", 0)
    )
    game_events = int(
        audit.get("event_counts", {}).get("game", {}).get("retained_events", 0)
    )
    candidate_features = set(audit.get("candidate_feature_definitions", []))

    rows = [
        {
            "requirement_domain": "session_count_load",
            "readiness_status": _status(training_events > 0 and duplicate_count == 0),
            "evidence_summary": (
                f"training_events={training_events}; game_events={game_events}; "
                f"duplicate_participation_keys={duplicate_count}"
            ),
            "recommended_first_pass_features": _available_features(
                COUNT_FEATURES,
                candidate_features,
            ),
            "modeling_action": (
                "attach time-safe prior-window session and game counts before "
                "adding minute-load terms"
            ),
        },
        {
            "requirement_domain": "participation_status",
            "readiness_status": _status(
                match_summary["overall_match_share"] >= 0.95
                and _has_status_variation(participations)
            ),
            "evidence_summary": (
                "overall_match_share="
                f"{match_summary['overall_match_share']:.4f}; "
                f"participation_categories={_participation_categories(participations)}"
            ),
            "recommended_first_pass_features": _available_features(
                PARTICIPATION_STATUS_FEATURES,
                candidate_features,
            ),
            "modeling_action": (
                "attach modified and no-participation history as managed-risk "
                "context"
            ),
        },
        {
            "requirement_domain": "duration_load",
            "readiness_status": _duration_status(duration),
            "evidence_summary": (
                "training_duration_missing_share="
                f"{duration['training']['missing_share']:.4f}; "
                "game_duration_missing_share="
                f"{duration['game']['missing_share']:.4f}"
            ),
            "recommended_first_pass_features": _available_features(
                DURATION_FEATURES,
                candidate_features,
            ),
            "modeling_action": (
                "treat minute-load features as secondary until participation "
                "duration missingness is reviewed by event type"
            ),
        },
        {
            "requirement_domain": "game_exposure",
            "readiness_status": _game_status(game_events, duration),
            "evidence_summary": (
                f"game_events={game_events}; "
                f"game_participation_duration_missing_share="
                f"{duration['game']['missing_share']:.4f}"
            ),
            "recommended_first_pass_features": _available_features(
                (
                    "prior_game_count",
                    "prior_game_participation_minutes",
                    "days_since_last_game",
                ),
                candidate_features,
            ),
            "modeling_action": (
                "use game counts and recency first; add game minutes only after "
                "duration semantics are confirmed"
            ),
        },
        {
            "requirement_domain": "category_specific_load",
            "readiness_status": _status(
                unclassified_count == 0
                and len(category_summary.get("training", {})) >= 2
            ),
            "evidence_summary": (
                f"training_category_count={len(category_summary.get('training', {}))}; "
                f"unclassified_training_session_types={unclassified_count}"
            ),
            "recommended_first_pass_features": _available_features(
                CATEGORY_FEATURES,
                candidate_features,
            ),
            "modeling_action": (
                "roll retained categories into coarse practice, lift, "
                "conditioning, RTP, and game channels"
            ),
        },
    ]
    return rows


def build_exposure_feature_requirements_summary(
    events: pd.DataFrame,
    participations: pd.DataFrame,
    audit: dict[str, Any],
    requirements: list[dict[str, object]],
) -> dict[str, object]:
    readiness = Counter(str(row["readiness_status"]) for row in requirements)
    return {
        "experiment_type": "exposure_feature_requirements_sprint",
        "overall_recommendation": _overall_recommendation(requirements),
        "production_readiness": "not_ready_feature_design_required",
        "event_count": int(len(events)),
        "participation_row_count": int(len(participations)),
        "category_summary": _category_summary(events),
        "duration_summary": _duration_summary(participations),
        "athlete_matching": audit.get("athlete_matching", {}),
        "excluded_training_event_reasons": (
            audit.get("event_counts", {})
            .get("training", {})
            .get("excluded_by_reason", {})
        ),
        "readiness_summary": dict(sorted(readiness.items())),
        "requirements": requirements,
    }


def build_exposure_category_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_type, categories in _category_summary(events).items():
        for category, count in categories.items():
            rows.append(
                {
                    "event_type": event_type,
                    "exposure_category": category,
                    "event_count": count,
                }
            )
    return pd.DataFrame(rows)


def build_exposure_duration_summary(participations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_type, summary in _duration_summary(participations).items():
        rows.append({"event_type": event_type, **summary})
    return pd.DataFrame(rows)


def write_exposure_feature_requirements_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Feature Requirements Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Events reviewed: {summary['event_count']}",
        f"Participation rows reviewed: {summary['participation_row_count']}",
        "",
        "## Feature Readiness",
        "",
        "| Requirement domain | Status | First-pass features | Modeling action |",
        "|---|---|---|---|",
    ]
    for row in summary["requirements"]:
        features = ", ".join(row["recommended_first_pass_features"])
        lines.append(
            "| "
            f"{row['requirement_domain']} | "
            f"{row['readiness_status']} | "
            f"{features} | "
            f"{row['modeling_action']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Proceed with count and participation-status features first. "
                "Minute-load features should remain secondary until the duration "
                "missingness pattern is reviewed by training and game context."
            ),
            "",
            (
                "This is still a research-readiness artifact, not pilot "
                "clearance. The next modeling sprint should attach only "
                "time-safe exposure context available before each graph snapshot."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _category_summary(events: pd.DataFrame) -> dict[str, dict[str, int]]:
    if events.empty:
        return {"training": {}, "game": {}}
    grouped = (
        events.assign(
            event_type=events["event_type"].astype(str),
            exposure_category=events["exposure_category"].astype(str),
        )
        .groupby(["event_type", "exposure_category"])
        .size()
    )
    out: dict[str, dict[str, int]] = {"training": {}, "game": {}}
    for (event_type, category), count in grouped.items():
        out.setdefault(str(event_type), {})[str(category)] = int(count)
    return {
        event_type: dict(sorted(categories.items()))
        for event_type, categories in sorted(out.items())
    }


def _duration_summary(participations: pd.DataFrame) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for event_type in ("training", "game"):
        subset = participations.loc[
            participations["event_type"].astype(str).eq(event_type)
        ]
        total = int(len(subset))
        duration = pd.to_numeric(subset.get("duration_minutes"), errors="coerce")
        missing = int(duration.isna().sum())
        out[event_type] = {
            "row_count": total,
            "missing_count": missing,
            "missing_share": round(missing / total, 4) if total else 0.0,
            "non_missing_count": int(total - missing),
        }
    return out


def _match_summary(audit: dict[str, Any]) -> dict[str, float]:
    matching = audit.get("athlete_matching", {})
    matched = 0
    unmatched = 0
    for event_type in ("training", "game"):
        row = matching.get(event_type, {})
        matched += int(row.get("matched_participation_rows", 0))
        unmatched += int(row.get("unmatched_participation_rows", 0))
    total = matched + unmatched
    return {"overall_match_share": matched / total if total else 0.0}


def _duplicate_count(audit: dict[str, Any]) -> int:
    duplicates = audit.get("duplicate_keys", {})
    return int(sum(int(value) for value in duplicates.values()))


def _duration_status(duration: dict[str, dict[str, object]]) -> str:
    max_missing = max(
        float(duration[event_type]["missing_share"])
        for event_type in ("training", "game")
    )
    if max_missing >= 0.25:
        return "caution"
    return "ready"


def _game_status(game_events: int, duration: dict[str, dict[str, object]]) -> str:
    if game_events <= 0:
        return "blocked"
    if float(duration["game"]["missing_share"]) >= 0.25:
        return "caution"
    return "ready"


def _status(condition: bool) -> str:
    return "ready" if condition else "blocked"


def _available_features(
    preferred_features: tuple[str, ...],
    candidate_features: set[object],
) -> list[str]:
    available = [
        feature for feature in preferred_features if feature in candidate_features
    ]
    return available or list(preferred_features)


def _has_status_variation(participations: pd.DataFrame) -> bool:
    categories = set(participations["participation_category"].astype(str))
    return bool(categories & {"modified", "no_participation", "partial"})


def _participation_categories(participations: pd.DataFrame) -> str:
    categories = sorted(set(participations["participation_category"].astype(str)))
    return ",".join(categories)


def _safe_dict(value: object) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(count) for key, count in value.items()}
    return {}


def _overall_recommendation(requirements: list[dict[str, object]]) -> str:
    statuses = {str(row["readiness_status"]) for row in requirements}
    if "blocked" in statuses:
        return "resolve_exposure_feature_blockers_before_modeling"
    if "caution" in statuses:
        return "proceed_with_count_and_status_features_first"
    return "proceed_to_time_safe_exposure_feature_sprint"
