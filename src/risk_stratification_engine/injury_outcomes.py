from __future__ import annotations

from collections import Counter
from typing import Callable

import pandas as pd


EXTREME_TIME_LOSS_DAYS = 365
MODERATE_TIME_LOSS_DAYS = 8
SEVERE_TIME_LOSS_DAYS = 29


def build_injury_severity_audit(detailed_events: pd.DataFrame) -> dict[str, object]:
    events = _normalized_events(detailed_events)
    rows = []
    for _, event in events.iterrows():
        rows.append(_severity_row(event))
    row_frame = pd.DataFrame(rows)
    return {
        "event_count": len(rows),
        "missing_time_loss_count": _count_flag(row_frame, "missing_time_loss"),
        "negative_time_loss_count": _count_flag(row_frame, "negative_time_loss"),
        "extreme_time_loss_count": _count_flag(row_frame, "extreme_time_loss"),
        "duration_resolution_mismatch_count": _count_flag(
            row_frame,
            "duration_resolution_mismatch",
        ),
        "time_loss_bucket_counts": _counter(row_frame, "time_loss_bucket"),
        "severity_semantics_flag_counts": _counter(
            row_frame,
            "severity_semantics_flag",
        ),
        "event_rows": rows,
    }


def build_outcome_policy_summary(detailed_events: pd.DataFrame) -> dict[str, object]:
    events = _normalized_events(detailed_events)
    policies = _outcome_policies()
    rows = []
    for policy_name, policy_fn in policies:
        subset = events.loc[events.apply(policy_fn, axis=1)]
        rows.append(_policy_row(policy_name, subset, len(events)))
    return {
        "event_count": int(len(events)),
        "policy_count": len(rows),
        "policy_rows": rows,
    }


def _severity_row(event: pd.Series) -> dict[str, object]:
    resolved_duration = _resolved_duration_days(event)
    duration_days = _number_or_none(event.get("duration_days"))
    time_loss_days = _number_or_none(event.get("time_loss_days"))
    mismatch = (
        resolved_duration is not None
        and duration_days is not None
        and abs(duration_days - resolved_duration) > 1
    )
    negative = time_loss_days is not None and time_loss_days < 0
    extreme = (
        time_loss_days is not None
        and time_loss_days > EXTREME_TIME_LOSS_DAYS
    )
    missing = time_loss_days is None
    return {
        "injury_event_id": str(event["injury_event_id"]),
        "athlete_id": str(event["athlete_id"]),
        "season_id": str(event["season_id"]),
        "injury_date": _date_string(event.get("injury_date")),
        "issue_resolved_date": _date_string(event.get("issue_resolved_date")),
        "injury_type": str(event["injury_type"]),
        "classification": str(event["classification"]),
        "pathology": str(event["pathology"]),
        "body_area": str(event["body_area"]),
        "activity_group": str(event["activity_group"]),
        "duration_days": duration_days,
        "resolved_duration_days": resolved_duration,
        "duration_resolution_delta_days": _delta(duration_days, resolved_duration),
        "time_loss_days": time_loss_days,
        "modified_available_days": _number_or_none(
            event.get("modified_available_days")
        ),
        "time_loss_bucket": _time_loss_bucket(time_loss_days),
        "missing_time_loss": missing,
        "negative_time_loss": negative,
        "extreme_time_loss": extreme,
        "duration_resolution_mismatch": mismatch,
        "caused_unavailability": str(event["caused_unavailability"]),
        "recurrent": str(event["recurrent"]),
        "severity_semantics_flag": _severity_flag(
            missing=missing,
            negative=negative,
            extreme=extreme,
            mismatch=mismatch,
        ),
    }


def _policy_row(
    policy_name: str,
    events: pd.DataFrame,
    denominator: int,
) -> dict[str, object]:
    return {
        "policy_name": policy_name,
        "event_count": int(len(events)),
        "event_share": _rate(len(events), denominator),
        "athlete_count": int(events["athlete_id"].nunique()) if not events.empty else 0,
        "median_time_loss_days": _median(events, "time_loss_days"),
        "median_duration_days": _median(events, "duration_days"),
        "caused_unavailability_count": _truthy_count(
            events,
            "caused_unavailability",
        ),
        "recurrent_event_count": _truthy_count(events, "recurrent"),
        "top_body_areas": _top_values(events, "body_area"),
        "top_injury_types": _top_values(events, "injury_type"),
        "recommended_use": _policy_recommendation(policy_name, events),
    }


def _outcome_policies() -> tuple[
    tuple[str, Callable[[pd.Series], bool]],
    ...,
]:
    return (
        ("any_injury", lambda row: True),
        ("time_loss_only", lambda row: _time_loss(row) > 0),
        (
            "model_safe_time_loss",
            lambda row: 0 < _time_loss(row) <= EXTREME_TIME_LOSS_DAYS,
        ),
        (
            "moderate_plus_time_loss",
            lambda row: MODERATE_TIME_LOSS_DAYS
            <= _time_loss(row)
            <= EXTREME_TIME_LOSS_DAYS,
        ),
        (
            "severe_time_loss",
            lambda row: SEVERE_TIME_LOSS_DAYS
            <= _time_loss(row)
            <= EXTREME_TIME_LOSS_DAYS,
        ),
        ("caused_unavailability", lambda row: _truthy(row["caused_unavailability"])),
        ("recurrent_only", lambda row: _truthy(row["recurrent"])),
        ("lower_extremity_only", _is_lower_extremity),
        ("soft_tissue_only", _is_soft_tissue),
        (
            "lower_extremity_soft_tissue",
            lambda row: _is_lower_extremity(row) and _is_soft_tissue(row),
        ),
        ("concussion_only", _is_concussion),
        ("exclude_concussion", lambda row: not _is_concussion(row)),
    )


def _normalized_events(events: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    if "injury_event_id" not in output:
        output["injury_event_id"] = [
            f"injury_event_{index + 1}" for index in range(len(output))
        ]
    for column in (
        "athlete_id",
        "season_id",
        "injury_type",
        "classification",
        "pathology",
        "body_area",
        "activity_group",
        "caused_unavailability",
        "recurrent",
    ):
        if column not in output:
            output[column] = "unknown"
        output[column] = output[column].map(_clean_text)
    for column in (
        "injury_date",
        "issue_resolved_date",
    ):
        if column not in output:
            output[column] = pd.NaT
        output[column] = pd.to_datetime(output[column], errors="coerce")
    for column in (
        "duration_days",
        "time_loss_days",
        "modified_available_days",
    ):
        if column not in output:
            output[column] = pd.NA
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.loc[
        output["athlete_id"].ne("unknown")
        & output["season_id"].ne("unknown")
        & output["injury_date"].notna()
    ].reset_index(drop=True)


def _is_lower_extremity(row: pd.Series) -> bool:
    body_area = str(row.get("body_area", "")).strip().lower()
    return body_area in {
        "ankle",
        "foot",
        "groin/hip",
        "hip",
        "knee",
        "lower leg",
        "thigh",
    }


def _is_soft_tissue(row: pd.Series) -> bool:
    text = " ".join(
        str(row.get(column, "")).lower()
        for column in ("injury_type", "classification", "pathology")
    )
    return any(
        token in text
        for token in (
            "soft tissue",
            "strain",
            "sprain",
            "muscle",
            "tendon",
            "tendinopathy",
            "ligament",
        )
    )


def _is_concussion(row: pd.Series) -> bool:
    text = " ".join(
        str(row.get(column, "")).lower()
        for column in ("injury_type", "classification", "pathology")
    )
    return "concussion" in text


def _time_loss(row: pd.Series) -> float:
    value = row.get("time_loss_days")
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _resolved_duration_days(event: pd.Series) -> float | None:
    injury_date = event.get("injury_date")
    resolved_date = event.get("issue_resolved_date")
    if pd.isna(injury_date) or pd.isna(resolved_date):
        return None
    return float((resolved_date - injury_date).days)


def _severity_flag(
    *,
    missing: bool,
    negative: bool,
    extreme: bool,
    mismatch: bool,
) -> str:
    if missing:
        return "missing_time_loss"
    if negative:
        return "review_negative_time_loss"
    if extreme:
        return "review_extreme_time_loss"
    if mismatch:
        return "review_duration_mismatch"
    return "usable"


def _policy_recommendation(policy_name: str, events: pd.DataFrame) -> str:
    if events.empty:
        return "insufficient_events"
    if policy_name in {"moderate_plus_time_loss", "severe_time_loss"}:
        return "candidate_severity_target"
    if policy_name in {"lower_extremity_soft_tissue", "soft_tissue_only"}:
        return "candidate_subtype_target"
    if policy_name in {"concussion_only", "exclude_concussion"}:
        return "candidate_subtype_sensitivity"
    return "candidate_policy"


def _time_loss_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0:
        return "negative"
    if value == 0:
        return "0d"
    if value <= 7:
        return "1-7d"
    if value <= 28:
        return "8-28d"
    if value <= EXTREME_TIME_LOSS_DAYS:
        return "29-365d"
    return "extreme_366d+"


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _date_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _number_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _count_flag(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].astype(bool).sum())


def _counter(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = Counter(str(value) for value in frame[column])
    return dict(sorted(counts.items()))


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _truthy_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].map(_truthy).sum())


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _top_values(frame: pd.DataFrame, column: str, limit: int = 5) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = Counter(str(value) for value in frame[column] if str(value))
    return dict(counts.most_common(limit))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)
