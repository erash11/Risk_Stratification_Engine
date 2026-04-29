from __future__ import annotations

from typing import Any

import pandas as pd


CONTEXT_FIELDS = (
    "injury_type",
    "classification",
    "pathology",
    "body_area",
    "tissue_type",
    "side",
    "recurrent",
    "caused_unavailability",
    "activity_group",
    "activity_group_type",
    "time_loss_bucket",
)


def build_injury_context_outcomes(
    detailed_events: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, object]:
    events = _normalize_events(detailed_events)
    if events.empty or episodes.empty:
        return {
            "event_profile_count": 0,
            "context_row_count": 0,
            "event_profile_rows": [],
            "context_rows": [],
            "lowest_capture_contexts": [],
            "high_time_loss_missed_contexts": [],
        }

    episode_groups = episodes.groupby(
        ["horizon_days", "threshold_kind", "threshold_value"],
        sort=True,
        dropna=False,
    )
    profile_rows: list[dict[str, object]] = []
    for (horizon, threshold_kind, threshold_value), group in episode_groups:
        horizon_days = int(horizon)
        threshold = _threshold_key(str(threshold_kind), float(threshold_value))
        for _, event in events.iterrows():
            profile_rows.append(
                _event_profile(
                    event=event,
                    episodes=group,
                    horizon_days=horizon_days,
                    threshold_kind=str(threshold_kind),
                    threshold_value=float(threshold_value),
                    threshold=threshold,
                )
            )

    context_rows = _context_rows(pd.DataFrame(profile_rows))
    lowest_capture = _lowest_capture_contexts(context_rows)
    high_time_loss_missed = _high_time_loss_missed_contexts(context_rows)
    return {
        "event_profile_count": len(profile_rows),
        "context_row_count": len(context_rows),
        "event_profile_rows": profile_rows,
        "context_rows": context_rows,
        "lowest_capture_contexts": lowest_capture,
        "high_time_loss_missed_contexts": high_time_loss_missed,
    }


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    if "injury_event_id" not in output:
        output["injury_event_id"] = [
            f"injury_event_{index + 1}" for index in range(len(output))
        ]
    for column in ("athlete_id", "season_id"):
        if column not in output:
            output[column] = ""
        output[column] = output[column].fillna("").astype(str)
    if "injury_date" not in output:
        output["injury_date"] = pd.NaT
    output["injury_date"] = pd.to_datetime(output["injury_date"], errors="coerce")
    for column in CONTEXT_FIELDS:
        if column == "time_loss_bucket":
            continue
        if column not in output:
            output[column] = "unknown"
        output[column] = output[column].map(_clean_context_value)
    for column in (
        "duration_days",
        "time_loss_days",
        "modified_available_days",
        "not_modified_available_days",
    ):
        if column not in output:
            output[column] = pd.NA
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["time_loss_bucket"] = output["time_loss_days"].map(_time_loss_bucket)
    return output.loc[
        output["athlete_id"].ne("")
        & output["season_id"].ne("")
        & output["injury_date"].notna()
    ].reset_index(drop=True)


def _event_profile(
    event: pd.Series,
    episodes: pd.DataFrame,
    horizon_days: int,
    threshold_kind: str,
    threshold_value: float,
    threshold: str,
) -> dict[str, object]:
    relevant = episodes[
        (episodes["athlete_id"].astype(str) == str(event["athlete_id"]))
        & (episodes["season_id"].astype(str) == str(event["season_id"]))
    ]
    start_gap = _nearest_gap(event["injury_date"], relevant, "start_date")
    peak_gap = _nearest_gap(event["injury_date"], relevant, "peak_date")
    end_gap = _nearest_gap(event["injury_date"], relevant, "end_date")
    row = {
        "horizon_days": horizon_days,
        "threshold_kind": threshold_kind,
        "threshold_value": threshold_value,
        "threshold": threshold,
        "injury_event_id": str(event["injury_event_id"]),
        "athlete_id": str(event["athlete_id"]),
        "season_id": str(event["season_id"]),
        "injury_date": event["injury_date"].strftime("%Y-%m-%d"),
        "duration_days": _number_or_none(event.get("duration_days")),
        "time_loss_days": _number_or_none(event.get("time_loss_days")),
        "modified_available_days": _number_or_none(
            event.get("modified_available_days")
        ),
        "not_modified_available_days": _number_or_none(
            event.get("not_modified_available_days")
        ),
        "time_loss_bucket": str(event["time_loss_bucket"]),
        "days_from_nearest_episode_start": start_gap,
        "days_from_nearest_episode_peak": peak_gap,
        "days_from_nearest_episode_end": end_gap,
        "captured_after_start": _captured(start_gap, horizon_days),
        "captured_after_peak": _captured(peak_gap, horizon_days),
        "captured_after_end": _captured(end_gap, horizon_days),
    }
    for column in CONTEXT_FIELDS:
        if column != "time_loss_bucket":
            row[column] = str(event[column])
    return row


def _nearest_gap(
    event_date: pd.Timestamp,
    episodes: pd.DataFrame,
    date_column: str,
) -> int | None:
    if episodes.empty or date_column not in episodes:
        return None
    dates = pd.to_datetime(episodes[date_column], errors="coerce").dropna()
    if dates.empty:
        return None
    gaps = (event_date - dates).dt.days
    positive = gaps[gaps >= 0]
    if positive.empty:
        return None
    return int(positive.min())


def _context_rows(profiles: pd.DataFrame) -> list[dict[str, object]]:
    if profiles.empty:
        return []
    rows: list[dict[str, object]] = []
    for context_field in CONTEXT_FIELDS:
        grouped = profiles.groupby(
            [
                "horizon_days",
                "threshold_kind",
                "threshold_value",
                "threshold",
                context_field,
            ],
            dropna=False,
            sort=True,
        )
        for (
            horizon,
            threshold_kind,
            threshold_value,
            threshold,
            context_value,
        ), group in grouped:
            event_count = int(len(group))
            captured = int(group["captured_after_start"].sum())
            missed = event_count - captured
            row = {
                "horizon_days": int(horizon),
                "threshold_kind": str(threshold_kind),
                "threshold_value": float(threshold_value),
                "threshold": str(threshold),
                "context_field": context_field,
                "context_value": str(context_value),
                "event_count": event_count,
                "captured_after_start_count": captured,
                "missed_after_start_count": missed,
                "start_capture_rate": _rate(captured, event_count),
                "median_time_loss_days": _median(group, "time_loss_days"),
                "median_duration_days": _median(group, "duration_days"),
                "recurrent_event_count": _truthy_count(group, "recurrent"),
                "caused_unavailability_event_count": _truthy_count(
                    group, "caused_unavailability"
                ),
            }
            row["recommended_next_action"] = _recommendation(row)
            rows.append(row)
    return rows


def _lowest_capture_contexts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if int(row["event_count"]) > 0
        and row["context_value"] != "unknown"
        and row["context_field"] != "time_loss_bucket"
    ]
    return sorted(
        candidates,
        key=lambda row: (
            float(row["start_capture_rate"]),
            -int(row["event_count"]),
            str(row["context_field"]),
            str(row["context_value"]),
        ),
    )[:10]


def _high_time_loss_missed_contexts(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if int(row["missed_after_start_count"]) > 0
        and row["median_time_loss_days"] is not None
        and float(row["median_time_loss_days"]) >= 8.0
    ]
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["median_time_loss_days"]),
            -int(row["missed_after_start_count"]),
            str(row["context_field"]),
        ),
    )[:10]


def _recommendation(row: dict[str, object]) -> str:
    capture_rate = float(row["start_capture_rate"])
    missed = int(row["missed_after_start_count"])
    median_time_loss = row["median_time_loss_days"]
    if missed and median_time_loss is not None and float(median_time_loss) >= 8.0:
        return "prioritize_severe_missed_context"
    if capture_rate < 0.25:
        return "review_missed_context"
    if capture_rate >= 0.75:
        return "context_signal_supported"
    return "monitor_context"


def _captured(gap_days: int | None, horizon_days: int) -> bool:
    return gap_days is not None and 0 <= gap_days <= horizon_days


def _time_loss_bucket(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    days = float(value)
    if days <= 0:
        return "0d"
    if days <= 7:
        return "1-7d"
    if days <= 28:
        return "8-28d"
    return "29d+"


def _clean_context_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _truthy_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].map(_truthy).sum())


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _number_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _threshold_key(threshold_kind: str, threshold_value: float) -> str:
    return f"{threshold_kind}:{threshold_value:g}"
