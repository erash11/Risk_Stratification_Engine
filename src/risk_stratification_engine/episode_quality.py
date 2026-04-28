from __future__ import annotations

import ast
from itertools import combinations
from typing import Any

import pandas as pd


def build_alert_episode_quality(
    episodes: pd.DataFrame,
    timeline: pd.DataFrame,
) -> dict[str, object]:
    observed_events = _observed_events(timeline)
    athlete_season_count = _athlete_season_count(timeline)
    quality_rows = []
    representative_cases: dict[str, dict[str, object]] = {}

    grouped = episodes.groupby(
        ["horizon_days", "threshold_kind", "threshold_value"],
        sort=True,
        dropna=False,
    )
    for (horizon, threshold_kind, threshold_value), group in grouped:
        horizon_int = int(horizon)
        threshold_key = _threshold_key(threshold_kind, threshold_value)
        captured_events = _captured_event_ids(group, observed_events)
        missed_events = [
            event for event in observed_events if event["event_id"] not in captured_events
        ]
        quality_rows.append(
            _quality_row(
                horizon_days=horizon_int,
                threshold_kind=str(threshold_kind),
                threshold_value=float(threshold_value),
                episodes=group,
                observed_events=observed_events,
                captured_event_count=len(captured_events),
                missed_event_count=len(missed_events),
                athlete_season_count=athlete_season_count,
            )
        )
        representative_cases.setdefault(str(horizon_int), {})[threshold_key] = (
            _representative_cases(group, missed_events)
        )

    return {
        "quality_rows": quality_rows,
        "threshold_overlaps": _threshold_overlaps(episodes),
        "representative_cases": representative_cases,
    }


def _quality_row(
    horizon_days: int,
    threshold_kind: str,
    threshold_value: float,
    episodes: pd.DataFrame,
    observed_events: list[dict[str, object]],
    captured_event_count: int,
    missed_event_count: int,
    athlete_season_count: int,
) -> dict[str, object]:
    true_positive = episodes[
        episodes["event_within_horizon_after_start"].map(_as_bool)
    ]
    false_positive = episodes[
        ~episodes["event_within_horizon_after_start"].map(_as_bool)
    ]
    episode_count = len(episodes)
    observed_event_count = len(observed_events)
    return {
        "horizon_days": horizon_days,
        "threshold_kind": threshold_kind,
        "threshold_value": threshold_value,
        "threshold": _threshold_key(threshold_kind, threshold_value),
        "episode_count": int(episode_count),
        "true_positive_episode_count": int(len(true_positive)),
        "true_positive_episode_rate": _rate(len(true_positive), episode_count),
        "false_positive_episode_count": int(len(false_positive)),
        "false_positive_episode_rate": _rate(len(false_positive), episode_count),
        "unique_observed_event_count": int(observed_event_count),
        "unique_captured_event_count": int(captured_event_count),
        "unique_event_capture_rate": _rate(captured_event_count, observed_event_count),
        "missed_event_count": int(missed_event_count),
        "episodes_per_athlete_season": _rate(episode_count, athlete_season_count),
        "median_start_lead_days": _median(true_positive, "days_from_start_to_event"),
        "median_peak_lead_days": _median(true_positive, "days_from_peak_to_event"),
        "median_end_lead_days": _median(true_positive, "days_from_end_to_event"),
        "median_duration_days": _median(episodes, "duration_days"),
        "median_snapshot_count": _median(episodes, "snapshot_count"),
        "true_positive_median_peak_risk": _median(true_positive, "peak_risk"),
        "false_positive_median_peak_risk": _median(false_positive, "peak_risk"),
        "true_positive_median_duration_days": _median(true_positive, "duration_days"),
        "false_positive_median_duration_days": _median(false_positive, "duration_days"),
        "true_positive_elevated_z_episode_rate": _elevated_z_rate(true_positive),
        "false_positive_elevated_z_episode_rate": _elevated_z_rate(false_positive),
        "true_positive_top_model_feature_counts": _top_feature_counts(true_positive),
        "false_positive_top_model_feature_counts": _top_feature_counts(false_positive),
    }


def _observed_events(timeline: pd.DataFrame) -> list[dict[str, object]]:
    required = {"athlete_id", "season_id", "event_observed", "injury_type"}
    if timeline.empty or not required.issubset(timeline.columns):
        return []
    date_column = "event_date" if "event_date" in timeline.columns else "injury_date"
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    observed = timeline[timeline["event_observed"].map(_as_bool)]
    for _, row in observed.iterrows():
        event_date = _clean_value(row.get(date_column))
        injury_type = _clean_value(row.get("injury_type"))
        event_id = (
            str(row["athlete_id"]),
            str(row["season_id"]),
            str(event_date or ""),
            str(injury_type or ""),
        )
        rows[event_id] = {
            "event_id": event_id,
            "athlete_id": event_id[0],
            "season_id": event_id[1],
            "event_date": event_id[2],
            "injury_type": event_id[3],
        }
    return sorted(
        rows.values(),
        key=lambda event: (
            str(event["event_date"]),
            str(event["athlete_id"]),
            str(event["season_id"]),
            str(event["injury_type"]),
        ),
    )


def _captured_event_ids(
    episodes: pd.DataFrame,
    observed_events: list[dict[str, object]],
) -> set[tuple[str, str, str, str]]:
    event_lookup = {}
    for event in observed_events:
        key = (
            str(event["athlete_id"]),
            str(event["season_id"]),
            str(event["injury_type"]),
        )
        event_lookup.setdefault(key, set()).add(event["event_id"])

    captured: set[tuple[str, str, str, str]] = set()
    true_positive = episodes[
        episodes["event_within_horizon_after_start"].map(_as_bool)
    ]
    for _, row in true_positive.iterrows():
        key = (
            str(row["athlete_id"]),
            str(row["season_id"]),
            str(_clean_value(row.get("injury_type")) or ""),
        )
        captured.update(event_lookup.get(key, set()))
    return captured


def _threshold_overlaps(episodes: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for horizon, horizon_group in episodes.groupby("horizon_days", sort=True):
        threshold_groups = {
            _threshold_key(kind, value): group
            for (kind, value), group in horizon_group.groupby(
                ["threshold_kind", "threshold_value"], sort=True, dropna=False
            )
        }
        for threshold_a, threshold_b in combinations(threshold_groups, 2):
            ids_a = _episode_ids(threshold_groups[threshold_a])
            ids_b = _episode_ids(threshold_groups[threshold_b])
            overlap_count = len(ids_a & ids_b)
            rows.append(
                {
                    "horizon_days": int(horizon),
                    "threshold_a": threshold_a,
                    "threshold_b": threshold_b,
                    "threshold_a_episode_count": int(len(ids_a)),
                    "threshold_b_episode_count": int(len(ids_b)),
                    "overlap_episode_count": int(overlap_count),
                    "threshold_a_overlap_rate": _rate(overlap_count, len(ids_a)),
                    "threshold_b_overlap_rate": _rate(overlap_count, len(ids_b)),
                }
            )
    return rows


def _episode_ids(episodes: pd.DataFrame) -> set[tuple[str, str, int, int, int]]:
    return {
        (
            str(row["athlete_id"]),
            str(row["season_id"]),
            int(row["horizon_days"]),
            int(row["start_time_index"]),
            int(row["end_time_index"]),
        )
        for _, row in episodes.iterrows()
    }


def _representative_cases(
    episodes: pd.DataFrame,
    missed_events: list[dict[str, object]],
) -> dict[str, object]:
    true_positive = episodes[
        episodes["event_within_horizon_after_start"].map(_as_bool)
    ]
    false_positive = episodes[
        ~episodes["event_within_horizon_after_start"].map(_as_bool)
    ]
    high_deviation = episodes[episodes["elevated_z_features"].map(_has_features)]
    missed = missed_events[0] if missed_events else None
    return {
        "true_positive_episode": _episode_case(_highest_peak_risk(true_positive)),
        "false_positive_episode": _episode_case(_highest_peak_risk(false_positive)),
        "missed_injury": _missed_case(missed),
        "high_intra_individual_deviation_episode": _episode_case(
            _highest_peak_risk(high_deviation)
        ),
    }


def _highest_peak_risk(episodes: pd.DataFrame) -> pd.Series | None:
    if episodes.empty:
        return None
    return episodes.loc[episodes["peak_risk"].astype(float).idxmax()]


def _episode_case(row: pd.Series | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "athlete_id": str(row["athlete_id"]),
        "season_id": str(row["season_id"]),
        "horizon_days": int(row["horizon_days"]),
        "threshold": _threshold_key(row["threshold_kind"], row["threshold_value"]),
        "start_date": str(row["start_date"]),
        "peak_date": str(row["peak_date"]),
        "end_date": str(row["end_date"]),
        "days_from_start_to_event": _clean_number(row.get("days_from_start_to_event")),
        "days_from_peak_to_event": _clean_number(row.get("days_from_peak_to_event")),
        "days_from_end_to_event": _clean_number(row.get("days_from_end_to_event")),
        "peak_risk": _clean_number(row.get("peak_risk")),
        "mean_risk": _clean_number(row.get("mean_risk")),
        "injury_type": _clean_value(row.get("injury_type")),
        "top_model_features": _feature_records(row.get("top_model_features")),
        "elevated_z_features": _feature_list(row.get("elevated_z_features")),
    }


def _missed_case(event: dict[str, object] | None) -> dict[str, object] | None:
    if event is None:
        return None
    return {
        "athlete_id": str(event["athlete_id"]),
        "season_id": str(event["season_id"]),
        "event_date": str(event["event_date"]),
        "injury_type": str(event["injury_type"]),
    }


def _athlete_season_count(timeline: pd.DataFrame) -> int:
    if timeline.empty or not {"athlete_id", "season_id"}.issubset(timeline.columns):
        return 0
    return int(timeline[["athlete_id", "season_id"]].drop_duplicates().shape[0])


def _top_feature_counts(episodes: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in episodes.get("top_model_features", pd.Series(dtype=object)):
        features = {record["feature"] for record in _feature_records(value)}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
    return dict(sorted(counts.items()))


def _elevated_z_rate(episodes: pd.DataFrame) -> float | None:
    if episodes.empty:
        return None
    count = sum(_has_features(value) for value in episodes["elevated_z_features"])
    return _rate(count, len(episodes))


def _has_features(value: object) -> bool:
    return bool(_feature_list(value))


def _feature_records(value: object) -> list[dict[str, object]]:
    parsed = _parse_collection(value)
    if not isinstance(parsed, list):
        return []
    records = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("feature"):
            continue
        records.append(
            {
                "feature": str(item["feature"]),
                "mean_abs_contribution": _clean_number(
                    item.get("mean_abs_contribution")
                ),
            }
        )
    return records


def _feature_list(value: object) -> list[str]:
    parsed = _parse_collection(value)
    if isinstance(parsed, list | tuple | set):
        return [str(item) for item in parsed if str(item)]
    if isinstance(parsed, str) and parsed:
        return [parsed]
    return []


def _parse_collection(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, list | tuple | set | dict):
        return value
    if not isinstance(value, str) and pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
    return value


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _rate(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _threshold_key(threshold_kind: object, threshold_value: object) -> str:
    return f"{threshold_kind}:{float(threshold_value):g}"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _clean_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _clean_number(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number
