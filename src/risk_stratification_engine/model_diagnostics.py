from __future__ import annotations

import ast
from collections import Counter
from typing import Any

import pandas as pd


COMPARISON_GROUPS = (
    "true_positive_episode",
    "false_positive_episode",
    "missed_event",
)


def build_model_improvement_diagnostics(
    episodes: pd.DataFrame,
    alert_timeline: pd.DataFrame,
    quality: dict[str, object],
) -> dict[str, object]:
    rows = []
    observed_events = _observed_events(alert_timeline)
    for quality_row in sorted(
        quality.get("quality_rows", []),
        key=lambda row: (
            int(row["horizon_days"]),
            str(row["threshold_kind"]),
            float(row["threshold_value"]),
        ),
    ):
        horizon = int(quality_row["horizon_days"])
        threshold_kind = str(quality_row["threshold_kind"])
        threshold_value = float(quality_row["threshold_value"])
        threshold = _threshold_key(threshold_kind, threshold_value)
        group = _episode_group(episodes, horizon, threshold_kind, threshold_value)
        true_positive = group[group["event_within_horizon_after_start"].map(_as_bool)]
        false_positive = group[
            ~group["event_within_horizon_after_start"].map(_as_bool)
        ]
        captured_events = _captured_event_ids(true_positive, observed_events)
        missed_events = [
            event for event in observed_events if event["event_id"] not in captured_events
        ]
        alert_cutoff = _alert_cutoff(group)

        rows.append(
            _episode_summary_row(
                episodes=true_positive,
                comparison_group="true_positive_episode",
                horizon=horizon,
                threshold_kind=threshold_kind,
                threshold_value=threshold_value,
                threshold=threshold,
                alert_timeline=alert_timeline,
            )
        )
        rows.append(
            _episode_summary_row(
                episodes=false_positive,
                comparison_group="false_positive_episode",
                horizon=horizon,
                threshold_kind=threshold_kind,
                threshold_value=threshold_value,
                threshold=threshold,
                alert_timeline=alert_timeline,
            )
        )
        rows.append(
            _missed_event_summary_row(
                events=missed_events,
                alert_timeline=alert_timeline,
                horizon=horizon,
                threshold_kind=threshold_kind,
                threshold_value=threshold_value,
                threshold=threshold,
                alert_cutoff=alert_cutoff,
            )
        )

    action_summary = dict(
        sorted(Counter(row["recommended_next_action"] for row in rows).items())
    )
    return {
        "diagnostic_row_count": len(rows),
        "diagnostic_rows": rows,
        "recommended_action_summary": action_summary,
    }


def _episode_group(
    episodes: pd.DataFrame,
    horizon: int,
    threshold_kind: str,
    threshold_value: float,
) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    return episodes[
        (episodes["horizon_days"].astype(int) == horizon)
        & (episodes["threshold_kind"].astype(str) == threshold_kind)
        & (episodes["threshold_value"].astype(float) == threshold_value)
    ]


def _episode_summary_row(
    episodes: pd.DataFrame,
    comparison_group: str,
    horizon: int,
    threshold_kind: str,
    threshold_value: float,
    threshold: str,
    alert_timeline: pd.DataFrame,
) -> dict[str, object]:
    event_rows = _episode_event_rows(episodes, alert_timeline)
    return {
        **_base_row(horizon, threshold_kind, threshold_value, threshold, comparison_group),
        "row_count": int(len(episodes)),
        "mean_peak_risk": _mean(episodes, "peak_risk"),
        "median_peak_risk": _median(episodes, "peak_risk"),
        "max_pre_event_risk": None,
        "median_pre_event_risk": None,
        "median_pre_event_snapshot_count": None,
        "median_duration_days": _median(episodes, "duration_days"),
        "median_lead_days": _median(episodes, "days_from_start_to_event"),
        "elevated_z_rate": _elevated_z_rate(episodes, "episode"),
        "top_feature_counts": _episode_top_feature_counts(episodes),
        "event_window_quality_counts": _value_counts(event_rows, "event_window_quality"),
        "median_nearest_measurement_gap_days": _median(
            event_rows, "nearest_measurement_gap_days"
        ),
        "recommended_next_action": _episode_recommendation(
            comparison_group, event_rows
        ),
    }


def _missed_event_summary_row(
    events: list[dict[str, object]],
    alert_timeline: pd.DataFrame,
    horizon: int,
    threshold_kind: str,
    threshold_value: float,
    threshold: str,
    alert_cutoff: float | None,
) -> dict[str, object]:
    profiles = [
        _missed_event_profile(event, alert_timeline, horizon) for event in events
    ]
    profile_frame = pd.DataFrame(profiles)
    return {
        **_base_row(horizon, threshold_kind, threshold_value, threshold, "missed_event"),
        "row_count": int(len(profiles)),
        "mean_peak_risk": None,
        "median_peak_risk": None,
        "max_pre_event_risk": _max(profile_frame, "max_pre_event_risk"),
        "median_pre_event_risk": _median(profile_frame, "median_pre_event_risk"),
        "median_pre_event_snapshot_count": _median(
            profile_frame, "pre_event_snapshot_count"
        ),
        "median_duration_days": None,
        "median_lead_days": _median(profile_frame, "lead_days_at_max_risk"),
        "elevated_z_rate": _mean(profile_frame, "elevated_z_rate"),
        "top_feature_counts": _merge_counts(
            profile.get("top_feature_counts", {}) for profile in profiles
        ),
        "event_window_quality_counts": _counter_from_values(
            profile.get("event_window_quality") for profile in profiles
        ),
        "median_nearest_measurement_gap_days": _median(
            profile_frame, "nearest_measurement_gap_days"
        ),
        "recommended_next_action": _missed_event_recommendation(
            profiles, alert_cutoff
        ),
    }


def _base_row(
    horizon: int,
    threshold_kind: str,
    threshold_value: float,
    threshold: str,
    comparison_group: str,
) -> dict[str, object]:
    return {
        "horizon_days": int(horizon),
        "threshold_kind": threshold_kind,
        "threshold_value": threshold_value,
        "threshold": threshold,
        "comparison_group": comparison_group,
    }


def _observed_events(timeline: pd.DataFrame) -> list[dict[str, object]]:
    required = {"athlete_id", "season_id", "event_observed", "injury_type"}
    if timeline.empty or not required.issubset(timeline.columns):
        return []
    date_column = "event_date" if "event_date" in timeline.columns else "injury_date"
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    observed = timeline[timeline["event_observed"].map(_as_bool)]
    for _, row in observed.iterrows():
        event_date = _date_key(row.get(date_column))
        injury_type = str(_clean_value(row.get("injury_type")) or "")
        event_id = (
            str(row["athlete_id"]),
            str(row["season_id"]),
            event_date,
            injury_type,
        )
        rows[event_id] = {
            "event_id": event_id,
            "athlete_id": event_id[0],
            "season_id": event_id[1],
            "event_date": event_id[2],
            "injury_type": event_id[3],
            "event_window_quality": _clean_value(row.get("event_window_quality")),
            "nearest_measurement_gap_days": _clean_number(
                row.get("nearest_measurement_gap_days")
            ),
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
    true_positive: pd.DataFrame,
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
    for _, row in true_positive.iterrows():
        key = (
            str(row["athlete_id"]),
            str(row["season_id"]),
            str(_clean_value(row.get("injury_type")) or ""),
        )
        captured.update(event_lookup.get(key, set()))
    return captured


def _episode_event_rows(
    episodes: pd.DataFrame,
    alert_timeline: pd.DataFrame,
) -> pd.DataFrame:
    if episodes.empty or alert_timeline.empty:
        return pd.DataFrame()
    frames = []
    for _, episode in episodes.iterrows():
        rows = alert_timeline[
            (alert_timeline["athlete_id"].astype(str) == str(episode["athlete_id"]))
            & (alert_timeline["season_id"].astype(str) == str(episode["season_id"]))
        ]
        if "injury_type" in rows.columns:
            rows = rows[
                rows["injury_type"].astype(str)
                == str(_clean_value(episode.get("injury_type")) or "")
            ]
        if not rows.empty:
            frames.append(rows.iloc[[0]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _missed_event_profile(
    event: dict[str, object],
    alert_timeline: pd.DataFrame,
    horizon: int,
) -> dict[str, object]:
    rows = alert_timeline[
        (alert_timeline["athlete_id"].astype(str) == str(event["athlete_id"]))
        & (alert_timeline["season_id"].astype(str) == str(event["season_id"]))
    ]
    if "event_date" in rows.columns:
        rows = rows[rows["event_date"].map(_date_key) == str(event["event_date"])]
    if "injury_type" in rows.columns:
        rows = rows[rows["injury_type"].astype(str) == str(event["injury_type"])]
    if "days_to_event" in rows.columns:
        days = pd.to_numeric(rows["days_to_event"], errors="coerce")
        rows = rows[(days >= 0) & (days <= horizon)]

    risk_column = f"risk_{horizon}d"
    max_risk = _max(rows, risk_column)
    lead_days = None
    if max_risk is not None and risk_column in rows.columns:
        risk_values = pd.to_numeric(rows[risk_column], errors="coerce")
        if not risk_values.dropna().empty:
            peak_row = rows.loc[risk_values.idxmax()]
            lead_days = _clean_number(peak_row.get("days_to_event"))

    return {
        "event_id": event["event_id"],
        "event_window_quality": _clean_value(event.get("event_window_quality")),
        "nearest_measurement_gap_days": _clean_number(
            event.get("nearest_measurement_gap_days")
        ),
        "pre_event_snapshot_count": int(len(rows)),
        "max_pre_event_risk": max_risk,
        "median_pre_event_risk": _median(rows, risk_column),
        "lead_days_at_max_risk": lead_days,
        "elevated_z_rate": _elevated_z_rate(rows, "snapshot"),
        "top_feature_counts": _timeline_top_feature_counts(rows, horizon),
    }


def _episode_recommendation(
    comparison_group: str,
    event_rows: pd.DataFrame,
) -> str:
    if comparison_group == "false_positive_episode":
        return "add_context_features"
    if comparison_group == "true_positive_episode":
        quality_counts = _value_counts(event_rows, "event_window_quality")
        if quality_counts.get("modelable", 0) >= max(1, len(event_rows) / 2):
            return "retain_policy_signal"
        return "improve_data_linkage"
    return "needs_review"


def _missed_event_recommendation(
    profiles: list[dict[str, object]],
    alert_cutoff: float | None,
) -> str:
    if not profiles:
        return "no_missed_events"
    data_limited = [
        profile
        for profile in profiles
        if profile.get("event_window_quality") != "modelable"
        or (
            profile.get("nearest_measurement_gap_days") is not None
            and float(profile["nearest_measurement_gap_days"]) > 30
        )
        or int(profile.get("pre_event_snapshot_count") or 0) == 0
    ]
    if len(data_limited) > len(profiles) / 2:
        return "improve_data_linkage"

    max_risk = max(
        (
            float(profile["max_pre_event_risk"])
            for profile in profiles
            if profile.get("max_pre_event_risk") is not None
        ),
        default=None,
    )
    if (
        max_risk is not None
        and alert_cutoff is not None
        and max_risk >= float(alert_cutoff) * 0.8
    ):
        return "review_threshold_policy"
    return "add_event_specific_features"


def _alert_cutoff(episodes: pd.DataFrame) -> float | None:
    return _min(episodes, "peak_risk")


def _episode_top_feature_counts(episodes: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in episodes.get("top_model_features", pd.Series(dtype=object)):
        features = {record["feature"] for record in _feature_records(value)}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
    return dict(sorted(counts.items()))


def _timeline_top_feature_counts(rows: pd.DataFrame, horizon: int) -> dict[str, int]:
    column = f"top_feature_{horizon}d"
    if rows.empty or column not in rows.columns:
        return {}
    return _counter_from_values(rows[column])


def _merge_counts(counts_iter: Any) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for counts in counts_iter:
        if not isinstance(counts, dict):
            continue
        merged.update({str(key): int(value) for key, value in counts.items()})
    return dict(sorted(merged.items()))


def _counter_from_values(values: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        clean = _clean_value(value)
        if clean:
            counter[clean] += 1
    return dict(sorted(counter.items()))


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return _counter_from_values(frame[column])


def _elevated_z_rate(frame: pd.DataFrame, mode: str) -> float | None:
    if frame.empty or "elevated_z_features" not in frame.columns:
        return None
    if mode == "episode":
        denominator = len(frame)
        numerator = sum(_has_features(value) for value in frame["elevated_z_features"])
    else:
        denominator = len(frame)
        numerator = sum(_has_features(value) for value in frame["elevated_z_features"])
    return _rate(numerator, denominator)


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


def _date_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return str(parsed.date())


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _max(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _min(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


def _rate(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _threshold_key(threshold_kind: object, threshold_value: object) -> str:
    return f"{threshold_kind}:{float(threshold_value):g}"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
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
