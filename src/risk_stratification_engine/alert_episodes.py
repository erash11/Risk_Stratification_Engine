from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from risk_stratification_engine.events import DEFAULT_HORIZONS


DEFAULT_ALERT_PERCENTILES = (0.05, 0.10)


def build_alert_episodes(
    timeline: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    percentile_thresholds: tuple[float, ...] = DEFAULT_ALERT_PERCENTILES,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        risk_column = f"risk_{horizon}d"
        _require_columns(timeline, (risk_column,))
        for percentile in sorted(percentile_thresholds):
            alert_indices = _percentile_alert_indices(timeline[risk_column], percentile)
            threshold_value = (
                float(timeline.loc[alert_indices, risk_column].min())
                if len(alert_indices)
                else None
            )
            rows.extend(
                _episodes_for_threshold(
                    timeline=timeline,
                    horizon=horizon,
                    percentile=percentile,
                    threshold_value=threshold_value,
                    alert_indices=set(alert_indices),
                )
            )
    return pd.DataFrame(rows, columns=_episode_columns())


def build_alert_episode_summary(episodes: pd.DataFrame) -> dict[str, object]:
    if episodes.empty:
        return {"episode_count": 0, "horizons": {}}

    horizons: dict[str, object] = {}
    grouped = episodes.groupby(
        ["horizon_days", "threshold_kind", "threshold_value"],
        sort=True,
        dropna=False,
    )
    for (horizon, threshold_kind, threshold_value), group in grouped:
        horizon_key = str(int(horizon))
        threshold_key = f"{threshold_kind}:{float(threshold_value):g}"
        horizons.setdefault(horizon_key, {"thresholds": {}})
        horizons[horizon_key]["thresholds"][threshold_key] = {
            "episode_count": int(len(group)),
            "episode_with_event_after_start_count": int(
                group["event_within_horizon_after_start"].sum()
            ),
            "episode_with_event_after_peak_count": int(
                group["event_within_horizon_after_peak"].sum()
            ),
            "episode_with_event_after_end_count": int(
                group["event_within_horizon_after_end"].sum()
            ),
            "event_capture_after_start_rate": _rate(
                group["event_within_horizon_after_start"].sum(), len(group)
            ),
            "event_capture_after_peak_rate": _rate(
                group["event_within_horizon_after_peak"].sum(), len(group)
            ),
            "event_capture_after_end_rate": _rate(
                group["event_within_horizon_after_end"].sum(), len(group)
            ),
            "median_snapshot_count": float(group["snapshot_count"].median()),
            "median_duration_days": float(group["duration_days"].median()),
        }
    return {"episode_count": int(len(episodes)), "horizons": horizons}


def _episodes_for_threshold(
    timeline: pd.DataFrame,
    horizon: int,
    percentile: float,
    threshold_value: float | None,
    alert_indices: set[object],
) -> list[dict[str, object]]:
    if not alert_indices:
        return []

    rows: list[dict[str, object]] = []
    ordered = timeline.sort_values(["athlete_id", "season_id", "time_index"])
    for (athlete_id, season_id), group in ordered.groupby(
        ["athlete_id", "season_id"], sort=False
    ):
        current_indices: list[object] = []
        previous_time_index: int | None = None
        for idx, row in group.iterrows():
            is_alert = idx in alert_indices
            time_index = int(row["time_index"])
            continues = (
                is_alert
                and previous_time_index is not None
                and time_index == previous_time_index + 1
            )
            if is_alert and (not current_indices or continues):
                current_indices.append(idx)
            else:
                if current_indices:
                    rows.append(
                        _episode_row(
                            timeline.loc[current_indices],
                            athlete_id=athlete_id,
                            season_id=season_id,
                            horizon=horizon,
                            percentile=percentile,
                            threshold_value=threshold_value,
                        )
                    )
                current_indices = [idx] if is_alert else []
            previous_time_index = time_index if is_alert else None

        if current_indices:
            rows.append(
                _episode_row(
                    timeline.loc[current_indices],
                    athlete_id=athlete_id,
                    season_id=season_id,
                    horizon=horizon,
                    percentile=percentile,
                    threshold_value=threshold_value,
                )
            )
    return rows


def _episode_row(
    episode: pd.DataFrame,
    athlete_id: object,
    season_id: object,
    horizon: int,
    percentile: float,
    threshold_value: float | None,
) -> dict[str, object]:
    risk_column = f"risk_{horizon}d"
    ordered = episode.sort_values("time_index")
    start = ordered.iloc[0]
    end = ordered.iloc[-1]
    peak = ordered.loc[ordered[risk_column].idxmax()]
    start_date = pd.to_datetime(start["snapshot_date"])
    end_date = pd.to_datetime(end["snapshot_date"])

    return {
        "athlete_id": str(athlete_id),
        "season_id": str(season_id),
        "horizon_days": int(horizon),
        "threshold_kind": "percentile",
        "threshold_value": float(percentile),
        "risk_threshold": threshold_value,
        "start_time_index": int(start["time_index"]),
        "end_time_index": int(end["time_index"]),
        "peak_time_index": int(peak["time_index"]),
        "start_date": str(start["snapshot_date"]),
        "end_date": str(end["snapshot_date"]),
        "peak_date": str(peak["snapshot_date"]),
        "snapshot_count": int(len(ordered)),
        "duration_days": int((end_date - start_date).days),
        "peak_risk": float(peak[risk_column]),
        "mean_risk": float(ordered[risk_column].mean()),
        "event_observed": bool(ordered["event_observed"].astype(bool).any())
        if "event_observed" in ordered.columns
        else None,
        "injury_type": _first_non_empty(ordered.get("injury_type", pd.Series())),
        "days_from_start_to_event": _days_to_event(start),
        "days_from_peak_to_event": _days_to_event(peak),
        "days_from_end_to_event": _days_to_event(end),
        "event_within_horizon_after_start": _event_within_horizon(start, horizon),
        "event_within_horizon_after_peak": _event_within_horizon(peak, horizon),
        "event_within_horizon_after_end": _event_within_horizon(end, horizon),
        "top_model_features": _top_model_features(ordered, horizon),
        "elevated_z_features": _elevated_z_features(ordered),
    }


def _percentile_alert_indices(predictions: pd.Series, percentile: float) -> pd.Index:
    if percentile <= 0:
        return pd.Index([])
    top_n = round(len(predictions) * percentile)
    if top_n <= 0:
        return pd.Index([])
    return predictions.nlargest(top_n).index


def _event_within_horizon(row: pd.Series, horizon: int) -> bool:
    if "event_observed" not in row.index or not bool(row["event_observed"]):
        return False
    days = _days_to_event(row)
    return days is not None and 0 <= days <= horizon


def _days_to_event(row: pd.Series) -> int | None:
    if "event_observed" in row.index and not bool(row["event_observed"]):
        return None
    if "days_to_event" not in row.index or pd.isna(row["days_to_event"]):
        return None
    return int(row["days_to_event"])


def _top_model_features(episode: pd.DataFrame, horizon: int) -> list[dict[str, object]]:
    feature_column = f"top_feature_{horizon}d"
    contribution_column = f"top_contribution_{horizon}d"
    if feature_column not in episode.columns or contribution_column not in episode.columns:
        return []

    rows: list[dict[str, object]] = []
    for feature, group in episode.groupby(feature_column, sort=False):
        if feature is None or str(feature) == "" or pd.isna(feature):
            continue
        mean_abs = group[contribution_column].astype(float).abs().mean()
        rows.append(
            {
                "feature": str(feature),
                "mean_abs_contribution": round(float(mean_abs), 6),
            }
        )
    return sorted(rows, key=lambda row: -float(row["mean_abs_contribution"]))[:3]


def _elevated_z_features(episode: pd.DataFrame) -> list[str]:
    if "elevated_z_features" not in episode.columns:
        return []

    features: list[str] = []
    for value in episode["elevated_z_features"]:
        for feature in _iter_features(value):
            if feature not in features:
                features.append(feature)
    return features


def _iter_features(value: object) -> Iterable[str]:
    if value is None or (not isinstance(value, list) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return []


def _first_non_empty(values: pd.Series) -> str | None:
    for value in values:
        if value is not None and not pd.isna(value) and str(value):
            return str(value)
    return None


def _rate(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"alert episode input missing required columns: {', '.join(missing)}")


def _episode_columns() -> list[str]:
    return [
        "athlete_id",
        "season_id",
        "horizon_days",
        "threshold_kind",
        "threshold_value",
        "risk_threshold",
        "start_time_index",
        "end_time_index",
        "peak_time_index",
        "start_date",
        "end_date",
        "peak_date",
        "snapshot_count",
        "duration_days",
        "peak_risk",
        "mean_risk",
        "event_observed",
        "injury_type",
        "days_from_start_to_event",
        "days_from_peak_to_event",
        "days_from_end_to_event",
        "event_within_horizon_after_start",
        "event_within_horizon_after_peak",
        "event_within_horizon_after_end",
        "top_model_features",
        "elevated_z_features",
    ]
