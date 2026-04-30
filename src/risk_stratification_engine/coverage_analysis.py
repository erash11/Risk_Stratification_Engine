from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


COVERAGE_TIER_LABELS = ("low", "medium", "high")

_TIER_COLUMNS = [
    "athlete_id",
    "season_id",
    "measurement_days",
    "measurement_row_count",
    "source_count",
    "median_days_between_measurements",
    "coverage_tier",
]


def build_coverage_tiers(measurements: pd.DataFrame) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame(columns=_TIER_COLUMNS)
    frame = measurements.copy()
    frame["season_id"] = frame["season_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    rows = []
    for (athlete_id, season_id), group in frame.groupby(
        ["athlete_id", "season_id"], sort=True
    ):
        rows.append(
            {
                "athlete_id": athlete_id,
                "season_id": str(season_id),
                "measurement_days": int(group["date"].nunique()),
                "measurement_row_count": int(len(group)),
                "source_count": int(group["source"].nunique()),
                "median_days_between_measurements": _median_days(group),
            }
        )
    tier_frame = pd.DataFrame(rows)
    tier_frame["coverage_tier"] = _assign_tiers(tier_frame["measurement_days"])
    return tier_frame[_TIER_COLUMNS]


def _assign_tiers(measurement_days: pd.Series) -> pd.Series:
    n = len(measurement_days)
    if n == 0:
        return pd.Series(dtype=str)
    if n < 3:
        return pd.Series(["low"] * n, index=measurement_days.index)
    try:
        result = pd.qcut(
            measurement_days,
            q=3,
            labels=list(COVERAGE_TIER_LABELS),
            duplicates="drop",
        )
        return result.fillna("low").astype(str)
    except ValueError:
        return pd.Series(["low"] * n, index=measurement_days.index)


def _median_days(group: pd.DataFrame) -> float | None:
    dates = group["date"].dropna().drop_duplicates().sort_values()
    if len(dates) < 2:
        return None
    deltas = dates.diff().dropna().dt.days.tolist()
    if not deltas:
        return None
    return round(float(pd.Series(deltas).median()), 3)


def build_coverage_stratified_evaluation(
    timeline_with_tiers: pd.DataFrame,
    channel: dict,
) -> dict:
    horizon = int(channel["horizon_days"])
    threshold_value = float(channel["threshold_value"])
    risk_col = f"risk_{horizon}d"
    event_col = f"event_within_{horizon}d"

    pop_threshold = (
        float(timeline_with_tiers[risk_col].quantile(1.0 - threshold_value))
        if not timeline_with_tiers.empty
        else 0.0
    )

    rows = []
    for tier in COVERAGE_TIER_LABELS:
        tier_frame = timeline_with_tiers[
            timeline_with_tiers["coverage_tier"] == tier
        ]
        rows.append(
            _stratified_row(
                frame=tier_frame,
                risk_col=risk_col,
                event_col=event_col,
                pop_threshold=pop_threshold,
                channel_name=str(channel["channel_name"]),
                coverage_tier=tier,
                season_id="all",
            )
        )
        for season_id, season_group in tier_frame.groupby("season_id", sort=True):
            rows.append(
                _stratified_row(
                    frame=season_group,
                    risk_col=risk_col,
                    event_col=event_col,
                    pop_threshold=pop_threshold,
                    channel_name=str(channel["channel_name"]),
                    coverage_tier=tier,
                    season_id=str(season_id),
                )
            )

    tier_capture_rates = {
        row["coverage_tier"]: row["capture_rate"]
        for row in rows
        if row["season_id"] == "all"
    }

    return {
        "channel_name": str(channel["channel_name"]),
        "population_threshold": _clean_value(pop_threshold),
        "tier_capture_rates": tier_capture_rates,
        "rows": rows,
    }


def build_coverage_flag(channel_results: list[dict]) -> str:
    diffs = []
    for ch in channel_results:
        rates = ch["tier_capture_rates"]
        high = rates.get("high")
        low = rates.get("low")
        if high is not None and low is not None:
            diffs.append(high - low)
    if not diffs:
        return "mixed"
    mean_diff = sum(diffs) / len(diffs)
    if mean_diff >= 0.15:
        return "coverage_confounded"
    if mean_diff < 0.05:
        return "coverage_independent"
    return "mixed"


def _stratified_row(
    frame: pd.DataFrame,
    risk_col: str,
    event_col: str,
    pop_threshold: float,
    channel_name: str,
    coverage_tier: str,
    season_id: str,
) -> dict:
    athlete_seasons = (
        frame[["athlete_id", "season_id"]].drop_duplicates()
        if not frame.empty
        else pd.DataFrame(columns=["athlete_id", "season_id"])
    )
    athlete_season_count = int(len(athlete_seasons))

    if frame.empty:
        return {
            "channel_name": channel_name,
            "coverage_tier": coverage_tier,
            "season_id": season_id,
            "athlete_season_count": 0,
            "observed_event_count": 0,
            "captured_event_count": 0,
            "capture_rate": None,
            "episodes_per_athlete_season": None,
            "mean_measurement_days": None,
        }

    observed_frame = frame[frame["event_observed"].astype(bool)]
    observed_athlete_seasons = observed_frame[
        ["athlete_id", "season_id"]
    ].drop_duplicates()
    observed_event_count = int(len(observed_athlete_seasons))

    # Captured: observed athlete-season had ≥1 snapshot where
    # event_within_{horizon}d == 1 AND risk >= population threshold.
    captured_count = 0
    if observed_event_count > 0:
        flagged = frame[
            (frame[risk_col] >= pop_threshold)
            & (frame[event_col].fillna(0).astype(int) == 1)
        ]
        flagged_pairs = flagged[["athlete_id", "season_id"]].drop_duplicates()
        captured_count = int(
            observed_athlete_seasons.merge(
                flagged_pairs, on=["athlete_id", "season_id"], how="inner"
            ).shape[0]
        )

    capture_rate = (
        round(float(captured_count) / float(observed_event_count), 6)
        if observed_event_count > 0
        else None
    )

    above_threshold = int((frame[risk_col] >= pop_threshold).sum())
    episodes_per_athlete_season = (
        round(float(above_threshold) / float(athlete_season_count), 6)
        if athlete_season_count > 0
        else None
    )

    mean_measurement_days = (
        round(float(frame["measurement_days"].mean()), 3)
        if "measurement_days" in frame.columns
        and not frame["measurement_days"].isna().all()
        else None
    )

    return {
        "channel_name": channel_name,
        "coverage_tier": coverage_tier,
        "season_id": season_id,
        "athlete_season_count": athlete_season_count,
        "observed_event_count": observed_event_count,
        "captured_event_count": captured_count,
        "capture_rate": capture_rate,
        "episodes_per_athlete_season": episodes_per_athlete_season,
        "mean_measurement_days": mean_measurement_days,
    }


def write_coverage_stratified_evaluation_report(
    path: Path,
    result: dict,
) -> None:
    raise NotImplementedError


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if not isinstance(value, str) and pd.isna(value):
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"
