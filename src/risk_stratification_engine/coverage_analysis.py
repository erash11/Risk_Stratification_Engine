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
        return result.astype(str).fillna("low")
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
    raise NotImplementedError


def build_coverage_flag(channel_results: list[dict]) -> str:
    raise NotImplementedError


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
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
