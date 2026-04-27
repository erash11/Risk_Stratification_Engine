from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


def build_data_quality_audit(
    measurements: pd.DataFrame,
    injury_events: pd.DataFrame,
    source_identities: dict[str, pd.DataFrame],
    *,
    sparse_measurement_dates_threshold: int = 4,
    large_gap_days: int = 21,
    injury_nearby_days: int = 14,
) -> dict[str, Any]:
    measurement_dates = measurements.copy()
    measurement_dates["date"] = pd.to_datetime(measurement_dates["date"])
    injuries = injury_events.copy()
    injuries["injury_date"] = pd.to_datetime(injuries["injury_date"], errors="coerce")
    return {
        "parameters": {
            "sparse_measurement_dates_threshold": sparse_measurement_dates_threshold,
            "large_gap_days": large_gap_days,
            "injury_nearby_days": injury_nearby_days,
        },
        "identity": _identity_audit(source_identities),
        "coverage": _coverage_audit(
            measurement_dates,
            sparse_measurement_dates_threshold,
        ),
        "date_gaps": _date_gap_audit(measurement_dates, large_gap_days),
        "duplicates": _duplicate_audit(measurement_dates),
        "injuries": _injury_audit(
            measurement_dates,
            injuries,
            injury_nearby_days,
        ),
    }


def _identity_audit(source_identities: dict[str, pd.DataFrame]) -> dict[str, Any]:
    source_sets = {
        source: _athlete_set(frame)
        for source, frame in sorted(source_identities.items())
    }
    athlete_sources: dict[str, list[str]] = {}
    for source, athlete_ids in source_sets.items():
        for athlete_id in athlete_ids:
            athlete_sources.setdefault(athlete_id, []).append(source)

    combination_counts: dict[str, int] = {}
    single_source_examples: dict[str, list[str]] = {}
    for athlete_id, sources in athlete_sources.items():
        source_key = "|".join(sorted(sources))
        combination_counts[source_key] = combination_counts.get(source_key, 0) + 1
        if len(sources) == 1:
            single_source_examples.setdefault(sources[0], []).append(athlete_id)

    pairwise_overlap_counts = {
        f"{left}|{right}": len(source_sets[left] & source_sets[right])
        for left, right in combinations(source_sets, 2)
    }
    return {
        "source_athlete_counts": {
            source: len(athlete_ids)
            for source, athlete_ids in source_sets.items()
        },
        "multi_source_athlete_count": sum(
            1 for sources in athlete_sources.values() if len(sources) > 1
        ),
        "source_combination_counts": dict(sorted(combination_counts.items())),
        "pairwise_overlap_counts": pairwise_overlap_counts,
        "single_source_athlete_counts": {
            source: len(examples)
            for source, examples in sorted(single_source_examples.items())
        },
        "single_source_athlete_examples": {
            source: sorted(examples)[:20]
            for source, examples in sorted(single_source_examples.items())
        },
    }


def _coverage_audit(
    measurements: pd.DataFrame,
    sparse_measurement_dates_threshold: int,
) -> dict[str, Any]:
    grouped = (
        measurements.groupby(["athlete_id", "season_id"], as_index=False)
        .agg(
            measurement_rows=("metric_value", "size"),
            measurement_dates=("date", "nunique"),
            first_measurement_date=("date", "min"),
            last_measurement_date=("date", "max"),
            source_count=("source", "nunique"),
        )
        .sort_values(["measurement_dates", "athlete_id", "season_id"])
    )
    sparse = grouped.loc[
        grouped["measurement_dates"] < sparse_measurement_dates_threshold
    ]
    return {
        "athlete_season_count": int(len(grouped)),
        "sparse_athlete_season_count": int(len(sparse)),
        "sparse_athlete_seasons": _records(
            sparse,
            [
                "athlete_id",
                "season_id",
                "measurement_rows",
                "measurement_dates",
                "source_count",
                "first_measurement_date",
                "last_measurement_date",
            ],
            limit=100,
        ),
    }


def _date_gap_audit(
    measurements: pd.DataFrame,
    large_gap_days: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (athlete_id, season_id), group in measurements.groupby(
        ["athlete_id", "season_id"],
        sort=True,
    ):
        dates = sorted(group["date"].drop_duplicates())
        for previous, current in zip(dates, dates[1:]):
            gap_days = int((current - previous).days)
            if gap_days > large_gap_days:
                rows.append(
                    {
                        "athlete_id": athlete_id,
                        "season_id": season_id,
                        "gap_start": previous,
                        "gap_end": current,
                        "gap_days": gap_days,
                    }
                )
    rows = sorted(rows, key=lambda row: (-row["gap_days"], row["athlete_id"]))
    return {
        "large_gap_days": large_gap_days,
        "large_gap_count": len(rows),
        "large_gaps": _json_ready(rows[:100]),
    }


def _duplicate_audit(measurements: pd.DataFrame) -> dict[str, Any]:
    duplicate_groups = (
        measurements.groupby(
            ["athlete_id", "season_id", "date", "source", "metric_name"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "row_count"})
    )
    duplicate_groups = duplicate_groups.loc[duplicate_groups["row_count"] > 1]
    duplicate_groups = duplicate_groups.sort_values(
        ["row_count", "athlete_id", "season_id", "date"],
        ascending=[False, True, True, True],
    )
    return {
        "duplicate_same_day_metric_count": int(len(duplicate_groups)),
        "duplicate_same_day_metrics": _records(
            duplicate_groups,
            [
                "athlete_id",
                "season_id",
                "date",
                "source",
                "metric_name",
                "row_count",
            ],
            limit=100,
        ),
    }


def _injury_audit(
    measurements: pd.DataFrame,
    injuries: pd.DataFrame,
    injury_nearby_days: int,
) -> dict[str, Any]:
    observed = injuries.loc[injuries["event_observed"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for injury in observed.itertuples(index=False):
        athlete_id = getattr(injury, "athlete_id")
        season_id = getattr(injury, "season_id")
        injury_date = getattr(injury, "injury_date")
        nearby = measurements.loc[
            (measurements["athlete_id"] == athlete_id)
            & (measurements["season_id"] == season_id)
        ]
        if nearby.empty:
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "season_id": season_id,
                    "injury_date": injury_date,
                    "nearest_measurement_date": None,
                    "nearest_measurement_gap_days": None,
                }
            )
            continue
        gaps = (nearby["date"] - injury_date).abs().dt.days
        nearest_index = gaps.idxmin()
        nearest_gap = int(gaps.loc[nearest_index])
        if nearest_gap > injury_nearby_days:
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "season_id": season_id,
                    "injury_date": injury_date,
                    "nearest_measurement_date": nearby.loc[nearest_index, "date"],
                    "nearest_measurement_gap_days": nearest_gap,
                }
            )
    rows = sorted(
        rows,
        key=lambda row: (
            row["nearest_measurement_gap_days"] is None,
            row["nearest_measurement_gap_days"] or 999999,
            row["athlete_id"],
        ),
        reverse=True,
    )
    return {
        "injury_nearby_days": injury_nearby_days,
        "observed_event_count": int(len(observed)),
        "events_without_nearby_measurements_count": len(rows),
        "events_without_nearby_measurements": _json_ready(rows[:100]),
    }


def _athlete_set(frame: pd.DataFrame) -> set[str]:
    if "athlete_id" not in frame.columns:
        return set()
    return {
        str(athlete_id)
        for athlete_id in frame["athlete_id"].dropna().unique()
        if str(athlete_id).strip()
    }


def _records(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return _json_ready(frame.loc[:, columns].head(limit).to_dict("records"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
