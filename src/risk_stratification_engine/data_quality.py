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
        "identity": {
            **_identity_audit(source_identities),
            "single_source_athlete_review": _identity_review(
                source_identities,
                measurement_dates,
                injuries,
            ),
        },
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


def _identity_review(
    source_identities: dict[str, pd.DataFrame],
    measurements: pd.DataFrame,
    injuries: pd.DataFrame,
) -> dict[str, Any]:
    source_sets = {
        source: _athlete_set(frame)
        for source, frame in sorted(source_identities.items())
    }
    athlete_sources: dict[str, list[str]] = {}
    for source, athlete_ids in source_sets.items():
        for athlete_id in athlete_ids:
            athlete_sources.setdefault(athlete_id, []).append(source)
    single_source_rows = [
        {"athlete_id": athlete_id, "source": sources[0]}
        for athlete_id, sources in athlete_sources.items()
        if len(sources) == 1
    ]
    single_source_rows = sorted(
        single_source_rows,
        key=lambda row: (row["source"], row["athlete_id"]),
    )
    return {
        "total_count": len(single_source_rows),
        "by_source": _count_by(single_source_rows, "source"),
        "examples": [
            _single_source_example(row, measurements, injuries)
            for row in single_source_rows[:100]
        ],
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


def _single_source_example(
    row: dict[str, str],
    measurements: pd.DataFrame,
    injuries: pd.DataFrame,
) -> dict[str, Any]:
    athlete_id = row["athlete_id"]
    athlete_measurements = measurements.loc[measurements["athlete_id"] == athlete_id]
    athlete_injuries = injuries.loc[injuries["athlete_id"] == athlete_id]
    observed_events = athlete_injuries.loc[
        athlete_injuries["event_observed"].astype(bool)
    ]
    if athlete_measurements.empty:
        return {
            "athlete_id": athlete_id,
            "source": row["source"],
            "measurement_rows": 0,
            "measurement_dates": 0,
            "athlete_seasons": 0,
            "first_measurement_date": None,
            "last_measurement_date": None,
            "modeled_observed_events": int(len(observed_events)),
        }
    return {
        "athlete_id": athlete_id,
        "source": row["source"],
        "measurement_rows": int(len(athlete_measurements)),
        "measurement_dates": int(athlete_measurements["date"].nunique()),
        "athlete_seasons": int(athlete_measurements["season_id"].nunique()),
        "first_measurement_date": _json_ready(athlete_measurements["date"].min()),
        "last_measurement_date": _json_ready(athlete_measurements["date"].max()),
        "modeled_observed_events": int(len(observed_events)),
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
                    "season_measurement_dates": 0,
                    "season_source_count": 0,
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
                    "season_measurement_dates": int(nearby["date"].nunique()),
                    "season_source_count": int(nearby["source"].nunique()),
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
        "events_without_nearby_measurements_by_gap_bucket": _injury_gap_buckets(rows),
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


def _injury_gap_buckets(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {
        "4-7d": 0,
        "8-14d": 0,
        "15-30d": 0,
        "31-90d": 0,
        "91d+": 0,
        "no_measurements": 0,
    }
    for row in rows:
        gap = row["nearest_measurement_gap_days"]
        if gap is None:
            buckets["no_measurements"] += 1
        elif gap <= 7:
            buckets["4-7d"] += 1
        elif gap <= 14:
            buckets["8-14d"] += 1
        elif gap <= 30:
            buckets["15-30d"] += 1
        elif gap <= 90:
            buckets["31-90d"] += 1
        else:
            buckets["91d+"] += 1
    return buckets


def _count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row[key]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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
