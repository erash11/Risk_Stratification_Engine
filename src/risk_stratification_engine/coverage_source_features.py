from __future__ import annotations

import pandas as pd

from risk_stratification_engine.domain import (
    CANONICAL_MEASUREMENT_COLUMNS,
    require_columns,
)


TRACKED_SOURCES = ("bodyweight", "forceplate", "gps", "perch")

COVERAGE_SOURCE_FEATURE_COLUMNS = (
    "coverage_measurement_days_to_date",
    "coverage_measurement_rows_to_date",
    "coverage_source_count_to_date",
    "coverage_days_since_previous_measurement",
    "coverage_seen_bodyweight_to_date",
    "coverage_seen_forceplate_to_date",
    "coverage_seen_gps_to_date",
    "coverage_seen_perch_to_date",
)


def attach_coverage_source_features(
    graph_features: pd.DataFrame,
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(measurements, CANONICAL_MEASUREMENT_COLUMNS, "measurements")
    out = graph_features.copy()
    for column in COVERAGE_SOURCE_FEATURE_COLUMNS:
        out[column] = 0
    if out.empty or measurements.empty:
        return out

    measurement_features = _coverage_feature_frame(measurements)
    out["season_id"] = out["season_id"].astype(str)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    enriched = out.merge(
        measurement_features,
        left_on=["athlete_id", "season_id", "snapshot_date"],
        right_on=["athlete_id", "season_id", "date"],
        how="left",
        suffixes=("", "_coverage"),
    )
    enriched = enriched.drop(columns=["date"], errors="ignore")
    for column in COVERAGE_SOURCE_FEATURE_COLUMNS:
        coverage_column = f"{column}_coverage"
        if coverage_column in enriched.columns:
            enriched[column] = enriched[coverage_column].fillna(enriched[column])
            enriched = enriched.drop(columns=[coverage_column])
        enriched[column] = (
            pd.to_numeric(enriched[column], errors="coerce").fillna(0).astype(int)
        )
    return enriched


def _coverage_feature_frame(measurements: pd.DataFrame) -> pd.DataFrame:
    frame = measurements.copy()
    frame["season_id"] = frame["season_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["source_key"] = frame["source"].map(_source_key)
    rows: list[dict[str, object]] = []
    for (athlete_id, season_id), group in frame.groupby(
        ["athlete_id", "season_id"],
        sort=True,
    ):
        prior_dates: list[pd.Timestamp] = []
        for date, date_group in group.sort_values("date").groupby("date", sort=True):
            history = group[group["date"] <= date]
            source_keys = set(history["source_key"].dropna())
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "season_id": str(season_id),
                    "date": date,
                    "coverage_measurement_days_to_date": int(
                        history["date"].nunique()
                    ),
                    "coverage_measurement_rows_to_date": int(len(history)),
                    "coverage_source_count_to_date": int(
                        history["source_key"].nunique()
                    ),
                    "coverage_days_since_previous_measurement": (
                        int((date - prior_dates[-1]).days) if prior_dates else 0
                    ),
                    **{
                        f"coverage_seen_{source}_to_date": int(source in source_keys)
                        for source in TRACKED_SOURCES
                    },
                }
            )
            prior_dates.append(date)
    return pd.DataFrame(rows)


def _source_key(value: object) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "")
