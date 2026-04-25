from __future__ import annotations

import pandas as pd


DEFAULT_HORIZONS = (7, 14, 30)


def attach_time_to_event_labels(
    snapshots: pd.DataFrame,
    injury_events: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    _validate_horizons(horizons)
    labeled = snapshots.merge(
        injury_events,
        on=["athlete_id", "season_id"],
        how="left",
        validate="many_to_one",
    )
    _require_event_rows(labeled)
    labeled["event_date"] = labeled["injury_date"].where(
        labeled["event_observed"], labeled["censor_date"]
    )
    labeled["days_to_event"] = (
        labeled["event_date"] - labeled["snapshot_date"]
    ).dt.days
    labeled = labeled.loc[labeled["days_to_event"] >= 0].copy()
    for horizon in horizons:
        labeled[f"event_within_{horizon}d"] = (
            labeled["event_observed"] & (labeled["days_to_event"] <= horizon)
        )
    return labeled


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
        for horizon in horizons
    ) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique positive integers")


def _require_event_rows(labeled: pd.DataFrame) -> None:
    missing = labeled.loc[labeled["event_observed"].isna(), ["athlete_id", "season_id"]]
    if missing.empty:
        return

    missing_keys = missing.drop_duplicates().sort_values(["athlete_id", "season_id"])
    key_list = ", ".join(
        f"({row.athlete_id}, {row.season_id})"
        for row in missing_keys.itertuples(index=False)
    )
    raise ValueError(f"missing injury event rows for athlete-season keys: {key_list}")
