from __future__ import annotations

import pandas as pd


DEFAULT_HORIZONS = (7, 14, 30)


def attach_time_to_event_labels(
    snapshots: pd.DataFrame,
    injury_events: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    labeled = snapshots.merge(
        injury_events,
        on=["athlete_id", "season_id"],
        how="left",
        validate="many_to_one",
    )
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
    boolean_columns = [
        "event_observed",
        *(f"event_within_{horizon}d" for horizon in horizons),
    ]
    for column in boolean_columns:
        labeled[column] = labeled[column].astype(object)
    return labeled
