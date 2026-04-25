from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


CANONICAL_MEASUREMENT_COLUMNS = (
    "athlete_id",
    "date",
    "season_id",
    "source",
    "metric_name",
    "metric_value",
)

INJURY_EVENT_COLUMNS = (
    "athlete_id",
    "season_id",
    "injury_date",
    "injury_type",
    "event_observed",
    "censor_date",
)


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns: {', '.join(sorted(missing))}"
        )
