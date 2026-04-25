from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_stratification_engine.domain import (
    CANONICAL_MEASUREMENT_COLUMNS,
    INJURY_EVENT_COLUMNS,
    require_columns,
)


REQUIRED_MEASUREMENT_TEXT_FIELDS = (
    "athlete_id",
    "season_id",
    "source",
    "metric_name",
)


def load_measurements(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, CANONICAL_MEASUREMENT_COLUMNS, "measurements")
    frame = frame.loc[:, list(CANONICAL_MEASUREMENT_COLUMNS)].copy()
    _require_nonblank_values(
        frame,
        REQUIRED_MEASUREMENT_TEXT_FIELDS,
        "measurements",
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("measurements contains unparseable date values")
    if frame["metric_value"].isna().any():
        raise ValueError("measurements contains non-numeric metric_value values")
    return frame


def load_injury_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, INJURY_EVENT_COLUMNS, "injury_events")
    frame = frame.loc[:, list(INJURY_EVENT_COLUMNS)].copy()
    raw_injury_date = frame["injury_date"]
    populated_injury_date = raw_injury_date.notna() & raw_injury_date.astype(
        str
    ).str.strip().ne("")
    frame["injury_date"] = pd.to_datetime(raw_injury_date, errors="coerce")
    frame["censor_date"] = pd.to_datetime(frame["censor_date"], errors="coerce")
    frame["event_observed"] = frame["event_observed"].map(_parse_bool)
    if frame.loc[populated_injury_date, "injury_date"].isna().any():
        raise ValueError("injury_events contains unparseable injury_date values")
    if frame["censor_date"].isna().any():
        raise ValueError("injury_events contains unparseable censor_date values")
    if frame.loc[frame["event_observed"], "injury_date"].isna().any():
        raise ValueError("observed injury events require injury_date")
    return frame


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def _require_nonblank_values(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    invalid_columns = [
        column
        for column in columns
        if (
            frame[column].isna()
            | frame[column].astype(str).str.strip().eq("")
        ).any()
    ]
    if invalid_columns:
        raise ValueError(
            f"{label} contains blank/null required fields: "
            f"{', '.join(invalid_columns)}"
        )


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value}")
