from __future__ import annotations

import pandas as pd

from risk_stratification_engine.domain import (
    CANONICAL_MEASUREMENT_COLUMNS,
    require_columns,
)


def build_measurement_matrix(measurements: pd.DataFrame) -> pd.DataFrame:
    require_columns(measurements, CANONICAL_MEASUREMENT_COLUMNS, "measurements")
    matrix = (
        measurements.pivot_table(
            index=["athlete_id", "season_id", "date"],
            columns="metric_name",
            values="metric_value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["athlete_id", "season_id", "date"])
    )
    matrix.columns.name = None
    metric_columns = sorted(
        column
        for column in matrix.columns
        if column not in {"athlete_id", "season_id", "date"}
    )
    matrix = matrix[["athlete_id", "season_id", "date", *metric_columns]]
    matrix.insert(
        3,
        "time_index",
        matrix.groupby(["athlete_id", "season_id"]).cumcount(),
    )
    return matrix
