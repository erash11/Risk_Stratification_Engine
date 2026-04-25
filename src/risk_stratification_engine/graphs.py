from __future__ import annotations

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {"athlete_id", "season_id", "date", "time_index"}
OUTPUT_COLUMNS = [
    "athlete_id",
    "season_id",
    "time_index",
    "snapshot_date",
    "node_count",
    "edge_count",
    "mean_abs_correlation",
]


def build_graph_snapshots(
    measurement_matrix: pd.DataFrame,
    window_size: int = 4,
    correlation_threshold: float = 0.3,
) -> pd.DataFrame:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    metric_columns = [
        column for column in measurement_matrix.columns if column not in IDENTIFIER_COLUMNS
    ]
    if measurement_matrix.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    _require_numeric_metrics(measurement_matrix, metric_columns)
    rows: list[dict[str, object]] = []
    grouped = measurement_matrix.sort_values(
        ["athlete_id", "season_id", "date"]
    ).groupby(["athlete_id", "season_id"], sort=False)

    for (athlete_id, season_id), group in grouped:
        for row_position, row in enumerate(group.itertuples(index=False)):
            history = group.iloc[max(0, row_position - window_size + 1) : row_position + 1]
            features = _graph_features(history[metric_columns], correlation_threshold)
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "season_id": season_id,
                    "time_index": int(getattr(row, "time_index")),
                    "snapshot_date": getattr(row, "date"),
                    **features,
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _require_numeric_metrics(
    measurement_matrix: pd.DataFrame,
    metric_columns: list[str],
) -> None:
    non_numeric_columns = [
        column
        for column in metric_columns
        if not pd.api.types.is_numeric_dtype(measurement_matrix[column])
    ]
    if non_numeric_columns:
        column_list = ", ".join(non_numeric_columns)
        raise ValueError(
            f"measurement_matrix metric columns must be numeric: {column_list}"
        )


def _graph_features(
    history: pd.DataFrame,
    correlation_threshold: float,
) -> dict[str, float | int]:
    node_count = len(history.columns)
    if len(history) < 2 or node_count < 2:
        return {
            "node_count": node_count,
            "edge_count": 0,
            "mean_abs_correlation": 0.0,
        }

    corr = history.corr(numeric_only=True).fillna(0.0).abs().to_numpy()
    upper = corr[np.triu_indices(node_count, k=1)]
    if len(upper) == 0:
        edge_count = 0
        mean_abs_correlation = 0.0
    else:
        edge_count = int((upper >= correlation_threshold).sum())
        mean_abs_correlation = float(round(upper.mean(), 6))
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "mean_abs_correlation": mean_abs_correlation,
    }
