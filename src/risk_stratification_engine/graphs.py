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
    "edge_density",
    "delta_edge_count",
    "delta_mean_abs_correlation",
    "delta_edge_density",
    "graph_instability",
    "z_mean_abs_correlation",
    "z_edge_density",
    "z_edge_count",
    "z_graph_instability",
]

_INSTABILITY_WINDOW = 3
_Z_SCORE_FEATURES = {
    "z_mean_abs_correlation": "mean_abs_correlation",
    "z_edge_density": "edge_density",
    "z_edge_count": "edge_count",
    "z_graph_instability": "graph_instability",
}


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
        group_rows: list[dict[str, object]] = []
        for row_position, row in enumerate(group.itertuples(index=False)):
            history = group.iloc[max(0, row_position - window_size + 1) : row_position + 1]
            features = _graph_features(history[metric_columns], correlation_threshold)
            group_rows.append(
                {
                    "athlete_id": athlete_id,
                    "season_id": season_id,
                    "time_index": int(getattr(row, "time_index")),
                    "snapshot_date": getattr(row, "date"),
                    **features,
                }
            )
        _add_temporal_features(group_rows, window_size)
        rows.extend(group_rows)
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


def _add_temporal_features(
    group_rows: list[dict[str, object]],
    window_size: int,
) -> None:
    for i, row in enumerate(group_rows):
        node_count = int(row["node_count"])
        max_edges = node_count * (node_count - 1) // 2
        edge_density = float(row["edge_count"]) / max_edges if max_edges > 0 else 0.0
        row["edge_density"] = round(edge_density, 6)

        if i == 0:
            row["delta_edge_count"] = 0
            row["delta_mean_abs_correlation"] = 0.0
            row["delta_edge_density"] = 0.0
        else:
            prev = group_rows[i - 1]
            row["delta_edge_count"] = int(row["edge_count"]) - int(prev["edge_count"])
            row["delta_mean_abs_correlation"] = round(
                float(row["mean_abs_correlation"]) - float(prev["mean_abs_correlation"]), 6
            )
            row["delta_edge_density"] = round(
                float(row["edge_density"]) - float(prev["edge_density"]), 6
            )

        window_start = max(0, i - _INSTABILITY_WINDOW + 1)
        window_corrs = [
            float(group_rows[j]["mean_abs_correlation"])
            for j in range(window_start, i + 1)
        ]
        if len(window_corrs) < 2:
            row["graph_instability"] = 0.0
        else:
            row["graph_instability"] = round(float(np.std(window_corrs)), 6)

        baseline_start = max(0, i - window_size + 1)
        baseline_rows = group_rows[baseline_start:i]
        for z_column, source_column in _Z_SCORE_FEATURES.items():
            row[z_column] = _prior_window_z_score(
                current_value=float(row[source_column]),
                baseline_values=[
                    float(baseline_row[source_column])
                    for baseline_row in baseline_rows
                ],
            )


def _prior_window_z_score(
    current_value: float,
    baseline_values: list[float],
) -> float:
    if len(baseline_values) < 2:
        return 0.0

    baseline_std = float(np.std(baseline_values))
    if baseline_std == 0.0:
        return 0.0

    baseline_mean = float(np.mean(baseline_values))
    z_score = (current_value - baseline_mean) / baseline_std
    return round(float(np.clip(z_score, -10.0, 10.0)), 6)
