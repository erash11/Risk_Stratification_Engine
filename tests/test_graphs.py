from pathlib import Path

import pandas as pd
import pytest

from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.io import load_measurements
from risk_stratification_engine.trajectories import build_measurement_matrix


FIXTURES = Path(__file__).parent / "fixtures"
OUTPUT_COLUMNS = [
    "athlete_id",
    "season_id",
    "time_index",
    "snapshot_date",
    "node_count",
    "edge_count",
    "mean_abs_correlation",
]


def test_build_graph_snapshots_returns_athlete_specific_snapshots():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)

    assert list(snapshots.columns) == OUTPUT_COLUMNS
    assert snapshots.shape[0] == 4
    assert set(snapshots["athlete_id"]) == {"a1", "a2"}


def test_build_graph_snapshots_preserves_early_history_with_zero_edges():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    first = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 0)
    ].iloc[0]

    assert first["node_count"] == 2
    assert first["edge_count"] == 0
    assert first["mean_abs_correlation"] == 0.0


def test_build_graph_snapshots_detects_relationship_after_history_accumulates():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    second = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 1)
    ].iloc[0]

    assert second["node_count"] == 2
    assert second["edge_count"] == 1
    assert second["mean_abs_correlation"] == 1.0


def test_build_graph_snapshots_returns_schema_for_empty_measurement_matrix():
    empty = pd.DataFrame(
        columns=[
            "athlete_id",
            "season_id",
            "date",
            "time_index",
            "jump_height",
            "eccentric_peak_force_asymmetry",
        ]
    )

    snapshots = build_graph_snapshots(empty, window_size=2)

    assert list(snapshots.columns) == OUTPUT_COLUMNS
    assert snapshots.empty


def test_build_graph_snapshots_rejects_non_numeric_metric_columns():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)
    matrix["training_note"] = ["ok", "sore", "ok", "fresh"]

    with pytest.raises(
        ValueError,
        match="measurement_matrix metric columns must be numeric: training_note",
    ):
        build_graph_snapshots(matrix, window_size=2)


def test_build_graph_snapshots_rejects_too_small_window_size():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    with pytest.raises(ValueError, match="window_size must be at least 2"):
        build_graph_snapshots(matrix, window_size=1)
