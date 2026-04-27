from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk_stratification_engine.graphs import OUTPUT_COLUMNS, build_graph_snapshots
from risk_stratification_engine.io import load_measurements
from risk_stratification_engine.trajectories import build_measurement_matrix


FIXTURES = Path(__file__).parent / "fixtures"


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


def test_build_graph_snapshots_edge_density_is_zero_when_no_edges():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    first = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 0)
    ].iloc[0]

    assert first["edge_density"] == 0.0


def test_build_graph_snapshots_edge_density_matches_edge_fraction():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    second = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 1)
    ].iloc[0]

    # 2 nodes → max_edges = 1; edge_count = 1 → density = 1.0
    assert second["edge_density"] == 1.0


def test_build_graph_snapshots_deltas_are_zero_at_first_snapshot():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    first = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 0)
    ].iloc[0]

    assert first["delta_edge_count"] == 0
    assert first["delta_mean_abs_correlation"] == 0.0
    assert first["delta_edge_density"] == 0.0


def test_build_graph_snapshots_computes_deltas_from_prior_snapshot():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    second = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 1)
    ].iloc[0]

    # edges: 0 → 1, correlation: 0.0 → 1.0, density: 0.0 → 1.0
    assert second["delta_edge_count"] == 1
    assert second["delta_mean_abs_correlation"] == pytest.approx(1.0)
    assert second["delta_edge_density"] == pytest.approx(1.0)


def test_build_graph_snapshots_graph_instability_is_zero_at_first_snapshot():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    first = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 0)
    ].iloc[0]

    assert first["graph_instability"] == 0.0


def test_build_graph_snapshots_graph_instability_reflects_correlation_variance():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    second = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 1)
    ].iloc[0]

    # population std of [0.0, 1.0] = 0.5
    assert second["graph_instability"] == pytest.approx(np.std([0.0, 1.0]))
