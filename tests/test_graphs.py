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


Z_SCORE_COLUMNS = [
    "z_mean_abs_correlation",
    "z_edge_density",
    "z_edge_count",
    "z_graph_instability",
]


def _snapshot_matrix(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "date": pd.Timestamp(f"2026-01-{index + 1:02d}"),
                "time_index": index,
                **row,
            }
            for index, row in enumerate(rows)
        ]
    )


def test_build_graph_snapshots_includes_z_score_feature_columns():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)

    assert set(Z_SCORE_COLUMNS).issubset(OUTPUT_COLUMNS)
    assert set(Z_SCORE_COLUMNS).issubset(snapshots.columns)


def test_build_graph_snapshots_z_scores_are_zero_at_first_snapshot():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    first = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 0)
    ].iloc[0]

    assert first[Z_SCORE_COLUMNS].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_build_graph_snapshots_z_scores_are_zero_at_second_snapshot():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    second = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 1)
    ].iloc[0]

    assert second[Z_SCORE_COLUMNS].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_build_graph_snapshots_z_score_nonzero_once_baseline_has_two_prior_snapshots():
    matrix = _snapshot_matrix(
        [
            {"jump_height": 1.0, "force_asymmetry": 1.0},
            {"jump_height": 2.0, "force_asymmetry": 2.0},
            {"jump_height": 3.0, "force_asymmetry": 1.0},
        ]
    )

    snapshots = build_graph_snapshots(matrix, window_size=4)
    third = snapshots.loc[snapshots["time_index"] == 2].iloc[0]

    assert third["z_mean_abs_correlation"] != 0.0
    assert third["z_mean_abs_correlation"] == pytest.approx(-1.0)


def test_build_graph_snapshots_z_score_is_zero_when_baseline_std_is_zero():
    matrix = _snapshot_matrix(
        [
            {"jump_height": 1.0, "force_asymmetry": 1.0},
            {"jump_height": 2.0, "force_asymmetry": 1.0},
            {"jump_height": 3.0, "force_asymmetry": 9.0},
        ]
    )

    snapshots = build_graph_snapshots(matrix, window_size=4)
    third = snapshots.loc[snapshots["time_index"] == 2].iloc[0]

    assert third["z_mean_abs_correlation"] == 0.0


def test_build_graph_snapshots_z_score_clips_extreme_departures():
    metric_count = 20
    early_sparse_row = {f"metric_{i}": 0.0 for i in range(metric_count)}
    early_sparse_row["metric_0"] = 1.0
    early_sparse_row["metric_1"] = 1.0
    second_sparse_row = {f"metric_{i}": 0.0 for i in range(metric_count)}
    second_sparse_row["metric_0"] = 2.0
    second_sparse_row["metric_1"] = 2.0
    departure_row = {f"metric_{i}": float(i + 1) for i in range(metric_count)}

    matrix = _snapshot_matrix([early_sparse_row, second_sparse_row, departure_row])

    snapshots = build_graph_snapshots(matrix, window_size=4)
    third = snapshots.loc[snapshots["time_index"] == 2].iloc[0]

    assert third["z_mean_abs_correlation"] == 10.0
