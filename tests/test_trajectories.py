from pathlib import Path

from risk_stratification_engine.io import load_measurements
from risk_stratification_engine.trajectories import build_measurement_matrix


FIXTURES = Path(__file__).parent / "fixtures"


def test_build_measurement_matrix_pivots_metrics_by_athlete_season_date():
    measurements = load_measurements(FIXTURES / "measurements.csv")

    matrix = build_measurement_matrix(measurements)

    assert list(matrix.columns) == [
        "athlete_id",
        "season_id",
        "date",
        "time_index",
        "eccentric_peak_force_asymmetry",
        "jump_height",
    ]
    assert matrix.shape[0] == 4
    first = matrix.iloc[0].to_dict()
    assert first["athlete_id"] == "a1"
    assert first["time_index"] == 0
    assert first["jump_height"] == 42.0
    assert first["eccentric_peak_force_asymmetry"] == 8.0


def test_build_measurement_matrix_time_index_resets_by_athlete_season():
    measurements = load_measurements(FIXTURES / "measurements.csv")

    matrix = build_measurement_matrix(measurements)

    indices = matrix.groupby(["athlete_id", "season_id"])["time_index"].apply(list)
    assert indices.loc[("a1", "2026")] == [0, 1]
    assert indices.loc[("a2", "2026")] == [0, 1]
