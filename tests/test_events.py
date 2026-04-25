from pathlib import Path

from risk_stratification_engine.events import attach_time_to_event_labels
from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.io import load_injury_events, load_measurements
from risk_stratification_engine.trajectories import build_measurement_matrix


FIXTURES = Path(__file__).parent / "fixtures"


def test_attach_time_to_event_labels_adds_observed_event_horizons():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)

    labeled = attach_time_to_event_labels(snapshots, injuries)

    a1_first = labeled.loc[
        (labeled["athlete_id"] == "a1") & (labeled["time_index"] == 0)
    ].iloc[0]
    assert a1_first["event_observed"] is True
    assert a1_first["days_to_event"] == 19
    assert a1_first["injury_type"] == "lower_extremity_soft_tissue"
    assert a1_first["event_within_7d"] is False
    assert a1_first["event_within_14d"] is False
    assert a1_first["event_within_30d"] is True


def test_attach_time_to_event_labels_preserves_censored_athletes():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)

    labeled = attach_time_to_event_labels(snapshots, injuries)

    a2_first = labeled.loc[
        (labeled["athlete_id"] == "a2") & (labeled["time_index"] == 0)
    ].iloc[0]
    assert a2_first["event_observed"] is False
    assert a2_first["days_to_event"] == 31
    assert a2_first["injury_type"] == "none"
    assert a2_first["event_within_30d"] is False
