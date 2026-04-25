from pathlib import Path

import pandas as pd
import pytest

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
    assert bool(a1_first["event_observed"])
    assert a1_first["days_to_event"] == 19
    assert a1_first["injury_type"] == "lower_extremity_soft_tissue"
    assert not bool(a1_first["event_within_7d"])
    assert not bool(a1_first["event_within_14d"])
    assert bool(a1_first["event_within_30d"])


def test_attach_time_to_event_labels_preserves_censored_athletes():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)

    labeled = attach_time_to_event_labels(snapshots, injuries)

    a2_first = labeled.loc[
        (labeled["athlete_id"] == "a2") & (labeled["time_index"] == 0)
    ].iloc[0]
    assert not bool(a2_first["event_observed"])
    assert a2_first["days_to_event"] == 31
    assert a2_first["injury_type"] == "none"
    assert not bool(a2_first["event_within_30d"])


def test_attach_time_to_event_labels_requires_event_metadata_for_every_snapshot():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)
    incomplete_injuries = injuries.loc[injuries["athlete_id"] != "a2"].copy()

    with pytest.raises(
        ValueError,
        match="missing injury event rows.*a2.*2026",
    ):
        attach_time_to_event_labels(snapshots, incomplete_injuries)


def test_attach_time_to_event_labels_keeps_boolean_columns_as_bool_dtype():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)

    labeled = attach_time_to_event_labels(snapshots, injuries)

    assert pd.api.types.is_bool_dtype(labeled["event_observed"])
    assert pd.api.types.is_bool_dtype(labeled["event_within_7d"])
    assert pd.api.types.is_bool_dtype(labeled["event_within_14d"])
    assert pd.api.types.is_bool_dtype(labeled["event_within_30d"])


@pytest.mark.parametrize("horizons", [(0,), (-1,), (7, 7), (7, 14.0), (True,)])
def test_attach_time_to_event_labels_rejects_invalid_horizons(horizons):
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)

    with pytest.raises(ValueError, match="horizons must be unique positive integers"):
        attach_time_to_event_labels(snapshots, injuries, horizons=horizons)


def test_attach_time_to_event_labels_filters_snapshots_after_event_or_censor_date():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)
    late_snapshot = snapshots.iloc[[0]].copy()
    late_snapshot["snapshot_date"] = pd.Timestamp("2026-01-21")
    snapshots = pd.concat([snapshots, late_snapshot], ignore_index=True)

    labeled = attach_time_to_event_labels(snapshots, injuries)

    assert not (
        (labeled["athlete_id"] == "a1")
        & (labeled["snapshot_date"] == pd.Timestamp("2026-01-21"))
    ).any()
