import pandas as pd

from risk_stratification_engine.coverage_source_features import (
    COVERAGE_SOURCE_FEATURE_COLUMNS,
    attach_coverage_source_features,
)
from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.trajectories import build_measurement_matrix


def test_attach_coverage_source_features_uses_only_measurements_up_to_snapshot():
    measurements = pd.DataFrame(
        [
            ("a1", "2026-01-01", "2026", "force_plate", "jump_height", 42.0),
            ("a1", "2026-01-01", "2026", "force_plate", "asymmetry", 8.0),
            ("a1", "2026-01-08", "2026", "gps", "distance", 5100.0),
            ("a1", "2026-01-08", "2026", "gps", "player_load", 18.0),
        ],
        columns=[
            "athlete_id",
            "date",
            "season_id",
            "source",
            "metric_name",
            "metric_value",
        ],
    )
    graph_features = build_graph_snapshots(
        build_measurement_matrix(measurements),
        window_size=2,
    )

    enriched = attach_coverage_source_features(graph_features, measurements)

    first = enriched.sort_values("snapshot_date").iloc[0]
    second = enriched.sort_values("snapshot_date").iloc[1]
    assert set(COVERAGE_SOURCE_FEATURE_COLUMNS).issubset(enriched.columns)
    assert first["coverage_measurement_days_to_date"] == 1
    assert first["coverage_measurement_rows_to_date"] == 2
    assert first["coverage_source_count_to_date"] == 1
    assert first["coverage_seen_forceplate_to_date"] == 1
    assert first["coverage_seen_gps_to_date"] == 0
    assert second["coverage_measurement_days_to_date"] == 2
    assert second["coverage_measurement_rows_to_date"] == 4
    assert second["coverage_source_count_to_date"] == 2
    assert second["coverage_days_since_previous_measurement"] == 7
    assert second["coverage_seen_gps_to_date"] == 1


def test_attach_coverage_source_features_fills_missing_coverage_context_with_zeroes():
    graph_features = pd.DataFrame(
        [
            {
                "athlete_id": "missing",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-01-01",
            }
        ]
    )
    measurements = pd.DataFrame(
        columns=[
            "athlete_id",
            "date",
            "season_id",
            "source",
            "metric_name",
            "metric_value",
        ]
    )

    enriched = attach_coverage_source_features(graph_features, measurements)

    assert enriched.loc[0, list(COVERAGE_SOURCE_FEATURE_COLUMNS)].eq(0).all()
