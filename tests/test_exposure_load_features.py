import json
from pathlib import Path

import pandas as pd

from risk_stratification_engine.exposure_load_features import (
    EXPOSURE_LOAD_FEATURE_COLUMNS,
    attach_exposure_load_features,
)


def _sample_graph_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "snapshot_date": "2026-01-10",
                "time_index": 0,
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "snapshot_date": "2026-01-20",
                "time_index": 1,
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "snapshot_date": "2026-01-20",
                "time_index": 1,
            },
        ]
    )


def _sample_participations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "training:old",
                "event_type": "training",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2026-01-01",
                "season_id": "2026",
                "exposure_category": "practice_shells",
                "participation_category": "full",
            },
            {
                "event_id": "training:lift",
                "event_type": "training",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2026-01-09",
                "season_id": "2026",
                "exposure_category": "weight_room",
                "participation_category": "modified",
            },
            {
                "event_id": "training:same_day",
                "event_type": "training",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2026-01-10",
                "season_id": "2026",
                "exposure_category": "conditioning",
                "participation_category": "no_participation",
            },
            {
                "event_id": "game:1",
                "event_type": "game",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2026-01-15",
                "season_id": "2026",
                "exposure_category": "game",
                "participation_category": "full",
            },
            {
                "event_id": "training:rtp",
                "event_type": "training",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2026-01-18",
                "season_id": "2026",
                "exposure_category": "rtp",
                "participation_category": "no_participation",
            },
            {
                "event_id": "training:unmatched",
                "event_type": "training",
                "athlete_id": "",
                "athlete_match_status": "unmatched_athlete_id",
                "date": "2026-01-18",
                "season_id": "2026",
                "exposure_category": "practice",
                "participation_category": "full",
            },
        ]
    )


def test_attach_exposure_load_features_uses_only_prior_participations():
    enriched = attach_exposure_load_features(
        _sample_graph_features(),
        _sample_participations(),
    ).sort_values(["athlete_id", "snapshot_date"])

    first = enriched.iloc[0]
    second = enriched.iloc[1]

    assert set(EXPOSURE_LOAD_FEATURE_COLUMNS).issubset(enriched.columns)
    assert first["exposure_training_sessions_7d"] == 1
    assert first["exposure_training_sessions_14d"] == 2
    assert first["exposure_modified_participations_28d"] == 1
    assert first["exposure_no_participations_28d"] == 0
    assert first["exposure_days_since_last_modified_or_no_participation"] == 1
    assert first["exposure_games_prior_count"] == 0
    assert first["exposure_conditioning_sessions_28d"] == 0

    assert second["exposure_training_sessions_7d"] == 1
    assert second["exposure_training_sessions_14d"] == 3
    assert second["exposure_games_prior_count"] == 1
    assert second["exposure_days_since_last_game"] == 5
    assert second["exposure_no_participations_28d"] == 2
    assert second["exposure_lift_sessions_28d"] == 1
    assert second["exposure_conditioning_sessions_28d"] == 1
    assert second["exposure_rtp_sessions_28d"] == 1
    assert second["exposure_game_events_28d"] == 1


def test_attach_exposure_load_features_fills_missing_context_with_zeroes():
    enriched = attach_exposure_load_features(
        _sample_graph_features().iloc[[2]],
        _sample_participations(),
    )

    assert enriched.iloc[0].loc[list(EXPOSURE_LOAD_FEATURE_COLUMNS)].eq(0).all()


def _write_exposure_participations(path: Path) -> None:
    _sample_participations().to_csv(path, index=False)


def test_run_exposure_load_feature_sprint_writes_artifacts(tmp_path):
    from risk_stratification_engine.experiments import (
        run_exposure_load_feature_sprint_experiment,
    )

    exposure_participations = tmp_path / "exposure_participations.csv"
    _write_exposure_participations(exposure_participations)

    result = run_exposure_load_feature_sprint_experiment(
        measurements_path=Path("tests/fixtures/measurements.csv"),
        injuries_path=Path("tests/fixtures/injuries.csv"),
        exposure_participations_path=exposure_participations,
        output_dir=tmp_path,
        experiment_id="exposure_load_feature_v1",
        graph_window_size=2,
        model_variant="l2",
    )

    assert (result / "exposure_load_features.csv").exists()
    assert (result / "exposure_load_model_comparison.csv").exists()
    summary = json.loads(
        (result / "exposure_load_model_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["experiment_type"] == "exposure_load_feature_sprint"
    assert "graph_plus_coverage_exposure_load" in summary["feature_sets"]
