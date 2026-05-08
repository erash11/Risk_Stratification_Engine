import pandas as pd

from risk_stratification_engine.injury_history_features import (
    INJURY_HISTORY_FEATURE_COLUMNS,
    attach_injury_history_features,
)


def test_attach_injury_history_features_uses_only_prior_injuries():
    graph_features = pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "snapshot_date": "2026-01-10",
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "snapshot_date": "2026-01-20",
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "snapshot_date": "2026-01-30",
            },
        ]
    )
    detailed_injuries = pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2025",
                "injury_date": "2025-12-01",
                "body_area": "Knee",
                "tissue_type": "Ligament/joint capsule",
                "activity_group": "Game",
                "time_loss_days": 10,
                "caused_unavailability": True,
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-01-15",
                "body_area": "Thigh",
                "tissue_type": "Muscle/tendon",
                "activity_group": "Practice",
                "time_loss_days": 20,
                "caused_unavailability": True,
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-02-01",
                "body_area": "Shoulder",
                "tissue_type": "Ligament/joint capsule",
                "activity_group": "S & C",
                "time_loss_days": 30,
                "caused_unavailability": False,
            },
        ]
    )

    enriched = attach_injury_history_features(graph_features, detailed_injuries)

    assert set(INJURY_HISTORY_FEATURE_COLUMNS).issubset(enriched.columns)
    first, second, third = enriched.sort_values("snapshot_date").to_dict("records")
    assert first["injury_history_prior_injury_count"] == 1
    assert first["injury_history_prior_same_season_injury_count"] == 0
    assert first["injury_history_days_since_last_injury"] == 40
    assert first["injury_history_prior_time_loss_days_sum"] == 10
    assert first["injury_history_prior_game_injury_count"] == 1

    assert second["injury_history_prior_injury_count"] == 2
    assert second["injury_history_prior_same_season_injury_count"] == 1
    assert second["injury_history_days_since_last_injury"] == 5
    assert second["injury_history_prior_time_loss_days_sum"] == 30
    assert second["injury_history_prior_time_loss_days_max"] == 20
    assert second["injury_history_prior_lower_extremity_injury_count"] == 2
    assert second["injury_history_prior_soft_tissue_injury_count"] == 1
    assert second["injury_history_prior_lower_extremity_soft_tissue_count"] == 1
    assert second["injury_history_prior_practice_injury_count"] == 1
    assert second["injury_history_prior_caused_unavailability_count"] == 2

    assert third["injury_history_prior_injury_count"] == 2
    assert third["injury_history_prior_s_and_c_injury_count"] == 0


def test_attach_injury_history_features_fills_missing_context_with_zeroes():
    graph_features = pd.DataFrame(
        [
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "snapshot_date": "2026-01-10",
            }
        ]
    )
    detailed_injuries = pd.DataFrame(
        columns=["athlete_id", "season_id", "injury_date"]
    )

    enriched = attach_injury_history_features(graph_features, detailed_injuries)

    assert enriched.loc[0, list(INJURY_HISTORY_FEATURE_COLUMNS)].eq(0).all()
