import pandas as pd

from risk_stratification_engine.injury_context import build_injury_context_outcomes


def test_build_injury_context_outcomes_profiles_capture_by_context():
    detailed_events = pd.DataFrame(
        [
            {
                "injury_event_id": "inj_1",
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-01-20",
                "injury_type": "soft tissue",
                "pathology": "hamstring strain",
                "classification": "muscle",
                "body_area": "thigh",
                "tissue_type": "muscle",
                "side": "left",
                "recurrent": "Yes",
                "caused_unavailability": "Yes",
                "activity_group": "training",
                "activity_group_type": "field",
                "duration_days": 12,
                "time_loss_days": 10,
                "modified_available_days": 2,
            },
            {
                "injury_event_id": "inj_2",
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-02-15",
                "injury_type": "joint",
                "pathology": "ankle sprain",
                "classification": "ligament",
                "body_area": "ankle",
                "tissue_type": "ligament",
                "side": "right",
                "recurrent": "No",
                "caused_unavailability": "No",
                "activity_group": "competition",
                "activity_group_type": "game",
                "duration_days": 3,
                "time_loss_days": 0,
                "modified_available_days": 3,
            },
        ]
    )
    episodes = pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "horizon_days": 30,
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
                "start_date": "2026-01-01",
                "peak_date": "2026-01-10",
                "end_date": "2026-01-15",
            }
        ]
    )

    outcomes = build_injury_context_outcomes(detailed_events, episodes)

    profiles = pd.DataFrame(outcomes["event_profile_rows"])
    assert profiles[
        [
            "injury_event_id",
            "captured_after_start",
            "days_from_nearest_episode_start",
            "time_loss_bucket",
        ]
    ].to_dict("records") == [
        {
            "injury_event_id": "inj_1",
            "captured_after_start": True,
            "days_from_nearest_episode_start": 19,
            "time_loss_bucket": "8-28d",
        },
        {
            "injury_event_id": "inj_2",
            "captured_after_start": False,
            "days_from_nearest_episode_start": 45,
            "time_loss_bucket": "0d",
        },
    ]

    context = pd.DataFrame(outcomes["context_rows"])
    body_area = context[context["context_field"] == "body_area"].set_index(
        "context_value"
    )
    assert body_area.loc["thigh", "event_count"] == 1
    assert body_area.loc["thigh", "captured_after_start_count"] == 1
    assert body_area.loc["thigh", "start_capture_rate"] == 1.0
    assert body_area.loc["ankle", "missed_after_start_count"] == 1
    assert body_area.loc["ankle", "median_time_loss_days"] == 0.0

    bucket = context[context["context_field"] == "time_loss_bucket"].set_index(
        "context_value"
    )
    assert bucket.loc["8-28d", "recurrent_event_count"] == 1
    assert bucket.loc["0d", "caused_unavailability_event_count"] == 0
