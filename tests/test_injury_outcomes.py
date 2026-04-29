import pandas as pd

from risk_stratification_engine.injury_outcomes import (
    build_injury_severity_audit,
    build_policy_injury_events,
    build_outcome_policy_summary,
)


def test_build_injury_severity_audit_flags_time_loss_semantics():
    detailed = pd.DataFrame(
        [
            {
                "injury_event_id": "inj_1",
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-01-01",
                "issue_resolved_date": "2026-01-11",
                "injury_type": "soft tissue",
                "classification": "strain",
                "body_area": "Thigh",
                "activity_group": "Practice",
                "duration_days": 10,
                "time_loss_days": 8,
                "modified_available_days": 2,
                "caused_unavailability": "Yes",
                "recurrent": "No",
            },
            {
                "injury_event_id": "inj_2",
                "athlete_id": "a2",
                "season_id": "2026",
                "injury_date": "2026-02-01",
                "issue_resolved_date": "2026-02-10",
                "injury_type": "fracture",
                "classification": "bone",
                "body_area": "Ankle",
                "activity_group": "Game",
                "duration_days": 9,
                "time_loss_days": 900,
                "modified_available_days": 0,
                "caused_unavailability": "Yes",
                "recurrent": "Yes",
            },
        ]
    )

    audit = build_injury_severity_audit(detailed)

    assert audit["event_count"] == 2
    assert audit["extreme_time_loss_count"] == 1
    assert audit["duration_resolution_mismatch_count"] == 0
    assert audit["time_loss_bucket_counts"] == {"8-28d": 1, "extreme_366d+": 1}

    rows = pd.DataFrame(audit["event_rows"]).set_index("injury_event_id")
    assert rows.loc["inj_1", "resolved_duration_days"] == 10
    assert rows.loc["inj_1", "time_loss_bucket"] == "8-28d"
    assert rows.loc["inj_1", "severity_semantics_flag"] == "usable"
    assert rows.loc["inj_2", "time_loss_bucket"] == "extreme_366d+"
    assert rows.loc["inj_2", "severity_semantics_flag"] == "review_extreme_time_loss"


def test_build_outcome_policy_summary_counts_context_policies():
    detailed = pd.DataFrame(
        [
            {
                "injury_event_id": "inj_1",
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-01-01",
                "injury_type": "Hamstring strain",
                "classification": "Soft tissue",
                "pathology": "Hamstring strain/tear",
                "body_area": "Thigh",
                "time_loss_days": 8,
                "caused_unavailability": "Yes",
                "recurrent": "No",
            },
            {
                "injury_event_id": "inj_2",
                "athlete_id": "a2",
                "season_id": "2026",
                "injury_date": "2026-02-01",
                "injury_type": "Concussion",
                "classification": "Concussion",
                "pathology": "Concussion",
                "body_area": "Head",
                "time_loss_days": 11,
                "caused_unavailability": "Yes",
                "recurrent": "No",
            },
            {
                "injury_event_id": "inj_3",
                "athlete_id": "a3",
                "season_id": "2026",
                "injury_date": "2026-03-01",
                "injury_type": "General soreness",
                "classification": "Other",
                "pathology": "Soreness",
                "body_area": "Shoulder",
                "time_loss_days": 0,
                "caused_unavailability": "No",
                "recurrent": "Yes",
            },
        ]
    )

    summary = build_outcome_policy_summary(detailed)
    rows = pd.DataFrame(summary["policy_rows"]).set_index("policy_name")

    assert summary["policy_count"] >= 8
    assert rows.loc["any_injury", "event_count"] == 3
    assert rows.loc["time_loss_only", "event_count"] == 2
    assert rows.loc["moderate_plus_time_loss", "event_count"] == 2
    assert rows.loc["severe_time_loss", "event_count"] == 0
    assert rows.loc["lower_extremity_only", "event_count"] == 1
    assert rows.loc["soft_tissue_only", "event_count"] == 1
    assert rows.loc["lower_extremity_soft_tissue", "event_count"] == 1
    assert rows.loc["concussion_only", "event_count"] == 1
    assert rows.loc["exclude_concussion", "event_count"] == 2
    assert rows.loc["recurrent_only", "event_count"] == 1


def test_build_policy_injury_events_relabels_canonical_seasons_by_policy():
    canonical = pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-02-01",
                "injury_type": "any injury",
                "event_observed": True,
                "censor_date": "2026-04-01",
                "nearest_measurement_date": "2026-01-30",
                "nearest_measurement_gap_days": 2,
                "event_window_quality": "modelable",
                "primary_model_event": True,
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "injury_date": "",
                "injury_type": "censored",
                "event_observed": False,
                "censor_date": "2026-04-01",
                "nearest_measurement_date": "",
                "nearest_measurement_gap_days": "",
                "event_window_quality": "censored",
                "primary_model_event": False,
            },
        ]
    )
    detailed = pd.DataFrame(
        [
            {
                "injury_event_id": "inj_1",
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-01-15",
                "injury_type": "Soreness",
                "classification": "Other",
                "pathology": "Soreness",
                "body_area": "Shoulder",
                "time_loss_days": 0,
                "caused_unavailability": "No",
                "recurrent": "No",
            },
            {
                "injury_event_id": "inj_2",
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-02-10",
                "injury_type": "Hamstring strain",
                "classification": "Soft tissue",
                "pathology": "Hamstring strain/tear",
                "body_area": "Thigh",
                "time_loss_days": 12,
                "caused_unavailability": "Yes",
                "recurrent": "No",
            },
        ]
    )

    relabeled = build_policy_injury_events(
        canonical,
        detailed,
        policy_name="time_loss_only",
    )

    observed, censored = relabeled.to_dict("records")
    assert observed == {
        "athlete_id": "a1",
        "season_id": "2026",
        "injury_date": pd.Timestamp("2026-02-10"),
        "injury_type": "Hamstring strain",
        "event_observed": True,
        "censor_date": pd.Timestamp("2026-04-01"),
        "nearest_measurement_date": pd.Timestamp("2026-01-30"),
        "nearest_measurement_gap_days": 2.0,
        "event_window_quality": "modelable",
        "primary_model_event": True,
    }
    assert censored["athlete_id"] == "a2"
    assert censored["season_id"] == "2026"
    assert pd.isna(censored["injury_date"])
    assert censored["injury_type"] == "censored"
    assert censored["event_observed"] is False
    assert censored["censor_date"] == pd.Timestamp("2026-04-01")
    assert pd.isna(censored["nearest_measurement_date"])
    assert pd.isna(censored["nearest_measurement_gap_days"])
    assert censored["event_window_quality"] == "censored"
    assert censored["primary_model_event"] is False
