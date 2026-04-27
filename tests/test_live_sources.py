import pandas as pd

from risk_stratification_engine.live_sources import (
    build_injury_event_rows,
    canonicalize_long_measurements,
    canonicalize_wide_measurements,
    stable_athlete_id,
)


def test_stable_athlete_id_normalizes_names_and_does_not_expose_name():
    athlete_id = stable_athlete_id("  Jane   Athlete-Smith ")

    assert athlete_id == stable_athlete_id("jane athlete smith")
    assert athlete_id.startswith("ath_")
    assert "jane" not in athlete_id
    assert "smith" not in athlete_id


def test_canonicalize_wide_measurements_builds_required_fields():
    raw = pd.DataFrame(
        [
            {
                "DATE": "2026-06-30",
                "NAME": "Jane Athlete",
                "WEIGHT": "190.5",
                "IGNORED": 1,
            },
            {
                "DATE": "2026-07-01",
                "NAME": "Jane Athlete",
                "WEIGHT": "191.0",
                "IGNORED": 2,
            },
        ]
    )

    frame = canonicalize_wide_measurements(
        raw,
        source="bodyweight",
        date_column="DATE",
        name_column="NAME",
        metric_columns=["WEIGHT"],
    )

    assert frame.to_dict("records") == [
        {
            "athlete_id": stable_athlete_id("Jane Athlete"),
            "date": pd.Timestamp("2026-06-30"),
            "season_id": "2025-2026",
            "source": "bodyweight",
            "metric_name": "bodyweight__weight",
            "metric_value": 190.5,
        },
        {
            "athlete_id": stable_athlete_id("Jane Athlete"),
            "date": pd.Timestamp("2026-07-01"),
            "season_id": "2026-2027",
            "source": "bodyweight",
            "metric_name": "bodyweight__weight",
            "metric_value": 191.0,
        },
    ]


def test_canonicalize_long_measurements_drops_bad_rows_and_slugs_metric_names():
    raw = pd.DataFrame(
        [
            {
                "athlete_name": "Jane Athlete",
                "test_date": "2026-08-01",
                "metric_name": "Concentric Impulse-100ms",
                "metric_value": "42.5",
            },
            {
                "athlete_name": "Jane Athlete",
                "test_date": "2026-08-02",
                "metric_name": "Concentric Impulse-100ms",
                "metric_value": "not numeric",
            },
        ]
    )

    frame = canonicalize_long_measurements(
        raw,
        source="forceplate",
        date_column="test_date",
        name_column="athlete_name",
        metric_name_column="metric_name",
        metric_value_column="metric_value",
    )

    assert frame.to_dict("records") == [
        {
            "athlete_id": stable_athlete_id("Jane Athlete"),
            "date": pd.Timestamp("2026-08-01"),
            "season_id": "2026-2027",
            "source": "forceplate",
            "metric_name": "forceplate__concentric_impulse_100ms",
            "metric_value": 42.5,
        }
    ]


def test_build_injury_event_rows_uses_earliest_event_and_censors_event_free_seasons():
    measurements = pd.DataFrame(
        [
            {
                "athlete_id": stable_athlete_id("Jane Athlete"),
                "date": pd.Timestamp("2026-08-01"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 100.0,
            },
            {
                "athlete_id": stable_athlete_id("Jane Athlete"),
                "date": pd.Timestamp("2026-08-10"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 120.0,
            },
            {
                "athlete_id": stable_athlete_id("No Event"),
                "date": pd.Timestamp("2026-09-15"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 90.0,
            },
        ]
    )
    injuries = pd.DataFrame(
        [
            {
                "Athlete": "Jane Athlete",
                "Issue Date": "2026-08-09",
                "Classification": "later event",
                "Pathology": "",
                "Type": "injury",
            },
            {
                "Athlete": "Jane Athlete",
                "Issue Date": "2026-08-04",
                "Classification": "earliest event",
                "Pathology": "",
                "Type": "injury",
            },
        ]
    )

    frame = build_injury_event_rows(measurements, injuries)

    assert frame.to_dict("records") == [
        {
            "athlete_id": stable_athlete_id("Jane Athlete"),
            "season_id": "2026-2027",
            "injury_date": pd.Timestamp("2026-08-04"),
            "injury_type": "earliest event",
            "event_observed": True,
            "censor_date": pd.Timestamp("2026-08-10"),
        },
        {
            "athlete_id": stable_athlete_id("No Event"),
            "season_id": "2026-2027",
            "injury_date": pd.NaT,
            "injury_type": "censored",
            "event_observed": False,
            "censor_date": pd.Timestamp("2026-09-15"),
        },
    ]
