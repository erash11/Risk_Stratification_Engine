import pandas as pd

from risk_stratification_engine.data_quality import build_data_quality_audit
from risk_stratification_engine.live_sources import stable_athlete_id


def test_build_data_quality_audit_reports_identity_and_coverage_findings():
    shared_id = stable_athlete_id("Shared Athlete")
    sparse_id = stable_athlete_id("Sparse Athlete")
    injury_only_id = stable_athlete_id("Injury Only")
    measurements = pd.DataFrame(
        [
            {
                "athlete_id": shared_id,
                "date": "2026-01-01",
                "season_id": "2025-2026",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 100.0,
            },
            {
                "athlete_id": shared_id,
                "date": "2026-01-01",
                "season_id": "2025-2026",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 101.0,
            },
            {
                "athlete_id": shared_id,
                "date": "2026-01-10",
                "season_id": "2025-2026",
                "source": "forceplate",
                "metric_name": "forceplate__peak_power",
                "metric_value": 200.0,
            },
            {
                "athlete_id": sparse_id,
                "date": "2026-02-01",
                "season_id": "2025-2026",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 90.0,
            },
        ]
    )
    injury_events = pd.DataFrame(
        [
            {
                "athlete_id": shared_id,
                "season_id": "2025-2026",
                "injury_date": "2026-01-20",
                "injury_type": "soft tissue",
                "event_observed": True,
                "censor_date": "2026-01-20",
            },
            {
                "athlete_id": sparse_id,
                "season_id": "2025-2026",
                "injury_date": "",
                "injury_type": "censored",
                "event_observed": False,
                "censor_date": "2026-02-01",
            },
        ]
    )
    source_identities = {
        "gps": pd.DataFrame({"athlete_id": [shared_id, sparse_id]}),
        "forceplate": pd.DataFrame({"athlete_id": [shared_id]}),
        "injury": pd.DataFrame({"athlete_id": [shared_id, injury_only_id]}),
    }

    audit = build_data_quality_audit(
        measurements,
        injury_events,
        source_identities,
        sparse_measurement_dates_threshold=2,
        large_gap_days=7,
        injury_nearby_days=3,
    )

    assert audit["identity"]["source_athlete_counts"] == {
        "forceplate": 1,
        "gps": 2,
        "injury": 2,
    }
    assert audit["identity"]["pairwise_overlap_counts"] == {
        "forceplate|gps": 1,
        "forceplate|injury": 1,
        "gps|injury": 1,
    }
    assert audit["identity"]["single_source_athlete_counts"] == {
        "gps": 1,
        "injury": 1,
    }
    assert audit["identity"]["single_source_athlete_review"]["total_count"] == 2
    assert audit["identity"]["single_source_athlete_review"]["by_source"] == {
        "gps": 1,
        "injury": 1,
    }
    assert audit["identity"]["single_source_athlete_review"]["examples"] == [
        {
            "athlete_id": sparse_id,
            "source": "gps",
            "measurement_rows": 1,
            "measurement_dates": 1,
            "athlete_seasons": 1,
            "first_measurement_date": "2026-02-01",
            "last_measurement_date": "2026-02-01",
            "modeled_observed_events": 0,
        },
        {
            "athlete_id": injury_only_id,
            "source": "injury",
            "measurement_rows": 0,
            "measurement_dates": 0,
            "athlete_seasons": 0,
            "first_measurement_date": None,
            "last_measurement_date": None,
            "modeled_observed_events": 0,
        },
    ]
    assert audit["coverage"]["sparse_athlete_season_count"] == 1
    assert audit["coverage"]["sparse_athlete_seasons"][0]["athlete_id"] == sparse_id
    assert audit["date_gaps"]["large_gap_count"] == 1
    assert audit["date_gaps"]["large_gaps"][0]["gap_days"] == 9
    assert audit["duplicates"]["duplicate_same_day_metric_count"] == 1
    assert audit["duplicates"]["duplicate_same_day_metrics"][0]["row_count"] == 2
    assert audit["injuries"]["events_without_nearby_measurements_count"] == 1
    assert audit["injuries"]["events_without_nearby_measurements_by_gap_bucket"] == {
        "4-7d": 0,
        "8-14d": 1,
        "15-30d": 0,
        "31-90d": 0,
        "91d+": 0,
        "no_measurements": 0,
    }
    assert audit["injuries"]["events_without_nearby_measurements"][0][
        "season_measurement_dates"
    ] == 2
    assert audit["injuries"]["events_without_nearby_measurements"][0][
        "season_source_count"
    ] == 2
    assert (
        audit["injuries"]["events_without_nearby_measurements"][0]["athlete_id"]
        == shared_id
    )
