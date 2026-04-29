import pandas as pd

from risk_stratification_engine.config import DataSourcePaths
from risk_stratification_engine.live_sources import (
    aggregate_same_day_measurements,
    build_detailed_injury_event_rows,
    build_injury_event_rows,
    canonicalize_long_measurements,
    canonicalize_wide_measurements,
    prepare_live_source_inputs,
    stable_athlete_id,
)


def test_stable_athlete_id_normalizes_names_and_does_not_expose_name():
    athlete_id = stable_athlete_id("  Jane   Athlete-Smith ")

    assert athlete_id == stable_athlete_id("jane athlete smith")
    assert athlete_id.startswith("ath_")
    assert "jane" not in athlete_id
    assert "smith" not in athlete_id


def test_stable_athlete_id_reconciles_last_comma_first_names():
    assert stable_athlete_id("Athlete, Jane") == stable_athlete_id("Jane Athlete")


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


def test_aggregate_same_day_measurements_averages_duplicate_metric_rows():
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
                "date": pd.Timestamp("2026-08-01"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 120.0,
            },
            {
                "athlete_id": stable_athlete_id("Jane Athlete"),
                "date": pd.Timestamp("2026-08-02"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 130.0,
            },
        ]
    )

    aggregated, summary = aggregate_same_day_measurements(measurements)

    assert aggregated.to_dict("records") == [
        {
            "athlete_id": stable_athlete_id("Jane Athlete"),
            "date": pd.Timestamp("2026-08-01"),
            "season_id": "2026-2027",
            "source": "gps",
            "metric_name": "gps__total_distance_m",
            "metric_value": 110.0,
        },
        {
            "athlete_id": stable_athlete_id("Jane Athlete"),
            "date": pd.Timestamp("2026-08-02"),
            "season_id": "2026-2027",
            "source": "gps",
            "metric_name": "gps__total_distance_m",
            "metric_value": 130.0,
        },
    ]
    assert summary == {
        "policy": "mean metric_value per athlete_id, season_id, date, source, metric_name",
        "input_rows": 3,
        "output_rows": 2,
        "duplicate_same_day_metric_groups": 1,
        "aggregated_rows_removed": 1,
    }


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
            "nearest_measurement_date": pd.Timestamp("2026-08-01"),
            "nearest_measurement_gap_days": 3,
            "event_window_quality": "modelable",
            "primary_model_event": True,
        },
        {
            "athlete_id": stable_athlete_id("No Event"),
            "season_id": "2026-2027",
            "injury_date": pd.NaT,
            "injury_type": "censored",
            "event_observed": False,
            "censor_date": pd.Timestamp("2026-09-15"),
            "nearest_measurement_date": pd.NaT,
            "nearest_measurement_gap_days": None,
            "event_window_quality": "censored",
            "primary_model_event": False,
        },
    ]


def test_build_injury_event_rows_assigns_event_window_quality_labels():
    measurements = pd.DataFrame(
        [
            {
                "athlete_id": stable_athlete_id("Modelable Athlete"),
                "date": pd.Timestamp("2026-08-01"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 100.0,
            },
            {
                "athlete_id": stable_athlete_id("Low Confidence Athlete"),
                "date": pd.Timestamp("2026-08-01"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 100.0,
            },
            {
                "athlete_id": stable_athlete_id("Out Window Athlete"),
                "date": pd.Timestamp("2026-08-01"),
                "season_id": "2026-2027",
                "source": "gps",
                "metric_name": "gps__total_distance_m",
                "metric_value": 100.0,
            },
        ]
    )
    injuries = pd.DataFrame(
        [
            {
                "Athlete": "Modelable Athlete",
                "Issue Date": "2026-08-10",
                "Classification": "injury",
            },
            {
                "Athlete": "Low Confidence Athlete",
                "Issue Date": "2026-08-25",
                "Classification": "injury",
            },
            {
                "Athlete": "Out Window Athlete",
                "Issue Date": "2026-09-15",
                "Classification": "injury",
            },
        ]
    )

    frame = build_injury_event_rows(measurements, injuries)

    assert frame.set_index("athlete_id")[
        ["nearest_measurement_gap_days", "event_window_quality", "primary_model_event"]
    ].to_dict("index") == {
        stable_athlete_id("Modelable Athlete"): {
            "nearest_measurement_gap_days": 9,
            "event_window_quality": "modelable",
            "primary_model_event": True,
        },
        stable_athlete_id("Low Confidence Athlete"): {
            "nearest_measurement_gap_days": 24,
            "event_window_quality": "low_confidence",
            "primary_model_event": False,
        },
        stable_athlete_id("Out Window Athlete"): {
            "nearest_measurement_gap_days": 45,
            "event_window_quality": "out_of_window",
            "primary_model_event": False,
        },
    }


def test_build_detailed_injury_event_rows_preserves_rich_context_without_names():
    injuries = pd.DataFrame(
        [
            {
                "Athlete": "Jane Athlete",
                "DOB": "2001-01-01",
                "Issue Date": "2026-08-04",
                "Issue Entry Date": "2026-08-05",
                "Issue Resolved Date": "2026-08-20",
                "Duration": "16.0",
                "Classification": "soft tissue",
                "Pathology": "hamstring strain",
                "Type": "injury",
                "Body Area": "thigh",
                "Tissue Type": "muscle",
                "Side": "left",
                "Recurrent": "Yes",
                "Caused Unavailability": "Yes",
                "Activity": "sprint",
                "Activity Group": "training",
                "Activity Group Type": "field",
                "Participation Level": "modified",
                "Session Completed": "No",
                "Competition": "",
                "Game Date/Time": "",
                "Training Session Type": "practice",
                "Surface Type": "grass",
                "Status duration: Unavailable - time-loss": "6",
                "Status duration: Available - modified": "10",
                "Status duration: Available - not modified": "0",
                "Code": "HAM-1",
                "ICD": "S76",
                "_source_file": "injuries-summary-export-test.csv",
                "_source_row_number": 1,
            },
            {
                "Athlete": "Jane Athlete",
                "Issue Date": "2026-09-10",
                "Issue Entry Date": "2026-09-10",
                "Classification": "",
                "Pathology": "ankle sprain",
                "Type": "injury",
                "Code": "ANK-1",
                "_source_file": "injuries-summary-export-test.csv",
                "_source_row_number": 2,
            },
        ]
    )

    frame = build_detailed_injury_event_rows(injuries)

    assert list(frame.columns) == [
        "injury_event_id",
        "athlete_id",
        "season_id",
        "injury_date",
        "issue_entry_date",
        "issue_resolved_date",
        "injury_type",
        "pathology",
        "classification",
        "body_area",
        "tissue_type",
        "side",
        "recurrent",
        "caused_unavailability",
        "activity",
        "activity_group",
        "activity_group_type",
        "participation_level",
        "session_completed",
        "competition",
        "game_date",
        "training_session_type",
        "surface_type",
        "duration_days",
        "time_loss_days",
        "modified_available_days",
        "not_modified_available_days",
        "raw_code",
        "icd_code",
        "source_file",
        "source_row_number",
    ]
    assert len(frame) == 2
    assert "Athlete" not in frame.columns
    assert "DOB" not in frame.columns

    first = frame.iloc[0].to_dict()
    assert first["athlete_id"] == stable_athlete_id("Jane Athlete")
    assert first["season_id"] == "2026-2027"
    assert first["injury_date"] == pd.Timestamp("2026-08-04")
    assert first["injury_type"] == "soft tissue"
    assert first["pathology"] == "hamstring strain"
    assert first["body_area"] == "thigh"
    assert first["recurrent"] == "Yes"
    assert first["duration_days"] == 16.0
    assert first["time_loss_days"] == 6.0
    assert first["modified_available_days"] == 10.0
    assert first["source_file"] == "injuries-summary-export-test.csv"
    assert first["source_row_number"] == 1


def test_prepare_live_source_inputs_writes_data_quality_audit(
    tmp_path,
    monkeypatch,
):
    paths = DataSourcePaths(
        forceplate_db=tmp_path / "forceplate.duckdb",
        gps_db=tmp_path / "gps.duckdb",
        bodyweight_csv=tmp_path / "bodyweight.csv",
        perch_db=tmp_path / "perch.duckdb",
        injury_csv=tmp_path / "injury.csv",
    )
    pd.DataFrame(
        [
            {
                "DATE": "2026-01-01",
                "NAME": "Athlete, Shared",
                "WEIGHT": 190.0,
            },
            {
                "DATE": "2026-01-01",
                "NAME": "Athlete, Shared",
                "WEIGHT": 192.0,
            }
        ]
    ).to_csv(paths.bodyweight_csv, index=False)
    pd.DataFrame(
        [
            {
                "Athlete": "Shared Athlete",
                "Issue Date": "2026-01-20",
                "Classification": "soft tissue",
                "Pathology": "",
                "Type": "injury",
            }
        ]
    ).to_csv(paths.injury_csv, index=False)

    def fake_read_duckdb(_duckdb, path, _query):
        if path == paths.gps_db:
            return pd.DataFrame(
                [
                    {
                        "name": "Shared Athlete",
                        "session_date": "2026-01-01",
                        "total_player_load": 10.0,
                        "total_distance_m": 100.0,
                    }
                ]
            )
        if path == paths.forceplate_db:
            return pd.DataFrame(
                [
                    {
                        "athlete_name": "Shared Athlete",
                        "test_date": "2026-01-10",
                        "metric_name": "Peak Power",
                        "metric_value": 200.0,
                    }
                ]
            )
        if path == paths.perch_db:
            return pd.DataFrame(
                [
                    {
                        "name_normalized": "Shared Athlete",
                        "test_date": "2026-01-12",
                        "exercise": "Bench Press",
                        "one_rm_lbs": 300.0,
                    }
                ]
            )
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(
        "risk_stratification_engine.live_sources._require_duckdb",
        lambda: object(),
    )
    monkeypatch.setattr(
        "risk_stratification_engine.live_sources._read_duckdb",
        fake_read_duckdb,
    )

    result = prepare_live_source_inputs(paths, tmp_path / "prepared")
    measurements = pd.read_csv(result.measurements_path)

    assert result.audit_path.exists()
    audit_text = result.audit_path.read_text(encoding="utf-8")
    assert "events_without_nearby_measurements_count" in audit_text
    assert result.metadata["aggregation"]["duplicate_same_day_metric_groups"] == 1
    assert result.metadata["event_window_quality_counts"] == {
        "censored": 0,
        "low_confidence": 0,
        "modelable": 1,
        "no_measurements": 0,
        "out_of_window": 0,
    }
    assert result.audit["duplicates"]["duplicate_same_day_metric_count"] == 0
    assert measurements.loc[
        measurements["metric_name"] == "bodyweight__weight",
        "metric_value",
    ].item() == 191.0
    assert result.audit["identity"]["source_athlete_counts"] == {
        "bodyweight": 1,
        "forceplate": 1,
        "gps": 1,
        "injury": 1,
        "perch": 1,
    }


def test_prepare_live_source_inputs_writes_detailed_events_from_sibling_exports(
    tmp_path,
    monkeypatch,
):
    paths = DataSourcePaths(
        forceplate_db=tmp_path / "forceplate.duckdb",
        gps_db=tmp_path / "gps.duckdb",
        bodyweight_csv=tmp_path / "bodyweight.csv",
        perch_db=tmp_path / "perch.duckdb",
        injury_csv=tmp_path / "injuries-summary-export-a.csv",
    )
    pd.DataFrame(
        [
            {
                "DATE": "2026-01-01",
                "NAME": "Shared Athlete",
                "WEIGHT": 190.0,
            }
        ]
    ).to_csv(paths.bodyweight_csv, index=False)
    pd.DataFrame(
        [
            {
                "Athlete": "Shared Athlete",
                "Issue Date": "2026-01-20",
                "Classification": "soft tissue",
                "Pathology": "hamstring strain",
                "Body Area": "thigh",
                "Type": "injury",
                "Duration": "12",
                "Status duration: Unavailable - time-loss": "4",
            }
        ]
    ).to_csv(paths.injury_csv, index=False)
    pd.DataFrame(
        [
            {
                "Athlete": "Shared Athlete",
                "Issue Date": "2026-03-01",
                "Classification": "joint",
                "Pathology": "ankle sprain",
                "Body Area": "ankle",
                "Type": "injury",
                "Duration": "8",
                "Status duration: Unavailable - time-loss": "2",
            }
        ]
    ).to_csv(tmp_path / "injuries-summary-export-b.csv", index=False)

    def fake_read_duckdb(_duckdb, path, _query):
        if path == paths.gps_db:
            return pd.DataFrame(
                [
                    {
                        "name": "Shared Athlete",
                        "session_date": "2026-01-01",
                        "total_player_load": 10.0,
                        "total_distance_m": 100.0,
                    }
                ]
            )
        if path == paths.forceplate_db:
            return pd.DataFrame(
                [
                    {
                        "athlete_name": "Shared Athlete",
                        "test_date": "2026-01-10",
                        "metric_name": "Peak Power",
                        "metric_value": 200.0,
                    }
                ]
            )
        if path == paths.perch_db:
            return pd.DataFrame(
                [
                    {
                        "name_normalized": "Shared Athlete",
                        "test_date": "2026-01-12",
                        "exercise": "Bench Press",
                        "one_rm_lbs": 300.0,
                    }
                ]
            )
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(
        "risk_stratification_engine.live_sources._require_duckdb",
        lambda: object(),
    )
    monkeypatch.setattr(
        "risk_stratification_engine.live_sources._read_duckdb",
        fake_read_duckdb,
    )

    result = prepare_live_source_inputs(paths, tmp_path / "prepared")
    detailed = pd.read_csv(result.detailed_injuries_path)
    canonical = pd.read_csv(result.injuries_path)

    assert result.detailed_injuries_path.name == "injury_events_detailed.csv"
    assert detailed[["source_file", "injury_type", "body_area"]].to_dict(
        "records"
    ) == [
        {
            "source_file": "injuries-summary-export-a.csv",
            "injury_type": "soft tissue",
            "body_area": "thigh",
        },
        {
            "source_file": "injuries-summary-export-b.csv",
            "injury_type": "joint",
            "body_area": "ankle",
        },
    ]
    assert len(canonical) == 1
    assert canonical.loc[0, "injury_type"] == "soft tissue"
    assert result.metadata["sources"]["injury"]["row_count"] == 2
    assert result.metadata["sources"]["injury"]["files"] == [
        "injuries-summary-export-a.csv",
        "injuries-summary-export-b.csv",
    ]
    assert result.metadata["canonical_rows"]["detailed_injury_events"] == 2
