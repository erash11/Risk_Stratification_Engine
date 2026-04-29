import pandas as pd

from risk_stratification_engine.season_drift import (
    build_season_drift_diagnostics,
)


def test_build_season_drift_diagnostics_joins_coverage_injury_mix_and_shadow_rows():
    diagnostics = build_season_drift_diagnostics(
        measurements=_measurement_fixture(),
        canonical_injuries=_canonical_injury_fixture(),
        detailed_injuries=_detailed_injury_fixture(),
        shadow_mode_rows=_shadow_row_fixture(),
    )

    rows = pd.DataFrame(diagnostics["season_rows"]).set_index("season_id")

    assert diagnostics["experiment_type"] == "season_drift_diagnostic"
    assert diagnostics["season_count"] == 2
    assert rows.loc["2024", "athlete_count"] == 1
    assert rows.loc["2025", "measurement_row_count"] == 8
    assert rows.loc["2025", "source_counts"] == {"bodyweight": 4, "force_plate": 4}
    assert rows.loc["2025", "metric_count"] == 3
    assert rows.loc["2024", "observed_event_count"] == 1
    assert rows.loc["2025", "detailed_event_count"] == 2
    assert rows.loc["2025", "model_safe_time_loss_event_count"] == 2
    assert rows.loc["2025", "lower_extremity_soft_tissue_event_count"] == 1
    assert rows.loc["2024", "time_loss_bucket_counts"] == {"1_7_days": 1}
    assert rows.loc["2025", "broad_30d_capture_rate"] == 0.6
    assert rows.loc["2024", "broad_30d_captured_event_count"] == 0


def test_build_season_drift_diagnostics_flags_low_capture_and_coverage():
    diagnostics = build_season_drift_diagnostics(
        measurements=_measurement_fixture(),
        canonical_injuries=_canonical_injury_fixture(),
        detailed_injuries=_detailed_injury_fixture(),
        shadow_mode_rows=_shadow_row_fixture(),
    )

    rows = pd.DataFrame(diagnostics["season_rows"]).set_index("season_id")
    summary = diagnostics["summary"]

    assert rows.loc["2024", "primary_drift_flag"] == "low_capture_with_events"
    assert rows.loc["2025", "primary_drift_flag"] == "reference_high_capture"
    assert summary["latest_season"] == "2025"
    assert summary["highest_capture_season_by_channel"]["broad_30d"] == "2025"
    assert summary["low_capture_seasons"] == ["2024"]
    assert "coverage and injury mix" in summary["overall_interpretation"]


def _measurement_fixture():
    return pd.DataFrame(
        [
            ("a1", "2024-01-01", "2024", "force_plate", "jump_height", 35.0),
            ("a1", "2024-01-01", "2024", "force_plate", "asymmetry", 8.0),
            ("a1", "2025-01-01", "2025", "force_plate", "jump_height", 36.0),
            ("a1", "2025-01-01", "2025", "force_plate", "asymmetry", 7.0),
            ("a1", "2025-01-08", "2025", "force_plate", "jump_height", 37.0),
            ("a1", "2025-01-08", "2025", "force_plate", "asymmetry", 6.0),
            ("a2", "2025-01-01", "2025", "bodyweight", "body_weight", 220.0),
            ("a2", "2025-01-08", "2025", "bodyweight", "body_weight", 221.0),
            ("a3", "2025-01-01", "2025", "bodyweight", "body_weight", 205.0),
            ("a3", "2025-01-08", "2025", "bodyweight", "body_weight", 206.0),
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


def _canonical_injury_fixture():
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2024",
                "injury_date": "2024-01-15",
                "injury_type": "quad strain",
                "event_observed": True,
                "primary_model_event": True,
                "censor_date": "2024-01-15",
            },
            {
                "athlete_id": "a1",
                "season_id": "2025",
                "injury_date": "2025-01-15",
                "injury_type": "hamstring strain",
                "event_observed": True,
                "primary_model_event": True,
                "censor_date": "2025-01-15",
            },
            {
                "athlete_id": "a2",
                "season_id": "2025",
                "injury_date": "",
                "injury_type": "censored",
                "event_observed": False,
                "primary_model_event": False,
                "censor_date": "2025-02-01",
            },
        ]
    )


def _detailed_injury_fixture():
    return pd.DataFrame(
        [
            {
                "injury_event_id": "inj_1",
                "athlete_id": "a1",
                "season_id": "2024",
                "injury_date": "2024-01-15",
                "injury_type": "quad strain",
                "classification": "Soft tissue",
                "pathology": "strain",
                "body_area": "Thigh",
                "time_loss_days": 5,
            },
            {
                "injury_event_id": "inj_2",
                "athlete_id": "a1",
                "season_id": "2025",
                "injury_date": "2025-01-15",
                "injury_type": "hamstring strain",
                "classification": "Soft tissue",
                "pathology": "strain",
                "body_area": "Thigh",
                "time_loss_days": 12,
            },
            {
                "injury_event_id": "inj_3",
                "athlete_id": "a3",
                "season_id": "2025",
                "injury_date": "2025-01-20",
                "injury_type": "ankle sprain",
                "classification": "Ligament",
                "pathology": "sprain",
                "body_area": "Ankle",
                "time_loss_days": 30,
            },
        ]
    )


def _shadow_row_fixture():
    return pd.DataFrame(
        [
            _shadow_row("broad_30d", "2024", capture=0.0, captured=0, burden=0.2),
            _shadow_row("broad_30d", "2025", capture=0.6, captured=3, burden=0.9),
            _shadow_row("severity_7d", "2024", capture=0.0, captured=0, burden=0.3),
            _shadow_row("severity_7d", "2025", capture=0.2, captured=1, burden=1.1),
        ]
    )


def _shadow_row(channel, season, *, capture, captured, burden):
    return {
        "channel_name": channel,
        "slice_type": "season",
        "slice_id": season,
        "unique_observed_event_count": 5,
        "unique_captured_event_count": captured,
        "unique_event_capture_rate": capture,
        "episodes_per_athlete_season": burden,
        "episode_count": 10,
    }
