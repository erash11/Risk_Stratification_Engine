import json

import pandas as pd

from risk_stratification_engine.exposure_load_shadow_event_crosswalk import (
    build_shadow_event_crosswalk_summary,
    clean_shadow_event_crosswalk_rows,
    write_exposure_load_shadow_event_crosswalk_report,
)


def test_shadow_event_crosswalk_identifies_captured_and_missed_events(tmp_path):
    packet = {
        "review_packet_id": "broad_30d__2024-2025",
        "channel_name": "broad_30d",
        "test_season_id": "2024-2025",
        "policy_name": "exclude_concussion",
        "horizon_days": 30,
        "threshold_policy": "burden_capped_percentile",
        "selected_threshold_value": 0.05,
    }
    timeline = pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2024-2025",
                "event_observed": True,
                "event_date": "2025-01-20",
                "injury_type": "Hamstring strain",
            },
            {
                "athlete_id": "a2",
                "season_id": "2024-2025",
                "event_observed": True,
                "event_date": "2025-01-24",
                "injury_type": "Ankle sprain",
            },
            {
                "athlete_id": "a3",
                "season_id": "2024-2025",
                "event_observed": False,
                "event_date": "2025-02-01",
                "injury_type": "censored",
            },
        ]
    )
    episodes = pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2024-2025",
                "injury_type": "Hamstring strain",
                "event_within_horizon_after_start": True,
                "start_date": "2025-01-10",
                "peak_date": "2025-01-12",
                "end_date": "2025-01-14",
                "days_from_start_to_event": 10,
                "days_from_peak_to_event": 8,
                "days_from_end_to_event": 6,
                "peak_risk": 0.72,
                "mean_risk": 0.63,
            },
            {
                "athlete_id": "a3",
                "season_id": "2024-2025",
                "injury_type": "censored",
                "event_within_horizon_after_start": False,
                "start_date": "2025-01-15",
                "peak_date": "2025-01-16",
                "end_date": "2025-01-18",
                "days_from_start_to_event": None,
                "days_from_peak_to_event": None,
                "days_from_end_to_event": None,
                "peak_risk": 0.60,
                "mean_risk": 0.58,
            },
        ]
    )
    detailed = pd.DataFrame(
        [
            {
                "injury_event_id": "injury_a1",
                "athlete_id": "a1",
                "season_id": "2024-2025",
                "injury_date": "2025-01-20",
                "injury_type": "Hamstring strain",
                "classification": "Soft tissue",
                "pathology": "Strain",
                "body_area": "Thigh",
                "tissue_type": "Muscle",
                "time_loss_days": 12,
                "duration_days": 18,
                "caused_unavailability": "Yes",
                "recurrent": "No",
                "activity": "Practice",
                "activity_group": "Training",
                "source_file": "injuries.csv",
                "source_row_number": 10,
            },
            {
                "injury_event_id": "injury_a2",
                "athlete_id": "a2",
                "season_id": "2024-2025",
                "injury_date": "2025-01-24",
                "injury_type": "Ankle sprain",
                "classification": "Ligament",
                "pathology": "Sprain",
                "body_area": "Ankle",
                "tissue_type": "Ligament",
                "time_loss_days": 3,
                "duration_days": 6,
                "caused_unavailability": "No",
                "recurrent": "No",
                "activity": "Game",
                "activity_group": "Competition",
                "source_file": "injuries.csv",
                "source_row_number": 11,
            },
        ]
    )

    summary = build_shadow_event_crosswalk_summary(
        [{"packet": packet, "episodes": episodes, "timeline": timeline}],
        detailed,
    )

    rows = summary["event_crosswalk_rows"]
    assert summary["overall_recommendation"] == (
        "use_event_crosswalk_for_independent_practitioner_adjudication"
    )
    assert summary["total_event_rows"] == 2
    assert summary["captured_event_rows"] == 1
    assert summary["missed_event_rows"] == 1

    by_status = {row["capture_status"]: row for row in rows}
    assert by_status["captured"]["injury_event_id"] == "injury_a1"
    assert by_status["captured"]["linked_alert_episode_count"] == 1
    assert by_status["captured"]["nearest_alert_start_date"] == "2025-01-10"
    assert by_status["captured"]["time_loss_days"] == 12
    assert by_status["missed"]["injury_event_id"] == "injury_a2"
    assert by_status["missed"]["linked_alert_episode_count"] == 0
    assert by_status["missed"]["body_area"] == "Ankle"

    report_path = tmp_path / "crosswalk.md"
    write_exposure_load_shadow_event_crosswalk_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Shadow Event Crosswalk Sprint" in report
    assert "captured/missed injury-event crosswalk" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(clean_shadow_event_crosswalk_rows(rows), allow_nan=False)
