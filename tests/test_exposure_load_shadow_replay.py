import json

from risk_stratification_engine.exposure_load_shadow_replay import (
    build_exposure_load_shadow_replay_package,
    clean_shadow_replay_rows,
    write_exposure_load_shadow_replay_report,
)


def test_shadow_replay_builds_review_packets_and_stop_rule_tracking(tmp_path):
    summary = build_exposure_load_shadow_replay_package(
        validation_rows=_validation_rows(),
        shadow_channel_lock=_channel_lock(),
        shadow_review_protocol=_review_protocol(),
    )

    assert summary["experiment_type"] == "exposure_load_shadow_replay_sprint"
    assert summary["overall_recommendation"] == (
        "historical_shadow_replay_ready_for_prospective_collection"
    )
    assert summary["production_readiness"] == "not_ready_for_probability_or_pilot"

    replay_rows = summary["replay_rows"]
    assert {row["test_season_id"] for row in replay_rows} == {
        "2023-2024",
        "2024-2025",
    }
    source_ineligible = [
        row for row in replay_rows if row["test_season_id"] == "2024-2025"
    ][0]
    assert source_ineligible["source_eligible"] is False
    assert source_ineligible["replay_status"] == "source_ineligible_stop"

    review_packets = summary["review_packet_rows"]
    assert len(review_packets) == 1
    assert review_packets[0]["review_packet_id"] == "broad_30d__2023-2024"
    assert review_packets[0]["required_evidence"] == (
        "frozen alert episodes, source eligibility, exposure capture status, "
        "outcome adjudication, and alert burden"
    )
    assert review_packets[0]["review_packet_status"] == (
        "ready_for_research_adjudication"
    )

    stop_rows = summary["stop_rule_rows"]
    assert {row["stop_rule_status"] for row in stop_rows} == {
        "no_stop_rule_triggered",
        "source_ineligible_stop",
    }

    report_path = tmp_path / "shadow_replay.md"
    write_exposure_load_shadow_replay_report(report_path, summary)
    report = report_path.read_text(encoding="utf-8")
    assert "Historical Shadow Replay Sprint" in report
    assert "review packets" in report
    assert "not pilot or dashboard clearance" in report

    json.dumps(summary, allow_nan=False)
    json.dumps(clean_shadow_replay_rows(summary["replay_rows"]), allow_nan=False)


def _channel_lock() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_shadow_channel_lock_sprint",
        "production_readiness": "not_ready_for_probability_or_pilot",
        "excluded_test_seasons": ["2024-2025"],
        "locked_channels": [
            {
                "channel_name": "broad_30d",
                "policy_name": "exclude_concussion",
                "horizon_days": 30,
                "threshold_policy": "burden_capped_percentile",
                "lock_status": "locked_for_research_shadow_review",
            }
        ],
    }


def _review_protocol() -> dict[str, object]:
    return {
        "experiment_type": "exposure_load_shadow_review_protocol_sprint",
        "protocol_rows": [
            {
                "channel_name": "broad_30d",
                "policy_name": "exclude_concussion",
                "horizon_days": 30,
                "threshold_policy": "burden_capped_percentile",
                "minimum_review_unit": "complete source-eligible athlete-season",
                "required_evidence": (
                    "frozen alert episodes, source eligibility, exposure capture "
                    "status, outcome adjudication, and alert burden"
                ),
                "stop_rule": (
                    "pause channel if prospective burden exceeds 1.0 episodes "
                    "per athlete-season or source eligibility fails"
                ),
            }
        ],
    }


def _validation_rows() -> list[dict[str, object]]:
    return [
        _alert_row(
            test_season_id="2023-2024",
            capture=0.25,
            burden=0.80,
        ),
        _alert_row(
            test_season_id="2024-2025",
            capture=0.45,
            burden=0.70,
        ),
    ]


def _alert_row(
    test_season_id: str,
    capture: float,
    burden: float,
) -> dict[str, object]:
    return {
        "row_type": "alert_policy",
        "test_season_id": test_season_id,
        "feature_set": "graph_plus_coverage_exposure_load",
        "threshold_policy": "burden_capped_percentile",
        "channel_name": "broad_30d",
        "policy_name": "exclude_concussion",
        "graph_window_size": 4,
        "horizon_days": 30,
        "role": "broad 30d early warning",
        "selected_threshold_value": 0.05,
        "episode_count": 10,
        "true_positive_episode_count": 4,
        "false_positive_episode_count": 6,
        "unique_event_capture_rate": capture,
        "unique_captured_event_count": 3,
        "unique_observed_event_count": 12,
        "missed_event_count": 9,
        "episodes_per_athlete_season": burden,
        "median_start_lead_days": 11,
    }
