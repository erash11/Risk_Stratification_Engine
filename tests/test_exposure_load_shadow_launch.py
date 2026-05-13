import json

from risk_stratification_engine.exposure_load_shadow_launch import (
    build_exposure_load_shadow_channel_lock,
    build_exposure_load_shadow_readiness_register,
    build_exposure_load_shadow_review_protocol,
    clean_shadow_launch_rows,
    write_exposure_load_shadow_channel_lock_report,
    write_exposure_load_shadow_readiness_register_report,
    write_exposure_load_shadow_review_protocol_report,
)


def test_shadow_launch_chain_locks_ready_channels_and_preserves_blockers(tmp_path):
    channel_lock = build_exposure_load_shadow_channel_lock(_monitoring_summary())

    assert channel_lock["experiment_type"] == "exposure_load_shadow_channel_lock_sprint"
    assert channel_lock["overall_recommendation"] == (
        "lock_source_eligible_burden_capped_channels_for_shadow_review"
    )
    assert channel_lock["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert [row["channel_name"] for row in channel_lock["locked_channels"]] == [
        "broad_30d",
        "severity_14d",
    ]
    assert channel_lock["held_channels"][0]["channel_name"] == (
        "subtype_lower_extremity_soft_tissue_30d"
    )
    assert channel_lock["held_channels"][0]["hold_reason"] == (
        "shadow_burden_guardrail_review_needed"
    )

    protocol = build_exposure_load_shadow_review_protocol(channel_lock)
    assert protocol["experiment_type"] == "exposure_load_shadow_review_protocol_sprint"
    assert protocol["overall_recommendation"] == (
        "launch_research_shadow_review_with_locked_channels"
    )
    assert len(protocol["protocol_rows"]) == 2
    assert protocol["protocol_rows"][0]["minimum_review_unit"] == (
        "complete source-eligible athlete-season"
    )

    readiness = build_exposure_load_shadow_readiness_register(
        channel_lock,
        protocol,
    )
    assert readiness["experiment_type"] == (
        "exposure_load_shadow_readiness_register_sprint"
    )
    assert readiness["overall_recommendation"] == (
        "launch_research_shadow_monitoring_without_product_escalation"
    )
    assert readiness["production_readiness"] == "not_ready_for_probability_or_pilot"
    assert readiness["readiness_rows"][0]["readiness_status"] == (
        "research_shadow_launch_ready"
    )

    channel_lock_report = tmp_path / "channel_lock.md"
    protocol_report = tmp_path / "protocol.md"
    readiness_report = tmp_path / "readiness.md"
    write_exposure_load_shadow_channel_lock_report(channel_lock_report, channel_lock)
    write_exposure_load_shadow_review_protocol_report(protocol_report, protocol)
    write_exposure_load_shadow_readiness_register_report(
        readiness_report,
        readiness,
    )

    assert "Shadow Channel Lock Sprint" in channel_lock_report.read_text()
    assert "not pilot or dashboard clearance" in protocol_report.read_text()
    assert "research shadow monitoring" in readiness_report.read_text()
    json.dumps(channel_lock, allow_nan=False)
    json.dumps(protocol, allow_nan=False)
    json.dumps(readiness, allow_nan=False)
    json.dumps(clean_shadow_launch_rows(readiness["readiness_rows"]), allow_nan=False)


def _monitoring_summary() -> dict[str, object]:
    return {
        "experiment_type": (
            "exposure_load_source_eligible_shadow_monitoring_sprint"
        ),
        "overall_recommendation": (
            "proceed_with_prospective_source_eligible_shadow_monitoring"
        ),
        "production_readiness": "not_ready_for_probability_or_pilot",
        "excluded_test_seasons": ["2024-2025"],
        "burden_cap_episodes_per_athlete_season": 1.0,
        "monitoring_rows": [
            {
                "channel_name": "broad_30d",
                "policy_name": "exclude_concussion",
                "horizon_days": 30,
                "threshold_policy": "burden_capped_percentile",
                "source_eligible_season_count": 4,
                "mean_capture_rate": 0.151,
                "max_episodes_per_athlete_season": 0.686,
                "mean_threshold_absolute_drift": 0.009,
                "monitoring_status": "ready_for_prospective_shadow_review",
            },
            {
                "channel_name": "severity_14d",
                "policy_name": "model_safe_time_loss",
                "horizon_days": 14,
                "threshold_policy": "burden_capped_percentile",
                "source_eligible_season_count": 4,
                "mean_capture_rate": 0.129,
                "max_episodes_per_athlete_season": 0.919,
                "mean_threshold_absolute_drift": 0.019,
                "monitoring_status": "ready_for_prospective_shadow_review",
            },
            {
                "channel_name": "subtype_lower_extremity_soft_tissue_30d",
                "policy_name": "lower_extremity_soft_tissue",
                "horizon_days": 30,
                "threshold_policy": "season_local_percentile",
                "source_eligible_season_count": 4,
                "mean_capture_rate": 0.186,
                "max_episodes_per_athlete_season": 2.488,
                "mean_threshold_absolute_drift": 0.0,
                "monitoring_status": "shadow_burden_guardrail_review_needed",
            },
        ],
    }
