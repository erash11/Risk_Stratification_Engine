import pandas as pd

from risk_stratification_engine.coverage_adjusted_thresholds import (
    DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON,
    build_coverage_adjusted_threshold_policy_rows,
    build_coverage_adjusted_threshold_summary,
    write_coverage_adjusted_threshold_report,
)


def test_build_coverage_adjusted_threshold_policy_rows_applies_tier_local_thresholds():
    timeline = _threshold_timeline()
    channel = _channel(threshold_value=0.5)

    rows = build_coverage_adjusted_threshold_policy_rows(
        timeline,
        channel,
        candidate_percentiles=(0.5, 0.25),
    )

    tier_rows = [
        row
        for row in rows
        if row["threshold_policy"] == "coverage_tier_local_percentile"
    ]
    assert {row["coverage_tier"] for row in tier_rows} == {"low", "high"}
    assert all(row["threshold_scope"] == "season_coverage_tier" for row in tier_rows)
    assert all(row["selected_threshold_value"] == 0.5 for row in tier_rows)


def test_build_coverage_adjusted_threshold_policy_rows_selects_stricter_burden_cap():
    timeline = _threshold_timeline()
    channel = _channel(threshold_value=0.5)

    rows = build_coverage_adjusted_threshold_policy_rows(
        timeline,
        channel,
        burden_cap_episodes_per_athlete_season=0.25,
        candidate_percentiles=(0.5, 0.25),
    )

    burden_row = next(
        row for row in rows if row["threshold_policy"] == "burden_capped_percentile"
    )
    assert burden_row["burden_cap_episodes_per_athlete_season"] == 0.25
    assert burden_row["selected_threshold_value"] == 0.25
    assert burden_row["episodes_per_athlete_season"] <= 0.25


def test_build_coverage_adjusted_threshold_summary_keeps_research_recommendation():
    rows = [
        {
            "channel_name": "severity_14d",
            "threshold_policy": "season_local_percentile",
            "unique_event_capture_rate": 0.20,
            "episodes_per_athlete_season": 2.0,
        },
        {
            "channel_name": "severity_14d",
            "threshold_policy": "burden_capped_percentile",
            "unique_event_capture_rate": 0.15,
            "episodes_per_athlete_season": 0.8,
        },
    ]

    summary = build_coverage_adjusted_threshold_summary(rows)

    assert summary["experiment_type"] == "coverage_adjusted_threshold_sprint"
    assert summary["overall_recommendation"] == "continue_threshold_research"
    assert (
        summary["channel_recommendations"]["severity_14d"]["recommended_policy"]
        == "burden_capped_percentile"
    )


def test_write_coverage_adjusted_threshold_report_names_peterson_guardrail(tmp_path):
    summary = {
        "experiment_type": "coverage_adjusted_threshold_sprint",
        "overall_recommendation": "continue_threshold_research",
        "burden_cap_episodes_per_athlete_season": (
            DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON
        ),
        "threshold_policies": [
            "season_local_percentile",
            "coverage_tier_local_percentile",
            "burden_capped_percentile",
        ],
        "channel_recommendations": {
            "severity_14d": {
                "recommended_policy": "burden_capped_percentile",
                "mean_capture_rate": 0.15,
                "mean_episodes_per_athlete_season": 0.8,
            }
        },
    }
    path = tmp_path / "coverage_adjusted_threshold_report.md"

    write_coverage_adjusted_threshold_report(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "Coverage-Adjusted Threshold Sprint" in text
    assert "complete athlete-season trajectories" in text
    assert "burden_capped_percentile" in text


def _threshold_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _snapshot("a1", "low", 0.90, observed=True, days_to_event=3),
            _snapshot("a2", "low", 0.80, observed=False, days_to_event=None),
            _snapshot("a3", "high", 0.70, observed=True, days_to_event=5),
            _snapshot("a4", "high", 0.60, observed=False, days_to_event=None),
        ]
    )


def _snapshot(
    athlete_id: str,
    coverage_tier: str,
    risk: float,
    *,
    observed: bool,
    days_to_event: int | None,
) -> dict[str, object]:
    return {
        "athlete_id": athlete_id,
        "season_id": "2026",
        "coverage_tier": coverage_tier,
        "time_index": 0,
        "snapshot_date": "2026-01-01",
        "risk_14d": risk,
        "event_observed": observed,
        "event_date": "2026-01-10" if observed else None,
        "injury_type": "hamstring" if observed else "censored",
        "days_to_event": days_to_event,
        "event_within_14d": observed,
    }


def _channel(threshold_value: float) -> dict[str, object]:
    return {
        "channel_name": "severity_14d",
        "role": "short-horizon severity triage",
        "policy_name": "model_safe_time_loss",
        "graph_window_size": 4,
        "horizon_days": 14,
        "threshold_value": threshold_value,
    }
