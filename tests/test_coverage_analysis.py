import pandas as pd
from risk_stratification_engine.coverage_analysis import (
    build_coverage_tiers,
    build_coverage_stratified_evaluation,
    build_coverage_flag,
)


def test_build_coverage_tiers_empty_measurements_returns_correct_columns():
    empty = pd.DataFrame(
        columns=["athlete_id", "date", "season_id", "source", "metric_name", "metric_value"]
    )
    result = build_coverage_tiers(empty)
    assert list(result.columns) == [
        "athlete_id",
        "season_id",
        "measurement_days",
        "measurement_row_count",
        "source_count",
        "median_days_between_measurements",
        "coverage_tier",
    ]
    assert len(result) == 0


def test_build_coverage_tiers_assigns_population_wide_tertile_tiers():
    # 3 athlete-seasons with clearly different measurement days
    rows = []
    # a_low: 1 unique date
    rows.append(("a_low", "2026-01-01", "s1", "fp", "m", 1.0))
    # a_med: 5 unique dates
    for d in range(1, 6):
        rows.append(("a_med", f"2026-01-{d:02d}", "s1", "fp", "m", 1.0))
    # a_high: 10 unique dates
    for d in range(1, 11):
        rows.append(("a_high", f"2026-01-{d:02d}", "s1", "fp", "m", 1.0))
    measurements = pd.DataFrame(
        rows, columns=["athlete_id", "date", "season_id", "source", "metric_name", "metric_value"]
    )
    result = build_coverage_tiers(measurements).set_index("athlete_id")

    assert result.loc["a_low", "coverage_tier"] == "low"
    assert result.loc["a_med", "coverage_tier"] == "medium"
    assert result.loc["a_high", "coverage_tier"] == "high"
    assert result.loc["a_low", "measurement_days"] == 1
    assert result.loc["a_high", "measurement_days"] == 10
    assert result.loc["a_med", "measurement_row_count"] == 5


def test_build_coverage_tiers_measurement_days_counts_unique_dates_not_rows():
    # Two rows on the same date should count as 1 measurement_day
    measurements = pd.DataFrame(
        [
            ("a1", "2026-01-01", "s1", "fp", "jump_height", 40.0),
            ("a1", "2026-01-01", "s1", "gps", "distance", 5000.0),  # same date, different source
        ],
        columns=["athlete_id", "date", "season_id", "source", "metric_name", "metric_value"],
    )
    result = build_coverage_tiers(measurements)
    assert result.loc[0, "measurement_days"] == 1
    assert result.loc[0, "measurement_row_count"] == 2


_TEST_CHANNEL = {
    "channel_name": "test_channel",
    "horizon_days": 30,
    "threshold_value": 0.50,  # top 50% → threshold = median of risk_30d
    "policy_name": "test",
    "graph_window_size": 4,
    "role": "test",
}


def _stratified_timeline_fixture():
    # Population risk_30d values: 0.10, 0.15, 0.80, 0.90, 0.50, 0.60
    # Sorted: 0.10, 0.15, 0.50, 0.60, 0.80, 0.90 → median (50th pct) ≈ 0.55
    # ev_low (low tier):  risk=[0.10, 0.15], event_within_30d=[0, 1], event_observed=True
    #   → 0.15 < 0.55 when event_within_30d=1 → NOT captured
    # ev_high (high tier): risk=[0.80, 0.90], event_within_30d=[1, 0], event_observed=True
    #   → 0.80 >= 0.55 when event_within_30d=1 → captured
    # no_ev (medium tier): risk=[0.50, 0.60], event_within_30d=[0, 0], event_observed=False
    #   → no event
    return pd.DataFrame(
        [
            {
                "athlete_id": "ev_low", "season_id": "s1", "coverage_tier": "low",
                "measurement_days": 2, "risk_30d": 0.10,
                "event_observed": True, "event_within_30d": 0,
            },
            {
                "athlete_id": "ev_low", "season_id": "s1", "coverage_tier": "low",
                "measurement_days": 2, "risk_30d": 0.15,
                "event_observed": True, "event_within_30d": 1,
            },
            {
                "athlete_id": "ev_high", "season_id": "s1", "coverage_tier": "high",
                "measurement_days": 10, "risk_30d": 0.80,
                "event_observed": True, "event_within_30d": 1,
            },
            {
                "athlete_id": "ev_high", "season_id": "s1", "coverage_tier": "high",
                "measurement_days": 10, "risk_30d": 0.90,
                "event_observed": True, "event_within_30d": 0,
            },
            {
                "athlete_id": "no_ev", "season_id": "s1", "coverage_tier": "medium",
                "measurement_days": 5, "risk_30d": 0.50,
                "event_observed": False, "event_within_30d": 0,
            },
            {
                "athlete_id": "no_ev", "season_id": "s1", "coverage_tier": "medium",
                "measurement_days": 5, "risk_30d": 0.60,
                "event_observed": False, "event_within_30d": 0,
            },
        ]
    )


def test_build_coverage_stratified_evaluation_capture_rates_by_tier():
    result = build_coverage_stratified_evaluation(
        _stratified_timeline_fixture(), _TEST_CHANNEL
    )
    rates = result["tier_capture_rates"]
    assert rates["low"] == 0.0      # 0 captured / 1 observed
    assert rates["high"] == 1.0     # 1 captured / 1 observed
    assert rates["medium"] is None  # 0 observed events


def test_build_coverage_stratified_evaluation_uses_population_wide_threshold():
    # The threshold should be the 50th percentile of ALL risk scores,
    # not per-tier. Verify via population_threshold field.
    result = build_coverage_stratified_evaluation(
        _stratified_timeline_fixture(), _TEST_CHANNEL
    )
    # 50th percentile of [0.10, 0.15, 0.50, 0.60, 0.80, 0.90] ≈ 0.55
    assert 0.50 <= result["population_threshold"] <= 0.60


def test_build_coverage_stratified_evaluation_rows_contain_tier_and_season_entries():
    result = build_coverage_stratified_evaluation(
        _stratified_timeline_fixture(), _TEST_CHANNEL
    )
    rows = result["rows"]
    tier_season_ids = {(r["coverage_tier"], r["season_id"]) for r in rows}
    # Should have "all" entries for each tier
    assert ("low", "all") in tier_season_ids
    assert ("high", "all") in tier_season_ids
    assert ("medium", "all") in tier_season_ids
    # Should have per-season entries for s1
    assert ("low", "s1") in tier_season_ids
    assert ("high", "s1") in tier_season_ids


def test_build_coverage_flag_confounded_when_high_much_greater_than_low():
    channel_results = [
        {"tier_capture_rates": {"low": 0.05, "medium": 0.10, "high": 0.25}},
        {"tier_capture_rates": {"low": 0.03, "medium": 0.08, "high": 0.22}},
    ]
    assert build_coverage_flag(channel_results) == "coverage_confounded"


def test_build_coverage_flag_independent_when_tiers_nearly_equal():
    channel_results = [
        {"tier_capture_rates": {"low": 0.10, "medium": 0.11, "high": 0.12}},
        {"tier_capture_rates": {"low": 0.09, "medium": 0.10, "high": 0.11}},
    ]
    assert build_coverage_flag(channel_results) == "coverage_independent"


def test_build_coverage_flag_mixed_when_difference_is_moderate():
    channel_results = [
        {"tier_capture_rates": {"low": 0.10, "medium": 0.12, "high": 0.18}},
    ]
    assert build_coverage_flag(channel_results) == "mixed"
