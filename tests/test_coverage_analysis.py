import pandas as pd
from risk_stratification_engine.coverage_analysis import build_coverage_tiers


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
