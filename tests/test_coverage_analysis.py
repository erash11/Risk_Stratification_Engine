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
