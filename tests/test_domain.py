from risk_stratification_engine import __version__

import pandas as pd
import pytest

from risk_stratification_engine.domain import (
    CANONICAL_MEASUREMENT_COLUMNS,
    INJURY_EVENT_COLUMNS,
    require_columns,
)


def test_package_imports():
    assert __version__ == "0.1.0"


def test_canonical_measurement_columns_match_spec():
    assert CANONICAL_MEASUREMENT_COLUMNS == (
        "athlete_id",
        "date",
        "season_id",
        "source",
        "metric_name",
        "metric_value",
    )


def test_injury_event_columns_include_censoring_contract():
    assert INJURY_EVENT_COLUMNS == (
        "athlete_id",
        "season_id",
        "injury_date",
        "injury_type",
        "event_observed",
        "censor_date",
    )


def test_require_columns_accepts_complete_dataframe():
    frame = pd.DataFrame(
        {
            "athlete_id": ["a1"],
            "date": ["2026-01-01"],
            "season_id": ["2026"],
            "source": ["force_plate"],
            "metric_name": ["jump_height"],
            "metric_value": [42.0],
        }
    )

    require_columns(frame, CANONICAL_MEASUREMENT_COLUMNS, "measurements")


def test_require_columns_reports_missing_columns():
    frame = pd.DataFrame({"athlete_id": ["a1"]})

    with pytest.raises(ValueError) as exc:
        require_columns(frame, CANONICAL_MEASUREMENT_COLUMNS, "measurements")

    assert "measurements missing required columns" in str(exc.value)
    assert "metric_value" in str(exc.value)
