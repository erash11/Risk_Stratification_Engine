from pathlib import Path

import pandas as pd
import pytest

from risk_stratification_engine.io import load_injury_events, load_measurements, write_frame


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_measurements_parses_dates_and_values():
    frame = load_measurements(FIXTURES / "measurements.csv")

    assert str(frame["date"].dtype).startswith("datetime64")
    assert frame["metric_value"].dtype.kind == "f"
    assert list(frame.columns) == [
        "athlete_id",
        "date",
        "season_id",
        "source",
        "metric_name",
        "metric_value",
    ]


def test_load_injury_events_parses_event_observed_and_dates():
    frame = load_injury_events(FIXTURES / "injuries.csv")

    assert str(frame["injury_date"].dtype).startswith("datetime64")
    assert str(frame["censor_date"].dtype).startswith("datetime64")
    assert frame.loc[frame["athlete_id"] == "a1", "event_observed"].item() is True
    assert frame.loc[frame["athlete_id"] == "a2", "event_observed"].item() is False


def test_load_measurements_rejects_bad_schema(tmp_path):
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("athlete_id,date\nx,2026-01-01\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_measurements(bad_path)

    assert "measurements missing required columns" in str(exc.value)


def test_write_frame_writes_csv(tmp_path):
    output = tmp_path / "frame.csv"
    frame = pd.DataFrame({"athlete_id": ["a1"], "risk": [0.25]})

    write_frame(frame, output)

    assert output.exists()
    loaded = pd.read_csv(output)
    assert loaded.to_dict("records") == [{"athlete_id": "a1", "risk": 0.25}]
