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


@pytest.mark.parametrize("column", ["athlete_id", "season_id", "source", "metric_name"])
@pytest.mark.parametrize("bad_value", ["", "   "])
def test_load_measurements_rejects_blank_or_null_required_fields(
    tmp_path,
    column,
    bad_value,
):
    row = {
        "athlete_id": "a1",
        "date": "2026-01-01",
        "season_id": "2026",
        "source": "force_plate",
        "metric_name": "jump_height",
        "metric_value": "42.0",
    }
    row[column] = bad_value
    bad_path = tmp_path / "bad_required_field.csv"
    pd.DataFrame([row]).to_csv(bad_path, index=False)

    with pytest.raises(ValueError) as exc:
        load_measurements(bad_path)

    message = str(exc.value)
    assert "measurements contains blank/null required fields" in message
    assert column in message


def test_load_injury_events_rejects_populated_invalid_injury_date(tmp_path):
    bad_path = tmp_path / "bad_injury_date.csv"
    bad_path.write_text(
        "athlete_id,season_id,injury_date,injury_type,event_observed,censor_date\n"
        "a1,2026,not-a-date,none,false,2026-02-01\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        load_injury_events(bad_path)

    assert "injury_events contains unparseable injury_date values" in str(exc.value)


def test_load_injury_events_rejects_invalid_event_observed(tmp_path):
    bad_path = tmp_path / "bad_event_observed.csv"
    bad_path.write_text(
        "athlete_id,season_id,injury_date,injury_type,event_observed,censor_date\n"
        "a1,2026,,none,maybe,2026-02-01\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        load_injury_events(bad_path)

    assert "cannot parse boolean value: maybe" in str(exc.value)


def test_load_injury_events_rejects_invalid_censor_date(tmp_path):
    bad_path = tmp_path / "bad_censor_date.csv"
    bad_path.write_text(
        "athlete_id,season_id,injury_date,injury_type,event_observed,censor_date\n"
        "a1,2026,,none,false,not-a-date\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        load_injury_events(bad_path)

    assert "injury_events contains unparseable censor_date values" in str(exc.value)


def test_load_injury_events_rejects_observed_event_missing_injury_date(tmp_path):
    bad_path = tmp_path / "missing_observed_injury_date.csv"
    bad_path.write_text(
        "athlete_id,season_id,injury_date,injury_type,event_observed,censor_date\n"
        "a1,2026,,lower_extremity_soft_tissue,true,2026-01-20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        load_injury_events(bad_path)

    assert "observed injury events require injury_date" in str(exc.value)


def test_write_frame_writes_csv(tmp_path):
    output = tmp_path / "frame.csv"
    frame = pd.DataFrame({"athlete_id": ["a1"], "risk": [0.25]})

    write_frame(frame, output)

    assert output.exists()
    loaded = pd.read_csv(output)
    assert loaded.to_dict("records") == [{"athlete_id": "a1", "risk": 0.25}]
