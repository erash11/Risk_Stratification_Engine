from pathlib import Path

import pytest

from risk_stratification_engine.cli import main


FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_runs_fixture_experiment(tmp_path):
    exit_code = main(
        [
            "--measurements",
            str(FIXTURES / "measurements.csv"),
            "--injuries",
            str(FIXTURES / "injuries.csv"),
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "cli_fixture_run",
            "--graph-window-size",
            "2",
        ]
    )

    assert exit_code == 0
    assert (
        tmp_path
        / "experiments"
        / "cli_fixture_run"
        / "athlete_risk_timeline.csv"
    ).exists()


def test_cli_requires_experiment_arguments():
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2
