from pathlib import Path

import pytest

import risk_stratification_engine.cli as cli
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


def test_cli_prepares_live_sources_before_running_experiment(tmp_path, monkeypatch):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(paths, output_dir):
        calls["paths"] = paths
        calls["prep_output_dir"] = output_dir
        output_dir.mkdir(parents=True)
        measurements = output_dir / "canonical_measurements.csv"
        injuries = output_dir / "canonical_injuries.csv"
        measurements.write_text("measurements", encoding="utf-8")
        injuries.write_text("injuries", encoding="utf-8")
        return cli.LiveSourcePreparationResult(
            measurements_path=measurements,
            injuries_path=injuries,
            metadata_path=output_dir / "prep_metadata.json",
            audit_path=output_dir / "data_quality_audit.json",
            metadata={"canonical_rows": {"measurements": 1, "injury_events": 1}},
            audit={"coverage": {"athlete_season_count": 1}},
        )

    def fake_run_research_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
    ):
        calls["experiment"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
        }
        experiment_dir = output_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True)
        return experiment_dir

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(cli, "run_research_experiment", fake_run_research_experiment)

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "live_run",
            "--graph-window-size",
            "5",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["prep_output_dir"] == tmp_path / "live_inputs" / "live_run"
    assert calls["experiment"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "live_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "live_run"
        / "canonical_injuries.csv",
        "output_dir": tmp_path,
        "experiment_id": "live_run",
        "graph_window_size": 5,
    }


def test_cli_runs_window_sensitivity_experiment(tmp_path, monkeypatch):
    calls = {}

    def fake_run_window_sensitivity_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_sizes,
    ):
        calls["window_sensitivity"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_sizes": graph_window_sizes,
        }
        experiment_dir = output_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True)
        return experiment_dir

    monkeypatch.setattr(
        cli,
        "run_window_sensitivity_experiment",
        fake_run_window_sensitivity_experiment,
    )

    exit_code = main(
        [
            "--measurements",
            str(FIXTURES / "measurements.csv"),
            "--injuries",
            str(FIXTURES / "injuries.csv"),
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "window_sensitivity",
            "--window-sensitivity-sizes",
            "2",
            "4",
            "7",
        ]
    )

    assert exit_code == 0
    assert calls["window_sensitivity"] == {
        "measurements_path": FIXTURES / "measurements.csv",
        "injuries_path": FIXTURES / "injuries.csv",
        "output_dir": tmp_path,
        "experiment_id": "window_sensitivity",
        "graph_window_sizes": (2, 4, 7),
    }
