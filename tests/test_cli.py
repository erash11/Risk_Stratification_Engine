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
        detailed_injuries = output_dir / "injury_events_detailed.csv"
        measurements.write_text("measurements", encoding="utf-8")
        injuries.write_text("injuries", encoding="utf-8")
        detailed_injuries.write_text("detailed injuries", encoding="utf-8")
        return cli.LiveSourcePreparationResult(
            measurements_path=measurements,
            injuries_path=injuries,
            detailed_injuries_path=detailed_injuries,
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
        model_variant,
    ):
        calls["experiment"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
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
        "model_variant": "baseline",
    }


def test_cli_runs_window_sensitivity_experiment(tmp_path, monkeypatch):
    calls = {}

    def fake_run_window_sensitivity_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_sizes,
        model_variant,
    ):
        calls["window_sensitivity"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_sizes": graph_window_sizes,
            "model_variant": model_variant,
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
        "model_variant": "baseline",
    }


def test_cli_runs_model_robustness_sprint(tmp_path, monkeypatch):
    calls = {}

    def fake_run_model_robustness_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        split_count,
    ):
        calls["robustness"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "split_count": split_count,
        }
        experiment_dir = output_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True)
        return experiment_dir

    monkeypatch.setattr(
        cli,
        "run_model_robustness_experiment",
        fake_run_model_robustness_experiment,
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
            "robustness",
            "--model-robustness-sprint",
            "--graph-window-size",
            "4",
            "--stability-splits",
            "3",
        ]
    )

    assert exit_code == 0
    assert calls["robustness"] == {
        "measurements_path": FIXTURES / "measurements.csv",
        "injuries_path": FIXTURES / "injuries.csv",
        "output_dir": tmp_path,
        "experiment_id": "robustness",
        "graph_window_size": 4,
        "split_count": 3,
    }


def test_cli_runs_window_model_robustness_sprint(tmp_path, monkeypatch):
    calls = {}

    def fake_run_window_model_robustness_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_sizes,
        split_count,
    ):
        calls["window_model_robustness"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_sizes": graph_window_sizes,
            "split_count": split_count,
        }
        experiment_dir = output_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True)
        return experiment_dir

    monkeypatch.setattr(
        cli,
        "run_window_model_robustness_experiment",
        fake_run_window_model_robustness_experiment,
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
            "window_model_robustness",
            "--model-robustness-sprint",
            "--window-sensitivity-sizes",
            "2",
            "4",
            "7",
            "--stability-splits",
            "3",
        ]
    )

    assert exit_code == 0
    assert calls["window_model_robustness"] == {
        "measurements_path": FIXTURES / "measurements.csv",
        "injuries_path": FIXTURES / "injuries.csv",
        "output_dir": tmp_path,
        "experiment_id": "window_model_robustness",
        "graph_window_sizes": (2, 4, 7),
        "split_count": 3,
    }


def test_cli_runs_calibration_thresholds_experiment(tmp_path, monkeypatch):
    calls = {}

    def fake_run_calibration_threshold_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
        split_count,
    ):
        calls["calibration"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "split_count": split_count,
        }
        experiment_dir = output_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True)
        return experiment_dir

    monkeypatch.setattr(
        cli,
        "run_calibration_threshold_experiment",
        fake_run_calibration_threshold_experiment,
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
            "calibration_run",
            "--calibration-thresholds",
            "--model-variant",
            "l2",
            "--graph-window-size",
            "4",
            "--stability-splits",
            "3",
        ]
    )

    assert exit_code == 0
    assert calls["calibration"] == {
        "measurements_path": FIXTURES / "measurements.csv",
        "injuries_path": FIXTURES / "injuries.csv",
        "output_dir": tmp_path,
        "experiment_id": "calibration_run",
        "graph_window_size": 4,
        "model_variant": "l2",
        "split_count": 3,
    }


def test_cli_runs_alert_episode_experiment(tmp_path, monkeypatch):
    calls = {}

    def fake_run_alert_episode_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
        detailed_injuries_path,
    ):
        calls["alert_episodes"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "detailed_injuries_path": detailed_injuries_path,
        }
        experiment_dir = output_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True)
        return experiment_dir

    monkeypatch.setattr(
        cli,
        "run_alert_episode_experiment",
        fake_run_alert_episode_experiment,
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
            "alert_episode_run",
            "--alert-episodes",
            "--model-variant",
            "l2",
            "--graph-window-size",
            "4",
        ]
    )

    assert exit_code == 0
    assert calls["alert_episodes"] == {
        "measurements_path": FIXTURES / "measurements.csv",
        "injuries_path": FIXTURES / "injuries.csv",
        "output_dir": tmp_path,
        "experiment_id": "alert_episode_run",
        "graph_window_size": 4,
        "model_variant": "l2",
        "detailed_injuries_path": None,
    }


def test_cli_runs_injury_outcome_policy_experiment_from_live_sources(
    tmp_path,
    monkeypatch,
):
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
        detailed_injuries = output_dir / "injury_events_detailed.csv"
        measurements.write_text("measurements", encoding="utf-8")
        injuries.write_text("injuries", encoding="utf-8")
        detailed_injuries.write_text("detailed injuries", encoding="utf-8")
        return cli.LiveSourcePreparationResult(
            measurements_path=measurements,
            injuries_path=injuries,
            detailed_injuries_path=detailed_injuries,
            metadata_path=output_dir / "prep_metadata.json",
            audit_path=output_dir / "data_quality_audit.json",
            metadata={"canonical_rows": {"measurements": 1, "injury_events": 1}},
            audit={"coverage": {"athlete_season_count": 1}},
        )

    def fake_run_injury_outcome_policy_experiment(
        detailed_injuries_path,
        output_dir,
        experiment_id,
    ):
        calls["injury_outcomes"] = {
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
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
    monkeypatch.setattr(
        cli,
        "run_injury_outcome_policy_experiment",
        fake_run_injury_outcome_policy_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "injury_outcome_policy_run",
            "--injury-outcome-policies",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["injury_outcomes"] == {
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "injury_outcome_policy_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "injury_outcome_policy_run",
    }


def test_cli_runs_outcome_policy_model_comparison_from_live_sources(
    tmp_path,
    monkeypatch,
):
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
        detailed_injuries = output_dir / "injury_events_detailed.csv"
        measurements.write_text("measurements", encoding="utf-8")
        injuries.write_text("injuries", encoding="utf-8")
        detailed_injuries.write_text("detailed injuries", encoding="utf-8")
        return cli.LiveSourcePreparationResult(
            measurements_path=measurements,
            injuries_path=injuries,
            detailed_injuries_path=detailed_injuries,
            metadata_path=output_dir / "prep_metadata.json",
            audit_path=output_dir / "data_quality_audit.json",
            metadata={"canonical_rows": {"measurements": 1, "injury_events": 1}},
            audit={"coverage": {"athlete_season_count": 1}},
        )

    def fake_run_outcome_policy_model_comparison_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["policy_model_comparison"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
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
    monkeypatch.setattr(
        cli,
        "run_outcome_policy_model_comparison_experiment",
        fake_run_outcome_policy_model_comparison_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "policy_model_comparison_run",
            "--outcome-policy-model-comparison",
            "--model-variant",
            "l2",
            "--graph-window-size",
            "4",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["policy_model_comparison"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "policy_model_comparison_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "policy_model_comparison_run"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "policy_model_comparison_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "policy_model_comparison_run",
        "graph_window_size": 4,
        "model_variant": "l2",
    }


def test_cli_runs_policy_decision_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
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
        detailed_injuries = output_dir / "injury_events_detailed.csv"
        measurements.write_text("measurements", encoding="utf-8")
        injuries.write_text("injuries", encoding="utf-8")
        detailed_injuries.write_text("detailed injuries", encoding="utf-8")
        return cli.LiveSourcePreparationResult(
            measurements_path=measurements,
            injuries_path=injuries,
            detailed_injuries_path=detailed_injuries,
            metadata_path=output_dir / "prep_metadata.json",
            audit_path=output_dir / "data_quality_audit.json",
            metadata={"canonical_rows": {"measurements": 1, "injury_events": 1}},
            audit={"coverage": {"athlete_season_count": 1}},
        )

    def fake_run_policy_decision_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        graph_window_sizes,
        model_variant,
    ):
        calls["policy_decision_sprint"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_sizes": graph_window_sizes,
            "model_variant": model_variant,
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
    monkeypatch.setattr(
        cli,
        "run_policy_decision_sprint_experiment",
        fake_run_policy_decision_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "policy_decision_sprint_run",
            "--policy-decision-sprint",
            "--policy-window-sizes",
            "2",
            "4",
            "7",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["policy_decision_sprint"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "policy_decision_sprint_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "policy_decision_sprint_run"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "policy_decision_sprint_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "policy_decision_sprint_run",
        "graph_window_sizes": (2, 4, 7),
        "model_variant": "l2",
    }
