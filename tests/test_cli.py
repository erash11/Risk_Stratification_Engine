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


def test_cli_runs_exposure_cleaning_audit_without_model_inputs(tmp_path, monkeypatch):
    calls = {}

    def fake_prepare_exposure_inputs(exposure_dir, output_dir):
        calls["exposure_dir"] = exposure_dir
        calls["output_dir"] = output_dir
        output_dir.mkdir(parents=True)
        events = output_dir / "exposure_events.csv"
        participations = output_dir / "exposure_participations.csv"
        audit = output_dir / "exposure_cleaning_audit.json"
        events.write_text("events", encoding="utf-8")
        participations.write_text("participations", encoding="utf-8")
        audit.write_text("{}", encoding="utf-8")
        return cli.ExposurePreparationResult(
            events_path=events,
            participations_path=participations,
            audit_path=audit,
            audit={},
        )

    monkeypatch.setattr(cli, "prepare_exposure_inputs", fake_prepare_exposure_inputs)

    exit_code = main(
        [
            "--exposure-cleaning-audit",
            "--exposure-dir",
            "C:/tmp/Baylor_Exposure_Data",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_audit_v1",
        ]
    )

    assert exit_code == 0
    assert calls["exposure_dir"] == Path("C:/tmp/Baylor_Exposure_Data")
    assert calls["output_dir"] == tmp_path / "exposure_inputs" / "exposure_audit_v1"


def test_cli_runs_exposure_feature_requirements_sprint_from_cleaned_inputs(
    tmp_path,
    monkeypatch,
):
    exposure_dir = tmp_path / "exposure_inputs" / "exposure_audit_v1"
    exposure_dir.mkdir(parents=True)
    events = exposure_dir / "exposure_events.csv"
    participations = exposure_dir / "exposure_participations.csv"
    audit = exposure_dir / "exposure_cleaning_audit.json"
    events.write_text("events", encoding="utf-8")
    participations.write_text("participations", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    calls = {}

    def fake_run_exposure_feature_requirements_sprint_experiment(
        exposure_events_path,
        exposure_participations_path,
        exposure_audit_path,
        output_dir,
        experiment_id,
    ):
        calls["requirements"] = {
            "exposure_events_path": exposure_events_path,
            "exposure_participations_path": exposure_participations_path,
            "exposure_audit_path": exposure_audit_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        experiment_dir = output_dir / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True)
        return experiment_dir

    monkeypatch.setattr(
        cli,
        "run_exposure_feature_requirements_sprint_experiment",
        fake_run_exposure_feature_requirements_sprint_experiment,
    )

    exit_code = main(
        [
            "--exposure-feature-requirements-sprint",
            "--exposure-events",
            str(events),
            "--exposure-participations",
            str(participations),
            "--exposure-audit",
            str(audit),
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_feature_requirements_v1",
        ]
    )

    assert exit_code == 0
    assert calls["requirements"] == {
        "exposure_events_path": events,
        "exposure_participations_path": participations,
        "exposure_audit_path": audit,
        "output_dir": tmp_path,
        "experiment_id": "exposure_feature_requirements_v1",
    }


def test_cli_runs_exposure_load_feature_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
    exposure_participations = tmp_path / "exposure_participations.csv"
    exposure_participations.write_text("exposure rows", encoding="utf-8")
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_exposure_load_feature_sprint_experiment(
        measurements_path,
        injuries_path,
        exposure_participations_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["exposure_load_feature"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "exposure_participations_path": exposure_participations_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_exposure_load_feature_sprint_experiment",
        fake_run_exposure_load_feature_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_feature_v1",
            "--exposure-load-feature-sprint",
            "--exposure-participations",
            str(exposure_participations),
            "--graph-window-size",
            "4",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["exposure_load_feature"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "exposure_load_feature_v1"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "exposure_load_feature_v1"
        / "canonical_injuries.csv",
        "exposure_participations_path": exposure_participations,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_feature_v1",
        "graph_window_size": 4,
        "model_variant": "l2",
    }


def test_cli_runs_exposure_load_season_forward_validation_from_live_sources(
    tmp_path,
    monkeypatch,
):
    exposure_participations = tmp_path / "exposure_participations.csv"
    exposure_participations.write_text("exposure rows", encoding="utf-8")
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_exposure_load_season_forward_validation_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        exposure_participations_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["exposure_load_season_forward"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "exposure_participations_path": exposure_participations_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_exposure_load_season_forward_validation_sprint_experiment",
        fake_run_exposure_load_season_forward_validation_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_season_forward",
            "--exposure-load-season-forward-validation",
            "--exposure-participations",
            str(exposure_participations),
            "--graph-window-size",
            "4",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["exposure_load_season_forward"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "exposure_load_season_forward"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "exposure_load_season_forward"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "exposure_load_season_forward"
        / "injury_events_detailed.csv",
        "exposure_participations_path": exposure_participations,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_season_forward",
        "graph_window_size": 4,
        "model_variant": "l2",
    }


def test_cli_runs_exposure_load_forward_diagnostic_from_validation_csv(
    tmp_path,
    monkeypatch,
):
    validation_path = tmp_path / "exposure_load_season_forward_validation.csv"
    validation_path.write_text("validation rows", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_forward_diagnostic_sprint_experiment(
        season_forward_validation_path,
        output_dir,
        experiment_id,
    ):
        calls["exposure_load_forward_diagnostic"] = {
            "season_forward_validation_path": season_forward_validation_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_forward_diagnostic_sprint_experiment",
        fake_run_exposure_load_forward_diagnostic_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_forward_diagnostic",
            "--exposure-load-forward-diagnostic-sprint",
            "--season-forward-validation-path",
            str(validation_path),
        ]
    )

    assert exit_code == 0
    assert calls["exposure_load_forward_diagnostic"] == {
        "season_forward_validation_path": validation_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_forward_diagnostic",
    }


def test_cli_runs_exposure_load_failure_mode_from_artifacts(
    tmp_path,
    monkeypatch,
):
    features_path = tmp_path / "exposure_load_features.csv"
    diagnostics_path = tmp_path / "exposure_load_calibration_diagnostics.csv"
    features_path.write_text("features", encoding="utf-8")
    diagnostics_path.write_text("diagnostics", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_failure_mode_sprint_experiment(
        exposure_load_features_path,
        exposure_load_diagnostics_path,
        output_dir,
        experiment_id,
    ):
        calls["failure_modes"] = {
            "exposure_load_features_path": exposure_load_features_path,
            "exposure_load_diagnostics_path": exposure_load_diagnostics_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_failure_mode_sprint_experiment",
        fake_run_exposure_load_failure_mode_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_failure_modes",
            "--exposure-load-failure-mode-sprint",
            "--exposure-load-features",
            str(features_path),
            "--exposure-load-diagnostics",
            str(diagnostics_path),
        ]
    )

    assert exit_code == 0
    assert calls["failure_modes"] == {
        "exposure_load_features_path": features_path,
        "exposure_load_diagnostics_path": diagnostics_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_failure_modes",
    }


def test_cli_runs_exposure_load_guardrail_policy_from_artifacts(
    tmp_path,
    monkeypatch,
):
    failure_modes_path = tmp_path / "exposure_load_failure_modes.json"
    diagnostics_path = tmp_path / "exposure_load_calibration_diagnostics.csv"
    failure_modes_path.write_text("failure modes", encoding="utf-8")
    diagnostics_path.write_text("diagnostics", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_guardrail_policy_sprint_experiment(
        exposure_load_failure_modes_path,
        exposure_load_diagnostics_path,
        output_dir,
        experiment_id,
    ):
        calls["guardrail_policy"] = {
            "exposure_load_failure_modes_path": exposure_load_failure_modes_path,
            "exposure_load_diagnostics_path": exposure_load_diagnostics_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_guardrail_policy_sprint_experiment",
        fake_run_exposure_load_guardrail_policy_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_guardrail_policy",
            "--exposure-load-guardrail-policy-sprint",
            "--exposure-load-failure-modes",
            str(failure_modes_path),
            "--exposure-load-diagnostics",
            str(diagnostics_path),
        ]
    )

    assert exit_code == 0
    assert calls["guardrail_policy"] == {
        "exposure_load_failure_modes_path": failure_modes_path,
        "exposure_load_diagnostics_path": diagnostics_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_guardrail_policy",
    }


def test_cli_runs_exposure_load_shift_context_from_artifacts(
    tmp_path,
    monkeypatch,
):
    events_path = tmp_path / "exposure_events.csv"
    participations_path = tmp_path / "exposure_participations.csv"
    features_path = tmp_path / "exposure_load_features.csv"
    diagnostics_path = tmp_path / "exposure_load_calibration_diagnostics.csv"
    failure_modes_path = tmp_path / "exposure_load_failure_modes.json"
    for path in (
        events_path,
        participations_path,
        features_path,
        diagnostics_path,
        failure_modes_path,
    ):
        path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shift_context_sprint_experiment(
        exposure_events_path,
        exposure_participations_path,
        exposure_load_features_path,
        exposure_load_diagnostics_path,
        exposure_load_failure_modes_path,
        output_dir,
        experiment_id,
    ):
        calls["shift_context"] = {
            "exposure_events_path": exposure_events_path,
            "exposure_participations_path": exposure_participations_path,
            "exposure_load_features_path": exposure_load_features_path,
            "exposure_load_diagnostics_path": exposure_load_diagnostics_path,
            "exposure_load_failure_modes_path": exposure_load_failure_modes_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shift_context_sprint_experiment",
        fake_run_exposure_load_shift_context_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shift_context",
            "--exposure-load-shift-context-sprint",
            "--exposure-events",
            str(events_path),
            "--exposure-participations",
            str(participations_path),
            "--exposure-load-features",
            str(features_path),
            "--exposure-load-diagnostics",
            str(diagnostics_path),
            "--exposure-load-failure-modes",
            str(failure_modes_path),
        ]
    )

    assert exit_code == 0
    assert calls["shift_context"] == {
        "exposure_events_path": events_path,
        "exposure_participations_path": participations_path,
        "exposure_load_features_path": features_path,
        "exposure_load_diagnostics_path": diagnostics_path,
        "exposure_load_failure_modes_path": failure_modes_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shift_context",
    }


def test_cli_runs_exposure_load_schedule_roster_from_artifacts(tmp_path, monkeypatch):
    events_path = tmp_path / "exposure_events.csv"
    participations_path = tmp_path / "exposure_participations.csv"
    shift_context_path = tmp_path / "exposure_load_shift_context.json"
    for path in (events_path, participations_path, shift_context_path):
        path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_schedule_roster_sprint_experiment(
        exposure_events_path,
        exposure_participations_path,
        exposure_load_shift_context_path,
        output_dir,
        experiment_id,
    ):
        calls["schedule_roster"] = {
            "exposure_events_path": exposure_events_path,
            "exposure_participations_path": exposure_participations_path,
            "exposure_load_shift_context_path": exposure_load_shift_context_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_schedule_roster_sprint_experiment",
        fake_run_exposure_load_schedule_roster_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_schedule_roster",
            "--exposure-load-schedule-roster-sprint",
            "--exposure-events",
            str(events_path),
            "--exposure-participations",
            str(participations_path),
            "--exposure-load-shift-context",
            str(shift_context_path),
        ]
    )

    assert exit_code == 0
    assert calls["schedule_roster"] == {
        "exposure_events_path": events_path,
        "exposure_participations_path": participations_path,
        "exposure_load_shift_context_path": shift_context_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_schedule_roster",
    }


def test_cli_runs_exposure_load_availability_capture_from_artifacts(
    tmp_path,
    monkeypatch,
):
    participations_path = tmp_path / "exposure_participations.csv"
    shift_context_path = tmp_path / "exposure_load_shift_context.json"
    for path in (participations_path, shift_context_path):
        path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_availability_capture_sprint_experiment(
        exposure_participations_path,
        exposure_load_shift_context_path,
        output_dir,
        experiment_id,
    ):
        calls["availability_capture"] = {
            "exposure_participations_path": exposure_participations_path,
            "exposure_load_shift_context_path": exposure_load_shift_context_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_availability_capture_sprint_experiment",
        fake_run_exposure_load_availability_capture_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_availability_capture",
            "--exposure-load-availability-capture-sprint",
            "--exposure-participations",
            str(participations_path),
            "--exposure-load-shift-context",
            str(shift_context_path),
        ]
    )

    assert exit_code == 0
    assert calls["availability_capture"] == {
        "exposure_participations_path": participations_path,
        "exposure_load_shift_context_path": shift_context_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_availability_capture",
    }


def test_cli_runs_exposure_load_context_decision_from_artifacts(tmp_path, monkeypatch):
    shift_context_path = tmp_path / "exposure_load_shift_context.json"
    schedule_roster_path = tmp_path / "exposure_load_schedule_roster_context.json"
    availability_path = tmp_path / "exposure_load_availability_capture.json"
    guardrail_path = tmp_path / "exposure_load_guardrail_policy.json"
    for path in (
        shift_context_path,
        schedule_roster_path,
        availability_path,
        guardrail_path,
    ):
        path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_context_decision_sprint_experiment(
        exposure_load_shift_context_path,
        exposure_load_schedule_roster_path,
        exposure_load_availability_capture_path,
        exposure_load_guardrail_policy_path,
        output_dir,
        experiment_id,
    ):
        calls["context_decision"] = {
            "exposure_load_shift_context_path": exposure_load_shift_context_path,
            "exposure_load_schedule_roster_path": exposure_load_schedule_roster_path,
            "exposure_load_availability_capture_path": (
                exposure_load_availability_capture_path
            ),
            "exposure_load_guardrail_policy_path": exposure_load_guardrail_policy_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_context_decision_sprint_experiment",
        fake_run_exposure_load_context_decision_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_context_decision",
            "--exposure-load-context-decision-sprint",
            "--exposure-load-shift-context",
            str(shift_context_path),
            "--exposure-load-schedule-roster",
            str(schedule_roster_path),
            "--exposure-load-availability-capture",
            str(availability_path),
            "--exposure-load-guardrail-policy",
            str(guardrail_path),
        ]
    )

    assert exit_code == 0
    assert calls["context_decision"] == {
        "exposure_load_shift_context_path": shift_context_path,
        "exposure_load_schedule_roster_path": schedule_roster_path,
        "exposure_load_availability_capture_path": availability_path,
        "exposure_load_guardrail_policy_path": guardrail_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_context_decision",
    }


def test_cli_runs_exposure_load_source_context_classification_from_artifacts(
    tmp_path,
    monkeypatch,
):
    events_path = tmp_path / "exposure_events.csv"
    participations_path = tmp_path / "exposure_participations.csv"
    shift_context_path = tmp_path / "exposure_load_shift_context.json"
    schedule_roster_path = tmp_path / "exposure_load_schedule_roster_context.json"
    availability_path = tmp_path / "exposure_load_availability_capture.json"
    decision_path = tmp_path / "exposure_load_context_decision.json"
    for path in (
        events_path,
        participations_path,
        shift_context_path,
        schedule_roster_path,
        availability_path,
        decision_path,
    ):
        path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_source_context_classification_sprint_experiment(
        exposure_events_path,
        exposure_participations_path,
        exposure_load_shift_context_path,
        exposure_load_schedule_roster_path,
        exposure_load_availability_capture_path,
        exposure_load_context_decision_path,
        output_dir,
        experiment_id,
    ):
        calls["source_context_classification"] = {
            "exposure_events_path": exposure_events_path,
            "exposure_participations_path": exposure_participations_path,
            "exposure_load_shift_context_path": exposure_load_shift_context_path,
            "exposure_load_schedule_roster_path": exposure_load_schedule_roster_path,
            "exposure_load_availability_capture_path": (
                exposure_load_availability_capture_path
            ),
            "exposure_load_context_decision_path": exposure_load_context_decision_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_source_context_classification_sprint_experiment",
        fake_run_exposure_load_source_context_classification_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_source_context_classification",
            "--exposure-load-source-context-classification-sprint",
            "--exposure-events",
            str(events_path),
            "--exposure-participations",
            str(participations_path),
            "--exposure-load-shift-context",
            str(shift_context_path),
            "--exposure-load-schedule-roster",
            str(schedule_roster_path),
            "--exposure-load-availability-capture",
            str(availability_path),
            "--exposure-load-context-decision",
            str(decision_path),
        ]
    )

    assert exit_code == 0
    assert calls["source_context_classification"] == {
        "exposure_events_path": events_path,
        "exposure_participations_path": participations_path,
        "exposure_load_shift_context_path": shift_context_path,
        "exposure_load_schedule_roster_path": schedule_roster_path,
        "exposure_load_availability_capture_path": availability_path,
        "exposure_load_context_decision_path": decision_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_source_context_classification",
    }


def test_cli_runs_exposure_load_source_resolution_from_artifacts(
    tmp_path,
    monkeypatch,
):
    source_context_path = tmp_path / "exposure_load_source_context_classification.json"
    source_context_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_source_resolution_sprint_experiment(
        exposure_load_source_context_classification_path,
        output_dir,
        experiment_id,
    ):
        calls["source_resolution"] = {
            "exposure_load_source_context_classification_path": (
                exposure_load_source_context_classification_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_source_resolution_sprint_experiment",
        fake_run_exposure_load_source_resolution_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_source_resolution",
            "--exposure-load-source-resolution-sprint",
            "--exposure-load-source-context-classification",
            str(source_context_path),
        ]
    )

    assert exit_code == 0
    assert calls["source_resolution"] == {
        "exposure_load_source_context_classification_path": source_context_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_source_resolution",
    }


def test_cli_runs_exposure_load_source_eligible_calibration_from_artifacts(
    tmp_path,
    monkeypatch,
):
    validation_path = tmp_path / "exposure_load_season_forward_validation.csv"
    source_resolution_path = tmp_path / "exposure_load_source_resolution_policy.json"
    validation_path.write_text("artifact", encoding="utf-8")
    source_resolution_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_source_eligible_calibration_sprint_experiment(
        season_forward_validation_path,
        exposure_load_source_resolution_policy_path,
        output_dir,
        experiment_id,
    ):
        calls["source_eligible_calibration"] = {
            "season_forward_validation_path": season_forward_validation_path,
            "exposure_load_source_resolution_policy_path": (
                exposure_load_source_resolution_policy_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_source_eligible_calibration_sprint_experiment",
        fake_run_exposure_load_source_eligible_calibration_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_source_eligible_calibration",
            "--exposure-load-source-eligible-calibration-sprint",
            "--season-forward-validation-path",
            str(validation_path),
            "--exposure-load-source-resolution-policy",
            str(source_resolution_path),
        ]
    )

    assert exit_code == 0
    assert calls["source_eligible_calibration"] == {
        "season_forward_validation_path": validation_path,
        "exposure_load_source_resolution_policy_path": source_resolution_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_source_eligible_calibration",
    }


def test_cli_runs_exposure_load_source_eligible_policy_from_artifacts(
    tmp_path,
    monkeypatch,
):
    validation_path = tmp_path / "exposure_load_season_forward_validation.csv"
    calibration_path = tmp_path / "exposure_load_source_eligible_calibration.json"
    validation_path.write_text("artifact", encoding="utf-8")
    calibration_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_source_eligible_policy_sprint_experiment(
        season_forward_validation_path,
        exposure_load_source_eligible_calibration_path,
        output_dir,
        experiment_id,
    ):
        calls["source_eligible_policy"] = {
            "season_forward_validation_path": season_forward_validation_path,
            "exposure_load_source_eligible_calibration_path": (
                exposure_load_source_eligible_calibration_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_source_eligible_policy_sprint_experiment",
        fake_run_exposure_load_source_eligible_policy_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_source_eligible_policy",
            "--exposure-load-source-eligible-policy-sprint",
            "--season-forward-validation-path",
            str(validation_path),
            "--exposure-load-source-eligible-calibration",
            str(calibration_path),
        ]
    )

    assert exit_code == 0
    assert calls["source_eligible_policy"] == {
        "season_forward_validation_path": validation_path,
        "exposure_load_source_eligible_calibration_path": calibration_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_source_eligible_policy",
    }


def test_cli_runs_exposure_load_source_eligible_shadow_monitoring_from_artifacts(
    tmp_path,
    monkeypatch,
):
    validation_path = tmp_path / "exposure_load_season_forward_validation.csv"
    policy_path = tmp_path / "exposure_load_source_eligible_policy.json"
    validation_path.write_text("artifact", encoding="utf-8")
    policy_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_source_eligible_shadow_monitoring_sprint_experiment(
        season_forward_validation_path,
        exposure_load_source_eligible_policy_path,
        output_dir,
        experiment_id,
    ):
        calls["source_eligible_shadow_monitoring"] = {
            "season_forward_validation_path": season_forward_validation_path,
            "exposure_load_source_eligible_policy_path": (
                exposure_load_source_eligible_policy_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_source_eligible_shadow_monitoring_sprint_experiment",
        fake_run_exposure_load_source_eligible_shadow_monitoring_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_source_eligible_shadow_monitoring",
            "--exposure-load-source-eligible-shadow-monitoring-sprint",
            "--season-forward-validation-path",
            str(validation_path),
            "--exposure-load-source-eligible-policy",
            str(policy_path),
        ]
    )

    assert exit_code == 0
    assert calls["source_eligible_shadow_monitoring"] == {
        "season_forward_validation_path": validation_path,
        "exposure_load_source_eligible_policy_path": policy_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_source_eligible_shadow_monitoring",
    }


def test_cli_runs_exposure_load_shadow_launch_chain_from_artifacts(
    tmp_path,
    monkeypatch,
):
    monitoring_path = tmp_path / "exposure_load_source_eligible_shadow_monitoring.json"
    channel_lock_path = tmp_path / "exposure_load_shadow_channel_lock.json"
    protocol_path = tmp_path / "exposure_load_shadow_review_protocol.json"
    monitoring_path.write_text("artifact", encoding="utf-8")
    channel_lock_path.write_text("artifact", encoding="utf-8")
    protocol_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_channel_lock(
        exposure_load_source_eligible_shadow_monitoring_path,
        output_dir,
        experiment_id,
    ):
        calls["channel_lock"] = {
            "exposure_load_source_eligible_shadow_monitoring_path": (
                exposure_load_source_eligible_shadow_monitoring_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    def fake_review_protocol(
        exposure_load_shadow_channel_lock_path,
        output_dir,
        experiment_id,
    ):
        calls["review_protocol"] = {
            "exposure_load_shadow_channel_lock_path": (
                exposure_load_shadow_channel_lock_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    def fake_readiness_register(
        exposure_load_shadow_channel_lock_path,
        exposure_load_shadow_review_protocol_path,
        output_dir,
        experiment_id,
    ):
        calls["readiness_register"] = {
            "exposure_load_shadow_channel_lock_path": (
                exposure_load_shadow_channel_lock_path
            ),
            "exposure_load_shadow_review_protocol_path": (
                exposure_load_shadow_review_protocol_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_channel_lock_sprint_experiment",
        fake_channel_lock,
    )
    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_review_protocol_sprint_experiment",
        fake_review_protocol,
    )
    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_readiness_register_sprint_experiment",
        fake_readiness_register,
    )

    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--experiment-id",
                "exposure_load_shadow_channel_lock",
                "--exposure-load-shadow-channel-lock-sprint",
                "--exposure-load-source-eligible-shadow-monitoring",
                str(monitoring_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--experiment-id",
                "exposure_load_shadow_review_protocol",
                "--exposure-load-shadow-review-protocol-sprint",
                "--exposure-load-shadow-channel-lock",
                str(channel_lock_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--experiment-id",
                "exposure_load_shadow_readiness_register",
                "--exposure-load-shadow-readiness-register-sprint",
                "--exposure-load-shadow-channel-lock",
                str(channel_lock_path),
                "--exposure-load-shadow-review-protocol",
                str(protocol_path),
            ]
        )
        == 0
    )

    assert calls["channel_lock"] == {
        "exposure_load_source_eligible_shadow_monitoring_path": monitoring_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_channel_lock",
    }
    assert calls["review_protocol"] == {
        "exposure_load_shadow_channel_lock_path": channel_lock_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_review_protocol",
    }
    assert calls["readiness_register"] == {
        "exposure_load_shadow_channel_lock_path": channel_lock_path,
        "exposure_load_shadow_review_protocol_path": protocol_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_readiness_register",
    }


def test_cli_runs_exposure_load_shadow_replay_from_artifacts(
    tmp_path,
    monkeypatch,
):
    validation_path = tmp_path / "exposure_load_season_forward_validation.csv"
    channel_lock_path = tmp_path / "exposure_load_shadow_channel_lock.json"
    protocol_path = tmp_path / "exposure_load_shadow_review_protocol.json"
    validation_path.write_text("artifact", encoding="utf-8")
    channel_lock_path.write_text("artifact", encoding="utf-8")
    protocol_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_replay_sprint_experiment(
        season_forward_validation_path,
        exposure_load_shadow_channel_lock_path,
        exposure_load_shadow_review_protocol_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_replay"] = {
            "season_forward_validation_path": season_forward_validation_path,
            "exposure_load_shadow_channel_lock_path": (
                exposure_load_shadow_channel_lock_path
            ),
            "exposure_load_shadow_review_protocol_path": (
                exposure_load_shadow_review_protocol_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_replay_sprint_experiment",
        fake_run_exposure_load_shadow_replay_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_replay",
            "--exposure-load-shadow-replay-sprint",
            "--season-forward-validation-path",
            str(validation_path),
            "--exposure-load-shadow-channel-lock",
            str(channel_lock_path),
            "--exposure-load-shadow-review-protocol",
            str(protocol_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_replay"] == {
        "season_forward_validation_path": validation_path,
        "exposure_load_shadow_channel_lock_path": channel_lock_path,
        "exposure_load_shadow_review_protocol_path": protocol_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_replay",
    }


def test_cli_runs_exposure_load_shadow_adjudication_from_replay_artifact(
    tmp_path,
    monkeypatch,
):
    replay_path = tmp_path / "exposure_load_shadow_replay.json"
    replay_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_adjudication_sprint_experiment(
        exposure_load_shadow_replay_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_adjudication"] = {
            "exposure_load_shadow_replay_path": exposure_load_shadow_replay_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_adjudication_sprint_experiment",
        fake_run_exposure_load_shadow_adjudication_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_adjudication",
            "--exposure-load-shadow-adjudication-sprint",
            "--exposure-load-shadow-replay",
            str(replay_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_adjudication"] == {
        "exposure_load_shadow_replay_path": replay_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_adjudication",
    }


def test_cli_runs_exposure_load_shadow_adjudication_summary_from_completed_file(
    tmp_path,
    monkeypatch,
):
    adjudication_path = tmp_path / "completed_adjudication.csv"
    adjudication_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_adjudication_summary_sprint_experiment(
        exposure_load_shadow_adjudication_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_adjudication_summary"] = {
            "exposure_load_shadow_adjudication_path": (
                exposure_load_shadow_adjudication_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_adjudication_summary_sprint_experiment",
        fake_run_exposure_load_shadow_adjudication_summary_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_adjudication_summary",
            "--exposure-load-shadow-adjudication-summary-sprint",
            "--exposure-load-shadow-adjudication",
            str(adjudication_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_adjudication_summary"] == {
        "exposure_load_shadow_adjudication_path": adjudication_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_adjudication_summary",
    }


def test_cli_runs_exposure_load_shadow_adjudication_decision_from_summary(
    tmp_path,
    monkeypatch,
):
    summary_path = tmp_path / "exposure_load_shadow_adjudication_summary.json"
    summary_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_adjudication_decision_sprint_experiment(
        exposure_load_shadow_adjudication_summary_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_adjudication_decision"] = {
            "exposure_load_shadow_adjudication_summary_path": (
                exposure_load_shadow_adjudication_summary_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_adjudication_decision_sprint_experiment",
        fake_run_exposure_load_shadow_adjudication_decision_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_adjudication_decision",
            "--exposure-load-shadow-adjudication-decision-sprint",
            "--exposure-load-shadow-adjudication-summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_adjudication_decision"] == {
        "exposure_load_shadow_adjudication_summary_path": summary_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_adjudication_decision",
    }


def test_cli_runs_exposure_load_shadow_calibration_sensitivity_from_artifacts(
    tmp_path,
    monkeypatch,
):
    readiness_path = tmp_path / "exposure_load_shadow_calibration_readiness.json"
    collection_path = tmp_path / "exposure_load_shadow_collection.csv"
    crosswalk_path = tmp_path / "exposure_load_shadow_event_crosswalk.csv"
    for path in (readiness_path, collection_path, crosswalk_path):
        path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_calibration_sensitivity_sprint_experiment(
        exposure_load_shadow_calibration_readiness_path,
        exposure_load_shadow_collection_path,
        exposure_load_shadow_event_crosswalk_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_calibration_sensitivity"] = {
            "exposure_load_shadow_calibration_readiness_path": (
                exposure_load_shadow_calibration_readiness_path
            ),
            "exposure_load_shadow_collection_path": (
                exposure_load_shadow_collection_path
            ),
            "exposure_load_shadow_event_crosswalk_path": (
                exposure_load_shadow_event_crosswalk_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_calibration_sensitivity_sprint_experiment",
        fake_run_exposure_load_shadow_calibration_sensitivity_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_calibration_sensitivity",
            "--exposure-load-shadow-calibration-sensitivity-sprint",
            "--exposure-load-shadow-calibration-readiness",
            str(readiness_path),
            "--exposure-load-shadow-collection",
            str(collection_path),
            "--exposure-load-shadow-event-crosswalk",
            str(crosswalk_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_calibration_sensitivity"] == {
        "exposure_load_shadow_calibration_readiness_path": readiness_path,
        "exposure_load_shadow_collection_path": collection_path,
        "exposure_load_shadow_event_crosswalk_path": crosswalk_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_calibration_sensitivity",
    }


def test_cli_runs_exposure_load_shadow_error_control_from_sensitivity(
    tmp_path,
    monkeypatch,
):
    sensitivity_path = tmp_path / "exposure_load_shadow_calibration_sensitivity.json"
    sensitivity_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_error_control_sprint_experiment(
        exposure_load_shadow_calibration_sensitivity_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_error_control"] = {
            "exposure_load_shadow_calibration_sensitivity_path": (
                exposure_load_shadow_calibration_sensitivity_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_error_control_sprint_experiment",
        fake_run_exposure_load_shadow_error_control_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_error_control",
            "--exposure-load-shadow-error-control-sprint",
            "--exposure-load-shadow-calibration-sensitivity",
            str(sensitivity_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_error_control"] == {
        "exposure_load_shadow_calibration_sensitivity_path": sensitivity_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_error_control",
    }


def test_cli_runs_exposure_load_shadow_bounded_calibration_protocol_from_policy(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "exposure_load_shadow_error_control_policy.json"
    policy_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_bounded_calibration_protocol_sprint_experiment(
        exposure_load_shadow_error_control_policy_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_bounded_calibration_protocol"] = {
            "exposure_load_shadow_error_control_policy_path": (
                exposure_load_shadow_error_control_policy_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_bounded_calibration_protocol_sprint_experiment",
        fake_run_exposure_load_shadow_bounded_calibration_protocol_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_bounded_calibration_protocol",
            "--exposure-load-shadow-bounded-calibration-protocol-sprint",
            "--exposure-load-shadow-error-control-policy",
            str(policy_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_bounded_calibration_protocol"] == {
        "exposure_load_shadow_error_control_policy_path": policy_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_bounded_calibration_protocol",
    }


def test_cli_runs_exposure_load_shadow_bounded_calibration_stress_test_from_protocol(
    tmp_path,
    monkeypatch,
):
    protocol_path = (
        tmp_path / "exposure_load_shadow_bounded_calibration_protocol.json"
    )
    protocol_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_bounded_calibration_stress_test_sprint_experiment(
        exposure_load_shadow_bounded_calibration_protocol_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_bounded_calibration_stress_test"] = {
            "exposure_load_shadow_bounded_calibration_protocol_path": (
                exposure_load_shadow_bounded_calibration_protocol_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_bounded_calibration_stress_test_sprint_experiment",
        fake_run_exposure_load_shadow_bounded_calibration_stress_test_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_bounded_calibration_stress_test",
            "--exposure-load-shadow-bounded-calibration-stress-test-sprint",
            "--exposure-load-shadow-bounded-calibration-protocol",
            str(protocol_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_bounded_calibration_stress_test"] == {
        "exposure_load_shadow_bounded_calibration_protocol_path": protocol_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_bounded_calibration_stress_test",
    }


def test_cli_runs_exposure_load_shadow_monitoring_plan_from_decision(
    tmp_path,
    monkeypatch,
):
    decision_path = tmp_path / "exposure_load_shadow_adjudication_decision.json"
    decision_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_monitoring_plan_sprint_experiment(
        exposure_load_shadow_adjudication_decision_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_monitoring_plan"] = {
            "exposure_load_shadow_adjudication_decision_path": (
                exposure_load_shadow_adjudication_decision_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_monitoring_plan_sprint_experiment",
        fake_run_exposure_load_shadow_monitoring_plan_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_monitoring_plan",
            "--exposure-load-shadow-monitoring-plan-sprint",
            "--exposure-load-shadow-adjudication-decision",
            str(decision_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_monitoring_plan"] == {
        "exposure_load_shadow_adjudication_decision_path": decision_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_monitoring_plan",
    }


def test_cli_runs_exposure_load_shadow_collection_template_from_monitoring_plan(
    tmp_path,
    monkeypatch,
):
    monitoring_plan_path = tmp_path / "exposure_load_shadow_monitoring_plan.json"
    monitoring_plan_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_collection_template_sprint_experiment(
        exposure_load_shadow_monitoring_plan_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_collection_template"] = {
            "exposure_load_shadow_monitoring_plan_path": (
                exposure_load_shadow_monitoring_plan_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_collection_template_sprint_experiment",
        fake_run_exposure_load_shadow_collection_template_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_collection_template",
            "--exposure-load-shadow-collection-template-sprint",
            "--exposure-load-shadow-monitoring-plan",
            str(monitoring_plan_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_collection_template"] == {
        "exposure_load_shadow_monitoring_plan_path": monitoring_plan_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_collection_template",
    }


def test_cli_runs_exposure_load_shadow_collection_summary_from_collection_csv(
    tmp_path,
    monkeypatch,
):
    collection_path = tmp_path / "exposure_load_shadow_collection_template.csv"
    collection_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_collection_summary_sprint_experiment(
        exposure_load_shadow_collection_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_collection_summary"] = {
            "exposure_load_shadow_collection_path": (
                exposure_load_shadow_collection_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_collection_summary_sprint_experiment",
        fake_run_exposure_load_shadow_collection_summary_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_collection_summary",
            "--exposure-load-shadow-collection-summary-sprint",
            "--exposure-load-shadow-collection",
            str(collection_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_collection_summary"] == {
        "exposure_load_shadow_collection_path": collection_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_collection_summary",
    }


def test_cli_runs_exposure_load_shadow_collection_packet_workflow_from_collection_csv(
    tmp_path,
    monkeypatch,
):
    collection_path = tmp_path / "exposure_load_shadow_collection_template.csv"
    collection_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_collection_packet_workflow_sprint_experiment(
        exposure_load_shadow_collection_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_collection_packet_workflow"] = {
            "exposure_load_shadow_collection_path": (
                exposure_load_shadow_collection_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_collection_packet_workflow_sprint_experiment",
        fake_run_exposure_load_shadow_collection_packet_workflow_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_collection_packet_workflow",
            "--exposure-load-shadow-collection-packet-workflow-sprint",
            "--exposure-load-shadow-collection",
            str(collection_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_collection_packet_workflow"] == {
        "exposure_load_shadow_collection_path": collection_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_collection_packet_workflow",
    }


def test_cli_runs_exposure_load_shadow_collection_evidence_prefill_from_review_packets(
    tmp_path,
    monkeypatch,
):
    review_packets_path = tmp_path / "exposure_load_shadow_review_packets.csv"
    review_packets_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_collection_evidence_prefill_sprint_experiment(
        exposure_load_shadow_review_packets_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_collection_evidence_prefill"] = {
            "exposure_load_shadow_review_packets_path": (
                exposure_load_shadow_review_packets_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        result = output_dir / "experiments" / experiment_id
        result.mkdir(parents=True)
        return result

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_collection_evidence_prefill_sprint_experiment",
        fake_run_exposure_load_shadow_collection_evidence_prefill_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_collection_evidence_prefill",
            "--exposure-load-shadow-collection-evidence-prefill-sprint",
            "--exposure-load-shadow-review-packets",
            str(review_packets_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_collection_evidence_prefill"] == {
        "exposure_load_shadow_review_packets_path": review_packets_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_collection_evidence_prefill",
    }


def test_cli_runs_exposure_load_shadow_calibration_readiness_from_collection_summary(
    tmp_path,
    monkeypatch,
):
    collection_summary_path = (
        tmp_path / "exposure_load_shadow_collection_summary.json"
    )
    collection_summary_path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_calibration_readiness_sprint_experiment(
        exposure_load_shadow_collection_summary_path,
        output_dir,
        experiment_id,
    ):
        calls["shadow_calibration_readiness"] = {
            "exposure_load_shadow_collection_summary_path": (
                exposure_load_shadow_collection_summary_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_calibration_readiness_sprint_experiment",
        fake_run_exposure_load_shadow_calibration_readiness_sprint_experiment,
    )

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_calibration_readiness",
            "--exposure-load-shadow-calibration-readiness-sprint",
            "--exposure-load-shadow-collection-summary",
            str(collection_summary_path),
        ]
    )

    assert exit_code == 0
    assert calls["shadow_calibration_readiness"] == {
        "exposure_load_shadow_collection_summary_path": collection_summary_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_calibration_readiness",
    }


def test_cli_runs_exposure_load_shadow_event_crosswalk_from_replay_and_live_inputs(
    tmp_path,
    monkeypatch,
):
    measurements_path = tmp_path / "canonical_measurements.csv"
    injuries_path = tmp_path / "canonical_injuries.csv"
    detailed_path = tmp_path / "injury_events_detailed.csv"
    exposure_participations_path = tmp_path / "exposure_participations.csv"
    shadow_replay_path = tmp_path / "exposure_load_shadow_replay.json"
    collection_path = tmp_path / "exposure_load_shadow_collection_prefilled.csv"
    for path in (
        measurements_path,
        injuries_path,
        detailed_path,
        exposure_participations_path,
        shadow_replay_path,
        collection_path,
    ):
        path.write_text("artifact", encoding="utf-8")
    calls = {}

    def fake_run_exposure_load_shadow_event_crosswalk_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        exposure_participations_path,
        exposure_load_shadow_replay_path,
        exposure_load_shadow_collection_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["shadow_event_crosswalk"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "exposure_participations_path": exposure_participations_path,
            "exposure_load_shadow_replay_path": exposure_load_shadow_replay_path,
            "exposure_load_shadow_collection_path": (
                exposure_load_shadow_collection_path
            ),
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(
        cli,
        "run_exposure_load_shadow_event_crosswalk_sprint_experiment",
        fake_run_exposure_load_shadow_event_crosswalk_sprint_experiment,
    )

    exit_code = main(
        [
            "--measurements",
            str(measurements_path),
            "--injuries",
            str(injuries_path),
            "--exposure-participations",
            str(exposure_participations_path),
            "--exposure-load-shadow-replay",
            str(shadow_replay_path),
            "--exposure-load-shadow-collection",
            str(collection_path),
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "exposure_load_shadow_event_crosswalk",
            "--graph-window-size",
            "2",
            "--model-variant",
            "l2",
            "--exposure-load-shadow-event-crosswalk-sprint",
        ]
    )

    assert exit_code == 0
    assert calls["shadow_event_crosswalk"] == {
        "measurements_path": measurements_path,
        "injuries_path": injuries_path,
        "detailed_injuries_path": detailed_path,
        "exposure_participations_path": exposure_participations_path,
        "exposure_load_shadow_replay_path": shadow_replay_path,
        "exposure_load_shadow_collection_path": collection_path,
        "output_dir": tmp_path,
        "experiment_id": "exposure_load_shadow_event_crosswalk",
        "graph_window_size": 2,
        "model_variant": "l2",
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


def test_cli_runs_shadow_mode_stability_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(paths, output_dir):
        calls["paths"] = paths
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

    def fake_run_shadow_mode_stability_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["shadow_mode"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
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
        "run_shadow_mode_stability_experiment",
        fake_run_shadow_mode_stability_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "shadow_mode_stability_run",
            "--shadow-mode-stability",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["shadow_mode"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "shadow_mode_stability_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "shadow_mode_stability_run"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "shadow_mode_stability_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "shadow_mode_stability_run",
        "model_variant": "l2",
    }


def test_cli_runs_season_drift_diagnostic_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(paths, output_dir):
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

    def fake_run_season_drift_diagnostic_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["season_drift"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
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
        "run_season_drift_diagnostic_experiment",
        fake_run_season_drift_diagnostic_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "season_drift_diagnostic_run",
            "--season-drift-diagnostic",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["season_drift"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "season_drift_diagnostic_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "season_drift_diagnostic_run"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "season_drift_diagnostic_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "season_drift_diagnostic_run",
        "model_variant": "l2",
    }


def test_cli_runs_coverage_stratified_evaluation_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_coverage_stratified_evaluation_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["coverage"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli, "prepare_live_source_inputs", fake_prepare_live_source_inputs
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_stratified_evaluation_experiment",
        fake_run_coverage_stratified_evaluation_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "coverage_eval_run",
            "--coverage-stratified-evaluation",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["coverage"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "coverage_eval_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "coverage_eval_run"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "coverage_eval_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "coverage_eval_run",
        "model_variant": "l2",
    }


def test_cli_runs_coverage_normalized_policy_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_coverage_normalized_policy_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["coverage_normalized"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli, "prepare_live_source_inputs", fake_prepare_live_source_inputs
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_normalized_policy_sprint_experiment",
        fake_run_coverage_normalized_policy_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "coverage_normalized_run",
            "--coverage-normalized-policy-sprint",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["coverage_normalized"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "coverage_normalized_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "coverage_normalized_run"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "coverage_normalized_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "coverage_normalized_run",
        "model_variant": "l2",
    }


def test_cli_runs_coverage_source_aware_model_sprint(tmp_path, monkeypatch):
    calls = {}

    def fake_run_coverage_source_aware_model_sprint_experiment(
        measurements_path,
        injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["coverage_source_model"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(
        cli,
        "run_coverage_source_aware_model_sprint_experiment",
        fake_run_coverage_source_aware_model_sprint_experiment,
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
            "coverage_source_model",
            "--coverage-source-aware-model-sprint",
            "--graph-window-size",
            "2",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["coverage_source_model"] == {
        "measurements_path": FIXTURES / "measurements.csv",
        "injuries_path": FIXTURES / "injuries.csv",
        "output_dir": tmp_path,
        "experiment_id": "coverage_source_model",
        "graph_window_size": 2,
        "model_variant": "l2",
    }


def test_cli_runs_coverage_adjusted_threshold_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_coverage_adjusted_threshold_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["coverage_adjusted_threshold"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_adjusted_threshold_sprint_experiment",
        fake_run_coverage_adjusted_threshold_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "coverage_adjusted_threshold",
            "--coverage-adjusted-threshold-sprint",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["coverage_adjusted_threshold"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "coverage_adjusted_threshold"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "coverage_adjusted_threshold"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "coverage_adjusted_threshold"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "coverage_adjusted_threshold",
        "model_variant": "l2",
    }


def test_cli_runs_season_forward_validation_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_season_forward_validation_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["season_forward_validation"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_season_forward_validation_sprint_experiment",
        fake_run_season_forward_validation_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "season_forward_validation",
            "--season-forward-validation",
            "--model-variant",
            "l2",
            "--graph-window-size",
            "4",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["season_forward_validation"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "season_forward_validation"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "season_forward_validation"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "season_forward_validation"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "season_forward_validation",
        "graph_window_size": 4,
        "model_variant": "l2",
    }


def test_cli_runs_forward_case_review_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_forward_case_review_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["forward_case_review"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_forward_case_review_sprint_experiment",
        fake_run_forward_case_review_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "forward_case_review",
            "--forward-case-review-sprint",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["forward_case_review"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "forward_case_review"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "forward_case_review"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "forward_case_review"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "forward_case_review",
        "model_variant": "l2",
    }


def test_cli_runs_case_diagnostic_requirements_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_case_diagnostic_requirements_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["case_diagnostic_requirements"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_case_diagnostic_requirements_sprint_experiment",
        fake_run_case_diagnostic_requirements_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "case_diagnostic_requirements",
            "--case-diagnostic-requirements-sprint",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["case_diagnostic_requirements"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "case_diagnostic_requirements"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "case_diagnostic_requirements"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "case_diagnostic_requirements"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "case_diagnostic_requirements",
        "model_variant": "l2",
    }


def test_cli_runs_injury_history_feature_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_injury_history_feature_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["injury_history_feature"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_injury_history_feature_sprint_experiment",
        fake_run_injury_history_feature_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "injury_history_feature",
            "--injury-history-feature-sprint",
            "--graph-window-size",
            "4",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["injury_history_feature"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "injury_history_feature"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "injury_history_feature"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "injury_history_feature"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "injury_history_feature",
        "graph_window_size": 4,
        "model_variant": "l2",
    }


def test_cli_runs_injury_history_season_forward_validation_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_injury_history_season_forward_validation_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["injury_history_season_forward"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_injury_history_season_forward_validation_sprint_experiment",
        fake_run_injury_history_season_forward_validation_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "injury_history_season_forward",
            "--injury-history-season-forward-validation",
            "--graph-window-size",
            "4",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["injury_history_season_forward"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "injury_history_season_forward"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "injury_history_season_forward"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "injury_history_season_forward"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "injury_history_season_forward",
        "graph_window_size": 4,
        "model_variant": "l2",
    }


def test_cli_runs_injury_history_forward_diagnostic_sprint_from_live_sources(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_load_data_source_paths(config_path):
        calls["config_path"] = config_path
        return object()

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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

    def fake_run_injury_history_forward_diagnostic_sprint_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        graph_window_size,
        model_variant,
    ):
        calls["injury_history_forward_diagnostic"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(cli, "load_data_source_paths", fake_load_data_source_paths)
    monkeypatch.setattr(
        cli,
        "prepare_live_source_inputs",
        fake_prepare_live_source_inputs,
    )
    monkeypatch.setattr(
        cli,
        "run_injury_history_forward_diagnostic_sprint_experiment",
        fake_run_injury_history_forward_diagnostic_sprint_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "injury_history_forward_diagnostic",
            "--injury-history-forward-diagnostic-sprint",
            "--graph-window-size",
            "4",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == Path("config/paths.local.yaml")
    assert calls["injury_history_forward_diagnostic"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "injury_history_forward_diagnostic"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "injury_history_forward_diagnostic"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "injury_history_forward_diagnostic"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "injury_history_forward_diagnostic",
        "graph_window_size": 4,
        "model_variant": "l2",
    }
