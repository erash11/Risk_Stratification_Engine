from pathlib import Path

import json
import pandas as pd
import pytest

from risk_stratification_engine.experiments import (
    run_research_experiment,
    run_window_sensitivity_experiment,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_run_research_experiment_writes_artifacts(tmp_path):
    result = run_research_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="fixture_research_run",
        graph_window_size=2,
    )

    experiment_dir = tmp_path / "experiments" / "fixture_research_run"
    assert result == experiment_dir
    assert (experiment_dir / "config.json").exists()
    assert (experiment_dir / "model_evaluation.json").exists()
    assert (experiment_dir / "model_metrics.json").exists()
    assert (experiment_dir / "model_summary.json").exists()
    assert (experiment_dir / "feature_attribution.json").exists()
    assert (experiment_dir / "feature_ablation_report.md").exists()
    assert (experiment_dir / "experiment_report.md").exists()
    assert (experiment_dir / "athlete_risk_timeline.csv").exists()
    assert (experiment_dir / "graph_snapshots" / "graph_features.csv").exists()
    assert (experiment_dir / "explanations" / "explanation_summary.csv").exists()

    metrics = json.loads((experiment_dir / "model_metrics.json").read_text())
    assert metrics["athlete_count"] == 2
    assert metrics["snapshot_count"] == 4
    assert metrics["observed_event_count"] == 1
    assert metrics["model_type"] == "discrete_time_logistic_baseline"

    model_summary = json.loads((experiment_dir / "model_summary.json").read_text())
    assert model_summary["feature_columns"] == [
        "time_index",
        "node_count",
        "edge_count",
        "mean_abs_correlation",
        "edge_density",
        "delta_edge_count",
        "delta_mean_abs_correlation",
        "delta_edge_density",
        "graph_instability",
        "z_mean_abs_correlation",
        "z_edge_density",
        "z_edge_count",
        "z_graph_instability",
    ]
    assert model_summary["event_policy"] in {
        "event_observed",
        "primary_model_event",
    }
    model_evaluation = json.loads(
        (experiment_dir / "model_evaluation.json").read_text()
    )
    assert model_evaluation["model_type"] == "discrete_time_logistic_baseline"
    assert model_evaluation["horizons"]["7"]["prevalence_baseline_risk"] is not None

    feature_attribution = json.loads(
        (experiment_dir / "feature_attribution.json").read_text()
    )
    assert set(feature_attribution["feature_sets"]) == {
        "full_13",
        "original_9",
        "z_score_only",
    }
    assert feature_attribution["feature_sets"]["full_13"]["feature_columns"] == (
        model_summary["feature_columns"]
    )
    assert feature_attribution["feature_sets"]["z_score_only"]["feature_columns"] == [
        "z_mean_abs_correlation",
        "z_edge_density",
        "z_edge_count",
        "z_graph_instability",
    ]
    full_7d = feature_attribution["feature_sets"]["full_13"]["horizons"]["7"]
    assert "evaluation" in full_7d
    assert "feature_attribution" in full_7d
    assert full_7d["feature_attribution"][0]["feature"] in model_summary[
        "feature_columns"
    ]

    timeline = pd.read_csv(experiment_dir / "athlete_risk_timeline.csv")
    assert {"risk_7d", "risk_14d", "risk_30d"}.issubset(timeline.columns)

    risk_columns = ["risk_7d", "risk_14d", "risk_30d"]
    assert timeline[risk_columns].ge(0.0).all().all()
    assert timeline[risk_columns].le(1.0).all().all()
    assert timeline.loc[~timeline["event_observed"], risk_columns].gt(0.0).any().any()

    report = (experiment_dir / "experiment_report.md").read_text()
    assert "discrete-time logistic baseline" in report
    assert "Prevalence baseline" in report
    assert "not calibrated clinical probabilities" in report

    ablation_report = (experiment_dir / "feature_ablation_report.md").read_text()
    assert "Feature Attribution And Ablation" in ablation_report
    assert "full_13" in ablation_report
    assert "z_score_only" in ablation_report


def test_run_research_experiment_counts_observed_events_in_modeled_cohort_only(
    tmp_path,
):
    measurements_path = tmp_path / "measurements.csv"
    measurements_path.write_text(
        "\n".join(
            [
                "athlete_id,date,season_id,source,metric_name,metric_value",
                "a1,2026-01-01,2026,force_plate,jump_height,42.0",
                "a1,2026-01-01,2026,force_plate,eccentric_peak_force_asymmetry,8.0",
                "a1,2026-01-02,2026,force_plate,jump_height,39.0",
                "a1,2026-01-02,2026,force_plate,eccentric_peak_force_asymmetry,14.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    injuries_path = tmp_path / "injuries.csv"
    injuries_path.write_text(
        "\n".join(
            [
                "athlete_id,season_id,injury_date,injury_type,event_observed,censor_date",
                "a1,2026,2026-01-10,lower_extremity_soft_tissue,true,2026-01-10",
                "extra,2026,2026-01-11,lower_extremity_soft_tissue,true,2026-01-11",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_research_experiment(
        measurements_path=measurements_path,
        injuries_path=injuries_path,
        output_dir=tmp_path,
        experiment_id="modeled_cohort_metrics",
        graph_window_size=2,
    )

    metrics = json.loads((result / "model_metrics.json").read_text())
    assert metrics["athlete_count"] == 1
    assert metrics["observed_event_count"] == 1


def test_run_window_sensitivity_experiment_writes_comparison_artifacts(tmp_path):
    result = run_window_sensitivity_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="window_sensitivity_fixture",
        graph_window_sizes=(2, 3),
    )

    assert result == tmp_path / "experiments" / "window_sensitivity_fixture"
    assert (result / "config.json").exists()
    assert (result / "window_sensitivity.json").exists()
    assert (result / "window_sensitivity_report.md").exists()

    sensitivity = json.loads((result / "window_sensitivity.json").read_text())
    assert sensitivity["graph_window_sizes"] == [2, 3]
    assert set(sensitivity["windows"]) == {"2", "3"}
    assert sensitivity["windows"]["2"]["feature_columns"] == [
        "time_index",
        "node_count",
        "edge_count",
        "mean_abs_correlation",
        "edge_density",
        "delta_edge_count",
        "delta_mean_abs_correlation",
        "delta_edge_density",
        "graph_instability",
        "z_mean_abs_correlation",
        "z_edge_density",
        "z_edge_count",
        "z_graph_instability",
    ]
    assert "model_brier_score" in sensitivity["best_by_horizon"]["7"]
    assert sensitivity["best_by_horizon"]["7"]["model_brier_score"][
        "graph_window_size"
    ] in {
        2,
        3,
    }

    report = (result / "window_sensitivity_report.md").read_text()
    assert "Window Sensitivity" in report
    assert "window 2" in report
    assert "window 3" in report


@pytest.mark.parametrize(
    "experiment_id",
    ["../bad", "bad/name", "bad\\name", "", Path.cwd().anchor + "bad"],
)
def test_run_research_experiment_rejects_unsafe_experiment_ids(
    tmp_path,
    experiment_id,
):
    with pytest.raises(ValueError, match="experiment_id must be a simple identifier"):
        run_research_experiment(
            measurements_path=FIXTURES / "measurements.csv",
            injuries_path=FIXTURES / "injuries.csv",
            output_dir=tmp_path,
            experiment_id=experiment_id,
            graph_window_size=2,
        )


def test_run_research_experiment_rejects_empty_measurement_input(tmp_path):
    measurements_path = tmp_path / "empty_measurements.csv"
    measurements_path.write_text(
        "athlete_id,date,season_id,source,metric_name,metric_value\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no graph snapshots produced"):
        run_research_experiment(
            measurements_path=measurements_path,
            injuries_path=FIXTURES / "injuries.csv",
            output_dir=tmp_path,
            experiment_id="empty_run",
            graph_window_size=2,
        )

    assert not (tmp_path / "experiments" / "empty_run").exists()


def test_run_research_experiment_rejects_empty_labeled_snapshots(tmp_path):
    measurements_path = tmp_path / "post_event_measurements.csv"
    measurements_path.write_text(
        "\n".join(
            [
                "athlete_id,date,season_id,source,metric_name,metric_value",
                "a1,2026-01-02,2026,force_plate,jump_height,42.0",
                "a1,2026-01-02,2026,force_plate,eccentric_peak_force_asymmetry,8.0",
                "a1,2026-01-03,2026,force_plate,jump_height,39.0",
                "a1,2026-01-03,2026,force_plate,eccentric_peak_force_asymmetry,14.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    injuries_path = tmp_path / "pre_snapshot_injuries.csv"
    injuries_path.write_text(
        "\n".join(
            [
                "athlete_id,season_id,injury_date,injury_type,event_observed,censor_date",
                "a1,2026,2026-01-01,lower_extremity_soft_tissue,true,2026-01-01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no labeled graph snapshots produced"):
        run_research_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=tmp_path,
            experiment_id="post_event_run",
            graph_window_size=2,
        )

    assert not (tmp_path / "experiments" / "post_event_run").exists()
