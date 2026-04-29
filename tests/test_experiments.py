from pathlib import Path

import json
import pandas as pd
import pytest

from risk_stratification_engine.experiments import (
    _athlete_explanations,
    _compute_snapshot_contributions,
    run_alert_episode_experiment,
    run_calibration_threshold_experiment,
    run_model_robustness_experiment,
    run_research_experiment,
    run_window_model_robustness_experiment,
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


def test_run_research_experiment_accepts_regularized_model_variant(tmp_path):
    result = run_research_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="regularized_research_run",
        graph_window_size=2,
        model_variant="l2",
    )

    config = json.loads((result / "config.json").read_text())
    model_summary = json.loads((result / "model_summary.json").read_text())
    feature_attribution = json.loads(
        (result / "feature_attribution.json").read_text()
    )

    assert config["model_variant"] == "l2"
    assert model_summary["model_variant"] == "l2"
    assert feature_attribution["model_variant"] == "l2"


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


def test_run_model_robustness_experiment_writes_stability_artifacts(tmp_path):
    result = run_model_robustness_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="robustness_fixture",
        graph_window_size=2,
        split_count=2,
    )

    assert result == tmp_path / "experiments" / "robustness_fixture"
    assert (result / "config.json").exists()
    assert (result / "model_robustness.json").exists()
    assert (result / "model_robustness_report.md").exists()

    robustness = json.loads((result / "model_robustness.json").read_text())
    assert robustness["model_variants"] == ["baseline", "l2", "l1", "elasticnet"]
    assert robustness["split_seeds"] == [0, 1]
    assert set(robustness["variants"]) == {
        "baseline",
        "l2",
        "l1",
        "elasticnet",
    }
    assert "summary_by_horizon" in robustness["variants"]["baseline"]
    assert "calibration" in robustness["decision_modes"]
    assert "model_brier_score" in robustness["decision_modes"]["calibration"]["7"]

    report = (result / "model_robustness_report.md").read_text()
    assert "Model Robustness Sprint" in report
    assert "baseline" in report
    assert "elasticnet" in report


def test_run_window_model_robustness_experiment_writes_cross_window_artifacts(
    tmp_path,
):
    result = run_window_model_robustness_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="window_model_robustness_fixture",
        graph_window_sizes=(2, 3),
        split_count=2,
    )

    assert result == tmp_path / "experiments" / "window_model_robustness_fixture"
    assert (result / "config.json").exists()
    assert (result / "window_model_robustness.json").exists()
    assert (result / "window_model_robustness_report.md").exists()

    robustness = json.loads((result / "window_model_robustness.json").read_text())
    assert robustness["experiment_type"] == "window_model_robustness_sprint"
    assert robustness["graph_window_sizes"] == [2, 3]
    assert set(robustness["windows"]) == {"2", "3"}
    assert set(robustness["windows"]["2"]["variants"]) == {
        "baseline",
        "l2",
        "l1",
        "elasticnet",
    }
    assert "ranking" in robustness["overall_decision_modes"]
    assert "model_brier_score" in robustness["overall_decision_modes"]["calibration"]["7"]

    report = (result / "window_model_robustness_report.md").read_text()
    assert "Window + Model Robustness Sprint" in report
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


def test_run_calibration_threshold_experiment_writes_artifacts(tmp_path):
    result = run_calibration_threshold_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="calibration_fixture",
        graph_window_size=2,
        model_variant="l2",
        split_count=2,
    )

    assert result == tmp_path / "experiments" / "calibration_fixture"
    assert (result / "config.json").exists()
    assert (result / "calibration_summary.json").exists()
    assert (result / "threshold_table.csv").exists()
    assert (result / "calibration_report.md").exists()

    import pandas as pd

    summary = json.loads((result / "calibration_summary.json").read_text())
    assert summary["experiment_type"] == "calibration_threshold"
    assert summary["model_variant"] == "l2"
    assert summary["graph_window_size"] == 2
    assert summary["split_count"] == 2
    assert set(summary["horizons"]) == {"7", "14", "30"}

    horizon_7 = summary["horizons"]["7"]
    assert "oof_snapshot_count" in horizon_7
    assert "oof_positive_count" in horizon_7
    assert "brier_score" in horizon_7
    assert "calibration_bins" in horizon_7
    assert isinstance(horizon_7["calibration_bins"], list)

    threshold_table = pd.read_csv(result / "threshold_table.csv")
    assert {"horizon", "threshold_kind", "threshold_value", "alert_count",
            "event_capture", "precision", "lift"}.issubset(threshold_table.columns)
    assert set(threshold_table["threshold_kind"]) == {"percentile", "probability"}
    assert set(threshold_table["horizon"]) == {7, 14, 30}

    report = (result / "calibration_report.md").read_text()
    assert "Calibration" in report
    assert "l2" in report


def test_run_alert_episode_experiment_writes_episode_artifacts(tmp_path):
    detailed_path = tmp_path / "injury_events_detailed.csv"
    pd.DataFrame(
        [
            {
                "injury_event_id": "inj_fixture",
                "athlete_id": "a1",
                "season_id": "2026",
                "injury_date": "2026-01-20",
                "injury_type": "lower_extremity_soft_tissue",
                "pathology": "hamstring strain",
                "classification": "soft tissue",
                "body_area": "thigh",
                "tissue_type": "muscle",
                "side": "left",
                "recurrent": "No",
                "caused_unavailability": "Yes",
                "activity_group": "training",
                "activity_group_type": "field",
                "duration_days": 14,
                "time_loss_days": 7,
                "modified_available_days": 3,
            }
        ]
    ).to_csv(detailed_path, index=False)

    result = run_alert_episode_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        detailed_injuries_path=detailed_path,
        output_dir=tmp_path,
        experiment_id="alert_episode_fixture",
        graph_window_size=2,
        model_variant="l2",
        percentile_thresholds=(0.50,),
    )

    assert result == tmp_path / "experiments" / "alert_episode_fixture"
    assert (result / "config.json").exists()
    assert (result / "model_summary.json").exists()
    assert (result / "athlete_risk_timeline.csv").exists()
    assert (result / "alert_episodes.csv").exists()
    assert (result / "alert_episodes.json").exists()
    assert (result / "alert_episode_summary.json").exists()
    assert (result / "alert_episode_report.md").exists()
    assert (result / "alert_episode_quality.csv").exists()
    assert (result / "alert_episode_quality.json").exists()
    assert (result / "alert_episode_quality_report.md").exists()
    assert (result / "qualitative_case_review.json").exists()
    assert (result / "qualitative_case_review_report.md").exists()
    assert (result / "model_improvement_diagnostics.csv").exists()
    assert (result / "model_improvement_diagnostics.json").exists()
    assert (result / "model_improvement_diagnostic_report.md").exists()
    assert (result / "injury_event_context_profiles.csv").exists()
    assert (result / "injury_context_outcomes.csv").exists()
    assert (result / "injury_context_outcomes.json").exists()
    assert (result / "injury_context_outcome_report.md").exists()

    config = json.loads((result / "config.json").read_text())
    assert config["experiment_type"] == "alert_episode_validation"
    assert config["model_variant"] == "l2"
    assert config["alert_percentile_thresholds"] == [0.5]
    assert config["detailed_injuries_path"] == str(detailed_path)

    summary = json.loads((result / "alert_episode_summary.json").read_text())
    assert summary["experiment_type"] == "alert_episode_validation"
    assert summary["model_variant"] == "l2"
    assert summary["episode_count"] >= 1
    assert set(summary["horizons"]) == {"7", "14", "30"}

    episodes_payload = json.loads((result / "alert_episodes.json").read_text())
    assert episodes_payload["episode_count"] == summary["episode_count"]
    assert "episodes" in episodes_payload
    assert {
        "athlete_id",
        "horizon_days",
        "threshold_kind",
        "start_date",
        "peak_risk",
        "top_model_features",
        "elevated_z_features",
    }.issubset(episodes_payload["episodes"][0])

    report = (result / "alert_episode_report.md").read_text()
    assert "Alert Episode Validation" in report
    assert "l2" in report

    quality = json.loads((result / "alert_episode_quality.json").read_text())
    assert quality["experiment_type"] == "alert_episode_quality_audit"
    assert quality["quality_row_count"] >= 1
    assert "threshold_overlaps" in quality
    assert "representative_cases" in quality
    assert {
        "horizon_days",
        "threshold",
        "true_positive_episode_count",
        "false_positive_episode_count",
        "unique_event_capture_rate",
        "episodes_per_athlete_season",
    }.issubset(quality["quality_rows"][0])

    quality_table = pd.read_csv(result / "alert_episode_quality.csv")
    assert set(quality_table["horizon_days"]) == {7, 14, 30}
    assert "false_positive_episode_rate" in quality_table.columns

    quality_report = (result / "alert_episode_quality_report.md").read_text()
    assert "Episode Quality Audit" in quality_report
    assert "Unique event capture" in quality_report

    case_review = json.loads((result / "qualitative_case_review.json").read_text())
    assert case_review["experiment_type"] == "qualitative_case_review"
    assert case_review["case_count"] >= 1
    assert "diagnostic_summary" in case_review
    assert {
        "case_type",
        "review_label",
        "diagnostic_label",
        "timeline_context",
    }.issubset(case_review["cases"][0])

    case_report = (result / "qualitative_case_review_report.md").read_text()
    assert "Qualitative Case Review" in case_report
    assert "Diagnostic Summary" in case_report

    improvement = json.loads(
        (result / "model_improvement_diagnostics.json").read_text()
    )
    assert improvement["experiment_type"] == "model_improvement_diagnostics"
    assert improvement["diagnostic_row_count"] >= 1
    assert {
        "comparison_group",
        "row_count",
        "recommended_next_action",
    }.issubset(improvement["diagnostic_rows"][0])

    improvement_table = pd.read_csv(result / "model_improvement_diagnostics.csv")
    assert {
        "comparison_group",
        "recommended_next_action",
    }.issubset(improvement_table.columns)

    improvement_report = (
        result / "model_improvement_diagnostic_report.md"
    ).read_text()
    assert "Model Improvement Diagnostics" in improvement_report
    assert "Recommended next action" in improvement_report

    context_payload = json.loads((result / "injury_context_outcomes.json").read_text())
    assert context_payload["experiment_type"] == "injury_context_outcomes"
    assert context_payload["event_profile_count"] >= 1
    assert context_payload["context_row_count"] >= 1
    assert {
        "context_field",
        "context_value",
        "event_count",
        "captured_after_start_count",
    }.issubset(context_payload["context_rows"][0])

    context_table = pd.read_csv(result / "injury_context_outcomes.csv")
    assert "time_loss_bucket" in pd.read_csv(
        result / "injury_event_context_profiles.csv"
    ).columns
    assert "body_area" in set(context_table["context_field"])

    context_report = (result / "injury_context_outcome_report.md").read_text()
    assert "Injury Context Outcomes" in context_report
    assert "Lowest capture contexts" in context_report


# ---------------------------------------------------------------------------
# Per-snapshot feature contributions
# ---------------------------------------------------------------------------

def _two_feature_attribution():
    return [
        {
            "feature": "edge_count",
            "coefficient": 0.5,
            "train_mean": 2.0,
            "train_std": 2.0,
            "standardized_coefficient": 1.0,
            "abs_standardized_coefficient": 1.0,
        },
        {
            "feature": "mean_abs_correlation",
            "coefficient": 0.2,
            "train_mean": 0.3,
            "train_std": 0.1,
            "standardized_coefficient": 0.02,
            "abs_standardized_coefficient": 0.02,
        },
    ]


def test_compute_snapshot_contributions_correct_math():
    row = pd.Series({"edge_count": 4.0, "mean_abs_correlation": 0.3})
    attribution = _two_feature_attribution()

    contribs = _compute_snapshot_contributions(row, attribution, ("edge_count", "mean_abs_correlation"))

    # edge_count: z = (4.0 - 2.0) / 2.0 = 1.0; contribution = 1.0 * 1.0 = 1.0
    assert contribs["edge_count"] == pytest.approx(1.0)
    # mean_abs_correlation: z = (0.3 - 0.3) / 0.1 = 0.0; contribution = 0.02 * 0.0 = 0.0
    assert contribs["mean_abs_correlation"] == pytest.approx(0.0)


def test_compute_snapshot_contributions_top_feature_by_absolute_contribution():
    row = pd.Series({"edge_count": 2.0, "mean_abs_correlation": 0.5})
    attribution = _two_feature_attribution()

    contribs = _compute_snapshot_contributions(row, attribution, ("edge_count", "mean_abs_correlation"))

    # edge_count: z = (2.0 - 2.0) / 2.0 = 0.0; contribution = 0.0
    assert contribs["edge_count"] == pytest.approx(0.0)
    # mean_abs_correlation: z = (0.5 - 0.3) / 0.1 = 2.0; contribution = 0.02 * 2.0 = 0.04
    assert contribs["mean_abs_correlation"] == pytest.approx(0.04)

    top_feature = max(contribs.items(), key=lambda kv: abs(kv[1]))[0]
    assert top_feature == "mean_abs_correlation"


def test_compute_snapshot_contributions_zero_std_returns_zero():
    attribution = [
        {
            "feature": "edge_count",
            "coefficient": 0.5,
            "train_mean": 2.0,
            "train_std": 0.0,
            "standardized_coefficient": 0.0,
            "abs_standardized_coefficient": 0.0,
        }
    ]
    row = pd.Series({"edge_count": 99.0})

    contribs = _compute_snapshot_contributions(row, attribution, ("edge_count",))

    assert contribs["edge_count"] == pytest.approx(0.0)


def test_compute_snapshot_contributions_negative_contribution():
    # Snapshot where feature is below train mean → negative contribution for positive coeff
    row = pd.Series({"edge_count": 0.0, "mean_abs_correlation": 0.3})
    attribution = _two_feature_attribution()

    contribs = _compute_snapshot_contributions(row, attribution, ("edge_count", "mean_abs_correlation"))

    # edge_count: z = (0.0 - 2.0) / 2.0 = -1.0; contribution = 1.0 * -1.0 = -1.0
    assert contribs["edge_count"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Explanation artifacts from run_research_experiment
# ---------------------------------------------------------------------------


def test_run_research_experiment_explanation_summary_uses_model_contributions(tmp_path):
    result = run_research_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="explanation_fixture",
        graph_window_size=2,
        model_variant="l2",
    )

    csv_path = result / "explanations" / "explanation_summary.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    # Model-informed contribution columns present
    assert "top_feature_7d" in df.columns
    assert "top_contribution_7d" in df.columns
    assert "top_feature_14d" in df.columns
    assert "top_feature_30d" in df.columns
    # Hard-coded signal column removed
    assert "primary_signal" not in df.columns
    # Contributions are finite numbers
    assert df["top_contribution_7d"].notna().all()


def test_run_research_experiment_writes_athlete_explanations_json(tmp_path):
    result = run_research_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="athlete_expl_fixture",
        graph_window_size=2,
        model_variant="l2",
    )

    expl_path = result / "explanations" / "athlete_explanations.json"
    assert expl_path.exists()

    payload = json.loads(expl_path.read_text())
    assert "athlete_count" in payload
    assert "athletes" in payload
    assert len(payload["athletes"]) == payload["athlete_count"]

    athlete = payload["athletes"][0]
    assert "athlete_id" in athlete
    assert "season_id" in athlete
    assert "snapshot_count" in athlete
    assert "peak_risk" in athlete
    assert set(athlete["peak_risk"]) == {"7", "14", "30"}
    assert "dominant_features" in athlete
    assert set(athlete["dominant_features"]) == {"7", "14", "30"}
    assert "snapshots" in athlete
    assert len(athlete["snapshots"]) == athlete["snapshot_count"]

    snap = athlete["snapshots"][0]
    assert "time_index" in snap
    assert "risk_7d" in snap
    assert "feature_contributions" in snap
    assert set(snap["feature_contributions"]) == {"7", "14", "30"}
    assert isinstance(snap["feature_contributions"]["7"], list)
    assert "feature" in snap["feature_contributions"]["7"][0]
    assert "contribution" in snap["feature_contributions"]["7"][0]


def _z_score_explanation_model_summary():
    feature_columns = [
        "edge_count",
        "z_mean_abs_correlation",
        "z_edge_density",
        "z_edge_count",
        "z_graph_instability",
    ]
    base_attribution = [
        {
            "feature": "edge_count",
            "coefficient": 0.1,
            "train_mean": 2.0,
            "train_std": 1.0,
            "standardized_coefficient": 0.1,
            "abs_standardized_coefficient": 0.1,
        },
        {
            "feature": "z_mean_abs_correlation",
            "coefficient": 0.4,
            "train_mean": 0.0,
            "train_std": 1.0,
            "standardized_coefficient": 0.4,
            "abs_standardized_coefficient": 0.4,
        },
        {
            "feature": "z_edge_density",
            "coefficient": -0.2,
            "train_mean": 0.0,
            "train_std": 1.0,
            "standardized_coefficient": -0.2,
            "abs_standardized_coefficient": 0.2,
        },
        {
            "feature": "z_edge_count",
            "coefficient": 0.3,
            "train_mean": 0.0,
            "train_std": 1.0,
            "standardized_coefficient": 0.3,
            "abs_standardized_coefficient": 0.3,
        },
        {
            "feature": "z_graph_instability",
            "coefficient": -0.5,
            "train_mean": 0.0,
            "train_std": 1.0,
            "standardized_coefficient": -0.5,
            "abs_standardized_coefficient": 0.5,
        },
    ]
    return {
        "feature_columns": feature_columns,
        "horizon_models": {
            str(horizon): {"feature_attribution": base_attribution}
            for horizon in (7, 14, 30)
        },
    }


def _z_score_explanation_timeline():
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 1,
                "snapshot_date": "2026-01-01",
                "edge_count": 2.0,
                "z_mean_abs_correlation": 2.5,
                "z_edge_density": -1.0,
                "z_edge_count": 0.0,
                "z_graph_instability": 2.0,
                "risk_7d": 0.1,
                "risk_14d": 0.2,
                "risk_30d": 0.3,
                "event_observed": False,
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 2,
                "snapshot_date": "2026-01-02",
                "edge_count": 3.0,
                "z_mean_abs_correlation": 0.5,
                "z_edge_density": -4.0,
                "z_edge_count": 1.5,
                "z_graph_instability": 0.0,
                "risk_7d": 0.4,
                "risk_14d": 0.5,
                "risk_30d": 0.6,
                "event_observed": False,
            },
        ]
    )


def test_athlete_explanations_reports_snapshot_intra_individual_deviations():
    payload = _athlete_explanations(
        _z_score_explanation_timeline(),
        _z_score_explanation_model_summary(),
    )

    snapshot = payload["athletes"][0]["snapshots"][0]
    deviations = snapshot["intra_individual_deviations"]

    assert [entry["feature"] for entry in deviations] == [
        "z_mean_abs_correlation",
        "z_edge_density",
        "z_edge_count",
        "z_graph_instability",
    ]
    assert deviations[0] == {
        "feature": "z_mean_abs_correlation",
        "value": 2.5,
        "elevated": True,
        "contributions": {"7": 1.0, "14": 1.0, "30": 1.0},
    }
    assert deviations[1]["value"] == -1.0
    assert deviations[1]["elevated"] is False
    assert deviations[1]["contributions"] == {"7": 0.2, "14": 0.2, "30": 0.2}
    assert deviations[3]["value"] == 2.0
    assert deviations[3]["elevated"] is False


def test_athlete_explanations_reports_peak_intra_individual_deviation():
    payload = _athlete_explanations(
        _z_score_explanation_timeline(),
        _z_score_explanation_model_summary(),
    )

    peak = payload["athletes"][0]["peak_intra_individual_deviation"]

    assert peak["time_index"] == 2
    assert peak["snapshot_date"] == "2026-01-02"
    assert peak["combined_abs_z_score"] == 6.0
    assert peak["flagged_features"] == ["z_edge_density"]
    assert peak["deviations"][0]["feature"] == "z_edge_density"
    assert peak["deviations"][0]["value"] == -4.0
    assert peak["deviations"][0]["elevated"] is True
