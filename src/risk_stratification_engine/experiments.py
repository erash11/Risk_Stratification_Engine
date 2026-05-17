from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import brier_score_loss

from risk_stratification_engine.alert_episodes import (
    DEFAULT_ALERT_PERCENTILES,
    build_alert_episode_summary,
    build_alert_episodes,
)
from risk_stratification_engine.calibration import (
    build_calibration_bins,
    build_threshold_table,
)
from risk_stratification_engine.case_review import build_qualitative_case_review
from risk_stratification_engine.case_diagnostic_requirements import (
    build_case_diagnostic_requirements,
    build_case_diagnostic_requirements_summary,
    write_case_diagnostic_requirements_report,
)
from risk_stratification_engine.evaluation import evaluate_risk_model
from risk_stratification_engine.events import DEFAULT_HORIZONS, attach_time_to_event_labels
from risk_stratification_engine.exposure_feature_requirements import (
    build_exposure_category_summary,
    build_exposure_duration_summary,
    build_exposure_feature_requirements,
    build_exposure_feature_requirements_summary,
    write_exposure_feature_requirements_report,
)
from risk_stratification_engine.exposure_load_features import (
    EXPOSURE_LOAD_FEATURE_COLUMNS,
    attach_exposure_load_features,
)
from risk_stratification_engine.exposure_load_failure_modes import (
    build_exposure_load_failure_mode_summary,
    write_exposure_load_failure_mode_report,
)
from risk_stratification_engine.exposure_load_forward_diagnostics import (
    build_exposure_load_calibration_diagnostics,
    build_exposure_load_forward_diagnostic_cases,
    build_exposure_load_forward_diagnostic_summary,
    write_exposure_load_forward_diagnostic_report,
)
from risk_stratification_engine.exposure_load_guardrail_policy import (
    build_exposure_load_guardrail_policy,
    clean_guardrail_rows,
    write_exposure_load_guardrail_policy_report,
)
from risk_stratification_engine.exposure_load_shift_context import (
    build_exposure_load_shift_context_summary,
    clean_shift_context_rows,
    write_exposure_load_shift_context_report,
)
from risk_stratification_engine.exposure_load_context_review import (
    build_exposure_load_availability_capture_summary,
    build_exposure_load_context_decision_summary,
    build_exposure_load_schedule_roster_summary,
    clean_context_review_rows,
    write_exposure_load_availability_capture_report,
    write_exposure_load_context_decision_report,
    write_exposure_load_schedule_roster_report,
)
from risk_stratification_engine.exposure_load_source_context_classification import (
    build_exposure_load_source_context_classification_summary,
    clean_source_context_rows,
    write_exposure_load_source_context_classification_report,
)
from risk_stratification_engine.exposure_load_source_eligible_calibration import (
    build_exposure_load_source_eligible_calibration_summary,
    clean_source_eligible_calibration_rows,
    write_exposure_load_source_eligible_calibration_report,
)
from risk_stratification_engine.exposure_load_source_eligible_policy import (
    build_exposure_load_source_eligible_policy_package,
    clean_source_eligible_policy_rows,
    write_exposure_load_source_eligible_policy_report,
)
from risk_stratification_engine.exposure_load_source_eligible_shadow_monitoring import (
    build_exposure_load_source_eligible_shadow_monitoring_review,
    clean_source_eligible_shadow_monitoring_rows,
    write_exposure_load_source_eligible_shadow_monitoring_report,
)
from risk_stratification_engine.exposure_load_source_resolution import (
    build_exposure_load_source_resolution_policy,
    clean_source_resolution_rows,
    write_exposure_load_source_resolution_report,
)
from risk_stratification_engine.exposure_load_shadow_launch import (
    build_exposure_load_shadow_channel_lock,
    build_exposure_load_shadow_readiness_register,
    build_exposure_load_shadow_review_protocol,
    clean_shadow_launch_rows,
    write_exposure_load_shadow_channel_lock_report,
    write_exposure_load_shadow_readiness_register_report,
    write_exposure_load_shadow_review_protocol_report,
)
from risk_stratification_engine.exposure_load_shadow_monitoring import (
    build_exposure_load_shadow_monitoring_plan,
    clean_shadow_monitoring_rows,
    write_exposure_load_shadow_monitoring_plan_report,
)
from risk_stratification_engine.exposure_load_shadow_collection import (
    build_exposure_load_shadow_collection_evidence_prefill,
    build_exposure_load_shadow_collection_packet_workflow,
    build_exposure_load_shadow_collection_template,
    build_exposure_load_shadow_collection_summary,
    clean_shadow_collection_rows,
    write_exposure_load_shadow_collection_evidence_prefill_report,
    write_exposure_load_shadow_collection_packet_workflow_report,
    write_exposure_load_shadow_collection_reviewer_instructions,
    write_exposure_load_shadow_collection_summary_report,
    write_exposure_load_shadow_collection_template_report,
)
from risk_stratification_engine.exposure_load_shadow_calibration_readiness import (
    build_exposure_load_shadow_calibration_readiness_review,
    clean_shadow_calibration_readiness_rows,
    write_exposure_load_shadow_calibration_readiness_report,
)
from risk_stratification_engine.exposure_load_shadow_calibration_sensitivity import (
    build_exposure_load_shadow_calibration_sensitivity_review,
    clean_shadow_calibration_sensitivity_rows,
    write_exposure_load_shadow_calibration_sensitivity_report,
)
from risk_stratification_engine.exposure_load_shadow_bounded_calibration_protocol import (
    build_exposure_load_shadow_bounded_calibration_protocol,
    clean_shadow_bounded_calibration_protocol_rows,
    write_exposure_load_shadow_bounded_calibration_protocol_report,
)
from risk_stratification_engine.exposure_load_shadow_bounded_calibration_stress_test import (
    build_exposure_load_shadow_bounded_calibration_stress_test,
    clean_shadow_bounded_calibration_stress_test_rows,
    write_exposure_load_shadow_bounded_calibration_stress_test_report,
)
from risk_stratification_engine.exposure_load_shadow_error_control import (
    build_exposure_load_shadow_error_control_review,
    clean_shadow_error_control_rows,
    write_exposure_load_shadow_error_control_report,
)
from risk_stratification_engine.exposure_load_shadow_event_crosswalk import (
    build_shadow_event_crosswalk_summary,
    clean_shadow_event_crosswalk_rows,
    write_exposure_load_shadow_event_crosswalk_report,
)
from risk_stratification_engine.exposure_load_shadow_adjudication import (
    build_exposure_load_shadow_adjudication_decision_package,
    build_exposure_load_shadow_adjudication_package,
    build_exposure_load_shadow_adjudication_summary,
    clean_shadow_adjudication_rows,
    write_exposure_load_shadow_adjudication_decision_report,
    write_exposure_load_shadow_adjudication_report,
    write_exposure_load_shadow_adjudication_summary_report,
)
from risk_stratification_engine.exposure_load_shadow_replay import (
    build_exposure_load_shadow_replay_package,
    clean_shadow_replay_rows,
    write_exposure_load_shadow_replay_report,
)
from risk_stratification_engine.exposure_load_modeling import (
    build_exposure_load_model_comparison_summary,
    write_exposure_load_model_comparison_report,
)
from risk_stratification_engine.episode_quality import build_alert_episode_quality
from risk_stratification_engine.forward_case_review import (
    build_forward_case_review_summary,
    write_forward_case_review_report,
)
from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.injury_context import build_injury_context_outcomes
from risk_stratification_engine.injury_history_features import (
    INJURY_HISTORY_FEATURE_COLUMNS,
    attach_injury_history_features,
)
from risk_stratification_engine.injury_history_forward_diagnostics import (
    build_injury_history_calibration_diagnostics,
    build_injury_history_forward_diagnostic_cases,
    build_injury_history_forward_diagnostic_summary,
    write_injury_history_forward_diagnostic_report,
)
from risk_stratification_engine.injury_history_modeling import (
    build_injury_history_model_comparison_summary,
    write_injury_history_model_comparison_report,
)
from risk_stratification_engine.injury_outcomes import (
    DEFAULT_MODEL_COMPARISON_POLICIES,
    build_injury_severity_audit,
    build_policy_injury_events,
    build_outcome_policy_summary,
    policy_event_count,
)
from risk_stratification_engine.io import load_injury_events, load_measurements, write_frame
from risk_stratification_engine.models import (
    GRAPH_SNAPSHOT_FEATURE_COLUMNS,
    MODEL_TYPE,
    MODEL_VARIANTS,
    train_discrete_time_risk_model,
)
from risk_stratification_engine.model_diagnostics import (
    build_model_improvement_diagnostics,
)
from risk_stratification_engine.policy_sprints import (
    DEFAULT_POLICY_DECISION_POLICIES,
    DEFAULT_POLICY_WINDOW_SIZES,
    build_operational_policy_package,
    build_policy_window_sensitivity,
    build_two_channel_alert_policy,
    write_operational_policy_package_report,
    write_policy_window_sensitivity_report,
    write_two_channel_alert_policy_report,
)
from risk_stratification_engine.season_drift import (
    build_season_drift_diagnostics,
    write_season_drift_diagnostic_report,
)
from risk_stratification_engine.season_forward_validation import (
    build_season_forward_validation_summary,
    write_season_forward_validation_report,
)
from risk_stratification_engine.shadow_mode import (
    DEFAULT_SHADOW_MODE_CHANNELS,
    build_shadow_mode_stability_audit,
    write_shadow_mode_stability_report,
)
from risk_stratification_engine.coverage_analysis import (
    build_coverage_flag,
    build_coverage_stratified_evaluation,
    build_coverage_tiers,
    write_coverage_stratified_evaluation_report,
)
from risk_stratification_engine.coverage_adjusted_thresholds import (
    DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON,
    build_coverage_adjusted_threshold_policy_rows,
    build_coverage_adjusted_threshold_summary,
    write_coverage_adjusted_threshold_report,
)
from risk_stratification_engine.coverage_policy import (
    COVERAGE_ELIGIBILITY_SCOPES,
    COVERAGE_SCOPE_TIERS,
    build_coverage_normalized_policy_summary,
    write_coverage_normalized_policy_report,
)
from risk_stratification_engine.coverage_source_features import (
    COVERAGE_SOURCE_FEATURE_COLUMNS,
    attach_coverage_source_features,
)
from risk_stratification_engine.coverage_source_modeling import (
    build_coverage_source_model_comparison_summary,
    write_coverage_source_model_comparison_report,
)
from risk_stratification_engine.trajectories import build_measurement_matrix

ORIGINAL_GRAPH_FEATURE_COLUMNS = (
    "time_index",
    "node_count",
    "edge_count",
    "mean_abs_correlation",
    "edge_density",
    "delta_edge_count",
    "delta_mean_abs_correlation",
    "delta_edge_density",
    "graph_instability",
)
Z_SCORE_GRAPH_FEATURE_COLUMNS = (
    "z_mean_abs_correlation",
    "z_edge_density",
    "z_edge_count",
    "z_graph_instability",
)
INTRA_INDIVIDUAL_ELEVATED_Z_THRESHOLD = 2.0
FEATURE_ABLATION_SETS = {
    "full_13": GRAPH_SNAPSHOT_FEATURE_COLUMNS,
    "original_9": ORIGINAL_GRAPH_FEATURE_COLUMNS,
    "z_score_only": Z_SCORE_GRAPH_FEATURE_COLUMNS,
}

COVERAGE_SOURCE_MODEL_FEATURE_SETS = {
    "graph_trajectory": GRAPH_SNAPSHOT_FEATURE_COLUMNS,
    "graph_plus_coverage_source": (
        *GRAPH_SNAPSHOT_FEATURE_COLUMNS,
        *COVERAGE_SOURCE_FEATURE_COLUMNS,
    ),
}
INJURY_HISTORY_MODEL_FEATURE_SETS = {
    "graph_plus_coverage_source": COVERAGE_SOURCE_MODEL_FEATURE_SETS[
        "graph_plus_coverage_source"
    ],
    "graph_plus_coverage_injury_history": (
        *COVERAGE_SOURCE_MODEL_FEATURE_SETS["graph_plus_coverage_source"],
        *INJURY_HISTORY_FEATURE_COLUMNS,
    ),
}
EXPOSURE_LOAD_MODEL_FEATURE_SETS = {
    "graph_plus_coverage_source": COVERAGE_SOURCE_MODEL_FEATURE_SETS[
        "graph_plus_coverage_source"
    ],
    "graph_plus_coverage_exposure_load": (
        *COVERAGE_SOURCE_MODEL_FEATURE_SETS["graph_plus_coverage_source"],
        *EXPOSURE_LOAD_FEATURE_COLUMNS,
    ),
}
FORWARD_CASE_REVIEW_TARGET_CHANNELS = (
    "broad_30d",
    "severity_14d",
    "subtype_lower_extremity_soft_tissue_30d",
)
FORWARD_CASE_REVIEW_PREFERRED_SEASONS = ("2023-2024", "2025-2026")


def run_research_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "baseline",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    labeled = attach_time_to_event_labels(graph_features, injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")
    model_result = train_discrete_time_risk_model(
        labeled,
        model_variant=model_variant,
    )
    timeline = model_result.timeline
    evaluation = evaluate_risk_model(timeline, model_result.summary)
    feature_attribution = _feature_attribution_and_ablation(
        labeled=labeled,
        full_timeline=timeline,
        full_model_summary=model_result.summary,
        full_evaluation=evaluation,
    )
    explanations = _explanation_summary(timeline, model_result.summary)
    athlete_expl = _athlete_explanations(timeline, model_result.summary)

    graph_dir = experiment_dir / "graph_snapshots"
    explanation_dir = experiment_dir / "explanations"
    graph_dir.mkdir(parents=True, exist_ok=True)
    explanation_dir.mkdir(parents=True, exist_ok=True)

    write_frame(graph_features, graph_dir / "graph_features.csv")
    write_frame(timeline, experiment_dir / "athlete_risk_timeline.csv")
    write_frame(explanations, explanation_dir / "explanation_summary.csv")
    _write_json(explanation_dir / "athlete_explanations.json", athlete_expl)
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "horizons": list(DEFAULT_HORIZONS),
        },
    )
    _write_json(
        experiment_dir / "model_metrics.json",
        _model_metrics(labeled, timeline, model_result.summary),
    )
    _write_json(experiment_dir / "model_evaluation.json", evaluation)
    _write_json(experiment_dir / "model_summary.json", model_result.summary)
    _write_json(experiment_dir / "feature_attribution.json", feature_attribution)
    _write_report(
        experiment_dir / "experiment_report.md",
        timeline,
        model_result.summary,
        evaluation,
    )
    _write_feature_ablation_report(
        experiment_dir / "feature_ablation_report.md",
        feature_attribution,
    )
    return experiment_dir


def run_alert_episode_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
    percentile_thresholds: tuple[float, ...] = DEFAULT_ALERT_PERCENTILES,
    detailed_injuries_path: str | Path | None = None,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    labeled = attach_time_to_event_labels(graph_features, injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")

    model_result = train_discrete_time_risk_model(
        labeled,
        model_variant=model_variant,
    )
    timeline = model_result.timeline
    explanation_summary = _explanation_summary(timeline, model_result.summary)
    alert_timeline = _alert_episode_timeline(timeline, explanation_summary)
    episodes = build_alert_episodes(
        alert_timeline,
        percentile_thresholds=percentile_thresholds,
    )
    alert_summary = build_alert_episode_summary(episodes)
    quality = build_alert_episode_quality(episodes, timeline)
    case_review = build_qualitative_case_review(
        episodes=episodes,
        alert_timeline=alert_timeline,
        quality=quality,
    )
    model_diagnostics = build_model_improvement_diagnostics(
        episodes=episodes,
        alert_timeline=alert_timeline,
        quality=quality,
    )
    resolved_detailed_injuries_path = _resolve_detailed_injuries_path(
        injuries_path,
        detailed_injuries_path,
    )
    injury_context_outcomes = None
    if resolved_detailed_injuries_path is not None:
        injury_context_outcomes = build_injury_context_outcomes(
            detailed_events=pd.read_csv(resolved_detailed_injuries_path),
            episodes=episodes,
        )
    alert_summary.update(
        {
            "experiment_type": "alert_episode_validation",
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "alert_percentile_thresholds": list(percentile_thresholds),
            "athlete_count": int(labeled["athlete_id"].nunique()),
            "snapshot_count": int(len(labeled)),
        }
    )

    write_frame(timeline, experiment_dir / "athlete_risk_timeline.csv")
    write_frame(episodes, experiment_dir / "alert_episodes.csv")
    write_frame(
        pd.DataFrame(quality["quality_rows"]),
        experiment_dir / "alert_episode_quality.csv",
    )
    write_frame(
        pd.DataFrame(model_diagnostics["diagnostic_rows"]),
        experiment_dir / "model_improvement_diagnostics.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "alert_episode_validation",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(resolved_detailed_injuries_path)
            if resolved_detailed_injuries_path is not None
            else None,
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "horizons": list(DEFAULT_HORIZONS),
            "alert_percentile_thresholds": list(percentile_thresholds),
        },
    )
    _write_json(experiment_dir / "model_summary.json", model_result.summary)
    _write_json(experiment_dir / "alert_episode_summary.json", alert_summary)
    _write_json(
        experiment_dir / "alert_episodes.json",
        {
            "experiment_type": "alert_episode_validation",
            "episode_count": int(len(episodes)),
            "episodes": _json_records(episodes),
        },
    )
    _write_json(
        experiment_dir / "alert_episode_quality.json",
        {
            "experiment_type": "alert_episode_quality_audit",
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "alert_percentile_thresholds": list(percentile_thresholds),
            "quality_row_count": len(quality["quality_rows"]),
            "quality_rows": quality["quality_rows"],
            "threshold_overlaps": quality["threshold_overlaps"],
            "representative_cases": quality["representative_cases"],
        },
    )
    _write_json(
        experiment_dir / "qualitative_case_review.json",
        {
            "experiment_type": "qualitative_case_review",
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "alert_percentile_thresholds": list(percentile_thresholds),
            **case_review,
        },
    )
    _write_json(
        experiment_dir / "model_improvement_diagnostics.json",
        {
            "experiment_type": "model_improvement_diagnostics",
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "alert_percentile_thresholds": list(percentile_thresholds),
            **model_diagnostics,
        },
    )
    if injury_context_outcomes is not None:
        write_frame(
            pd.DataFrame(injury_context_outcomes["event_profile_rows"]),
            experiment_dir / "injury_event_context_profiles.csv",
        )
        write_frame(
            pd.DataFrame(injury_context_outcomes["context_rows"]),
            experiment_dir / "injury_context_outcomes.csv",
        )
        _write_json(
            experiment_dir / "injury_context_outcomes.json",
            {
                "experiment_type": "injury_context_outcomes",
                "model_variant": model_variant,
                "graph_window_size": graph_window_size,
                "detailed_injuries_path": str(resolved_detailed_injuries_path),
                "alert_percentile_thresholds": list(percentile_thresholds),
                **injury_context_outcomes,
            },
        )
    _write_alert_episode_report(
        experiment_dir / "alert_episode_report.md",
        alert_summary,
    )
    _write_alert_episode_quality_report(
        experiment_dir / "alert_episode_quality_report.md",
        {
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            **quality,
        },
    )
    _write_qualitative_case_review_report(
        experiment_dir / "qualitative_case_review_report.md",
        {
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            **case_review,
        },
    )
    _write_model_improvement_diagnostic_report(
        experiment_dir / "model_improvement_diagnostic_report.md",
        {
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            **model_diagnostics,
        },
    )
    if injury_context_outcomes is not None:
        _write_injury_context_outcome_report(
            experiment_dir / "injury_context_outcome_report.md",
            {
                "model_variant": model_variant,
                "graph_window_size": graph_window_size,
                **injury_context_outcomes,
            },
        )
    return experiment_dir


def run_injury_outcome_policy_experiment(
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    severity_audit = build_injury_severity_audit(detailed_injuries)
    policy_summary = build_outcome_policy_summary(detailed_injuries)

    write_frame(
        pd.DataFrame(severity_audit["event_rows"]),
        experiment_dir / "injury_severity_audit.csv",
    )
    write_frame(
        pd.DataFrame(policy_summary["policy_rows"]),
        experiment_dir / "outcome_policy_table.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "injury_outcome_policy",
            "detailed_injuries_path": str(detailed_injuries_path),
        },
    )
    _write_json(
        experiment_dir / "injury_severity_audit.json",
        {
            "experiment_type": "injury_severity_audit",
            "detailed_injuries_path": str(detailed_injuries_path),
            **severity_audit,
        },
    )
    _write_json(
        experiment_dir / "outcome_policy_summary.json",
        {
            "experiment_type": "outcome_policy_summary",
            "detailed_injuries_path": str(detailed_injuries_path),
            **policy_summary,
        },
    )
    _write_injury_severity_audit_report(
        experiment_dir / "injury_severity_audit_report.md",
        severity_audit,
    )
    _write_outcome_policy_report(
        experiment_dir / "outcome_policy_report.md",
        policy_summary,
    )
    return experiment_dir


def run_outcome_policy_model_comparison_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
    policy_names: tuple[str, ...] = DEFAULT_MODEL_COMPARISON_POLICIES,
    percentile_thresholds: tuple[float, ...] = DEFAULT_ALERT_PERCENTILES,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")

    rows: list[dict[str, object]] = []
    policy_summaries: dict[str, object] = {}
    for policy_name in policy_names:
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=policy_name,
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        if labeled.empty:
            raise ValueError(f"no labeled graph snapshots produced for {policy_name}")
        model_result = train_discrete_time_risk_model(
            labeled,
            model_variant=model_variant,
        )
        evaluation = evaluate_risk_model(model_result.timeline, model_result.summary)
        explanation_summary = _explanation_summary(
            model_result.timeline,
            model_result.summary,
        )
        alert_timeline = _alert_episode_timeline(
            model_result.timeline,
            explanation_summary,
        )
        episodes = build_alert_episodes(
            alert_timeline,
            percentile_thresholds=percentile_thresholds,
        )
        quality = build_alert_episode_quality(episodes, model_result.timeline)
        policy_event_total = policy_event_count(detailed_injuries, policy_name)
        policy_summaries[policy_name] = {
            "policy_event_count": policy_event_total,
            "observed_athlete_season_count": int(
                policy_injuries["event_observed"].sum()
            ),
            "timeline_snapshot_count": int(len(model_result.timeline)),
            "episode_count": int(len(episodes)),
        }
        rows.extend(
            _policy_comparison_rows(
                policy_name=policy_name,
                policy_event_count=policy_event_total,
                graph_window_size=graph_window_size,
                evaluation=evaluation,
                quality_rows=quality["quality_rows"],
            )
        )

    comparison = {
        "experiment_type": "context_policy_model_comparison",
        "model_variant": model_variant,
        "graph_window_size": graph_window_size,
        "policy_names": list(policy_names),
        "policy_count": len(policy_names),
        "comparison_row_count": len(rows),
        "policy_summaries": policy_summaries,
        "comparison_rows": rows,
        "best_by_horizon": _best_policies_by_horizon(rows),
    }
    write_frame(
        pd.DataFrame(rows),
        experiment_dir / "context_policy_model_comparison.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "context_policy_model_comparison",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "policy_names": list(policy_names),
            "alert_percentile_thresholds": list(percentile_thresholds),
        },
    )
    _write_json(
        experiment_dir / "context_policy_model_comparison.json",
        comparison,
    )
    _write_context_policy_model_comparison_report(
        experiment_dir / "context_policy_model_comparison_report.md",
        comparison,
    )
    return experiment_dir


def run_policy_decision_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_sizes: tuple[int, ...] = DEFAULT_POLICY_WINDOW_SIZES,
    model_variant: str = "l2",
    policy_names: tuple[str, ...] = DEFAULT_POLICY_DECISION_POLICIES,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    window_sizes = _normalize_window_sizes(graph_window_sizes)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    matrix = build_measurement_matrix(measurements)

    rows: list[dict[str, object]] = []
    policy_summaries: dict[str, object] = {}
    for window_size in window_sizes:
        graph_features = build_graph_snapshots(matrix, window_size=window_size)
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        for policy_name in policy_names:
            policy_injuries = build_policy_injury_events(
                canonical_injuries,
                detailed_injuries,
                policy_name=policy_name,
            )
            labeled = attach_time_to_event_labels(graph_features, policy_injuries)
            if labeled.empty:
                raise ValueError(
                    f"no labeled graph snapshots produced for {policy_name}"
                )
            model_result = train_discrete_time_risk_model(
                labeled,
                model_variant=model_variant,
            )
            evaluation = evaluate_risk_model(
                model_result.timeline,
                model_result.summary,
            )
            explanation_summary = _explanation_summary(
                model_result.timeline,
                model_result.summary,
            )
            alert_timeline = _alert_episode_timeline(
                model_result.timeline,
                explanation_summary,
            )
            episodes = build_alert_episodes(alert_timeline)
            quality = build_alert_episode_quality(episodes, model_result.timeline)
            policy_event_total = policy_event_count(detailed_injuries, policy_name)
            summary_key = f"window_{window_size}:{policy_name}"
            policy_summaries[summary_key] = {
                "graph_window_size": window_size,
                "policy_name": policy_name,
                "policy_event_count": policy_event_total,
                "observed_athlete_season_count": int(
                    policy_injuries["event_observed"].sum()
                ),
                "timeline_snapshot_count": int(len(model_result.timeline)),
                "episode_count": int(len(episodes)),
            }
            rows.extend(
                _policy_comparison_rows(
                    policy_name=policy_name,
                    policy_event_count=policy_event_total,
                    graph_window_size=window_size,
                    evaluation=evaluation,
                    quality_rows=quality["quality_rows"],
                )
            )

    comparison_rows = pd.DataFrame(rows)
    primary_window = 4 if 4 in window_sizes else window_sizes[0]
    primary_rows = comparison_rows[
        comparison_rows["graph_window_size"].astype(int).eq(primary_window)
    ]
    two_channel = build_two_channel_alert_policy(primary_rows, policy_summaries)
    window_sensitivity = build_policy_window_sensitivity(comparison_rows)
    operational_package = build_operational_policy_package(
        two_channel,
        window_sensitivity,
    )

    write_frame(comparison_rows, experiment_dir / "policy_window_sensitivity.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "policy_decision_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "graph_window_sizes": list(window_sizes),
            "model_variant": model_variant,
            "policy_names": list(policy_names),
        },
    )
    _write_json(experiment_dir / "two_channel_alert_policy.json", two_channel)
    _write_json(
        experiment_dir / "policy_window_sensitivity.json",
        window_sensitivity,
    )
    _write_json(
        experiment_dir / "operational_policy_package.json",
        operational_package,
    )
    write_two_channel_alert_policy_report(
        experiment_dir / "two_channel_alert_policy_report.md",
        two_channel,
    )
    write_policy_window_sensitivity_report(
        experiment_dir / "policy_window_sensitivity_report.md",
        window_sensitivity,
    )
    write_operational_policy_package_report(
        experiment_dir / "operational_policy_package_report.md",
        operational_package,
    )
    return experiment_dir


def run_shadow_mode_stability_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    stability_frame = _shadow_mode_stability_frame(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        model_variant=model_variant,
    )
    audit = build_shadow_mode_stability_audit(stability_frame)
    write_frame(stability_frame, experiment_dir / "shadow_mode_stability.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "shadow_mode_policy_stability",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(experiment_dir / "shadow_mode_stability.json", audit)
    write_shadow_mode_stability_report(
        experiment_dir / "shadow_mode_stability_report.md",
        audit,
    )
    return experiment_dir


def run_season_drift_diagnostic_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    stability_frame = _shadow_mode_stability_frame(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        model_variant=model_variant,
    )
    diagnostics = build_season_drift_diagnostics(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        shadow_mode_rows=stability_frame,
    )
    diagnostics["model_variant"] = model_variant

    write_frame(
        pd.DataFrame(diagnostics["season_rows"]),
        experiment_dir / "season_drift_diagnostics.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "season_drift_diagnostic",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(experiment_dir / "season_drift_diagnostics.json", diagnostics)
    write_season_drift_diagnostic_report(
        experiment_dir / "season_drift_diagnostic_report.md",
        diagnostics,
    )
    return experiment_dir


def run_coverage_stratified_evaluation_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)

    coverage_tiers = build_coverage_tiers(measurements)

    matrix = build_measurement_matrix(measurements)
    graph_cache: dict[int, pd.DataFrame] = {}
    channel_results = []

    for channel in DEFAULT_SHADOW_MODE_CHANNELS:
        window_size = int(channel["graph_window_size"])
        if window_size not in graph_cache:
            graph_cache[window_size] = build_graph_snapshots(
                matrix, window_size=window_size
            )
        graph_features = graph_cache[window_size]
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        if labeled.empty:
            raise ValueError(
                f"no labeled graph snapshots produced for {channel['policy_name']}"
            )
        model_result = train_discrete_time_risk_model(
            labeled, model_variant=model_variant
        )
        timeline = model_result.timeline

        tier_cols = coverage_tiers[
            ["athlete_id", "season_id", "coverage_tier", "measurement_days"]
        ]
        # Ensure season_id dtype matches before merge
        timeline = timeline.copy()
        timeline["season_id"] = timeline["season_id"].astype(str)
        timeline_with_tiers = timeline.merge(
            tier_cols, on=["athlete_id", "season_id"], how="left"
        )
        timeline_with_tiers["coverage_tier"] = (
            timeline_with_tiers["coverage_tier"].fillna("low")
        )

        channel_result = build_coverage_stratified_evaluation(
            timeline_with_tiers, channel
        )
        channel_results.append(channel_result)

    coverage_flag = build_coverage_flag(channel_results)
    tier_distribution = (
        coverage_tiers["coverage_tier"]
        .value_counts()
        .reindex(["low", "medium", "high"], fill_value=0)
        .astype(int)
        .to_dict()
    ) if not coverage_tiers.empty else {"low": 0, "medium": 0, "high": 0}

    result = {
        "experiment_type": "coverage_stratified_evaluation",
        "tier_distribution": tier_distribution,
        "coverage_flag": coverage_flag,
        "channel_results": channel_results,
    }

    csv_rows = [row for ch in channel_results for row in ch["rows"]]

    write_frame(coverage_tiers, experiment_dir / "coverage_tiers.csv")
    write_frame(
        pd.DataFrame(csv_rows),
        experiment_dir / "coverage_stratified_evaluation.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "coverage_stratified_evaluation",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(
        experiment_dir / "coverage_stratified_evaluation.json", result
    )
    write_coverage_stratified_evaluation_report(
        experiment_dir / "coverage_stratified_evaluation_report.md",
        result,
    )
    return experiment_dir


def run_coverage_normalized_policy_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    coverage_tiers = build_coverage_tiers(measurements)
    stability_rows = _coverage_normalized_stability_frame(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        coverage_tiers=coverage_tiers,
        model_variant=model_variant,
    )

    scope_audits = {}
    for scope in COVERAGE_ELIGIBILITY_SCOPES:
        scope_rows = stability_rows[
            stability_rows["coverage_eligibility_scope"].eq(scope)
        ]
        scope_audits[scope] = build_shadow_mode_stability_audit(scope_rows)
        scope_audits[scope]["coverage_eligibility_scope"] = scope
        scope_audits[scope]["eligible_coverage_tiers"] = list(COVERAGE_SCOPE_TIERS[scope])

    summary = build_coverage_normalized_policy_summary(scope_audits)
    result = {
        "experiment_type": "coverage_normalized_policy_sprint",
        "model_variant": model_variant,
        "coverage_eligibility_scopes": list(COVERAGE_ELIGIBILITY_SCOPES),
        "scope_audits": scope_audits,
        **summary,
    }

    write_frame(stability_rows, experiment_dir / "coverage_normalized_policy.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "coverage_normalized_policy_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
            "coverage_eligibility_scopes": list(COVERAGE_ELIGIBILITY_SCOPES),
        },
    )
    _write_json(experiment_dir / "coverage_normalized_policy.json", result)
    write_coverage_normalized_policy_report(
        experiment_dir / "coverage_normalized_policy_report.md",
        result,
    )
    return experiment_dir


def run_coverage_source_aware_model_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    coverage_features = attach_coverage_source_features(graph_features, measurements)
    labeled = attach_time_to_event_labels(coverage_features, injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")

    comparison_rows: list[dict[str, object]] = []
    feature_summaries: dict[str, object] = {}
    for feature_set_name, feature_columns in COVERAGE_SOURCE_MODEL_FEATURE_SETS.items():
        model_result = train_discrete_time_risk_model(
            labeled,
            feature_columns=feature_columns,
            model_variant=model_variant,
        )
        evaluation = evaluate_risk_model(model_result.timeline, model_result.summary)
        feature_summaries[feature_set_name] = {
            "feature_columns": list(feature_columns),
            "model_summary": model_result.summary,
            "evaluation": evaluation,
        }
        comparison_rows.extend(
            _coverage_source_comparison_rows(
                feature_set_name=feature_set_name,
                evaluation=evaluation,
                model_summary=model_result.summary,
            )
        )

    summary = build_coverage_source_model_comparison_summary(comparison_rows)
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "graph_feature_columns": list(GRAPH_SNAPSHOT_FEATURE_COLUMNS),
            "coverage_source_feature_columns": list(
                COVERAGE_SOURCE_FEATURE_COLUMNS
            ),
            "feature_set_summaries": feature_summaries,
        }
    )

    write_frame(coverage_features, experiment_dir / "coverage_source_features.csv")
    write_frame(
        pd.DataFrame(comparison_rows),
        experiment_dir / "coverage_source_model_comparison.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "coverage_source_aware_model_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "feature_sets": list(COVERAGE_SOURCE_MODEL_FEATURE_SETS),
            "coverage_source_feature_columns": list(
                COVERAGE_SOURCE_FEATURE_COLUMNS
            ),
        },
    )
    _write_json(
        experiment_dir / "coverage_source_model_comparison.json",
        summary,
    )
    write_coverage_source_model_comparison_report(
        experiment_dir / "coverage_source_model_comparison_report.md",
        summary,
    )
    return experiment_dir


def run_coverage_adjusted_threshold_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
    burden_cap_episodes_per_athlete_season: float = (
        DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON
    ),
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    coverage_tiers = build_coverage_tiers(measurements)
    rows = _coverage_adjusted_threshold_frame(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        coverage_tiers=coverage_tiers,
        model_variant=model_variant,
        burden_cap_episodes_per_athlete_season=(
            burden_cap_episodes_per_athlete_season
        ),
    )
    row_records = _json_records(rows)
    summary = build_coverage_adjusted_threshold_summary(
        row_records,
        burden_cap_episodes_per_athlete_season=(
            burden_cap_episodes_per_athlete_season
        ),
    )
    result = {
        "experiment_type": "coverage_adjusted_threshold_sprint",
        "model_variant": model_variant,
        "burden_cap_episodes_per_athlete_season": (
            burden_cap_episodes_per_athlete_season
        ),
        **summary,
    }

    write_frame(rows, experiment_dir / "coverage_adjusted_threshold_policy.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "coverage_adjusted_threshold_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
            "burden_cap_episodes_per_athlete_season": (
                burden_cap_episodes_per_athlete_season
            ),
        },
    )
    _write_json(
        experiment_dir / "coverage_adjusted_threshold_policy.json",
        result,
    )
    write_coverage_adjusted_threshold_report(
        experiment_dir / "coverage_adjusted_threshold_report.md",
        result,
    )
    return experiment_dir


def run_injury_history_feature_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    coverage_features = attach_coverage_source_features(graph_features, measurements)
    injury_history_features = attach_injury_history_features(
        coverage_features,
        detailed_injuries,
    )
    labeled = attach_time_to_event_labels(injury_history_features, injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")

    comparison_rows: list[dict[str, object]] = []
    feature_summaries: dict[str, object] = {}
    for feature_set_name, feature_columns in INJURY_HISTORY_MODEL_FEATURE_SETS.items():
        model_result = train_discrete_time_risk_model(
            labeled,
            feature_columns=feature_columns,
            model_variant=model_variant,
        )
        evaluation = evaluate_risk_model(model_result.timeline, model_result.summary)
        feature_summaries[feature_set_name] = {
            "feature_columns": list(feature_columns),
            "model_summary": model_result.summary,
            "evaluation": evaluation,
        }
        comparison_rows.extend(
            _coverage_source_comparison_rows(
                feature_set_name=feature_set_name,
                evaluation=evaluation,
                model_summary=model_result.summary,
            )
        )

    summary = build_injury_history_model_comparison_summary(comparison_rows)
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "graph_feature_columns": list(GRAPH_SNAPSHOT_FEATURE_COLUMNS),
            "coverage_source_feature_columns": list(COVERAGE_SOURCE_FEATURE_COLUMNS),
            "injury_history_feature_columns": list(INJURY_HISTORY_FEATURE_COLUMNS),
            "feature_set_summaries": feature_summaries,
        }
    )

    write_frame(injury_history_features, experiment_dir / "injury_history_features.csv")
    write_frame(
        pd.DataFrame(comparison_rows),
        experiment_dir / "injury_history_model_comparison.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "injury_history_feature_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "feature_sets": list(INJURY_HISTORY_MODEL_FEATURE_SETS),
            "injury_history_feature_columns": list(INJURY_HISTORY_FEATURE_COLUMNS),
        },
    )
    _write_json(
        experiment_dir / "injury_history_model_comparison.json",
        summary,
    )
    write_injury_history_model_comparison_report(
        experiment_dir / "injury_history_model_comparison_report.md",
        summary,
    )
    return experiment_dir


def run_injury_history_season_forward_validation_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    coverage_features = attach_coverage_source_features(graph_features, measurements)
    injury_history_features = attach_injury_history_features(
        coverage_features,
        detailed_injuries,
    )
    rows = _season_forward_validation_rows(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        graph_window_size=graph_window_size,
        model_variant=model_variant,
        feature_sets=INJURY_HISTORY_MODEL_FEATURE_SETS,
        prepared_features=injury_history_features,
        alert_policy_feature_set_name="graph_plus_coverage_injury_history",
        alert_policy_feature_columns=INJURY_HISTORY_MODEL_FEATURE_SETS[
            "graph_plus_coverage_injury_history"
        ],
        attach_injury_history_to_alert_features=True,
    )
    row_records = _json_records(rows)
    summary = build_season_forward_validation_summary(
        row_records,
        experiment_type="injury_history_season_forward_validation_sprint",
    )
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "feature_sets": list(INJURY_HISTORY_MODEL_FEATURE_SETS),
            "coverage_source_feature_columns": list(COVERAGE_SOURCE_FEATURE_COLUMNS),
            "injury_history_feature_columns": list(INJURY_HISTORY_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        }
    )

    write_frame(injury_history_features, experiment_dir / "injury_history_features.csv")
    write_frame(
        rows,
        experiment_dir / "injury_history_season_forward_validation.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "injury_history_season_forward_validation_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "feature_sets": list(INJURY_HISTORY_MODEL_FEATURE_SETS),
            "injury_history_feature_columns": list(INJURY_HISTORY_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(
        experiment_dir / "injury_history_season_forward_validation.json",
        summary,
    )
    write_season_forward_validation_report(
        experiment_dir / "injury_history_season_forward_validation_report.md",
        summary,
    )
    return experiment_dir


def run_injury_history_forward_diagnostic_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    coverage_features = attach_coverage_source_features(graph_features, measurements)
    injury_history_features = attach_injury_history_features(
        coverage_features,
        detailed_injuries,
    )
    rows = _season_forward_validation_rows(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        graph_window_size=graph_window_size,
        model_variant=model_variant,
        feature_sets=INJURY_HISTORY_MODEL_FEATURE_SETS,
        prepared_features=injury_history_features,
        alert_policy_feature_set_name="graph_plus_coverage_injury_history",
        alert_policy_feature_columns=INJURY_HISTORY_MODEL_FEATURE_SETS[
            "graph_plus_coverage_injury_history"
        ],
        attach_injury_history_to_alert_features=True,
    )
    row_records = _json_records(rows)
    calibration_diagnostics = build_injury_history_calibration_diagnostics(
        row_records,
    )
    cases = pd.DataFrame(
        build_injury_history_forward_diagnostic_cases(calibration_diagnostics)
    )
    case_records = _json_records(cases)
    summary = build_injury_history_forward_diagnostic_summary(
        row_records,
        case_records,
    )
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "feature_sets": list(INJURY_HISTORY_MODEL_FEATURE_SETS),
            "coverage_source_feature_columns": list(COVERAGE_SOURCE_FEATURE_COLUMNS),
            "injury_history_feature_columns": list(INJURY_HISTORY_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        }
    )

    write_frame(injury_history_features, experiment_dir / "injury_history_features.csv")
    write_frame(rows, experiment_dir / "injury_history_season_forward_validation.csv")
    write_frame(
        pd.DataFrame(calibration_diagnostics),
        experiment_dir / "injury_history_calibration_diagnostics.csv",
    )
    write_frame(cases, experiment_dir / "injury_history_forward_diagnostic_cases.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "injury_history_forward_diagnostic_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "feature_sets": list(INJURY_HISTORY_MODEL_FEATURE_SETS),
            "injury_history_feature_columns": list(INJURY_HISTORY_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
            "source_validation_artifact": (
                "injury_history_season_forward_validation.csv"
            ),
        },
    )
    _write_json(experiment_dir / "injury_history_forward_diagnostic.json", summary)
    write_injury_history_forward_diagnostic_report(
        experiment_dir / "injury_history_forward_diagnostic_report.md",
        summary,
    )
    return experiment_dir


def run_season_forward_validation_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    rows = _season_forward_validation_rows(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        graph_window_size=graph_window_size,
        model_variant=model_variant,
        feature_sets=COVERAGE_SOURCE_MODEL_FEATURE_SETS,
        alert_policy_feature_set_name="graph_plus_coverage_source",
        alert_policy_feature_columns=COVERAGE_SOURCE_MODEL_FEATURE_SETS[
            "graph_plus_coverage_source"
        ],
    )
    row_records = _json_records(rows)
    summary = build_season_forward_validation_summary(row_records)
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "feature_sets": list(COVERAGE_SOURCE_MODEL_FEATURE_SETS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        }
    )

    write_frame(rows, experiment_dir / "season_forward_validation.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "season_forward_validation_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "feature_sets": list(COVERAGE_SOURCE_MODEL_FEATURE_SETS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(experiment_dir / "season_forward_validation.json", summary)
    write_season_forward_validation_report(
        experiment_dir / "season_forward_validation_report.md",
        summary,
    )
    return experiment_dir


def run_forward_case_review_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    cases = _forward_case_review_cases(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        model_variant=model_variant,
    )
    case_records = _json_records(cases)
    summary = build_forward_case_review_summary(case_records)
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "feature_set": "graph_plus_coverage_source",
            "target_channels": list(FORWARD_CASE_REVIEW_TARGET_CHANNELS),
        }
    )

    write_frame(cases, experiment_dir / "forward_case_review_cases.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "forward_case_review_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "feature_set": "graph_plus_coverage_source",
            "target_channels": list(FORWARD_CASE_REVIEW_TARGET_CHANNELS),
            "preferred_test_seasons": list(FORWARD_CASE_REVIEW_PREFERRED_SEASONS),
        },
    )
    _write_json(experiment_dir / "forward_case_review.json", summary)
    write_forward_case_review_report(
        experiment_dir / "forward_case_review_report.md",
        summary,
    )
    return experiment_dir


def run_case_diagnostic_requirements_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    cases = _forward_case_review_cases(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        model_variant=model_variant,
    )
    case_records = _json_records(cases)
    requirements = build_case_diagnostic_requirements(case_records)
    summary = build_case_diagnostic_requirements_summary(requirements, case_records)
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "feature_set": "graph_plus_coverage_source",
            "source_case_artifact": "forward_case_review_cases.csv",
        }
    )

    write_frame(cases, experiment_dir / "forward_case_review_cases.csv")
    write_frame(
        pd.DataFrame(requirements),
        experiment_dir / "case_diagnostic_requirements.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "case_diagnostic_requirements_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "feature_set": "graph_plus_coverage_source",
            "source_case_artifact": "forward_case_review_cases.csv",
        },
    )
    _write_json(experiment_dir / "case_diagnostic_requirements.json", summary)
    write_case_diagnostic_requirements_report(
        experiment_dir / "case_diagnostic_requirements_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_feature_requirements_sprint_experiment(
    exposure_events_path: str | Path,
    exposure_participations_path: str | Path,
    exposure_audit_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    events = pd.read_csv(exposure_events_path)
    participations = pd.read_csv(exposure_participations_path)
    audit = json.loads(Path(exposure_audit_path).read_text(encoding="utf-8"))

    requirements = build_exposure_feature_requirements(
        events,
        participations,
        audit,
    )
    summary = build_exposure_feature_requirements_summary(
        events,
        participations,
        audit,
        requirements,
    )

    write_frame(
        build_exposure_category_summary(events),
        experiment_dir / "exposure_category_summary.csv",
    )
    write_frame(
        build_exposure_duration_summary(participations),
        experiment_dir / "exposure_duration_summary.csv",
    )
    write_frame(
        pd.DataFrame(requirements),
        experiment_dir / "exposure_feature_requirements.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_feature_requirements_sprint",
            "exposure_events_path": str(exposure_events_path),
            "exposure_participations_path": str(exposure_participations_path),
            "exposure_audit_path": str(exposure_audit_path),
            "source_cleaning_artifacts": [
                "exposure_events.csv",
                "exposure_participations.csv",
                "exposure_cleaning_audit.json",
            ],
        },
    )
    _write_json(experiment_dir / "exposure_feature_requirements.json", summary)
    write_exposure_feature_requirements_report(
        experiment_dir / "exposure_feature_requirements_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_feature_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    exposure_participations_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    exposure_participations = pd.read_csv(exposure_participations_path)

    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    coverage_features = attach_coverage_source_features(graph_features, measurements)
    exposure_features = attach_exposure_load_features(
        coverage_features,
        exposure_participations,
    )
    labeled = attach_time_to_event_labels(exposure_features, injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")

    comparison_rows: list[dict[str, object]] = []
    feature_summaries: dict[str, object] = {}
    for feature_set_name, feature_columns in EXPOSURE_LOAD_MODEL_FEATURE_SETS.items():
        model_result = train_discrete_time_risk_model(
            labeled,
            feature_columns=feature_columns,
            model_variant=model_variant,
        )
        evaluation = evaluate_risk_model(model_result.timeline, model_result.summary)
        feature_summaries[feature_set_name] = {
            "feature_columns": list(feature_columns),
            "model_summary": model_result.summary,
            "evaluation": evaluation,
        }
        comparison_rows.extend(
            _coverage_source_comparison_rows(
                feature_set_name=feature_set_name,
                evaluation=evaluation,
                model_summary=model_result.summary,
            )
        )

    summary = build_exposure_load_model_comparison_summary(comparison_rows)
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "graph_feature_columns": list(GRAPH_SNAPSHOT_FEATURE_COLUMNS),
            "coverage_source_feature_columns": list(COVERAGE_SOURCE_FEATURE_COLUMNS),
            "exposure_load_feature_columns": list(EXPOSURE_LOAD_FEATURE_COLUMNS),
            "feature_set_summaries": feature_summaries,
        }
    )

    write_frame(exposure_features, experiment_dir / "exposure_load_features.csv")
    write_frame(
        pd.DataFrame(comparison_rows),
        experiment_dir / "exposure_load_model_comparison.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_feature_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "exposure_participations_path": str(exposure_participations_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "feature_sets": list(EXPOSURE_LOAD_MODEL_FEATURE_SETS),
            "coverage_source_feature_columns": list(COVERAGE_SOURCE_FEATURE_COLUMNS),
            "exposure_load_feature_columns": list(EXPOSURE_LOAD_FEATURE_COLUMNS),
        },
    )
    _write_json(experiment_dir / "exposure_load_model_comparison.json", summary)
    write_exposure_load_model_comparison_report(
        experiment_dir / "exposure_load_model_comparison_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_season_forward_validation_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    exposure_participations_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    exposure_participations = pd.read_csv(exposure_participations_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    coverage_features = attach_coverage_source_features(graph_features, measurements)
    exposure_features = attach_exposure_load_features(
        coverage_features,
        exposure_participations,
    )
    rows = _season_forward_validation_rows(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        graph_window_size=graph_window_size,
        model_variant=model_variant,
        feature_sets=EXPOSURE_LOAD_MODEL_FEATURE_SETS,
        prepared_features=exposure_features,
        alert_policy_feature_set_name="graph_plus_coverage_exposure_load",
        alert_policy_feature_columns=EXPOSURE_LOAD_MODEL_FEATURE_SETS[
            "graph_plus_coverage_exposure_load"
        ],
        exposure_participations_for_alert_features=exposure_participations,
    )
    row_records = _json_records(rows)
    summary = build_season_forward_validation_summary(
        row_records,
        experiment_type="exposure_load_season_forward_validation_sprint",
    )
    summary.update(
        {
            "model_type": MODEL_TYPE,
            "model_variant": model_variant,
            "graph_window_size": graph_window_size,
            "feature_sets": list(EXPOSURE_LOAD_MODEL_FEATURE_SETS),
            "coverage_source_feature_columns": list(COVERAGE_SOURCE_FEATURE_COLUMNS),
            "exposure_load_feature_columns": list(EXPOSURE_LOAD_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        }
    )

    write_frame(exposure_features, experiment_dir / "exposure_load_features.csv")
    write_frame(
        rows,
        experiment_dir / "exposure_load_season_forward_validation.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_season_forward_validation_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "exposure_participations_path": str(exposure_participations_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "feature_sets": list(EXPOSURE_LOAD_MODEL_FEATURE_SETS),
            "exposure_load_feature_columns": list(EXPOSURE_LOAD_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_season_forward_validation.json",
        summary,
    )
    write_season_forward_validation_report(
        experiment_dir / "exposure_load_season_forward_validation_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_forward_diagnostic_sprint_experiment(
    season_forward_validation_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    validation_rows = pd.read_csv(season_forward_validation_path)
    row_records = _json_records(validation_rows)
    diagnostics = build_exposure_load_calibration_diagnostics(row_records)
    cases = build_exposure_load_forward_diagnostic_cases(diagnostics)
    summary = build_exposure_load_forward_diagnostic_summary(row_records, cases)
    summary.update(
        {
            "source_validation_path": str(season_forward_validation_path),
            "source_validation_artifact": "exposure_load_season_forward_validation.csv",
            "feature_sets": list(EXPOSURE_LOAD_MODEL_FEATURE_SETS),
            "exposure_load_feature_columns": list(EXPOSURE_LOAD_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        }
    )

    write_frame(
        validation_rows,
        experiment_dir / "exposure_load_season_forward_validation.csv",
    )
    write_frame(
        pd.DataFrame(diagnostics),
        experiment_dir / "exposure_load_calibration_diagnostics.csv",
    )
    write_frame(
        pd.DataFrame(cases),
        experiment_dir / "exposure_load_forward_diagnostic_cases.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_forward_diagnostic_sprint",
            "season_forward_validation_path": str(season_forward_validation_path),
            "feature_sets": list(EXPOSURE_LOAD_MODEL_FEATURE_SETS),
            "exposure_load_feature_columns": list(EXPOSURE_LOAD_FEATURE_COLUMNS),
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(experiment_dir / "exposure_load_forward_diagnostic.json", summary)
    write_exposure_load_forward_diagnostic_report(
        experiment_dir / "exposure_load_forward_diagnostic_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_failure_mode_sprint_experiment(
    exposure_load_features_path: str | Path,
    exposure_load_diagnostics_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    exposure_features = pd.read_csv(exposure_load_features_path)
    diagnostics = _load_records_from_path(exposure_load_diagnostics_path)
    summary = build_exposure_load_failure_mode_summary(
        _json_records(exposure_features),
        diagnostics,
    )
    feature_shift_rows = pd.DataFrame(summary["feature_shift_summary"])
    domain_shift_rows = pd.DataFrame(summary["domain_shift_summary"])

    write_frame(
        feature_shift_rows,
        experiment_dir / "exposure_load_failure_mode_features.csv",
    )
    write_frame(
        domain_shift_rows,
        experiment_dir / "exposure_load_failure_mode_domains.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_failure_mode_sprint",
            "exposure_load_features_path": str(exposure_load_features_path),
            "exposure_load_diagnostics_path": str(exposure_load_diagnostics_path),
            "exposure_load_feature_columns": list(EXPOSURE_LOAD_FEATURE_COLUMNS),
        },
    )
    _write_json(experiment_dir / "exposure_load_failure_modes.json", summary)
    write_exposure_load_failure_mode_report(
        experiment_dir / "exposure_load_failure_mode_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_guardrail_policy_sprint_experiment(
    exposure_load_failure_modes_path: str | Path,
    exposure_load_diagnostics_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    failure_mode_summary = _load_json_payload(exposure_load_failure_modes_path)
    diagnostics = _load_records_from_path(exposure_load_diagnostics_path)
    policy = build_exposure_load_guardrail_policy(failure_mode_summary, diagnostics)
    guardrail_rows = pd.DataFrame(clean_guardrail_rows(policy["guardrail_rows"]))

    write_frame(
        guardrail_rows,
        experiment_dir / "exposure_load_guardrail_policy.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_guardrail_policy_sprint",
            "exposure_load_failure_modes_path": str(exposure_load_failure_modes_path),
            "exposure_load_diagnostics_path": str(exposure_load_diagnostics_path),
        },
    )
    _write_json(experiment_dir / "exposure_load_guardrail_policy.json", policy)
    write_exposure_load_guardrail_policy_report(
        experiment_dir / "exposure_load_guardrail_policy_report.md",
        policy,
    )
    return experiment_dir


def run_exposure_load_shift_context_sprint_experiment(
    exposure_events_path: str | Path,
    exposure_participations_path: str | Path,
    exposure_load_features_path: str | Path,
    exposure_load_diagnostics_path: str | Path,
    exposure_load_failure_modes_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    exposure_events = pd.read_csv(exposure_events_path)
    exposure_participations = pd.read_csv(exposure_participations_path)
    exposure_features = pd.read_csv(exposure_load_features_path)
    diagnostics = _load_records_from_path(exposure_load_diagnostics_path)
    failure_modes = _load_json_payload(exposure_load_failure_modes_path)

    summary = build_exposure_load_shift_context_summary(
        exposure_events=_json_records(exposure_events),
        exposure_participations=_json_records(exposure_participations),
        exposure_load_features=_json_records(exposure_features),
        exposure_load_diagnostics=diagnostics,
        exposure_load_failure_modes=failure_modes,
    )

    write_frame(
        pd.DataFrame(clean_shift_context_rows(summary["shift_context_rows"])),
        experiment_dir / "exposure_load_shift_context.csv",
    )
    write_frame(
        pd.DataFrame(clean_shift_context_rows(summary["driver_context_rows"])),
        experiment_dir / "exposure_load_shift_context_drivers.csv",
    )
    write_frame(
        pd.DataFrame(clean_shift_context_rows(summary["shift_context_cases"])),
        experiment_dir / "exposure_load_shift_context_cases.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shift_context_sprint",
            "exposure_events_path": str(exposure_events_path),
            "exposure_participations_path": str(exposure_participations_path),
            "exposure_load_features_path": str(exposure_load_features_path),
            "exposure_load_diagnostics_path": str(exposure_load_diagnostics_path),
            "exposure_load_failure_modes_path": str(exposure_load_failure_modes_path),
        },
    )
    _write_json(experiment_dir / "exposure_load_shift_context.json", summary)
    write_exposure_load_shift_context_report(
        experiment_dir / "exposure_load_shift_context_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_schedule_roster_sprint_experiment(
    exposure_events_path: str | Path,
    exposure_participations_path: str | Path,
    exposure_load_shift_context_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    exposure_events = pd.read_csv(exposure_events_path)
    exposure_participations = pd.read_csv(exposure_participations_path)
    shift_context = _load_json_payload(exposure_load_shift_context_path)
    summary = build_exposure_load_schedule_roster_summary(
        exposure_events=_json_records(exposure_events),
        exposure_participations=_json_records(exposure_participations),
        exposure_load_shift_context=shift_context,
    )
    write_frame(
        pd.DataFrame(clean_context_review_rows(summary["schedule_roster_rows"])),
        experiment_dir / "exposure_load_schedule_roster_context.csv",
    )
    write_frame(
        pd.DataFrame(clean_context_review_rows(summary["schedule_roster_drivers"])),
        experiment_dir / "exposure_load_schedule_roster_drivers.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_schedule_roster_sprint",
            "exposure_events_path": str(exposure_events_path),
            "exposure_participations_path": str(exposure_participations_path),
            "exposure_load_shift_context_path": str(exposure_load_shift_context_path),
        },
    )
    _write_json(experiment_dir / "exposure_load_schedule_roster_context.json", summary)
    write_exposure_load_schedule_roster_report(
        experiment_dir / "exposure_load_schedule_roster_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_availability_capture_sprint_experiment(
    exposure_participations_path: str | Path,
    exposure_load_shift_context_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    exposure_participations = pd.read_csv(exposure_participations_path)
    shift_context = _load_json_payload(exposure_load_shift_context_path)
    summary = build_exposure_load_availability_capture_summary(
        exposure_participations=_json_records(exposure_participations),
        exposure_load_shift_context=shift_context,
    )
    write_frame(
        pd.DataFrame(clean_context_review_rows(summary["availability_capture_rows"])),
        experiment_dir / "exposure_load_availability_capture.csv",
    )
    write_frame(
        pd.DataFrame(clean_context_review_rows(summary["availability_capture_drivers"])),
        experiment_dir / "exposure_load_availability_capture_drivers.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_availability_capture_sprint",
            "exposure_participations_path": str(exposure_participations_path),
            "exposure_load_shift_context_path": str(exposure_load_shift_context_path),
        },
    )
    _write_json(experiment_dir / "exposure_load_availability_capture.json", summary)
    write_exposure_load_availability_capture_report(
        experiment_dir / "exposure_load_availability_capture_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_context_decision_sprint_experiment(
    exposure_load_shift_context_path: str | Path,
    exposure_load_schedule_roster_path: str | Path,
    exposure_load_availability_capture_path: str | Path,
    exposure_load_guardrail_policy_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    shift_context = _load_json_payload(exposure_load_shift_context_path)
    schedule_roster = _load_json_payload(exposure_load_schedule_roster_path)
    availability_capture = _load_json_payload(exposure_load_availability_capture_path)
    guardrail_policy = _load_json_payload(exposure_load_guardrail_policy_path)
    summary = build_exposure_load_context_decision_summary(
        exposure_load_shift_context=shift_context,
        schedule_roster_summary=schedule_roster,
        availability_capture_summary=availability_capture,
        guardrail_policy=guardrail_policy,
    )
    write_frame(
        pd.DataFrame(clean_context_review_rows(summary["decision_rows"])),
        experiment_dir / "exposure_load_context_decision.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_context_decision_sprint",
            "exposure_load_shift_context_path": str(exposure_load_shift_context_path),
            "exposure_load_schedule_roster_path": str(
                exposure_load_schedule_roster_path
            ),
            "exposure_load_availability_capture_path": str(
                exposure_load_availability_capture_path
            ),
            "exposure_load_guardrail_policy_path": str(
                exposure_load_guardrail_policy_path
            ),
        },
    )
    _write_json(experiment_dir / "exposure_load_context_decision.json", summary)
    write_exposure_load_context_decision_report(
        experiment_dir / "exposure_load_context_decision_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_source_context_classification_sprint_experiment(
    exposure_events_path: str | Path,
    exposure_participations_path: str | Path,
    exposure_load_shift_context_path: str | Path,
    exposure_load_schedule_roster_path: str | Path,
    exposure_load_availability_capture_path: str | Path,
    exposure_load_context_decision_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    exposure_events = pd.read_csv(exposure_events_path)
    exposure_participations = pd.read_csv(exposure_participations_path)
    shift_context = _load_json_payload(exposure_load_shift_context_path)
    schedule_roster = _load_json_payload(exposure_load_schedule_roster_path)
    availability_capture = _load_json_payload(exposure_load_availability_capture_path)
    context_decision = _load_json_payload(exposure_load_context_decision_path)
    summary = build_exposure_load_source_context_classification_summary(
        exposure_events=_json_records(exposure_events),
        exposure_participations=_json_records(exposure_participations),
        exposure_load_shift_context=shift_context,
        exposure_load_schedule_roster=schedule_roster,
        exposure_load_availability_capture=availability_capture,
        exposure_load_context_decision=context_decision,
    )
    write_frame(
        pd.DataFrame(
            clean_source_context_rows(summary["source_context_classification_rows"])
        ),
        experiment_dir / "exposure_load_source_context_classification.csv",
    )
    write_frame(
        pd.DataFrame(clean_source_context_rows(summary["source_evidence_rows"])),
        experiment_dir / "exposure_load_source_context_evidence.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": (
                "exposure_load_source_context_classification_sprint"
            ),
            "exposure_events_path": str(exposure_events_path),
            "exposure_participations_path": str(exposure_participations_path),
            "exposure_load_shift_context_path": str(exposure_load_shift_context_path),
            "exposure_load_schedule_roster_path": str(
                exposure_load_schedule_roster_path
            ),
            "exposure_load_availability_capture_path": str(
                exposure_load_availability_capture_path
            ),
            "exposure_load_context_decision_path": str(
                exposure_load_context_decision_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_source_context_classification.json",
        summary,
    )
    write_exposure_load_source_context_classification_report(
        experiment_dir / "exposure_load_source_context_classification_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_source_resolution_sprint_experiment(
    exposure_load_source_context_classification_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    source_context = _load_json_payload(exposure_load_source_context_classification_path)
    summary = build_exposure_load_source_resolution_policy(
        source_context_classification=source_context,
    )
    write_frame(
        pd.DataFrame(clean_source_resolution_rows(summary["policy_rows"])),
        experiment_dir / "exposure_load_source_resolution.csv",
    )
    write_frame(
        pd.DataFrame(clean_source_resolution_rows(summary["resolution_actions"])),
        experiment_dir / "exposure_load_source_resolution_actions.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_source_resolution_sprint",
            "exposure_load_source_context_classification_path": str(
                exposure_load_source_context_classification_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_source_resolution_policy.json",
        summary,
    )
    write_exposure_load_source_resolution_report(
        experiment_dir / "exposure_load_source_resolution_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_source_eligible_calibration_sprint_experiment(
    season_forward_validation_path: str | Path,
    exposure_load_source_resolution_policy_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    validation_rows = pd.read_csv(season_forward_validation_path)
    source_resolution_policy = _load_json_payload(
        exposure_load_source_resolution_policy_path
    )
    summary = build_exposure_load_source_eligible_calibration_summary(
        validation_rows=_json_records(validation_rows),
        source_resolution_policy=source_resolution_policy,
    )
    write_frame(
        pd.DataFrame(
            clean_source_eligible_calibration_rows(summary["calibration_rows"])
        ),
        experiment_dir / "exposure_load_source_eligible_calibration.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_source_eligible_calibration_rows(
                summary["calibration_diagnostics"]
            )
        ),
        experiment_dir / "exposure_load_source_eligible_calibration_diagnostics.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_source_eligible_calibration_sprint",
            "season_forward_validation_path": str(season_forward_validation_path),
            "exposure_load_source_resolution_policy_path": str(
                exposure_load_source_resolution_policy_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_source_eligible_calibration.json",
        summary,
    )
    write_exposure_load_source_eligible_calibration_report(
        experiment_dir / "exposure_load_source_eligible_calibration_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_source_eligible_policy_sprint_experiment(
    season_forward_validation_path: str | Path,
    exposure_load_source_eligible_calibration_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    validation_rows = pd.read_csv(season_forward_validation_path)
    source_eligible_calibration = _load_json_payload(
        exposure_load_source_eligible_calibration_path
    )
    summary = build_exposure_load_source_eligible_policy_package(
        validation_rows=_json_records(validation_rows),
        source_eligible_calibration=source_eligible_calibration,
    )
    write_frame(
        pd.DataFrame(clean_source_eligible_policy_rows(summary["policy_rows"])),
        experiment_dir / "exposure_load_source_eligible_policy.csv",
    )
    write_frame(
        pd.DataFrame(clean_source_eligible_policy_rows(summary["threshold_rows"])),
        experiment_dir / "exposure_load_source_eligible_thresholds.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_source_eligible_policy_sprint",
            "season_forward_validation_path": str(season_forward_validation_path),
            "exposure_load_source_eligible_calibration_path": str(
                exposure_load_source_eligible_calibration_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_source_eligible_policy.json",
        summary,
    )
    write_exposure_load_source_eligible_policy_report(
        experiment_dir / "exposure_load_source_eligible_policy_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_source_eligible_shadow_monitoring_sprint_experiment(
    season_forward_validation_path: str | Path,
    exposure_load_source_eligible_policy_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    validation_rows = pd.read_csv(season_forward_validation_path)
    source_eligible_policy = _load_json_payload(
        exposure_load_source_eligible_policy_path
    )
    summary = build_exposure_load_source_eligible_shadow_monitoring_review(
        validation_rows=_json_records(validation_rows),
        source_eligible_policy=source_eligible_policy,
    )
    write_frame(
        pd.DataFrame(
            clean_source_eligible_shadow_monitoring_rows(
                summary["monitoring_rows"]
            )
        ),
        experiment_dir / "exposure_load_source_eligible_shadow_monitoring.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_source_eligible_shadow_monitoring_rows(
                summary["monitoring_season_rows"]
            )
        ),
        experiment_dir
        / "exposure_load_source_eligible_shadow_monitoring_seasons.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": (
                "exposure_load_source_eligible_shadow_monitoring_sprint"
            ),
            "season_forward_validation_path": str(season_forward_validation_path),
            "exposure_load_source_eligible_policy_path": str(
                exposure_load_source_eligible_policy_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_source_eligible_shadow_monitoring.json",
        summary,
    )
    write_exposure_load_source_eligible_shadow_monitoring_report(
        experiment_dir
        / "exposure_load_source_eligible_shadow_monitoring_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_channel_lock_sprint_experiment(
    exposure_load_source_eligible_shadow_monitoring_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    source_eligible_shadow_monitoring = _load_json_payload(
        exposure_load_source_eligible_shadow_monitoring_path
    )
    summary = build_exposure_load_shadow_channel_lock(
        source_eligible_shadow_monitoring
    )
    write_frame(
        pd.DataFrame(clean_shadow_launch_rows(summary["locked_channels"])),
        experiment_dir / "exposure_load_shadow_channel_lock.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_launch_rows(summary["held_channels"])),
        experiment_dir / "exposure_load_shadow_channel_lock_held_channels.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_channel_lock_sprint",
            "exposure_load_source_eligible_shadow_monitoring_path": str(
                exposure_load_source_eligible_shadow_monitoring_path
            ),
        },
    )
    _write_json(experiment_dir / "exposure_load_shadow_channel_lock.json", summary)
    write_exposure_load_shadow_channel_lock_report(
        experiment_dir / "exposure_load_shadow_channel_lock_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_review_protocol_sprint_experiment(
    exposure_load_shadow_channel_lock_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    shadow_channel_lock = _load_json_payload(exposure_load_shadow_channel_lock_path)
    summary = build_exposure_load_shadow_review_protocol(shadow_channel_lock)
    write_frame(
        pd.DataFrame(clean_shadow_launch_rows(summary["protocol_rows"])),
        experiment_dir / "exposure_load_shadow_review_protocol.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_review_protocol_sprint",
            "exposure_load_shadow_channel_lock_path": str(
                exposure_load_shadow_channel_lock_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_review_protocol.json",
        summary,
    )
    write_exposure_load_shadow_review_protocol_report(
        experiment_dir / "exposure_load_shadow_review_protocol_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_readiness_register_sprint_experiment(
    exposure_load_shadow_channel_lock_path: str | Path,
    exposure_load_shadow_review_protocol_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    shadow_channel_lock = _load_json_payload(exposure_load_shadow_channel_lock_path)
    shadow_review_protocol = _load_json_payload(
        exposure_load_shadow_review_protocol_path
    )
    summary = build_exposure_load_shadow_readiness_register(
        shadow_channel_lock,
        shadow_review_protocol,
    )
    write_frame(
        pd.DataFrame(clean_shadow_launch_rows(summary["readiness_rows"])),
        experiment_dir / "exposure_load_shadow_readiness_register.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_readiness_register_sprint",
            "exposure_load_shadow_channel_lock_path": str(
                exposure_load_shadow_channel_lock_path
            ),
            "exposure_load_shadow_review_protocol_path": str(
                exposure_load_shadow_review_protocol_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_readiness_register.json",
        summary,
    )
    write_exposure_load_shadow_readiness_register_report(
        experiment_dir / "exposure_load_shadow_readiness_register_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_replay_sprint_experiment(
    season_forward_validation_path: str | Path,
    exposure_load_shadow_channel_lock_path: str | Path,
    exposure_load_shadow_review_protocol_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    validation_rows = pd.read_csv(season_forward_validation_path)
    shadow_channel_lock = _load_json_payload(exposure_load_shadow_channel_lock_path)
    shadow_review_protocol = _load_json_payload(
        exposure_load_shadow_review_protocol_path
    )
    summary = build_exposure_load_shadow_replay_package(
        validation_rows=_json_records(validation_rows),
        shadow_channel_lock=shadow_channel_lock,
        shadow_review_protocol=shadow_review_protocol,
    )
    write_frame(
        pd.DataFrame(clean_shadow_replay_rows(summary["replay_rows"])),
        experiment_dir / "exposure_load_shadow_replay_log.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_replay_rows(summary["review_packet_rows"])),
        experiment_dir / "exposure_load_shadow_review_packets.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_replay_rows(summary["stop_rule_rows"])),
        experiment_dir / "exposure_load_shadow_stop_rules.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_replay_sprint",
            "season_forward_validation_path": str(season_forward_validation_path),
            "exposure_load_shadow_channel_lock_path": str(
                exposure_load_shadow_channel_lock_path
            ),
            "exposure_load_shadow_review_protocol_path": str(
                exposure_load_shadow_review_protocol_path
            ),
        },
    )
    _write_json(experiment_dir / "exposure_load_shadow_replay.json", summary)
    write_exposure_load_shadow_replay_report(
        experiment_dir / "exposure_load_shadow_replay_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_event_crosswalk_sprint_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    exposure_participations_path: str | Path,
    exposure_load_shadow_replay_path: str | Path,
    exposure_load_shadow_collection_path: str | Path | None,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)
    exposure_participations = pd.read_csv(exposure_participations_path)
    shadow_replay = _load_json_payload(exposure_load_shadow_replay_path)
    retained_packet_ids = _shadow_collection_packet_ids(
        exposure_load_shadow_collection_path
    )
    packet_rows = [
        row
        for row in shadow_replay.get("replay_rows", [])
        if str(row.get("replay_status")) == "ready_for_research_adjudication"
        and bool(row.get("source_eligible", True))
        and (
            retained_packet_ids is None
            or str(row.get("review_packet_id")) in retained_packet_ids
        )
    ]
    packet_contexts = _shadow_event_crosswalk_packet_contexts(
        measurements=measurements,
        canonical_injuries=canonical_injuries,
        detailed_injuries=detailed_injuries,
        exposure_participations=exposure_participations,
        packet_rows=packet_rows,
        graph_window_size=graph_window_size,
        model_variant=model_variant,
    )
    summary = build_shadow_event_crosswalk_summary(
        packet_contexts,
        detailed_injuries,
    )

    write_frame(
        pd.DataFrame(
            clean_shadow_event_crosswalk_rows(summary["event_crosswalk_rows"])
        ),
        experiment_dir / "exposure_load_shadow_event_crosswalk.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_event_crosswalk_rows(summary["packet_summary_rows"])
        ),
        experiment_dir / "exposure_load_shadow_event_crosswalk_summary.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_event_crosswalk_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "exposure_participations_path": str(exposure_participations_path),
            "exposure_load_shadow_replay_path": str(exposure_load_shadow_replay_path),
            "exposure_load_shadow_collection_path": (
                str(exposure_load_shadow_collection_path)
                if exposure_load_shadow_collection_path is not None
                else None
            ),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_event_crosswalk.json",
        summary,
    )
    write_exposure_load_shadow_event_crosswalk_report(
        experiment_dir / "exposure_load_shadow_event_crosswalk_report.md",
        summary,
    )
    return experiment_dir


def _shadow_collection_packet_ids(
    collection_path: str | Path | None,
) -> set[str] | None:
    if collection_path is None:
        return None
    collection_rows = pd.read_csv(collection_path)
    if "collection_packet_id" not in collection_rows:
        return None
    packet_ids = {
        str(value)
        for value in collection_rows["collection_packet_id"].dropna().tolist()
        if str(value)
    }
    return packet_ids or None


def run_exposure_load_shadow_adjudication_sprint_experiment(
    exposure_load_shadow_replay_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    shadow_replay = _load_json_payload(exposure_load_shadow_replay_path)
    summary = build_exposure_load_shadow_adjudication_package(shadow_replay)
    write_frame(
        pd.DataFrame(clean_shadow_adjudication_rows(summary["schema_rows"])),
        experiment_dir / "exposure_load_shadow_adjudication_schema.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_adjudication_rows(summary["adjudication_template_rows"])
        ),
        experiment_dir / "exposure_load_shadow_adjudication_template.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_adjudication_rows(summary["completion_check_rows"])
        ),
        experiment_dir / "exposure_load_shadow_adjudication_completion.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_adjudication_sprint",
            "exposure_load_shadow_replay_path": str(
                exposure_load_shadow_replay_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_adjudication.json",
        summary,
    )
    write_exposure_load_shadow_adjudication_report(
        experiment_dir / "exposure_load_shadow_adjudication_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_adjudication_summary_sprint_experiment(
    exposure_load_shadow_adjudication_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    adjudication_rows = pd.read_csv(exposure_load_shadow_adjudication_path)
    summary = build_exposure_load_shadow_adjudication_summary(
        _json_records(adjudication_rows)
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_adjudication_rows(summary["validation_rows"])
        ),
        experiment_dir / "exposure_load_shadow_adjudication_validation.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_adjudication_rows(summary["channel_summary_rows"])
        ),
        experiment_dir / "exposure_load_shadow_adjudication_channel_summary.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_adjudication_summary",
            "exposure_load_shadow_adjudication_path": str(
                exposure_load_shadow_adjudication_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_adjudication_summary.json",
        summary,
    )
    write_exposure_load_shadow_adjudication_summary_report(
        experiment_dir / "exposure_load_shadow_adjudication_summary_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_adjudication_decision_sprint_experiment(
    exposure_load_shadow_adjudication_summary_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    adjudication_summary = _load_json_payload(
        exposure_load_shadow_adjudication_summary_path
    )
    decision = build_exposure_load_shadow_adjudication_decision_package(
        adjudication_summary
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_adjudication_rows(decision["channel_decision_rows"])
        ),
        experiment_dir / "exposure_load_shadow_adjudication_channel_decisions.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_adjudication_decision",
            "exposure_load_shadow_adjudication_summary_path": str(
                exposure_load_shadow_adjudication_summary_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_adjudication_decision.json",
        decision,
    )
    write_exposure_load_shadow_adjudication_decision_report(
        experiment_dir / "exposure_load_shadow_adjudication_decision_report.md",
        decision,
    )
    return experiment_dir


def run_exposure_load_shadow_monitoring_plan_sprint_experiment(
    exposure_load_shadow_adjudication_decision_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    adjudication_decision = _load_json_payload(
        exposure_load_shadow_adjudication_decision_path
    )
    plan = build_exposure_load_shadow_monitoring_plan(adjudication_decision)
    write_frame(
        pd.DataFrame(clean_shadow_monitoring_rows(plan["retained_channel_rows"])),
        experiment_dir / "exposure_load_shadow_monitoring_plan.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_monitoring_rows(plan["paused_channel_rows"])),
        experiment_dir / "exposure_load_shadow_monitoring_paused_channels.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_monitoring_rows(plan["evidence_gate_rows"])),
        experiment_dir / "exposure_load_shadow_monitoring_evidence_gates.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_monitoring_plan",
            "exposure_load_shadow_adjudication_decision_path": str(
                exposure_load_shadow_adjudication_decision_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_monitoring_plan.json",
        plan,
    )
    write_exposure_load_shadow_monitoring_plan_report(
        experiment_dir / "exposure_load_shadow_monitoring_plan_report.md",
        plan,
    )
    return experiment_dir


def run_exposure_load_shadow_collection_template_sprint_experiment(
    exposure_load_shadow_monitoring_plan_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    monitoring_plan = _load_json_payload(exposure_load_shadow_monitoring_plan_path)
    template = build_exposure_load_shadow_collection_template(monitoring_plan)
    write_frame(
        pd.DataFrame(clean_shadow_collection_rows(template["schema_rows"])),
        experiment_dir / "exposure_load_shadow_collection_schema.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_collection_rows(template["collection_template_rows"])
        ),
        experiment_dir / "exposure_load_shadow_collection_template.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_collection_rows(template["completion_check_rows"])
        ),
        experiment_dir / "exposure_load_shadow_collection_completion.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_collection_template",
            "exposure_load_shadow_monitoring_plan_path": str(
                exposure_load_shadow_monitoring_plan_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_collection_template.json",
        template,
    )
    write_exposure_load_shadow_collection_template_report(
        experiment_dir / "exposure_load_shadow_collection_template_report.md",
        template,
    )
    return experiment_dir


def run_exposure_load_shadow_collection_summary_sprint_experiment(
    exposure_load_shadow_collection_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    collection_rows = pd.read_csv(exposure_load_shadow_collection_path)
    summary = build_exposure_load_shadow_collection_summary(
        _json_records(collection_rows)
    )
    write_frame(
        pd.DataFrame(clean_shadow_collection_rows(summary["validation_rows"])),
        experiment_dir / "exposure_load_shadow_collection_validation.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_collection_rows(summary["channel_summary_rows"])),
        experiment_dir / "exposure_load_shadow_collection_channel_summary.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_collection_summary",
            "exposure_load_shadow_collection_path": str(
                exposure_load_shadow_collection_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_collection_summary.json",
        summary,
    )
    write_exposure_load_shadow_collection_summary_report(
        experiment_dir / "exposure_load_shadow_collection_summary_report.md",
        summary,
    )
    return experiment_dir


def run_exposure_load_shadow_collection_packet_workflow_sprint_experiment(
    exposure_load_shadow_collection_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    collection_rows = pd.read_csv(exposure_load_shadow_collection_path)
    workflow = build_exposure_load_shadow_collection_packet_workflow(
        _json_records(collection_rows)
    )
    write_frame(
        pd.DataFrame(clean_shadow_collection_rows(workflow["packet_manifest_rows"])),
        experiment_dir / "exposure_load_shadow_collection_packet_manifest.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_collection_rows(workflow["packet_checklist_rows"])),
        experiment_dir / "exposure_load_shadow_collection_packet_checklist.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_collection_rows(workflow["packet_audit_trail_rows"])
        ),
        experiment_dir / "exposure_load_shadow_collection_packet_audit_trail.csv",
    )
    for document in workflow["packet_documents"]:
        packet_path = experiment_dir / str(document["packet_filename"])
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(str(document["content"]), encoding="utf-8")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_collection_packet_workflow",
            "exposure_load_shadow_collection_path": str(
                exposure_load_shadow_collection_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_collection_packet_workflow.json",
        workflow,
    )
    write_exposure_load_shadow_collection_reviewer_instructions(
        experiment_dir / "exposure_load_shadow_collection_reviewer_instructions.md",
        workflow,
    )
    write_exposure_load_shadow_collection_packet_workflow_report(
        experiment_dir / "exposure_load_shadow_collection_packet_workflow_report.md",
        workflow,
    )
    return experiment_dir


def run_exposure_load_shadow_collection_evidence_prefill_sprint_experiment(
    exposure_load_shadow_review_packets_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    review_packets = pd.read_csv(exposure_load_shadow_review_packets_path)
    prefill = build_exposure_load_shadow_collection_evidence_prefill(
        _json_records(review_packets)
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_collection_rows(prefill["prefilled_collection_rows"])
        ),
        experiment_dir / "exposure_load_shadow_collection_prefilled.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_collection_rows(prefill["excluded_rows"])),
        experiment_dir / "exposure_load_shadow_collection_prefill_excluded.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_collection_rows(prefill["validation_rows"])),
        experiment_dir / "exposure_load_shadow_collection_prefill_validation.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_collection_evidence_prefill",
            "exposure_load_shadow_review_packets_path": str(
                exposure_load_shadow_review_packets_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_collection_evidence_prefill.json",
        prefill,
    )
    write_exposure_load_shadow_collection_evidence_prefill_report(
        experiment_dir / "exposure_load_shadow_collection_evidence_prefill_report.md",
        prefill,
    )
    return experiment_dir


def run_exposure_load_shadow_calibration_readiness_sprint_experiment(
    exposure_load_shadow_collection_summary_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    collection_summary = _load_json_payload(
        exposure_load_shadow_collection_summary_path
    )
    readiness = build_exposure_load_shadow_calibration_readiness_review(
        collection_summary
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_calibration_readiness_rows(
                readiness["channel_readiness_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_calibration_readiness_channels.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_calibration_readiness_rows(
                readiness["evidence_gap_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_calibration_readiness_gaps.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_calibration_readiness",
            "exposure_load_shadow_collection_summary_path": str(
                exposure_load_shadow_collection_summary_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_calibration_readiness.json",
        readiness,
    )
    write_exposure_load_shadow_calibration_readiness_report(
        experiment_dir / "exposure_load_shadow_calibration_readiness_report.md",
        readiness,
    )
    return experiment_dir


def run_exposure_load_shadow_calibration_sensitivity_sprint_experiment(
    exposure_load_shadow_calibration_readiness_path: str | Path,
    exposure_load_shadow_collection_path: str | Path,
    exposure_load_shadow_event_crosswalk_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    calibration_readiness = _load_json_payload(
        exposure_load_shadow_calibration_readiness_path
    )
    collection_rows = pd.read_csv(exposure_load_shadow_collection_path)
    event_crosswalk_rows = pd.read_csv(exposure_load_shadow_event_crosswalk_path)
    sensitivity = build_exposure_load_shadow_calibration_sensitivity_review(
        calibration_readiness=calibration_readiness,
        collection_rows=_json_records(collection_rows),
        event_crosswalk_rows=_json_records(event_crosswalk_rows),
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_calibration_sensitivity_rows(
                sensitivity["sensitivity_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_calibration_sensitivity.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_calibration_sensitivity_rows(
                sensitivity["evidence_dossier_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_evidence_dossier.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_calibration_sensitivity_rows(
                sensitivity["error_mode_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_error_modes.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_calibration_sensitivity",
            "exposure_load_shadow_calibration_readiness_path": str(
                exposure_load_shadow_calibration_readiness_path
            ),
            "exposure_load_shadow_collection_path": str(
                exposure_load_shadow_collection_path
            ),
            "exposure_load_shadow_event_crosswalk_path": str(
                exposure_load_shadow_event_crosswalk_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_calibration_sensitivity.json",
        sensitivity,
    )
    write_exposure_load_shadow_calibration_sensitivity_report(
        experiment_dir / "exposure_load_shadow_calibration_sensitivity_report.md",
        sensitivity,
    )
    return experiment_dir


def run_exposure_load_shadow_error_control_sprint_experiment(
    exposure_load_shadow_calibration_sensitivity_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    sensitivity = _load_json_payload(exposure_load_shadow_calibration_sensitivity_path)
    review = build_exposure_load_shadow_error_control_review(sensitivity)
    write_frame(
        pd.DataFrame(clean_shadow_error_control_rows(review["decision_rows"])),
        experiment_dir / "exposure_load_shadow_error_control_decisions.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_error_control_rows(review["refined_evidence_dossier_rows"])
        ),
        experiment_dir / "exposure_load_shadow_error_control_evidence_dossier.csv",
    )
    write_frame(
        pd.DataFrame(clean_shadow_error_control_rows(review["error_control_rows"])),
        experiment_dir / "exposure_load_shadow_error_controls.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_error_control",
            "exposure_load_shadow_calibration_sensitivity_path": str(
                exposure_load_shadow_calibration_sensitivity_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_error_control_policy.json",
        review,
    )
    write_exposure_load_shadow_error_control_report(
        experiment_dir / "exposure_load_shadow_error_control_report.md",
        review,
    )
    return experiment_dir


def run_exposure_load_shadow_bounded_calibration_protocol_sprint_experiment(
    exposure_load_shadow_error_control_policy_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    error_control_policy = _load_json_payload(
        exposure_load_shadow_error_control_policy_path
    )
    protocol = build_exposure_load_shadow_bounded_calibration_protocol(
        error_control_policy
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_bounded_calibration_protocol_rows(
                protocol["channel_protocol_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_bounded_calibration_protocol_channels.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_bounded_calibration_protocol_rows(
                protocol["evidence_use_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_bounded_calibration_evidence_use.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_bounded_calibration_protocol_rows(
                protocol["protocol_gate_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_bounded_calibration_protocol_gates.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_bounded_calibration_protocol",
            "exposure_load_shadow_error_control_policy_path": str(
                exposure_load_shadow_error_control_policy_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_bounded_calibration_protocol.json",
        protocol,
    )
    write_exposure_load_shadow_bounded_calibration_protocol_report(
        experiment_dir
        / "exposure_load_shadow_bounded_calibration_protocol_report.md",
        protocol,
    )
    return experiment_dir


def run_exposure_load_shadow_bounded_calibration_stress_test_sprint_experiment(
    exposure_load_shadow_bounded_calibration_protocol_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    protocol = _load_json_payload(exposure_load_shadow_bounded_calibration_protocol_path)
    stress_test = build_exposure_load_shadow_bounded_calibration_stress_test(protocol)
    write_frame(
        pd.DataFrame(
            clean_shadow_bounded_calibration_stress_test_rows(
                stress_test["channel_stress_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_bounded_calibration_stress_channels.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_bounded_calibration_stress_test_rows(
                stress_test["stress_scenario_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_bounded_calibration_stress_scenarios.csv",
    )
    write_frame(
        pd.DataFrame(
            clean_shadow_bounded_calibration_stress_test_rows(
                stress_test["stress_gate_rows"]
            )
        ),
        experiment_dir / "exposure_load_shadow_bounded_calibration_stress_gates.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "exposure_load_shadow_bounded_calibration_stress_test",
            "exposure_load_shadow_bounded_calibration_protocol_path": str(
                exposure_load_shadow_bounded_calibration_protocol_path
            ),
        },
    )
    _write_json(
        experiment_dir / "exposure_load_shadow_bounded_calibration_stress_test.json",
        stress_test,
    )
    write_exposure_load_shadow_bounded_calibration_stress_test_report(
        experiment_dir
        / "exposure_load_shadow_bounded_calibration_stress_test_report.md",
        stress_test,
    )
    return experiment_dir


def run_window_sensitivity_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_sizes: tuple[int, ...],
    model_variant: str = "baseline",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    window_sizes = _normalize_window_sizes(graph_window_sizes)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)

    windows: dict[str, object] = {}
    first_summary: dict[str, object] | None = None
    for window_size in window_sizes:
        graph_features = build_graph_snapshots(matrix, window_size=window_size)
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        labeled = attach_time_to_event_labels(graph_features, injuries)
        if labeled.empty:
            raise ValueError("no labeled graph snapshots produced")
        model_result = train_discrete_time_risk_model(
            labeled,
            model_variant=model_variant,
        )
        evaluation = evaluate_risk_model(model_result.timeline, model_result.summary)
        if first_summary is None:
            first_summary = model_result.summary
        windows[str(window_size)] = {
            "graph_window_size": window_size,
            "athlete_count": int(labeled["athlete_id"].nunique()),
            "snapshot_count": int(len(labeled)),
            "feature_columns": model_result.summary["feature_columns"],
            "horizons": evaluation["horizons"],
        }

    if first_summary is None:
        raise ValueError("no graph window sizes provided")

    sensitivity = {
        "model_type": MODEL_TYPE,
        "event_policy": first_summary["event_policy"],
        "model_variant": model_variant,
        "split_policy": first_summary["split_policy"],
        "graph_window_sizes": list(window_sizes),
        "windows": windows,
        "best_by_horizon": _best_windows_by_horizon(windows),
    }
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "window_sensitivity",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_sizes": list(window_sizes),
            "model_variant": model_variant,
            "horizons": list(DEFAULT_HORIZONS),
        },
    )
    _write_json(experiment_dir / "window_sensitivity.json", sensitivity)
    _write_window_sensitivity_report(
        experiment_dir / "window_sensitivity_report.md",
        sensitivity,
    )
    return experiment_dir


def run_model_robustness_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    split_count: int = 5,
) -> Path:
    if split_count < 1:
        raise ValueError("split_count must be at least 1")
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    labeled = attach_time_to_event_labels(graph_features, injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")

    split_test_ids = _rotating_holdout_splits(labeled["athlete_id"], split_count)
    variants: dict[str, object] = {}
    first_summary: dict[str, object] | None = None
    for variant in MODEL_VARIANTS:
        split_payloads: dict[str, object] = {}
        for split_seed, test_ids in enumerate(split_test_ids):
            model_result = train_discrete_time_risk_model(
                labeled,
                test_athlete_ids=test_ids,
                model_variant=variant,
            )
            evaluation = evaluate_risk_model(model_result.timeline, model_result.summary)
            if first_summary is None:
                first_summary = model_result.summary
            split_payloads[str(split_seed)] = {
                "test_athlete_ids": list(test_ids),
                "test_athlete_count": len(test_ids),
                "horizons": evaluation["horizons"],
            }
        variants[variant] = {
            "splits": split_payloads,
            "summary_by_horizon": _aggregate_variant_splits(split_payloads),
        }

    if first_summary is None:
        raise ValueError("no model variants evaluated")

    robustness = {
        "experiment_type": "model_robustness_sprint",
        "model_type": MODEL_TYPE,
        "event_policy": first_summary["event_policy"],
        "split_policy": "athlete_level_deterministic_rotating_holdout_20pct",
        "graph_window_size": graph_window_size,
        "model_variants": list(MODEL_VARIANTS),
        "split_count": split_count,
        "split_seeds": list(range(split_count)),
        "horizons": list(DEFAULT_HORIZONS),
        "feature_columns": list(GRAPH_SNAPSHOT_FEATURE_COLUMNS),
        "variants": variants,
        "decision_modes": _decision_mode_winners(variants),
    }
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "model_robustness_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_size": graph_window_size,
            "split_count": split_count,
            "model_variants": list(MODEL_VARIANTS),
            "horizons": list(DEFAULT_HORIZONS),
        },
    )
    _write_json(experiment_dir / "model_robustness.json", robustness)
    _write_model_robustness_report(
        experiment_dir / "model_robustness_report.md",
        robustness,
    )
    return experiment_dir


def run_window_model_robustness_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_sizes: tuple[int, ...],
    split_count: int = 5,
) -> Path:
    if split_count < 1:
        raise ValueError("split_count must be at least 1")
    experiment_dir = _experiment_path(output_dir, experiment_id)
    window_sizes = _normalize_window_sizes(graph_window_sizes)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)

    windows: dict[str, object] = {}
    first_summary: dict[str, object] | None = None
    for window_size in window_sizes:
        graph_features = build_graph_snapshots(matrix, window_size=window_size)
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        labeled = attach_time_to_event_labels(graph_features, injuries)
        if labeled.empty:
            raise ValueError("no labeled graph snapshots produced")
        split_test_ids = _rotating_holdout_splits(labeled["athlete_id"], split_count)
        variants: dict[str, object] = {}
        for variant in MODEL_VARIANTS:
            split_payloads: dict[str, object] = {}
            for split_seed, test_ids in enumerate(split_test_ids):
                model_result = train_discrete_time_risk_model(
                    labeled,
                    test_athlete_ids=test_ids,
                    model_variant=variant,
                )
                evaluation = evaluate_risk_model(
                    model_result.timeline,
                    model_result.summary,
                )
                if first_summary is None:
                    first_summary = model_result.summary
                split_payloads[str(split_seed)] = {
                    "test_athlete_ids": list(test_ids),
                    "test_athlete_count": len(test_ids),
                    "horizons": evaluation["horizons"],
                }
            variants[variant] = {
                "splits": split_payloads,
                "summary_by_horizon": _aggregate_variant_splits(split_payloads),
            }
        windows[str(window_size)] = {
            "graph_window_size": window_size,
            "athlete_count": int(labeled["athlete_id"].nunique()),
            "snapshot_count": int(len(labeled)),
            "variants": variants,
            "decision_modes": _decision_mode_winners(variants),
        }

    if first_summary is None:
        raise ValueError("no graph window sizes provided")

    robustness = {
        "experiment_type": "window_model_robustness_sprint",
        "model_type": MODEL_TYPE,
        "event_policy": first_summary["event_policy"],
        "split_policy": "athlete_level_deterministic_rotating_holdout_20pct",
        "graph_window_sizes": list(window_sizes),
        "model_variants": list(MODEL_VARIANTS),
        "split_count": split_count,
        "split_seeds": list(range(split_count)),
        "horizons": list(DEFAULT_HORIZONS),
        "feature_columns": list(GRAPH_SNAPSHOT_FEATURE_COLUMNS),
        "windows": windows,
        "overall_decision_modes": _overall_window_model_decision_winners(windows),
    }
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "window_model_robustness_sprint",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_sizes": list(window_sizes),
            "split_count": split_count,
            "model_variants": list(MODEL_VARIANTS),
            "horizons": list(DEFAULT_HORIZONS),
        },
    )
    _write_json(experiment_dir / "window_model_robustness.json", robustness)
    _write_window_model_robustness_report(
        experiment_dir / "window_model_robustness_report.md",
        robustness,
    )
    return experiment_dir


def run_calibration_threshold_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
    model_variant: str = "l2",
    split_count: int = 5,
) -> Path:
    if split_count < 1:
        raise ValueError("split_count must be at least 1")
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    if graph_features.empty:
        raise ValueError("no graph snapshots produced")
    labeled = attach_time_to_event_labels(graph_features, injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")

    split_test_ids = _rotating_holdout_splits(labeled["athlete_id"], split_count)

    oof_predictions: dict[int, dict[str, float]] = {}
    event_policy: str | None = None

    for test_ids in split_test_ids:
        model_result = train_discrete_time_risk_model(
            labeled,
            test_athlete_ids=test_ids,
            model_variant=model_variant,
        )
        if event_policy is None:
            event_policy = model_result.summary["event_policy"]
        test_mask = model_result.timeline["athlete_id"].astype(str).isin(
            [str(i) for i in test_ids]
        )
        for idx in model_result.timeline.index[test_mask]:
            oof_predictions[idx] = {
                str(horizon): float(
                    model_result.timeline.at[idx, f"risk_{horizon}d"]
                )
                for horizon in DEFAULT_HORIZONS
            }

    event_policy = event_policy or "event_observed"
    horizon_summaries: dict[str, object] = {}
    threshold_rows: list[dict[str, object]] = []

    for horizon in DEFAULT_HORIZONS:
        label_column = f"event_within_{horizon}d"
        oof_indices = sorted(oof_predictions.keys())
        predictions = pd.Series(
            [oof_predictions[idx][str(horizon)] for idx in oof_indices],
            index=oof_indices,
        )
        labels = _oof_labels(labeled, oof_indices, label_column, event_policy)

        model_brier = (
            float(brier_score_loss(labels.astype(int), predictions))
            if len(labels) > 0
            else None
        )
        oof_positive_rate = float(labels.mean()) if len(labels) else 0.0
        prevalence_brier = (
            float(brier_score_loss(
                labels.astype(int),
                pd.Series(oof_positive_rate, index=labels.index),
            ))
            if len(labels) > 0
            else None
        )
        brier_skill = _brier_skill(model_brier, prevalence_brier)

        n_bins = min(10, max(1, len(predictions) // 3))
        bins = build_calibration_bins(predictions, labels, n_bins=n_bins)
        horizon_threshold_rows = build_threshold_table(predictions, labels)
        for row in horizon_threshold_rows:
            threshold_rows.append({"horizon": horizon, **row})

        horizon_summaries[str(horizon)] = {
            "oof_snapshot_count": len(oof_indices),
            "oof_positive_count": int(labels.sum()),
            "oof_positive_rate": oof_positive_rate if len(labels) else None,
            "brier_score": model_brier,
            "brier_skill_score": brier_skill,
            "calibration_bins": bins,
            "threshold_rows": horizon_threshold_rows,
        }

    calibration_summary = {
        "experiment_type": "calibration_threshold",
        "model_type": MODEL_TYPE,
        "model_variant": model_variant,
        "graph_window_size": graph_window_size,
        "split_count": split_count,
        "event_policy": event_policy,
        "split_policy": "athlete_level_deterministic_rotating_holdout_20pct",
        "feature_columns": list(GRAPH_SNAPSHOT_FEATURE_COLUMNS),
        "horizons": horizon_summaries,
    }

    threshold_frame = pd.DataFrame(threshold_rows)

    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "calibration_threshold",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_size": graph_window_size,
            "model_variant": model_variant,
            "split_count": split_count,
            "horizons": list(DEFAULT_HORIZONS),
        },
    )
    _write_json(experiment_dir / "calibration_summary.json", calibration_summary)
    write_frame(threshold_frame, experiment_dir / "threshold_table.csv")
    _write_calibration_report(
        experiment_dir / "calibration_report.md",
        calibration_summary,
    )
    return experiment_dir


def _oof_labels(
    labeled: pd.DataFrame,
    oof_indices: list[int],
    label_column: str,
    event_policy: str,
) -> pd.Series:
    labels = labeled.loc[oof_indices, label_column].astype(bool)
    if event_policy == "primary_model_event" and "primary_model_event" in labeled.columns:
        labels = labels & labeled.loc[oof_indices, "primary_model_event"].astype(bool)
    return labels


def _brier_skill(
    model_brier: float | None,
    prevalence_brier: float | None,
) -> float | None:
    if model_brier is None or prevalence_brier in {None, 0.0}:
        return None
    return float(1.0 - model_brier / prevalence_brier)


def _rotating_holdout_splits(
    athlete_ids: pd.Series,
    split_count: int,
) -> list[list[str]]:
    unique_ids = sorted(str(athlete_id) for athlete_id in athlete_ids.dropna().unique())
    if len(unique_ids) <= 1:
        return [[] for _ in range(split_count)]
    test_count = max(1, ceil(len(unique_ids) * 0.2))
    splits: list[list[str]] = []
    for split_seed in range(split_count):
        start = (split_seed * test_count) % len(unique_ids)
        split_ids = [
            unique_ids[(start + offset) % len(unique_ids)]
            for offset in range(test_count)
        ]
        splits.append(sorted(split_ids))
    return splits


ROBUSTNESS_METRICS = (
    "roc_auc",
    "brier_skill_score",
    "model_brier_score",
    "top_decile_lift",
)


def _aggregate_variant_splits(split_payloads: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for horizon in DEFAULT_HORIZONS:
        horizon_summary: dict[str, object] = {}
        for metric_name in ROBUSTNESS_METRICS:
            values = [
                split_payload["horizons"][str(horizon)][metric_name]
                for split_payload in split_payloads.values()
                if split_payload["horizons"][str(horizon)][metric_name] is not None
            ]
            horizon_summary[metric_name] = _metric_distribution(values)
        summary[str(horizon)] = horizon_summary
    return summary


def _metric_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "defined_split_count": 0,
        }
    numeric = [float(value) for value in values]
    mean_value = sum(numeric) / len(numeric)
    variance = sum((value - mean_value) ** 2 for value in numeric) / len(numeric)
    return {
        "mean": float(mean_value),
        "std": float(variance**0.5),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
        "defined_split_count": len(numeric),
    }


def _decision_mode_winners(variants: dict[str, object]) -> dict[str, object]:
    decision_modes = {
        "ranking": ("roc_auc", max),
        "calibration": ("model_brier_score", min),
        "triage": ("top_decile_lift", max),
    }
    output: dict[str, object] = {}
    for mode_name, (metric_name, selector) in decision_modes.items():
        mode_payload: dict[str, object] = {}
        for horizon in DEFAULT_HORIZONS:
            candidates = []
            for variant_name, variant_payload in variants.items():
                distribution = variant_payload["summary_by_horizon"][str(horizon)][
                    metric_name
                ]
                if distribution["mean"] is not None:
                    candidates.append(
                        {
                            "model_variant": variant_name,
                            metric_name: distribution,
                        }
                    )
            if candidates:
                mode_payload[str(horizon)] = selector(
                    candidates,
                    key=lambda candidate: candidate[metric_name]["mean"],
                )
        output[mode_name] = mode_payload
    return output


def _overall_window_model_decision_winners(
    windows: dict[str, object],
) -> dict[str, object]:
    decision_modes = {
        "ranking": ("roc_auc", max),
        "calibration": ("model_brier_score", min),
        "triage": ("top_decile_lift", max),
    }
    output: dict[str, object] = {}
    for mode_name, (metric_name, selector) in decision_modes.items():
        mode_payload: dict[str, object] = {}
        for horizon in DEFAULT_HORIZONS:
            candidates = []
            for window_payload in windows.values():
                for variant_name, variant_payload in window_payload[
                    "variants"
                ].items():
                    distribution = variant_payload["summary_by_horizon"][
                        str(horizon)
                    ][metric_name]
                    if distribution["mean"] is not None:
                        candidates.append(
                            {
                                "graph_window_size": window_payload[
                                    "graph_window_size"
                                ],
                                "model_variant": variant_name,
                                metric_name: distribution,
                            }
                        )
            if candidates:
                mode_payload[str(horizon)] = selector(
                    candidates,
                    key=lambda candidate: candidate[metric_name]["mean"],
                )
        output[mode_name] = mode_payload
    return output


def _normalize_window_sizes(graph_window_sizes: tuple[int, ...]) -> tuple[int, ...]:
    if not graph_window_sizes:
        raise ValueError("at least one graph window size is required")
    normalized = tuple(dict.fromkeys(int(size) for size in graph_window_sizes))
    invalid = [size for size in normalized if size < 2]
    if invalid:
        raise ValueError("graph window sizes must be at least 2")
    return normalized


def _best_windows_by_horizon(windows: dict[str, object]) -> dict[str, object]:
    metric_policies = {
        "roc_auc": max,
        "brier_skill_score": max,
        "top_decile_lift": max,
        "model_brier_score": min,
    }
    best: dict[str, object] = {}
    for horizon in DEFAULT_HORIZONS:
        horizon_best: dict[str, object] = {}
        for metric_name, selector in metric_policies.items():
            candidates = []
            for window_payload in windows.values():
                metrics = window_payload["horizons"][str(horizon)]
                metric_value = metrics[metric_name]
                if metric_value is not None:
                    candidates.append(
                        {
                            "graph_window_size": window_payload["graph_window_size"],
                            "value": float(metric_value),
                        }
                    )
            if candidates:
                horizon_best[metric_name] = selector(
                    candidates,
                    key=lambda candidate: candidate["value"],
                )
        best[str(horizon)] = horizon_best
    return best


def _policy_comparison_rows(
    policy_name: str,
    policy_event_count: int,
    graph_window_size: int,
    evaluation: dict[str, object],
    quality_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    quality_by_horizon_threshold = {
        (
            int(row["horizon_days"]),
            str(row["threshold"]),
        ): row
        for row in quality_rows
    }
    for horizon in DEFAULT_HORIZONS:
        metrics = evaluation["horizons"][str(horizon)]
        for threshold in ("percentile:0.05", "percentile:0.1"):
            quality = quality_by_horizon_threshold.get((horizon, threshold), {})
            rows.append(
                {
                    "graph_window_size": int(graph_window_size),
                    "policy_name": policy_name,
                    "policy_event_count": int(policy_event_count),
                    "horizon_days": horizon,
                    "threshold": threshold,
                    "test_positive_count": metrics["test_positive_count"],
                    "test_positive_rate": metrics["test_positive_rate"],
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "brier_skill_score": metrics["brier_skill_score"],
                    "top_decile_lift": metrics["top_decile_lift"],
                    "episode_count": quality.get("episode_count"),
                    "true_positive_episode_count": quality.get(
                        "true_positive_episode_count"
                    ),
                    "false_positive_episode_count": quality.get(
                        "false_positive_episode_count"
                    ),
                    "unique_observed_event_count": quality.get(
                        "unique_observed_event_count"
                    ),
                    "unique_captured_event_count": quality.get(
                        "unique_captured_event_count"
                    ),
                    "unique_event_capture_rate": quality.get(
                        "unique_event_capture_rate"
                    ),
                    "missed_event_count": quality.get("missed_event_count"),
                    "episodes_per_athlete_season": quality.get(
                        "episodes_per_athlete_season"
                    ),
                    "median_start_lead_days": quality.get("median_start_lead_days"),
                }
            )
    return rows


def _coverage_normalized_stability_frame(
    measurements: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    coverage_tiers: pd.DataFrame,
    model_variant: str,
) -> pd.DataFrame:
    matrix = build_measurement_matrix(measurements)
    rows: list[dict[str, object]] = []
    graph_cache: dict[int, pd.DataFrame] = {}
    tier_cols = coverage_tiers[
        ["athlete_id", "season_id", "coverage_tier", "measurement_days"]
    ].copy()
    tier_cols["season_id"] = tier_cols["season_id"].astype(str)

    for channel in DEFAULT_SHADOW_MODE_CHANNELS:
        window_size = int(channel["graph_window_size"])
        if window_size not in graph_cache:
            graph_cache[window_size] = build_graph_snapshots(
                matrix,
                window_size=window_size,
            )
        graph_features = graph_cache[window_size]
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        if labeled.empty:
            raise ValueError(
                f"no labeled graph snapshots produced for {channel['policy_name']}"
            )
        model_result = train_discrete_time_risk_model(
            labeled,
            model_variant=model_variant,
        )
        explanation_summary = _explanation_summary(
            model_result.timeline,
            model_result.summary,
        )
        alert_timeline = _alert_episode_timeline(
            model_result.timeline,
            explanation_summary,
        )
        alert_timeline = alert_timeline.copy()
        alert_timeline["season_id"] = alert_timeline["season_id"].astype(str)
        alert_timeline = alert_timeline.merge(
            tier_cols,
            on=["athlete_id", "season_id"],
            how="left",
        )
        alert_timeline["coverage_tier"] = alert_timeline[
            "coverage_tier"
        ].fillna("low")

        for scope in COVERAGE_ELIGIBILITY_SCOPES:
            eligible_tiers = COVERAGE_SCOPE_TIERS[scope]
            scoped_timeline = alert_timeline[
                alert_timeline["coverage_tier"].isin(eligible_tiers)
            ].copy()
            scoped_rows = _shadow_mode_stability_rows(
                channel=channel,
                timeline=scoped_timeline,
            )
            eligible_athlete_season_count = int(
                scoped_timeline[["athlete_id", "season_id"]]
                .drop_duplicates()
                .shape[0]
            )
            for row in scoped_rows:
                row["coverage_eligibility_scope"] = scope
                row["eligible_coverage_tiers"] = ",".join(eligible_tiers)
                row["eligible_athlete_season_count"] = eligible_athlete_season_count
            rows.extend(scoped_rows)
    return pd.DataFrame(rows, columns=_coverage_normalized_stability_columns())


def _coverage_normalized_stability_columns() -> list[str]:
    return [
        "coverage_eligibility_scope",
        "eligible_coverage_tiers",
        "eligible_athlete_season_count",
        "channel_name",
        "role",
        "slice_type",
        "slice_id",
        "policy_name",
        "graph_window_size",
        "horizon_days",
        "threshold_scope",
        "threshold",
        "episode_count",
        "true_positive_episode_count",
        "false_positive_episode_count",
        "unique_observed_event_count",
        "unique_captured_event_count",
        "unique_event_capture_rate",
        "missed_event_count",
        "episodes_per_athlete_season",
        "median_start_lead_days",
    ]


def _coverage_source_comparison_rows(
    feature_set_name: str,
    evaluation: dict[str, object],
    model_summary: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for horizon in model_summary["horizons"]:
        metrics = evaluation["horizons"][str(horizon)]
        rows.append(
            {
                "feature_set": feature_set_name,
                "horizon_days": int(horizon),
                "test_snapshot_count": metrics["test_snapshot_count"],
                "test_positive_count": metrics["test_positive_count"],
                "test_positive_rate": metrics["test_positive_rate"],
                "mean_predicted_risk": metrics["mean_predicted_risk"],
                "model_brier_score": metrics["model_brier_score"],
                "prevalence_brier_score": metrics["prevalence_brier_score"],
                "brier_skill_score": metrics["brier_skill_score"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "top_decile_lift": metrics["top_decile_lift"],
            }
        )
    return rows


def _coverage_adjusted_threshold_frame(
    *,
    measurements: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    coverage_tiers: pd.DataFrame,
    model_variant: str,
    burden_cap_episodes_per_athlete_season: float,
) -> pd.DataFrame:
    matrix = build_measurement_matrix(measurements)
    rows: list[dict[str, object]] = []
    graph_cache: dict[int, pd.DataFrame] = {}
    tier_cols = coverage_tiers[
        ["athlete_id", "season_id", "coverage_tier", "measurement_days"]
    ].copy()
    tier_cols["season_id"] = tier_cols["season_id"].astype(str)

    for channel in DEFAULT_SHADOW_MODE_CHANNELS:
        window_size = int(channel["graph_window_size"])
        if window_size not in graph_cache:
            graph_cache[window_size] = build_graph_snapshots(
                matrix,
                window_size=window_size,
            )
        graph_features = graph_cache[window_size]
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        if labeled.empty:
            raise ValueError(
                f"no labeled graph snapshots produced for {channel['policy_name']}"
            )
        model_result = train_discrete_time_risk_model(
            labeled,
            model_variant=model_variant,
        )
        explanation_summary = _explanation_summary(
            model_result.timeline,
            model_result.summary,
        )
        alert_timeline = _alert_episode_timeline(
            model_result.timeline,
            explanation_summary,
        )
        alert_timeline = alert_timeline.copy()
        alert_timeline["season_id"] = alert_timeline["season_id"].astype(str)
        alert_timeline = alert_timeline.merge(
            tier_cols,
            on=["athlete_id", "season_id"],
            how="left",
        )
        alert_timeline["coverage_tier"] = alert_timeline[
            "coverage_tier"
        ].fillna("low")

        rows.extend(
            build_coverage_adjusted_threshold_policy_rows(
                alert_timeline,
                channel,
                burden_cap_episodes_per_athlete_season=(
                    burden_cap_episodes_per_athlete_season
                ),
            )
        )
    return pd.DataFrame(rows)


def _season_forward_validation_rows(
    *,
    measurements: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    graph_window_size: int,
    model_variant: str,
    feature_sets: dict[str, tuple[str, ...]],
    alert_policy_feature_set_name: str,
    alert_policy_feature_columns: tuple[str, ...],
    prepared_features: pd.DataFrame | None = None,
    attach_injury_history_to_alert_features: bool = False,
    exposure_participations_for_alert_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    matrix = build_measurement_matrix(measurements)
    if prepared_features is None:
        graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        prepared_features = attach_coverage_source_features(graph_features, measurements)
    labeled = attach_time_to_event_labels(prepared_features, canonical_injuries)
    if labeled.empty:
        raise ValueError("no labeled graph snapshots produced")
    season_ids = _forward_season_ids(labeled)
    rows: list[dict[str, object]] = []
    for feature_set_name, feature_columns in feature_sets.items():
        rows.extend(
            _season_forward_model_metric_rows(
                labeled=labeled,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                season_ids=season_ids,
                model_variant=model_variant,
            )
        )
    rows.extend(
        _season_forward_alert_policy_rows(
            measurements=measurements,
            matrix=matrix,
            canonical_injuries=canonical_injuries,
            detailed_injuries=detailed_injuries,
            season_ids=season_ids,
            model_variant=model_variant,
            feature_set_name=alert_policy_feature_set_name,
            feature_columns=alert_policy_feature_columns,
            attach_injury_history_features_to_graph=(
                attach_injury_history_to_alert_features
            ),
            exposure_participations_for_graph=(
                exposure_participations_for_alert_features
            ),
        )
    )
    return pd.DataFrame(rows)


def _forward_case_review_cases(
    *,
    measurements: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    model_variant: str,
    feature_set_name: str = "graph_plus_coverage_source",
    feature_columns: tuple[str, ...] = COVERAGE_SOURCE_MODEL_FEATURE_SETS[
        "graph_plus_coverage_source"
    ],
    attach_injury_history_features_to_graph: bool = False,
) -> pd.DataFrame:
    matrix = build_measurement_matrix(measurements)
    season_ids = sorted(str(value) for value in matrix["season_id"].dropna().unique())
    target_test_seasons = _forward_case_review_target_seasons(season_ids)
    target_channels = {
        channel["channel_name"]: channel
        for channel in DEFAULT_SHADOW_MODE_CHANNELS
        if channel["channel_name"] in FORWARD_CASE_REVIEW_TARGET_CHANNELS
    }
    rows: list[dict[str, object]] = []
    for channel_name in FORWARD_CASE_REVIEW_TARGET_CHANNELS:
        channel = target_channels[channel_name]
        window_size = int(channel["graph_window_size"])
        graph_features = attach_coverage_source_features(
            build_graph_snapshots(matrix, window_size=window_size),
            measurements,
        )
        if attach_injury_history_features_to_graph:
            graph_features = attach_injury_history_features(
                graph_features,
                detailed_injuries,
            )
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        if labeled.empty:
            continue
        for test_season in target_test_seasons:
            train_seasons = [season for season in season_ids if season < test_season]
            if not train_seasons:
                continue
            train_mask = labeled["season_id"].astype(str).isin(train_seasons)
            timeline = _fit_forward_risk_timeline(
                labeled=labeled,
                train_mask=train_mask,
                feature_columns=feature_columns,
                model_variant=model_variant,
            )
            explanation_summary = _explanation_summary(
                timeline,
                _forward_case_review_model_summary(feature_columns),
            )
            alert_timeline = _alert_episode_timeline(timeline, explanation_summary)
            test_timeline = alert_timeline[
                alert_timeline["season_id"].astype(str).eq(test_season)
            ].copy()
            rows.extend(
                _forward_case_review_rows_for_target(
                    channel=channel,
                    test_timeline=test_timeline,
                    train_seasons=train_seasons,
                    test_season=test_season,
                    feature_set_name=feature_set_name,
                )
            )
    return pd.DataFrame(rows)


def _forward_case_review_rows_for_target(
    *,
    channel: dict[str, object],
    test_timeline: pd.DataFrame,
    train_seasons: list[str],
    test_season: str,
    feature_set_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if test_timeline.empty:
        return rows
    policy_rows = build_coverage_adjusted_threshold_policy_rows(
        test_timeline,
        channel,
        burden_cap_episodes_per_athlete_season=(
            DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON
        ),
    )
    horizon = int(channel["horizon_days"])
    for policy_row in policy_rows:
        threshold_policy = str(policy_row["threshold_policy"])
        if threshold_policy not in {
            "season_local_percentile",
            "burden_capped_percentile",
        }:
            continue
        threshold_value = float(policy_row["selected_threshold_value"])
        episodes = build_alert_episodes(
            test_timeline,
            horizons=(horizon,),
            percentile_thresholds=(threshold_value,),
        )
        quality = build_alert_episode_quality(episodes, test_timeline)
        review = build_qualitative_case_review(episodes, test_timeline, quality)
        for case in review["cases"]:
            case.update(
                {
                    "target_reason": _forward_case_review_target_reason(
                        str(channel["channel_name"]),
                        test_season,
                    ),
                    "channel_name": str(channel["channel_name"]),
                    "role": str(channel["role"]),
                    "policy_name": str(channel["policy_name"]),
                    "feature_set": feature_set_name,
                    "threshold_policy": threshold_policy,
                    "selected_threshold_value": threshold_value,
                    "graph_window_size": int(channel["graph_window_size"]),
                    "train_season_ids": ",".join(train_seasons),
                    "test_season_id": test_season,
                }
            )
            rows.append(case)
    return rows


def _forward_case_review_model_summary(
    feature_columns: tuple[str, ...],
) -> dict[str, object]:
    return {
        "feature_columns": list(feature_columns),
        "horizon_models": {
            str(horizon): {
                "feature_attribution": [
                    {
                        "feature": feature,
                        "coefficient": 0.0,
                        "train_mean": 0.0,
                        "train_std": 1.0,
                        "standardized_coefficient": 0.0,
                        "abs_standardized_coefficient": 0.0,
                    }
                    for feature in feature_columns
                ]
            }
            for horizon in DEFAULT_HORIZONS
        },
    }


def _forward_case_review_target_seasons(season_ids: list[str]) -> list[str]:
    preferred = [
        season
        for season in FORWARD_CASE_REVIEW_PREFERRED_SEASONS
        if season in season_ids
    ]
    if len(preferred) == len(FORWARD_CASE_REVIEW_PREFERRED_SEASONS):
        return preferred
    return season_ids[1:][-2:]


def _forward_case_review_target_reason(channel_name: str, test_season: str) -> str:
    if test_season == "2023-2024":
        return "forward_ranking_survivor"
    if test_season == "2025-2026":
        return "forward_calibration_survivor"
    if channel_name == "broad_30d":
        return "low_burden_reference"
    return "forward_survivor"


def _season_forward_model_metric_rows(
    *,
    labeled: pd.DataFrame,
    feature_set_name: str,
    feature_columns: tuple[str, ...],
    season_ids: list[str],
    model_variant: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for test_index in range(1, len(season_ids)):
        train_seasons = season_ids[:test_index]
        test_season = season_ids[test_index]
        train_mask = labeled["season_id"].astype(str).isin(train_seasons)
        test_mask = labeled["season_id"].astype(str).eq(test_season)
        timeline = _fit_forward_risk_timeline(
            labeled=labeled,
            train_mask=train_mask,
            feature_columns=feature_columns,
            model_variant=model_variant,
        )
        for horizon in DEFAULT_HORIZONS:
            metrics = _forward_horizon_metrics(
                timeline=timeline,
                train_mask=train_mask,
                test_mask=test_mask,
                horizon=horizon,
            )
            rows.append(
                {
                    "row_type": "model_metric",
                    "train_season_ids": ",".join(train_seasons),
                    "train_season_count": len(train_seasons),
                    "test_season_id": test_season,
                    "feature_set": feature_set_name,
                    "threshold_policy": None,
                    "channel_name": None,
                    "policy_name": "canonical_any_injury",
                    "graph_window_size": None,
                    "horizon_days": horizon,
                    **metrics,
                }
            )
    return rows


def _season_forward_alert_policy_rows(
    *,
    measurements: pd.DataFrame,
    matrix: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    season_ids: list[str],
    model_variant: str,
    feature_set_name: str,
    feature_columns: tuple[str, ...],
    attach_injury_history_features_to_graph: bool = False,
    exposure_participations_for_graph: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    graph_cache: dict[int, pd.DataFrame] = {}
    for channel in DEFAULT_SHADOW_MODE_CHANNELS:
        window_size = int(channel["graph_window_size"])
        if window_size not in graph_cache:
            features = attach_coverage_source_features(
                build_graph_snapshots(matrix, window_size=window_size),
                measurements,
            )
            if attach_injury_history_features_to_graph:
                features = attach_injury_history_features(features, detailed_injuries)
            if exposure_participations_for_graph is not None:
                features = attach_exposure_load_features(
                    features,
                    exposure_participations_for_graph,
                )
            graph_cache[window_size] = features
        graph_features = graph_cache[window_size]
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        for test_index in range(1, len(season_ids)):
            train_seasons = season_ids[:test_index]
            test_season = season_ids[test_index]
            train_mask = labeled["season_id"].astype(str).isin(train_seasons)
            timeline = _fit_forward_risk_timeline(
                labeled=labeled,
                train_mask=train_mask,
                feature_columns=feature_columns,
                model_variant=model_variant,
            )
            test_timeline = timeline[
                timeline["season_id"].astype(str).eq(test_season)
            ].copy()
            policy_rows = build_coverage_adjusted_threshold_policy_rows(
                test_timeline,
                channel,
                burden_cap_episodes_per_athlete_season=(
                    DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON
                ),
            )
            for row in policy_rows:
                if row["threshold_policy"] not in {
                    "season_local_percentile",
                    "burden_capped_percentile",
                }:
                    continue
                row.update(
                    {
                        "row_type": "alert_policy",
                        "train_season_ids": ",".join(train_seasons),
                        "train_season_count": len(train_seasons),
                        "test_season_id": test_season,
                        "feature_set": feature_set_name,
                    }
                )
                rows.append(row)
    return rows


def _shadow_event_crosswalk_packet_contexts(
    *,
    measurements: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    exposure_participations: pd.DataFrame,
    packet_rows: list[dict[str, object]],
    graph_window_size: int,
    model_variant: str,
) -> list[dict[str, object]]:
    matrix = build_measurement_matrix(measurements)
    contexts: list[dict[str, object]] = []
    graph_cache: dict[int, pd.DataFrame] = {}
    channel_defaults = {
        str(channel["channel_name"]): channel for channel in DEFAULT_SHADOW_MODE_CHANNELS
    }
    for packet in packet_rows:
        channel_name = str(packet.get("channel_name"))
        channel = dict(channel_defaults.get(channel_name, {}))
        channel.update(
            {
                "channel_name": channel_name,
                "policy_name": packet.get(
                    "policy_name",
                    channel.get("policy_name", ""),
                ),
                "horizon_days": int(
                    packet.get("horizon_days") or channel.get("horizon_days") or 0
                ),
                "threshold_value": float(
                    packet.get("selected_threshold_value")
                    or channel.get("threshold_value")
                    or 0.0
                ),
                "graph_window_size": int(
                    packet.get("graph_window_size")
                    or channel.get("graph_window_size")
                    or graph_window_size
                ),
                "role": channel.get("role", "shadow event crosswalk"),
            }
        )
        window_size = int(channel["graph_window_size"])
        if window_size not in graph_cache:
            graph_features = attach_coverage_source_features(
                build_graph_snapshots(matrix, window_size=window_size),
                measurements,
            )
            graph_cache[window_size] = attach_exposure_load_features(
                graph_features,
                exposure_participations,
            )
        graph_features = graph_cache[window_size]
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        season_ids = _forward_season_ids(labeled)
        test_season = str(packet.get("test_season_id"))
        if test_season not in season_ids:
            continue
        test_index = season_ids.index(test_season)
        if test_index == 0:
            continue
        train_seasons = season_ids[:test_index]
        train_mask = labeled["season_id"].astype(str).isin(train_seasons)
        timeline = _fit_forward_risk_timeline(
            labeled=labeled,
            train_mask=train_mask,
            feature_columns=EXPOSURE_LOAD_MODEL_FEATURE_SETS[
                "graph_plus_coverage_exposure_load"
            ],
            model_variant=model_variant,
        )
        explanation_summary = _explanation_summary(
            timeline,
            _forward_case_review_model_summary(
                EXPOSURE_LOAD_MODEL_FEATURE_SETS[
                    "graph_plus_coverage_exposure_load"
                ]
            ),
        )
        alert_timeline = _alert_episode_timeline(timeline, explanation_summary)
        test_timeline = alert_timeline[
            alert_timeline["season_id"].astype(str).eq(test_season)
        ].copy()
        episodes = build_alert_episodes(
            test_timeline,
            horizons=(int(channel["horizon_days"]),),
            percentile_thresholds=(float(channel["threshold_value"]),),
        )
        contexts.append(
            {
                "packet": packet,
                "episodes": episodes,
                "timeline": test_timeline,
            }
        )
    return contexts


def _fit_forward_risk_timeline(
    *,
    labeled: pd.DataFrame,
    train_mask: pd.Series,
    feature_columns: tuple[str, ...],
    model_variant: str,
) -> pd.DataFrame:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            f"model_variant must be one of: {', '.join(MODEL_VARIANTS)}"
        )
    timeline = labeled.copy()
    features = _forward_feature_frame(timeline, feature_columns)
    train_features = features.loc[train_mask]
    fit_features, fit_train_features = _forward_fit_feature_frames(
        features,
        train_features,
        standardize=model_variant != "baseline",
    )
    for horizon in DEFAULT_HORIZONS:
        label_column = f"event_within_{horizon}d"
        labels = _forward_training_labels(timeline, label_column)
        train_labels = labels.loc[train_mask]
        if train_labels.nunique(dropna=False) < 2:
            probability = float(train_labels.mean()) if len(train_labels) else 0.0
            predictions = pd.Series(probability, index=timeline.index)
        else:
            model = _forward_logistic_model(model_variant)
            model.fit(fit_train_features, train_labels)
            predictions = pd.Series(
                model.predict_proba(fit_features)[:, 1],
                index=timeline.index,
            )
        timeline[f"risk_{horizon}d"] = predictions.clip(
            lower=0.0,
            upper=1.0,
        ).round(6)
    return timeline


def _forward_horizon_metrics(
    *,
    timeline: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    horizon: int,
) -> dict[str, object]:
    labels = _forward_training_labels(timeline, f"event_within_{horizon}d")
    train_labels = labels.loc[train_mask]
    test_labels = labels.loc[test_mask]
    predictions = timeline.loc[test_mask, f"risk_{horizon}d"].astype(float)
    train_positive_rate = float(train_labels.mean()) if len(train_labels) else 0.0
    baseline_predictions = pd.Series(train_positive_rate, index=test_labels.index)
    model_brier = _metric_brier_score(test_labels, predictions)
    prevalence_brier = _metric_brier_score(test_labels, baseline_predictions)
    return {
        "train_snapshot_count": int(train_mask.sum()),
        "test_snapshot_count": int(test_mask.sum()),
        "train_positive_count": int(train_labels.sum()),
        "test_positive_count": int(test_labels.sum()),
        "test_positive_rate": float(test_labels.mean()) if len(test_labels) else None,
        "mean_predicted_risk": float(predictions.mean()) if len(predictions) else None,
        "prevalence_baseline_risk": train_positive_rate,
        "model_brier_score": model_brier,
        "prevalence_brier_score": prevalence_brier,
        "brier_skill_score": _metric_brier_skill(model_brier, prevalence_brier),
        "roc_auc": _metric_roc_auc(test_labels, predictions),
        "average_precision": _metric_average_precision(test_labels, predictions),
        "top_decile_lift": _metric_top_decile_lift(test_labels, predictions),
    }


def _forward_feature_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError(f"model input missing required columns: {', '.join(missing)}")
    return (
        frame.loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def _forward_fit_feature_frames(
    features: pd.DataFrame,
    train_features: pd.DataFrame,
    *,
    standardize: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not standardize:
        return features, train_features
    means = train_features.mean()
    stds = train_features.std(ddof=0).replace(0.0, 1.0)
    return (features - means) / stds, (train_features - means) / stds


def _forward_training_labels(frame: pd.DataFrame, label_column: str) -> pd.Series:
    labels = frame[label_column].astype(bool)
    if "primary_model_event" in frame:
        labels = labels & frame["primary_model_event"].astype(bool)
    return labels


def _forward_logistic_model(model_variant: str) -> LogisticRegression:
    if model_variant == "baseline":
        return LogisticRegression(max_iter=1000, random_state=0)
    if model_variant == "l2":
        return LogisticRegression(
            C=0.2,
            solver="liblinear",
            max_iter=5000,
            random_state=0,
        )
    if model_variant == "l1":
        return LogisticRegression(
            C=0.2,
            l1_ratio=1,
            solver="saga",
            max_iter=5000,
            random_state=0,
        )
    if model_variant == "elasticnet":
        return LogisticRegression(
            C=0.2,
            l1_ratio=0.5,
            solver="saga",
            max_iter=5000,
            random_state=0,
        )
    raise ValueError(f"unsupported model_variant: {model_variant}")


def _forward_season_ids(labeled: pd.DataFrame) -> list[str]:
    season_ids = sorted(str(value) for value in labeled["season_id"].dropna().unique())
    if len(season_ids) < 2:
        raise ValueError("season-forward validation requires at least two seasons")
    return season_ids


def _metric_brier_score(
    labels: pd.Series,
    predictions: pd.Series,
) -> float | None:
    if labels.empty:
        return None
    return float(brier_score_loss(labels.astype(int), predictions))


def _metric_brier_skill(
    model_brier: float | None,
    prevalence_brier: float | None,
) -> float | None:
    if model_brier is None or prevalence_brier in {None, 0.0}:
        return None
    return float(1.0 - (model_brier / prevalence_brier))


def _metric_roc_auc(labels: pd.Series, predictions: pd.Series) -> float | None:
    if labels.nunique(dropna=False) < 2:
        return None
    return float(roc_auc_score(labels.astype(int), predictions))


def _metric_average_precision(
    labels: pd.Series,
    predictions: pd.Series,
) -> float | None:
    if labels.nunique(dropna=False) < 2:
        return None
    return float(average_precision_score(labels.astype(int), predictions))


def _metric_top_decile_lift(
    labels: pd.Series,
    predictions: pd.Series,
) -> float | None:
    positive_rate = float(labels.mean()) if len(labels) else 0.0
    if positive_rate <= 0.0:
        return None
    top_count = max(1, int(len(labels) * 0.1))
    top_indices = predictions.sort_values(ascending=False).head(top_count).index
    top_positive_rate = float(labels.loc[top_indices].mean())
    return float(top_positive_rate / positive_rate)


def _shadow_mode_stability_rows(
    channel: dict[str, object],
    timeline: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []
    for season_id, season_timeline in timeline.groupby("season_id", sort=True):
        horizon = int(channel["horizon_days"])
        threshold_value = float(channel["threshold_value"])
        season_episodes = build_alert_episodes(
            season_timeline,
            horizons=(horizon,),
            percentile_thresholds=(threshold_value,),
        )
        quality_rows = build_alert_episode_quality(
            season_episodes,
            season_timeline,
        )["quality_rows"]
        quality = (
            quality_rows[0]
            if quality_rows
            else _empty_shadow_quality_row(season_timeline)
        )
        rows.append(
            {
                "channel_name": str(channel["channel_name"]),
                "role": str(channel["role"]),
                "slice_type": "season",
                "slice_id": str(season_id),
                "policy_name": str(channel["policy_name"]),
                "graph_window_size": int(channel["graph_window_size"]),
                "horizon_days": int(channel["horizon_days"]),
                "threshold_scope": "season_local",
                "threshold": (
                    f"percentile:{float(channel['threshold_value']):g}"
                ),
                "episode_count": quality["episode_count"],
                "true_positive_episode_count": quality[
                    "true_positive_episode_count"
                ],
                "false_positive_episode_count": quality[
                    "false_positive_episode_count"
                ],
                "unique_observed_event_count": quality[
                    "unique_observed_event_count"
                ],
                "unique_captured_event_count": quality[
                    "unique_captured_event_count"
                ],
                "unique_event_capture_rate": quality[
                    "unique_event_capture_rate"
                ],
                "missed_event_count": quality["missed_event_count"],
                "episodes_per_athlete_season": quality[
                    "episodes_per_athlete_season"
                ],
                "median_start_lead_days": quality["median_start_lead_days"],
            }
        )
    return rows


def _shadow_mode_stability_frame(
    measurements: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    model_variant: str,
) -> pd.DataFrame:
    matrix = build_measurement_matrix(measurements)
    rows: list[dict[str, object]] = []
    graph_cache: dict[int, pd.DataFrame] = {}
    for channel in DEFAULT_SHADOW_MODE_CHANNELS:
        window_size = int(channel["graph_window_size"])
        if window_size not in graph_cache:
            graph_cache[window_size] = build_graph_snapshots(
                matrix,
                window_size=window_size,
            )
        graph_features = graph_cache[window_size]
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        if labeled.empty:
            raise ValueError(
                f"no labeled graph snapshots produced for {channel['policy_name']}"
            )
        model_result = train_discrete_time_risk_model(
            labeled,
            model_variant=model_variant,
        )
        explanation_summary = _explanation_summary(
            model_result.timeline,
            model_result.summary,
        )
        alert_timeline = _alert_episode_timeline(
            model_result.timeline,
            explanation_summary,
        )
        rows.extend(
            _shadow_mode_stability_rows(
                channel=channel,
                timeline=alert_timeline,
            )
        )
    return pd.DataFrame(rows)


def _empty_shadow_quality_row(timeline: pd.DataFrame) -> dict[str, object]:
    observed = timeline[timeline["event_observed"].astype(bool)]
    event_count = int(
        observed[["athlete_id", "season_id", "event_date", "injury_type"]]
        .drop_duplicates()
        .shape[0]
    )
    athlete_seasons = int(
        timeline[["athlete_id", "season_id"]].drop_duplicates().shape[0]
    )
    return {
        "episode_count": 0,
        "true_positive_episode_count": 0,
        "false_positive_episode_count": 0,
        "unique_observed_event_count": event_count,
        "unique_captured_event_count": 0,
        "unique_event_capture_rate": 0.0 if event_count else None,
        "missed_event_count": event_count,
        "episodes_per_athlete_season": 0.0 if athlete_seasons else None,
        "median_start_lead_days": None,
    }


def _best_policies_by_horizon(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    best: dict[str, object] = {}
    for horizon, group in frame.groupby("horizon_days", sort=True):
        horizon_best: dict[str, object] = {}
        for metric_name in (
            "roc_auc",
            "brier_skill_score",
            "top_decile_lift",
            "unique_event_capture_rate",
        ):
            values = group.dropna(subset=[metric_name])
            if values.empty:
                continue
            row = values.sort_values(
                [metric_name, "policy_name"],
                ascending=[False, True],
            ).iloc[0]
            horizon_best[metric_name] = {
                "policy_name": str(row["policy_name"]),
                "threshold": str(row["threshold"]),
                "value": float(row[metric_name]),
            }
        best[str(int(horizon))] = horizon_best
    return best


def _experiment_path(output_dir: str | Path, experiment_id: str) -> Path:
    experiment_path = Path(experiment_id)
    if (
        not experiment_id.strip()
        or experiment_path.is_absolute()
        or experiment_path.name != experiment_id
        or ".." in experiment_path.parts
        or "/" in experiment_id
        or "\\" in experiment_id
    ):
        raise ValueError("experiment_id must be a simple identifier")
    return Path(output_dir) / "experiments" / experiment_id


def _resolve_detailed_injuries_path(
    injuries_path: str | Path,
    detailed_injuries_path: str | Path | None,
) -> Path | None:
    if detailed_injuries_path is not None:
        path = Path(detailed_injuries_path)
        return path if path.exists() else None
    sibling = Path(injuries_path).parent / "injury_events_detailed.csv"
    return sibling if sibling.exists() else None


def _model_metrics(
    labeled: pd.DataFrame,
    timeline: pd.DataFrame,
    model_summary: dict[str, object],
) -> dict[str, int | float | str]:
    observed_events = labeled.loc[labeled["event_observed"]].drop_duplicates(
        ["athlete_id", "season_id"]
    )
    primary_model_events = _primary_model_events(labeled)
    return {
        "model_type": str(model_summary["model_type"]),
        "athlete_count": int(labeled["athlete_id"].nunique()),
        "snapshot_count": int(len(labeled)),
        "observed_event_count": int(len(observed_events)),
        "primary_model_event_count": int(len(primary_model_events)),
        "mean_risk_7d": float(timeline["risk_7d"].mean()),
        "mean_risk_14d": float(timeline["risk_14d"].mean()),
        "mean_risk_30d": float(timeline["risk_30d"].mean()),
    }


def _primary_model_events(labeled: pd.DataFrame) -> pd.DataFrame:
    if "primary_model_event" not in labeled.columns:
        return labeled.loc[labeled["event_observed"]].drop_duplicates(
            ["athlete_id", "season_id"]
        )
    return labeled.loc[labeled["primary_model_event"].astype(bool)].drop_duplicates(
        ["athlete_id", "season_id"]
    )


def _compute_snapshot_contributions(
    row: pd.Series,
    feature_attribution: list[dict],
    feature_columns: tuple[str, ...],
) -> dict[str, float]:
    """Compute per-feature log-odds contributions for one snapshot.

    contribution_k = standardized_coefficient_k × (value_k − train_mean_k) / train_std_k

    Zero is returned for features missing from the row or with zero train_std.
    """
    contributions: dict[str, float] = {}
    for entry in feature_attribution:
        feature = entry["feature"]
        if feature not in feature_columns:
            continue
        std_coeff = float(entry["standardized_coefficient"])
        train_std = float(entry["train_std"])
        if train_std == 0.0 or std_coeff == 0.0:
            contributions[feature] = 0.0
            continue
        value = float(row[feature]) if feature in row.index else 0.0
        train_mean = float(entry["train_mean"])
        z = (value - train_mean) / train_std
        raw = std_coeff * z
        contributions[feature] = 0.0 if (raw != raw) else raw  # NaN guard
    return contributions


def _intra_individual_deviations(
    row: pd.Series,
    horizon_contribs: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    deviations: list[dict[str, object]] = []
    for feature in Z_SCORE_GRAPH_FEATURE_COLUMNS:
        value = float(row[feature]) if feature in row.index else 0.0
        deviations.append(
            {
                "feature": feature,
                "value": round(value, 6),
                "elevated": abs(value) > INTRA_INDIVIDUAL_ELEVATED_Z_THRESHOLD,
                "contributions": {
                    str(horizon): round(
                        horizon_contribs.get(str(horizon), {}).get(feature, 0.0),
                        6,
                    )
                    for horizon in DEFAULT_HORIZONS
                },
            }
        )
    return deviations


def _peak_intra_individual_deviation(
    group: pd.DataFrame,
    snap_contribs: list[dict[str, dict[str, float]]],
) -> dict[str, object]:
    peak_row: pd.Series | None = None
    peak_contribs: dict[str, dict[str, float]] = {}
    peak_combined = -1.0
    for idx, (_, row) in enumerate(group.iterrows()):
        combined = sum(
            abs(float(row[feature])) if feature in row.index else 0.0
            for feature in Z_SCORE_GRAPH_FEATURE_COLUMNS
        )
        if combined > peak_combined:
            peak_combined = combined
            peak_row = row
            peak_contribs = snap_contribs[idx]

    if peak_row is None:
        return {
            "time_index": None,
            "snapshot_date": None,
            "combined_abs_z_score": 0.0,
            "flagged_features": [],
            "deviations": [],
        }

    deviations = _intra_individual_deviations(peak_row, peak_contribs)
    ranked_deviations = sorted(
        deviations,
        key=lambda entry: -abs(float(entry["value"])),
    )
    return {
        "time_index": int(peak_row["time_index"]),
        "snapshot_date": str(peak_row["snapshot_date"]),
        "combined_abs_z_score": round(peak_combined, 6),
        "flagged_features": [
            str(entry["feature"]) for entry in deviations if bool(entry["elevated"])
        ],
        "deviations": ranked_deviations,
    }


def _explanation_summary(
    timeline: pd.DataFrame,
    model_summary: dict,
) -> pd.DataFrame:
    columns = [
        "athlete_id",
        "season_id",
        "time_index",
        "snapshot_date",
        "mean_abs_correlation",
        "edge_count",
        "risk_7d",
        "risk_14d",
        "risk_30d",
    ]
    explanations = timeline.loc[:, columns].copy()
    feature_columns = tuple(model_summary["feature_columns"])
    for horizon in DEFAULT_HORIZONS:
        attribution = model_summary["horizon_models"][str(horizon)]["feature_attribution"]
        top_features = []
        top_contributions = []
        for _, row in timeline.iterrows():
            contribs = _compute_snapshot_contributions(row, attribution, feature_columns)
            if contribs:
                top_feature, top_contrib = max(contribs.items(), key=lambda kv: abs(kv[1]))
            else:
                top_feature, top_contrib = "", 0.0
            top_features.append(top_feature)
            top_contributions.append(round(top_contrib, 6))
        explanations[f"top_feature_{horizon}d"] = top_features
        explanations[f"top_contribution_{horizon}d"] = top_contributions
    return explanations


def _athlete_explanations(
    timeline: pd.DataFrame,
    model_summary: dict,
) -> dict:
    feature_columns = tuple(model_summary["feature_columns"])
    athletes = []
    for (athlete_id, season_id), group in timeline.groupby(
        ["athlete_id", "season_id"], sort=False
    ):
        group = group.sort_values("time_index")

        # Compute full contribution dicts for all snapshots and horizons
        snap_contribs: list[dict[str, dict[str, float]]] = []
        for _, row in group.iterrows():
            horizon_contribs: dict[str, dict[str, float]] = {}
            for horizon in DEFAULT_HORIZONS:
                attribution = model_summary["horizon_models"][str(horizon)][
                    "feature_attribution"
                ]
                horizon_contribs[str(horizon)] = _compute_snapshot_contributions(
                    row, attribution, feature_columns
                )
            snap_contribs.append(horizon_contribs)

        # Peak risk per horizon
        peak_risk: dict[str, object] = {}
        for horizon in DEFAULT_HORIZONS:
            risk_col = f"risk_{horizon}d"
            peak_idx = group[risk_col].idxmax()
            peak_row = group.loc[peak_idx]
            peak_risk[str(horizon)] = {
                "time_index": int(peak_row["time_index"]),
                "snapshot_date": str(peak_row["snapshot_date"]),
                "risk": float(peak_row[risk_col]),
            }

        # Dominant features: average |contribution| across all snapshots
        dominant_features: dict[str, list[str]] = {}
        n_snaps = max(1, len(snap_contribs))
        for horizon in DEFAULT_HORIZONS:
            feature_sum: dict[str, float] = {}
            for hc in snap_contribs:
                for feat, val in hc[str(horizon)].items():
                    feature_sum[feat] = feature_sum.get(feat, 0.0) + abs(val)
            ranked = sorted(feature_sum.items(), key=lambda kv: -kv[1] / n_snaps)
            dominant_features[str(horizon)] = [f for f, _ in ranked[:3]]

        peak_intra_individual_deviation = _peak_intra_individual_deviation(
            group, snap_contribs
        )

        # Per-snapshot payload: top-3 contributions per horizon
        snapshots = []
        for i, (_, row) in enumerate(group.iterrows()):
            snap_feature_contribs: dict[str, list[dict]] = {}
            for horizon in DEFAULT_HORIZONS:
                top3 = sorted(
                    snap_contribs[i][str(horizon)].items(),
                    key=lambda kv: -abs(kv[1]),
                )[:3]
                snap_feature_contribs[str(horizon)] = [
                    {"feature": k, "contribution": round(v, 6)} for k, v in top3
                ]
            snapshots.append(
                {
                    "time_index": int(row["time_index"]),
                    "snapshot_date": str(row["snapshot_date"]),
                    "risk_7d": float(row["risk_7d"]),
                    "risk_14d": float(row["risk_14d"]),
                    "risk_30d": float(row["risk_30d"]),
                    "feature_contributions": snap_feature_contribs,
                    "intra_individual_deviations": _intra_individual_deviations(
                        row, snap_contribs[i]
                    ),
                }
            )

        event_observed = (
            bool(group["event_observed"].any())
            if "event_observed" in group.columns
            else None
        )
        athletes.append(
            {
                "athlete_id": str(athlete_id),
                "season_id": str(season_id),
                "snapshot_count": int(len(group)),
                "event_observed": event_observed,
                "peak_risk": peak_risk,
                "dominant_features": dominant_features,
                "peak_intra_individual_deviation": peak_intra_individual_deviation,
                "snapshots": snapshots,
            }
        )
    return {"athlete_count": len(athletes), "athletes": athletes}


def _alert_episode_timeline(
    timeline: pd.DataFrame,
    explanation_summary: pd.DataFrame,
) -> pd.DataFrame:
    alert_timeline = timeline.copy()
    for horizon in DEFAULT_HORIZONS:
        for prefix in ("top_feature", "top_contribution"):
            column = f"{prefix}_{horizon}d"
            alert_timeline[column] = explanation_summary[column].to_list()
    alert_timeline["elevated_z_features"] = [
        [
            feature
            for feature in Z_SCORE_GRAPH_FEATURE_COLUMNS
            if feature in row.index
            and abs(float(row[feature])) > INTRA_INDIVIDUAL_ELEVATED_Z_THRESHOLD
        ]
        for _, row in alert_timeline.iterrows()
    ]
    return alert_timeline


def _feature_attribution_and_ablation(
    labeled: pd.DataFrame,
    full_timeline: pd.DataFrame,
    full_model_summary: dict[str, object],
    full_evaluation: dict[str, object],
) -> dict[str, object]:
    feature_sets: dict[str, object] = {}
    model_variant = str(full_model_summary["model_variant"])
    for name, feature_columns in FEATURE_ABLATION_SETS.items():
        if name == "full_13":
            timeline = full_timeline
            model_summary = full_model_summary
            evaluation = full_evaluation
        else:
            model_result = train_discrete_time_risk_model(
                labeled,
                feature_columns=feature_columns,
                model_variant=model_variant,
            )
            timeline = model_result.timeline
            model_summary = model_result.summary
            evaluation = evaluate_risk_model(timeline, model_summary)

        feature_sets[name] = {
            "feature_columns": list(feature_columns),
            "horizons": {
                str(horizon): {
                    "model_kind": model_summary["horizon_models"][str(horizon)][
                        "model_kind"
                    ],
                    "evaluation": evaluation["horizons"][str(horizon)],
                    "feature_attribution": model_summary["horizon_models"][
                        str(horizon)
                    ]["feature_attribution"],
                }
                for horizon in model_summary["horizons"]
            },
        }
    return {
        "model_type": MODEL_TYPE,
        "model_variant": model_variant,
        "event_policy": full_model_summary["event_policy"],
        "split_policy": full_model_summary["split_policy"],
        "feature_sets": feature_sets,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _load_json_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _load_records_from_path(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = _load_json_payload(source)
        for key in (
            "calibration_diagnostics",
            "feature_shift_summary",
            "guardrail_rows",
            "validation_rows",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]
        raise ValueError(f"JSON artifact at {path} does not contain record rows")
    return _json_records(pd.read_csv(source))


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _write_report(
    path: Path,
    timeline: pd.DataFrame,
    model_summary: dict[str, object],
    evaluation: dict[str, object],
) -> None:
    evaluation_horizons = evaluation["horizons"]
    lines = [
        "# Experiment Report",
        "",
        f"Model: {MODEL_TYPE}",
        f"Model variant: {model_summary['model_variant']}",
        f"Event policy: {model_summary['event_policy']}",
        f"Split policy: {model_summary['split_policy']}",
        f"Snapshots: {len(timeline)}",
        f"Athletes: {timeline['athlete_id'].nunique()}",
        f"Mean +7 day risk: {timeline['risk_7d'].mean():.3f}",
        f"Mean +14 day risk: {timeline['risk_14d'].mean():.3f}",
        f"Mean +30 day risk: {timeline['risk_30d'].mean():.3f}",
        "",
        "## Holdout Evaluation",
        "",
        *[
            _horizon_report_line(
                horizon,
                evaluation_horizons[str(horizon)],
            )
            for horizon in DEFAULT_HORIZONS
        ],
        "",
        "These risk values come from a discrete-time logistic baseline over graph snapshot features. They are not calibrated clinical probabilities, but they preserve the longitudinal time-to-event contract for later model replacement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_ablation_report(
    path: Path,
    feature_attribution: dict[str, object],
) -> None:
    feature_sets = feature_attribution["feature_sets"]
    lines = [
        "# Feature Attribution And Ablation",
        "",
        "All feature sets use the same athlete-level deterministic holdout split and event policy as the primary experiment.",
        "",
        "## Ablation Metrics",
        "",
        "| Feature set | Horizon | AUROC | Brier skill | Top-decile lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature_set_name, feature_set in feature_sets.items():
        horizons = feature_set["horizons"]
        for horizon, horizon_payload in horizons.items():
            evaluation = horizon_payload["evaluation"]
            lines.append(
                "| "
                f"{feature_set_name} | "
                f"{horizon}d | "
                f"{_format_metric(evaluation['roc_auc'])} | "
                f"{_format_metric(evaluation['brier_skill_score'])} | "
                f"{_format_metric(evaluation['top_decile_lift'])} |"
            )

    lines.extend(["", "## Top Standardized Coefficients", ""])
    for feature_set_name, feature_set in feature_sets.items():
        lines.append(f"### {feature_set_name}")
        for horizon, horizon_payload in feature_set["horizons"].items():
            top_features = horizon_payload["feature_attribution"][:5]
            formatted = ", ".join(
                f"{entry['feature']} ({float(entry['standardized_coefficient']):+.3f})"
                for entry in top_features
            )
            lines.append(f"- {horizon}d: {formatted}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_alert_episode_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Alert Episode Validation",
        "",
        f"Model variant: {summary['model_variant']}",
        f"Graph window size: {summary['graph_window_size']}",
        f"Episodes: {summary['episode_count']}",
        "",
        "Episodes collapse contiguous high-risk snapshots selected by percentile thresholds within each athlete-season.",
        "",
        "| Horizon | Threshold | Episodes | Start capture | Peak capture | End capture | Median snapshots | Median days |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, horizon_payload in summary["horizons"].items():
        for threshold, row in horizon_payload["thresholds"].items():
            lines.append(
                "| "
                f"{horizon}d | "
                f"{threshold} | "
                f"{row['episode_count']} | "
                f"{_format_metric(row['event_capture_after_start_rate'])} | "
                f"{_format_metric(row['event_capture_after_peak_rate'])} | "
                f"{_format_metric(row['event_capture_after_end_rate'])} | "
                f"{_format_metric(row['median_snapshot_count'])} | "
                f"{_format_metric(row['median_duration_days'])} |"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_alert_episode_quality_report(
    path: Path,
    quality: dict[str, object],
) -> None:
    lines = [
        "# Episode Quality Audit",
        "",
        f"Model variant: {quality['model_variant']}",
        f"Graph window size: {quality['graph_window_size']}",
        "",
        "Start-based capture is used as the default true-positive definition: "
        "an episode is useful when its start date falls within the forecast "
        "horizon before an observed event.",
        "",
        "## Quality Metrics",
        "",
        "| Horizon | Threshold | Episodes | TP episodes | FP episodes | "
        "Unique event capture | Missed events | Episodes / athlete-season | "
        "Median start lead | TP peak risk | FP peak risk |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in quality["quality_rows"]:
        lines.append(
            "| "
            f"{row['horizon_days']}d | "
            f"{row['threshold']} | "
            f"{row['episode_count']} | "
            f"{row['true_positive_episode_count']} "
            f"({_format_metric(row['true_positive_episode_rate'])}) | "
            f"{row['false_positive_episode_count']} "
            f"({_format_metric(row['false_positive_episode_rate'])}) | "
            f"{row['unique_captured_event_count']}/"
            f"{row['unique_observed_event_count']} "
            f"({_format_metric(row['unique_event_capture_rate'])}) | "
            f"{row['missed_event_count']} | "
            f"{_format_metric(row['episodes_per_athlete_season'])} | "
            f"{_format_metric(row['median_start_lead_days'])} | "
            f"{_format_metric(row['true_positive_median_peak_risk'])} | "
            f"{_format_metric(row['false_positive_median_peak_risk'])} |"
        )

    if quality["threshold_overlaps"]:
        lines.extend(
            [
                "",
                "## Threshold Overlap",
                "",
                "| Horizon | Threshold A | Threshold B | Overlap | A overlap | B overlap |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for row in quality["threshold_overlaps"]:
            lines.append(
                "| "
                f"{row['horizon_days']}d | "
                f"{row['threshold_a']} | "
                f"{row['threshold_b']} | "
                f"{row['overlap_episode_count']} | "
                f"{_format_metric(row['threshold_a_overlap_rate'])} | "
                f"{_format_metric(row['threshold_b_overlap_rate'])} |"
            )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_qualitative_case_review_report(
    path: Path,
    review: dict[str, object],
) -> None:
    lines = [
        "# Qualitative Case Review",
        "",
        f"Model variant: {review['model_variant']}",
        f"Graph window size: {review['graph_window_size']}",
        f"Cases: {review['case_count']}",
        "",
        "This report samples representative alert and missed-injury cases so the "
        "next sprint can distinguish model issues from label, data-quality, or "
        "missing-context issues.",
        "",
        "## Diagnostic Summary",
        "",
        "| Diagnostic | Cases |",
        "|---|---:|",
    ]
    for label, count in review["diagnostic_summary"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Type | Label | Diagnosis | Athlete | Season | Horizon | Threshold |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for case in review["cases"]:
        lines.append(
            "| "
            f"{case['case_type']} | "
            f"{case['review_label']} | "
            f"{case['diagnostic_label']} | "
            f"{case['athlete_id']} | "
            f"{case['season_id']} | "
            f"{case['horizon_days']}d | "
            f"{case['threshold']} |"
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_model_improvement_diagnostic_report(
    path: Path,
    diagnostics: dict[str, object],
) -> None:
    lines = [
        "# Model Improvement Diagnostics",
        "",
        f"Model variant: {diagnostics['model_variant']}",
        f"Graph window size: {diagnostics['graph_window_size']}",
        f"Rows: {diagnostics['diagnostic_row_count']}",
        "",
        "This report compares useful alerts, noisy alerts, and missed observed "
        "injuries so the next modeling step can target the right limitation.",
        "",
        "## Recommended Actions",
        "",
        "| Recommended next action | Rows |",
        "|---|---:|",
    ]
    for action, count in diagnostics["recommended_action_summary"].items():
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Diagnostic Table",
            "",
            "| Horizon | Threshold | Group | Count | Median risk | Max pre-event risk | "
            "Z-rate | Recommended next action |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in diagnostics["diagnostic_rows"]:
        lines.append(
            "| "
            f"{row['horizon_days']}d | "
            f"{row['threshold']} | "
            f"{row['comparison_group']} | "
            f"{row['row_count']} | "
            f"{_format_metric(row['median_peak_risk'])} | "
            f"{_format_metric(row['max_pre_event_risk'])} | "
            f"{_format_metric(row['elevated_z_rate'])} | "
            f"{row['recommended_next_action']} |"
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_injury_context_outcome_report(
    path: Path,
    outcomes: dict[str, object],
) -> None:
    lines = [
        "# Injury Context Outcomes",
        "",
        f"Model variant: {outcomes['model_variant']}",
        f"Graph window size: {outcomes['graph_window_size']}",
        f"Event profiles: {outcomes['event_profile_count']}",
        f"Context rows: {outcomes['context_row_count']}",
        "",
        "This report checks whether injury severity, subtype, body area, "
        "recurrence, unavailability, and activity context explain which "
        "events the alert policy captures or misses.",
        "",
        "## Lowest capture contexts",
        "",
        "| Horizon | Threshold | Field | Value | Events | Captured | Capture rate | Median time-loss |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in outcomes["lowest_capture_contexts"]:
        lines.append(
            "| "
            f"{row['horizon_days']}d | "
            f"{row['threshold']} | "
            f"{row['context_field']} | "
            f"{row['context_value']} | "
            f"{row['event_count']} | "
            f"{row['captured_after_start_count']} | "
            f"{_format_metric(row['start_capture_rate'])} | "
            f"{_format_metric(row['median_time_loss_days'])} |"
        )

    lines.extend(
        [
            "",
            "## High time-loss missed contexts",
            "",
            "| Horizon | Threshold | Field | Value | Missed | Capture rate | Median time-loss | Action |",
            "|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in outcomes["high_time_loss_missed_contexts"]:
        lines.append(
            "| "
            f"{row['horizon_days']}d | "
            f"{row['threshold']} | "
            f"{row['context_field']} | "
            f"{row['context_value']} | "
            f"{row['missed_after_start_count']} | "
            f"{_format_metric(row['start_capture_rate'])} | "
            f"{_format_metric(row['median_time_loss_days'])} | "
            f"{row['recommended_next_action']} |"
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_injury_severity_audit_report(
    path: Path,
    audit: dict[str, object],
) -> None:
    lines = [
        "# Injury Severity Audit",
        "",
        f"Events: {audit['event_count']}",
        f"Missing time-loss values: {audit['missing_time_loss_count']}",
        f"Negative time-loss values: {audit['negative_time_loss_count']}",
        f"Extreme time-loss values: {audit['extreme_time_loss_count']}",
        "Extreme time-loss threshold: >365 days",
        f"Duration/resolved-date mismatches: {audit['duration_resolution_mismatch_count']}",
        "",
        "## Time-Loss Buckets",
        "",
        "| Bucket | Events |",
        "|---|---:|",
    ]
    for bucket, count in audit["time_loss_bucket_counts"].items():
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Severity Semantics Flags",
            "",
            "| Flag | Events |",
            "|---|---:|",
        ]
    )
    for flag, count in audit["severity_semantics_flag_counts"].items():
        lines.append(f"| {flag} | {count} |")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_outcome_policy_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Outcome Policy Summary",
        "",
        f"Detailed injury events: {summary['event_count']}",
        f"Outcome policies: {summary['policy_count']}",
        "",
        "This report defines candidate injury outcome policies before changing "
        "the model target.",
        "",
        "| Policy | Events | Share | Median time-loss | Athletes | Recommended use |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["policy_rows"]:
        lines.append(
            "| "
            f"{row['policy_name']} | "
            f"{row['event_count']} | "
            f"{_format_metric(row['event_share'])} | "
            f"{_format_metric(row['median_time_loss_days'])} | "
            f"{row['athlete_count']} | "
            f"{row['recommended_use']} |"
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_context_policy_model_comparison_report(
    path: Path,
    comparison: dict[str, object],
) -> None:
    lines = [
        "# Context Policy Model Comparison",
        "",
        f"Model variant: {comparison['model_variant']}",
        f"Graph window size: {comparison['graph_window_size']}",
        f"Policies: {comparison['policy_count']}",
        f"Rows: {comparison['comparison_row_count']}",
        "",
        "This report compares the same graph model and alert policy across "
        "candidate injury outcome definitions.",
        "",
        "## Best Policies By Horizon",
        "",
    ]
    for horizon in DEFAULT_HORIZONS:
        best = comparison["best_by_horizon"].get(str(horizon), {})
        lines.append(f"### {horizon}d")
        if not best:
            lines.append("- n/a")
            continue
        for metric_name, row in best.items():
            lines.append(
                f"- {metric_name}: {row['policy_name']} "
                f"({row['threshold']}, {_format_metric(row['value'])})"
            )
        lines.append("")

    lines.extend(
        [
            "## Comparison Rows",
            "",
            "| Policy | Horizon | Threshold | AUROC | Brier skill | Top-decile lift | Capture | Missed |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["comparison_rows"]:
        lines.append(
            "| "
            f"{row['policy_name']} | "
            f"{row['horizon_days']}d | "
            f"{row['threshold']} | "
            f"{_format_metric(row['roc_auc'])} | "
            f"{_format_metric(row['brier_skill_score'])} | "
            f"{_format_metric(row['top_decile_lift'])} | "
            f"{_format_metric(row['unique_event_capture_rate'])} | "
            f"{row['missed_event_count']} |"
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_window_sensitivity_report(
    path: Path,
    sensitivity: dict[str, object],
) -> None:
    windows = sensitivity["windows"]
    best_by_horizon = sensitivity["best_by_horizon"]
    lines = [
        "# Window Sensitivity",
        "",
        f"Model variant: {sensitivity['model_variant']}",
        "All graph window sizes use the same athlete-level deterministic holdout split and event policy.",
        "",
        "## Holdout Metrics",
        "",
        "| Window | Horizon | AUROC | Brier skill | Top-decile lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for window_size in sensitivity["graph_window_sizes"]:
        window_payload = windows[str(window_size)]
        for horizon in DEFAULT_HORIZONS:
            metrics = window_payload["horizons"][str(horizon)]
            lines.append(
                "| "
                f"window {window_size} | "
                f"{horizon}d | "
                f"{_format_metric(metrics['roc_auc'])} | "
                f"{_format_metric(metrics['brier_skill_score'])} | "
                f"{_format_metric(metrics['top_decile_lift'])} |"
            )

    lines.extend(["", "## Best Windows", ""])
    for horizon in DEFAULT_HORIZONS:
        horizon_best = best_by_horizon[str(horizon)]
        for label, metric_name in (
            ("AUROC", "roc_auc"),
            ("Brier skill", "brier_skill_score"),
            ("top-decile lift", "top_decile_lift"),
        ):
            if metric_name not in horizon_best:
                lines.append(f"- {horizon}d {label}: n/a")
                continue
            best_metric = horizon_best[metric_name]
            lines.append(
                f"- {horizon}d {label}: window "
                f"{best_metric['graph_window_size']} "
                f"({_format_metric(best_metric['value'])})"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_robustness_report(
    path: Path,
    robustness: dict[str, object],
) -> None:
    variants = robustness["variants"]
    decision_modes = robustness["decision_modes"]
    lines = [
        "# Model Robustness Sprint",
        "",
        f"Graph window size: {robustness['graph_window_size']}",
        f"Split count: {robustness['split_count']}",
        "",
        "## Stability Summary",
        "",
        "| Variant | Horizon | AUROC mean/std | Brier skill mean/std | Brier mean/std | Top-decile lift mean/std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant_name, variant_payload in variants.items():
        for horizon in DEFAULT_HORIZONS:
            summary = variant_payload["summary_by_horizon"][str(horizon)]
            lines.append(
                "| "
                f"{variant_name} | "
                f"{horizon}d | "
                f"{_format_distribution(summary['roc_auc'])} | "
                f"{_format_distribution(summary['brier_skill_score'])} | "
                f"{_format_distribution(summary['model_brier_score'])} | "
                f"{_format_distribution(summary['top_decile_lift'])} |"
            )

    lines.extend(["", "## Decision Mode Winners", ""])
    for mode_name, mode_payload in decision_modes.items():
        lines.append(f"### {mode_name}")
        for horizon in DEFAULT_HORIZONS:
            if str(horizon) not in mode_payload:
                lines.append(f"- {horizon}d: n/a")
                continue
            winner = mode_payload[str(horizon)]
            metric_name = next(key for key in winner if key != "model_variant")
            lines.append(
                f"- {horizon}d: {winner['model_variant']} "
                f"({_format_distribution(winner[metric_name])})"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_window_model_robustness_report(
    path: Path,
    robustness: dict[str, object],
) -> None:
    windows = robustness["windows"]
    overall_decision_modes = robustness["overall_decision_modes"]
    lines = [
        "# Window + Model Robustness Sprint",
        "",
        f"Graph window sizes: {', '.join(str(size) for size in robustness['graph_window_sizes'])}",
        f"Split count: {robustness['split_count']}",
        "",
        "## Stability Summary",
        "",
        "| Window | Variant | Horizon | AUROC mean/std | Brier skill mean/std | Brier mean/std | Top-decile lift mean/std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for window_size in robustness["graph_window_sizes"]:
        window_payload = windows[str(window_size)]
        for variant_name, variant_payload in window_payload["variants"].items():
            for horizon in DEFAULT_HORIZONS:
                summary = variant_payload["summary_by_horizon"][str(horizon)]
                lines.append(
                    "| "
                    f"window {window_size} | "
                    f"{variant_name} | "
                    f"{horizon}d | "
                    f"{_format_distribution(summary['roc_auc'])} | "
                    f"{_format_distribution(summary['brier_skill_score'])} | "
                    f"{_format_distribution(summary['model_brier_score'])} | "
                    f"{_format_distribution(summary['top_decile_lift'])} |"
                )

    lines.extend(["", "## Overall Decision Mode Winners", ""])
    for mode_name, mode_payload in overall_decision_modes.items():
        lines.append(f"### {mode_name}")
        for horizon in DEFAULT_HORIZONS:
            if str(horizon) not in mode_payload:
                lines.append(f"- {horizon}d: n/a")
                continue
            winner = mode_payload[str(horizon)]
            metric_name = next(
                key
                for key in winner
                if key not in {"graph_window_size", "model_variant"}
            )
            lines.append(
                f"- {horizon}d: window {winner['graph_window_size']} "
                f"{winner['model_variant']} "
                f"({_format_distribution(winner[metric_name])})"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _format_distribution(distribution: dict[str, object]) -> str:
    if distribution["mean"] is None:
        return "n/a"
    return f"{float(distribution['mean']):.3f}/{float(distribution['std']):.3f}"


def _horizon_report_line(horizon: int, metrics: dict[str, object]) -> str:
    model_brier = _format_metric(metrics["model_brier_score"])
    prevalence_brier = _format_metric(metrics["prevalence_brier_score"])
    auc = _format_metric(metrics["roc_auc"])
    return (
        f"- +{horizon} days: Brier {model_brier}; "
        f"Prevalence baseline Brier {prevalence_brier}; AUROC {auc}"
    )


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _write_calibration_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Calibration and Threshold Tables",
        "",
        f"Model variant: {summary['model_variant']}",
        f"Graph window size: {summary['graph_window_size']}",
        f"Split count: {summary['split_count']}",
        f"Event policy: {summary['event_policy']}",
        f"Split policy: {summary['split_policy']}",
        "",
        "All predictions are out-of-fold: each athlete-season snapshot is scored by a model that did not train on that athlete.",
        "",
    ]

    for horizon in DEFAULT_HORIZONS:
        h = summary["horizons"][str(horizon)]
        lines.extend(
            [
                f"## Horizon {horizon}d",
                "",
                f"OOF snapshots: {h['oof_snapshot_count']} | "
                f"Positives: {h['oof_positive_count']} "
                f"({_format_metric(h['oof_positive_rate'])})",
                f"Brier score: {_format_metric(h['brier_score'])} | "
                f"Brier skill: {_format_metric(h['brier_skill_score'])}",
                "",
                "### Calibration Bins",
                "",
                "| Bin | Predicted mean | Observed rate | Count | Positives |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in h["calibration_bins"]:
            lines.append(
                f"| {row['bin_index']} | "
                f"{_format_metric(row['predicted_risk_mean'])} | "
                f"{_format_metric(row['observed_event_rate'])} | "
                f"{row['snapshot_count']} | "
                f"{row['positive_count']} |"
            )

        lines.extend(
            [
                "",
                "### Alert Thresholds",
                "",
                "| Kind | Threshold | Alerts | Recall | Precision | Lift |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in h["threshold_rows"]:
            lines.append(
                f"| {row['threshold_kind']} | "
                f"{row['threshold_value']:.2f} | "
                f"{row['alert_count']} | "
                f"{_format_metric(row['event_capture'])} | "
                f"{_format_metric(row['precision'])} | "
                f"{_format_metric(row['lift'])} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
