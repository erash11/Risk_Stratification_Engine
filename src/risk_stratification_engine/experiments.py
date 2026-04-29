from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import pandas as pd
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
from risk_stratification_engine.evaluation import evaluate_risk_model
from risk_stratification_engine.events import DEFAULT_HORIZONS, attach_time_to_event_labels
from risk_stratification_engine.episode_quality import build_alert_episode_quality
from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.injury_context import build_injury_context_outcomes
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
