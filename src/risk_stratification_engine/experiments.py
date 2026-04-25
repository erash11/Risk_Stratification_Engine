from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from risk_stratification_engine.events import DEFAULT_HORIZONS, attach_time_to_event_labels
from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.io import load_injury_events, load_measurements, write_frame
from risk_stratification_engine.trajectories import build_measurement_matrix


def run_research_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
) -> Path:
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    labeled = attach_time_to_event_labels(graph_features, injuries)
    timeline = _risk_timeline(labeled)
    explanations = _explanation_summary(timeline)

    experiment_dir = Path(output_dir) / "experiments" / experiment_id
    graph_dir = experiment_dir / "graph_snapshots"
    explanation_dir = experiment_dir / "explanations"
    graph_dir.mkdir(parents=True, exist_ok=True)
    explanation_dir.mkdir(parents=True, exist_ok=True)

    write_frame(graph_features, graph_dir / "graph_features.csv")
    write_frame(timeline, experiment_dir / "athlete_risk_timeline.csv")
    write_frame(explanations, explanation_dir / "explanation_summary.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_size": graph_window_size,
            "horizons": list(DEFAULT_HORIZONS),
        },
    )
    _write_json(
        experiment_dir / "model_metrics.json",
        {
            "athlete_count": int(labeled["athlete_id"].nunique()),
            "snapshot_count": int(len(labeled)),
            "observed_event_count": int(labeled["event_observed"].sum()),
            "mean_risk_7d": float(timeline["risk_7d"].mean()),
            "mean_risk_14d": float(timeline["risk_14d"].mean()),
            "mean_risk_30d": float(timeline["risk_30d"].mean()),
        },
    )
    _write_report(experiment_dir / "experiment_report.md", timeline)
    return experiment_dir


def _risk_timeline(labeled: pd.DataFrame) -> pd.DataFrame:
    timeline = labeled.copy()
    denominator = timeline["days_to_event"].clip(lower=1)
    for horizon in DEFAULT_HORIZONS:
        temporal_pressure = (horizon / denominator).clip(upper=1.0)
        graph_pressure = timeline["mean_abs_correlation"].clip(lower=0.0, upper=1.0)
        observed_pressure = timeline[f"event_within_{horizon}d"].astype(float)
        timeline[f"risk_{horizon}d"] = (
            0.5 * observed_pressure + 0.3 * temporal_pressure + 0.2 * graph_pressure
        ).round(6)
    return timeline


def _explanation_summary(timeline: pd.DataFrame) -> pd.DataFrame:
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
    explanations["primary_signal"] = explanations.apply(_primary_signal, axis=1)
    return explanations


def _primary_signal(row: pd.Series) -> str:
    if row["edge_count"] == 0:
        return "insufficient_history"
    if row["mean_abs_correlation"] >= 0.7:
        return "strong_metric_relationship_shift"
    return "moderate_metric_relationship_shift"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_report(path: Path, timeline: pd.DataFrame) -> None:
    lines = [
        "# Experiment Report",
        "",
        f"Snapshots: {len(timeline)}",
        f"Athletes: {timeline['athlete_id'].nunique()}",
        f"Mean +7 day risk: {timeline['risk_7d'].mean():.3f}",
        f"Mean +14 day risk: {timeline['risk_14d'].mean():.3f}",
        f"Mean +30 day risk: {timeline['risk_30d'].mean():.3f}",
        "",
        "This first artifact is a deterministic research baseline over graph snapshot features. It preserves the longitudinal time-to-event contract for later model replacement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
