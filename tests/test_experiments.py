from pathlib import Path

import json
import pandas as pd

from risk_stratification_engine.experiments import run_research_experiment


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
    assert (experiment_dir / "model_metrics.json").exists()
    assert (experiment_dir / "experiment_report.md").exists()
    assert (experiment_dir / "athlete_risk_timeline.csv").exists()
    assert (experiment_dir / "graph_snapshots" / "graph_features.csv").exists()
    assert (experiment_dir / "explanations" / "explanation_summary.csv").exists()

    metrics = json.loads((experiment_dir / "model_metrics.json").read_text())
    assert metrics["athlete_count"] == 2
    assert metrics["snapshot_count"] == 4
    assert metrics["observed_event_count"] == 2

    timeline = pd.read_csv(experiment_dir / "athlete_risk_timeline.csv")
    assert {"risk_7d", "risk_14d", "risk_30d"}.issubset(timeline.columns)
