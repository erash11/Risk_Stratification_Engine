from __future__ import annotations

import argparse
from pathlib import Path

from risk_stratification_engine.experiments import run_research_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Peterson-inspired risk stratification research pipeline."
    )
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--injuries", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--graph-window-size", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = run_research_experiment(
        measurements_path=args.measurements,
        injuries_path=args.injuries,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        graph_window_size=args.graph_window_size,
    )
    print(f"Experiment artifacts written to {experiment_dir}")
    return 0
