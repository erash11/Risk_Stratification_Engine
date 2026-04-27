from __future__ import annotations

import argparse
from pathlib import Path

from risk_stratification_engine.config import (
    DEFAULT_PATHS_CONFIG,
    load_data_source_paths,
)
from risk_stratification_engine.experiments import run_research_experiment
from risk_stratification_engine.live_sources import (
    LiveSourcePreparationResult,
    prepare_live_source_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Peterson-inspired risk stratification research pipeline."
    )
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--injuries", type=Path)
    parser.add_argument(
        "--from-live-sources",
        action="store_true",
        help="Prepare canonical inputs from config/paths.local.yaml before running.",
    )
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--graph-window-size", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.from_live_sources:
        data_paths = load_data_source_paths(args.paths_config)
        prepared = prepare_live_source_inputs(
            data_paths,
            args.output_dir / "live_inputs" / args.experiment_id,
        )
        measurements_path = prepared.measurements_path
        injuries_path = prepared.injuries_path
        print(f"Canonical live inputs written to {prepared.measurements_path.parent}")
        print(f"Data quality audit written to {prepared.audit_path}")
    else:
        if args.measurements is None or args.injuries is None:
            parser.error(
                "--measurements and --injuries are required unless "
                "--from-live-sources is set"
            )
        measurements_path = args.measurements
        injuries_path = args.injuries

    experiment_dir = run_research_experiment(
        measurements_path=measurements_path,
        injuries_path=injuries_path,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        graph_window_size=args.graph_window_size,
    )
    print(f"Experiment artifacts written to {experiment_dir}")
    return 0
