from __future__ import annotations

import argparse
from pathlib import Path

from risk_stratification_engine.config import (
    DEFAULT_PATHS_CONFIG,
    load_data_source_paths,
)
from risk_stratification_engine.experiments import (
    run_alert_episode_experiment,
    run_calibration_threshold_experiment,
    run_injury_outcome_policy_experiment,
    run_model_robustness_experiment,
    run_outcome_policy_model_comparison_experiment,
    run_research_experiment,
    run_window_model_robustness_experiment,
    run_window_sensitivity_experiment,
)
from risk_stratification_engine.live_sources import (
    LiveSourcePreparationResult,
    prepare_live_source_inputs,
)
from risk_stratification_engine.models import MODEL_VARIANTS


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
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="baseline")
    parser.add_argument("--window-sensitivity-sizes", nargs="+", type=int)
    parser.add_argument("--model-robustness-sprint", action="store_true")
    parser.add_argument("--calibration-thresholds", action="store_true")
    parser.add_argument("--alert-episodes", action="store_true")
    parser.add_argument("--injury-outcome-policies", action="store_true")
    parser.add_argument("--outcome-policy-model-comparison", action="store_true")
    parser.add_argument("--stability-splits", type=int, default=5)
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
        detailed_injuries_path = prepared.detailed_injuries_path
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
        detailed_injuries_path = None

    if args.outcome_policy_model_comparison:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--outcome-policy-model-comparison requires live-source "
                    "detailed injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_outcome_policy_model_comparison_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
        )
        print(f"Outcome policy model comparison artifacts written to {experiment_dir}")
    elif args.injury_outcome_policies:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--injury-outcome-policies requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_injury_outcome_policy_experiment(
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Injury outcome policy artifacts written to {experiment_dir}")
    elif args.alert_episodes:
        experiment_dir = run_alert_episode_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
            detailed_injuries_path=detailed_injuries_path,
        )
        print(f"Alert episode artifacts written to {experiment_dir}")
    elif args.calibration_thresholds:
        experiment_dir = run_calibration_threshold_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
            split_count=args.stability_splits,
        )
        print(f"Calibration and threshold artifacts written to {experiment_dir}")
    elif args.model_robustness_sprint and args.window_sensitivity_sizes:
        experiment_dir = run_window_model_robustness_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_sizes=tuple(args.window_sensitivity_sizes),
            split_count=args.stability_splits,
        )
        print(f"Window + model robustness artifacts written to {experiment_dir}")
    elif args.model_robustness_sprint:
        experiment_dir = run_model_robustness_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            split_count=args.stability_splits,
        )
        print(f"Model robustness artifacts written to {experiment_dir}")
    elif args.window_sensitivity_sizes:
        experiment_dir = run_window_sensitivity_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_sizes=tuple(args.window_sensitivity_sizes),
            model_variant=args.model_variant,
        )
        print(f"Window sensitivity artifacts written to {experiment_dir}")
    else:
        experiment_dir = run_research_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
        )
        print(f"Experiment artifacts written to {experiment_dir}")
    return 0
