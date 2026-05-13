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
    run_case_diagnostic_requirements_sprint_experiment,
    run_coverage_adjusted_threshold_sprint_experiment,
    run_coverage_normalized_policy_sprint_experiment,
    run_coverage_source_aware_model_sprint_experiment,
    run_coverage_stratified_evaluation_experiment,
    run_exposure_feature_requirements_sprint_experiment,
    run_exposure_load_feature_sprint_experiment,
    run_exposure_load_failure_mode_sprint_experiment,
    run_exposure_load_forward_diagnostic_sprint_experiment,
    run_exposure_load_guardrail_policy_sprint_experiment,
    run_exposure_load_season_forward_validation_sprint_experiment,
    run_exposure_load_shift_context_sprint_experiment,
    run_exposure_load_schedule_roster_sprint_experiment,
    run_exposure_load_availability_capture_sprint_experiment,
    run_exposure_load_context_decision_sprint_experiment,
    run_exposure_load_source_context_classification_sprint_experiment,
    run_exposure_load_source_eligible_calibration_sprint_experiment,
    run_exposure_load_source_eligible_policy_sprint_experiment,
    run_exposure_load_source_eligible_shadow_monitoring_sprint_experiment,
    run_exposure_load_source_resolution_sprint_experiment,
    run_exposure_load_shadow_adjudication_decision_sprint_experiment,
    run_exposure_load_shadow_adjudication_summary_sprint_experiment,
    run_exposure_load_shadow_adjudication_sprint_experiment,
    run_exposure_load_shadow_channel_lock_sprint_experiment,
    run_exposure_load_shadow_collection_template_sprint_experiment,
    run_exposure_load_shadow_monitoring_plan_sprint_experiment,
    run_exposure_load_shadow_readiness_register_sprint_experiment,
    run_exposure_load_shadow_replay_sprint_experiment,
    run_exposure_load_shadow_review_protocol_sprint_experiment,
    run_forward_case_review_sprint_experiment,
    run_injury_history_feature_sprint_experiment,
    run_injury_history_forward_diagnostic_sprint_experiment,
    run_injury_history_season_forward_validation_sprint_experiment,
    run_injury_outcome_policy_experiment,
    run_model_robustness_experiment,
    run_outcome_policy_model_comparison_experiment,
    run_policy_decision_sprint_experiment,
    run_research_experiment,
    run_season_forward_validation_sprint_experiment,
    run_season_drift_diagnostic_experiment,
    run_shadow_mode_stability_experiment,
    run_window_model_robustness_experiment,
    run_window_sensitivity_experiment,
)
from risk_stratification_engine.exposure_sources import (
    ExposurePreparationResult,
    prepare_exposure_inputs,
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
    parser.add_argument("--exposure-dir", type=Path)
    parser.add_argument("--exposure-events", type=Path)
    parser.add_argument("--exposure-participations", type=Path)
    parser.add_argument("--exposure-audit", type=Path)
    parser.add_argument("--exposure-load-features", type=Path)
    parser.add_argument("--exposure-load-diagnostics", type=Path)
    parser.add_argument("--exposure-load-failure-modes", type=Path)
    parser.add_argument("--exposure-load-shift-context", type=Path)
    parser.add_argument("--exposure-load-schedule-roster", type=Path)
    parser.add_argument("--exposure-load-availability-capture", type=Path)
    parser.add_argument("--exposure-load-guardrail-policy", type=Path)
    parser.add_argument("--exposure-load-context-decision", type=Path)
    parser.add_argument("--exposure-load-source-context-classification", type=Path)
    parser.add_argument("--exposure-load-source-eligible-calibration", type=Path)
    parser.add_argument("--exposure-load-source-eligible-policy", type=Path)
    parser.add_argument("--exposure-load-source-eligible-shadow-monitoring", type=Path)
    parser.add_argument("--exposure-load-shadow-channel-lock", type=Path)
    parser.add_argument("--exposure-load-shadow-review-protocol", type=Path)
    parser.add_argument("--exposure-load-shadow-replay", type=Path)
    parser.add_argument("--exposure-load-shadow-adjudication", type=Path)
    parser.add_argument("--exposure-load-shadow-adjudication-summary", type=Path)
    parser.add_argument("--exposure-load-shadow-adjudication-decision", type=Path)
    parser.add_argument("--exposure-load-shadow-monitoring-plan", type=Path)
    parser.add_argument("--exposure-load-source-resolution-policy", type=Path)
    parser.add_argument("--season-forward-validation-path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--graph-window-size", type=int, default=4)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="baseline")
    parser.add_argument("--window-sensitivity-sizes", nargs="+", type=int)
    parser.add_argument("--policy-window-sizes", nargs="+", type=int, default=[2, 4, 7])
    parser.add_argument("--model-robustness-sprint", action="store_true")
    parser.add_argument("--calibration-thresholds", action="store_true")
    parser.add_argument("--alert-episodes", action="store_true")
    parser.add_argument("--injury-outcome-policies", action="store_true")
    parser.add_argument("--outcome-policy-model-comparison", action="store_true")
    parser.add_argument("--policy-decision-sprint", action="store_true")
    parser.add_argument("--shadow-mode-stability", action="store_true")
    parser.add_argument("--season-drift-diagnostic", action="store_true")
    parser.add_argument("--coverage-stratified-evaluation", action="store_true")
    parser.add_argument("--coverage-normalized-policy-sprint", action="store_true")
    parser.add_argument("--coverage-source-aware-model-sprint", action="store_true")
    parser.add_argument("--coverage-adjusted-threshold-sprint", action="store_true")
    parser.add_argument("--season-forward-validation", action="store_true")
    parser.add_argument("--forward-case-review-sprint", action="store_true")
    parser.add_argument("--case-diagnostic-requirements-sprint", action="store_true")
    parser.add_argument("--injury-history-feature-sprint", action="store_true")
    parser.add_argument(
        "--injury-history-season-forward-validation",
        action="store_true",
    )
    parser.add_argument("--injury-history-forward-diagnostic-sprint", action="store_true")
    parser.add_argument("--exposure-cleaning-audit", action="store_true")
    parser.add_argument("--exposure-feature-requirements-sprint", action="store_true")
    parser.add_argument("--exposure-load-feature-sprint", action="store_true")
    parser.add_argument(
        "--exposure-load-season-forward-validation",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-forward-diagnostic-sprint",
        action="store_true",
    )
    parser.add_argument("--exposure-load-failure-mode-sprint", action="store_true")
    parser.add_argument("--exposure-load-guardrail-policy-sprint", action="store_true")
    parser.add_argument("--exposure-load-shift-context-sprint", action="store_true")
    parser.add_argument("--exposure-load-schedule-roster-sprint", action="store_true")
    parser.add_argument(
        "--exposure-load-availability-capture-sprint",
        action="store_true",
    )
    parser.add_argument("--exposure-load-context-decision-sprint", action="store_true")
    parser.add_argument(
        "--exposure-load-source-context-classification-sprint",
        action="store_true",
    )
    parser.add_argument("--exposure-load-source-resolution-sprint", action="store_true")
    parser.add_argument(
        "--exposure-load-source-eligible-calibration-sprint",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-source-eligible-policy-sprint",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-source-eligible-shadow-monitoring-sprint",
        action="store_true",
    )
    parser.add_argument("--exposure-load-shadow-channel-lock-sprint", action="store_true")
    parser.add_argument(
        "--exposure-load-shadow-review-protocol-sprint",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-shadow-readiness-register-sprint",
        action="store_true",
    )
    parser.add_argument("--exposure-load-shadow-replay-sprint", action="store_true")
    parser.add_argument(
        "--exposure-load-shadow-adjudication-sprint",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-shadow-adjudication-summary-sprint",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-shadow-adjudication-decision-sprint",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-shadow-monitoring-plan-sprint",
        action="store_true",
    )
    parser.add_argument(
        "--exposure-load-shadow-collection-template-sprint",
        action="store_true",
    )
    parser.add_argument("--stability-splits", type=int, default=5)
    return parser


def _resolve_exposure_participations(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    flag_name: str,
) -> Path:
    exposure_participations = args.exposure_participations
    if exposure_participations is not None:
        return exposure_participations
    exposure_dir = args.exposure_dir
    if exposure_dir is None:
        data_paths = load_data_source_paths(args.paths_config)
        exposure_dir = data_paths.exposure_dir
    if exposure_dir is None:
        parser.error(
            f"{flag_name} requires --exposure-participations, --exposure-dir, "
            "or exposure_dir in --paths-config"
        )
    prepared_exposure = prepare_exposure_inputs(
        exposure_dir,
        args.output_dir / "exposure_inputs" / args.experiment_id,
    )
    return prepared_exposure.participations_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.exposure_cleaning_audit:
        exposure_dir = args.exposure_dir
        if exposure_dir is None:
            data_paths = load_data_source_paths(args.paths_config)
            exposure_dir = data_paths.exposure_dir
        if exposure_dir is None:
            parser.error(
                "--exposure-cleaning-audit requires --exposure-dir or "
                "exposure_dir in --paths-config"
            )
        prepared_exposure = prepare_exposure_inputs(
            exposure_dir,
            args.output_dir / "exposure_inputs" / args.experiment_id,
        )
        print(f"Exposure events written to {prepared_exposure.events_path}")
        print(
            "Exposure participations written to "
            f"{prepared_exposure.participations_path}"
        )
        print(f"Exposure cleaning audit written to {prepared_exposure.audit_path}")
        return 0
    if args.exposure_feature_requirements_sprint:
        exposure_events = args.exposure_events
        exposure_participations = args.exposure_participations
        exposure_audit = args.exposure_audit
        explicit_paths = (exposure_events, exposure_participations, exposure_audit)
        if any(path is not None for path in explicit_paths) and not all(
            path is not None for path in explicit_paths
        ):
            parser.error(
                "--exposure-feature-requirements-sprint requires all of "
                "--exposure-events, --exposure-participations, and --exposure-audit "
                "when using cleaned artifact paths"
            )
        if not all(path is not None for path in explicit_paths):
            exposure_dir = args.exposure_dir
            if exposure_dir is None:
                data_paths = load_data_source_paths(args.paths_config)
                exposure_dir = data_paths.exposure_dir
            if exposure_dir is None:
                parser.error(
                    "--exposure-feature-requirements-sprint requires cleaned "
                    "exposure artifact paths, --exposure-dir, or exposure_dir "
                    "in --paths-config"
                )
            prepared_exposure = prepare_exposure_inputs(
                exposure_dir,
                args.output_dir / "exposure_inputs" / args.experiment_id,
            )
            exposure_events = prepared_exposure.events_path
            exposure_participations = prepared_exposure.participations_path
            exposure_audit = prepared_exposure.audit_path
        experiment_dir = run_exposure_feature_requirements_sprint_experiment(
            exposure_events_path=exposure_events,
            exposure_participations_path=exposure_participations,
            exposure_audit_path=exposure_audit,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure feature requirements artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_forward_diagnostic_sprint:
        if args.season_forward_validation_path is None:
            parser.error(
                "--exposure-load-forward-diagnostic-sprint requires "
                "--season-forward-validation-path"
            )
        experiment_dir = run_exposure_load_forward_diagnostic_sprint_experiment(
            season_forward_validation_path=args.season_forward_validation_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load forward diagnostic artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_failure_mode_sprint:
        if args.exposure_load_features is None:
            parser.error(
                "--exposure-load-failure-mode-sprint requires "
                "--exposure-load-features"
            )
        if args.exposure_load_diagnostics is None:
            parser.error(
                "--exposure-load-failure-mode-sprint requires "
                "--exposure-load-diagnostics"
            )
        experiment_dir = run_exposure_load_failure_mode_sprint_experiment(
            exposure_load_features_path=args.exposure_load_features,
            exposure_load_diagnostics_path=args.exposure_load_diagnostics,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load failure mode artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_guardrail_policy_sprint:
        if args.exposure_load_failure_modes is None:
            parser.error(
                "--exposure-load-guardrail-policy-sprint requires "
                "--exposure-load-failure-modes"
            )
        if args.exposure_load_diagnostics is None:
            parser.error(
                "--exposure-load-guardrail-policy-sprint requires "
                "--exposure-load-diagnostics"
            )
        experiment_dir = run_exposure_load_guardrail_policy_sprint_experiment(
            exposure_load_failure_modes_path=args.exposure_load_failure_modes,
            exposure_load_diagnostics_path=args.exposure_load_diagnostics,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load guardrail policy artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_shift_context_sprint:
        required_paths = {
            "--exposure-events": args.exposure_events,
            "--exposure-participations": args.exposure_participations,
            "--exposure-load-features": args.exposure_load_features,
            "--exposure-load-diagnostics": args.exposure_load_diagnostics,
            "--exposure-load-failure-modes": args.exposure_load_failure_modes,
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-shift-context-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = run_exposure_load_shift_context_sprint_experiment(
            exposure_events_path=args.exposure_events,
            exposure_participations_path=args.exposure_participations,
            exposure_load_features_path=args.exposure_load_features,
            exposure_load_diagnostics_path=args.exposure_load_diagnostics,
            exposure_load_failure_modes_path=args.exposure_load_failure_modes,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load shift context artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_schedule_roster_sprint:
        required_paths = {
            "--exposure-events": args.exposure_events,
            "--exposure-participations": args.exposure_participations,
            "--exposure-load-shift-context": args.exposure_load_shift_context,
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-schedule-roster-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = run_exposure_load_schedule_roster_sprint_experiment(
            exposure_events_path=args.exposure_events,
            exposure_participations_path=args.exposure_participations,
            exposure_load_shift_context_path=args.exposure_load_shift_context,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load schedule roster artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_availability_capture_sprint:
        required_paths = {
            "--exposure-participations": args.exposure_participations,
            "--exposure-load-shift-context": args.exposure_load_shift_context,
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-availability-capture-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = run_exposure_load_availability_capture_sprint_experiment(
            exposure_participations_path=args.exposure_participations,
            exposure_load_shift_context_path=args.exposure_load_shift_context,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(
            "Exposure load availability capture artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_context_decision_sprint:
        required_paths = {
            "--exposure-load-shift-context": args.exposure_load_shift_context,
            "--exposure-load-schedule-roster": args.exposure_load_schedule_roster,
            "--exposure-load-availability-capture": (
                args.exposure_load_availability_capture
            ),
            "--exposure-load-guardrail-policy": args.exposure_load_guardrail_policy,
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-context-decision-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = run_exposure_load_context_decision_sprint_experiment(
            exposure_load_shift_context_path=args.exposure_load_shift_context,
            exposure_load_schedule_roster_path=args.exposure_load_schedule_roster,
            exposure_load_availability_capture_path=(
                args.exposure_load_availability_capture
            ),
            exposure_load_guardrail_policy_path=args.exposure_load_guardrail_policy,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load context decision artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_source_context_classification_sprint:
        required_paths = {
            "--exposure-events": args.exposure_events,
            "--exposure-participations": args.exposure_participations,
            "--exposure-load-shift-context": args.exposure_load_shift_context,
            "--exposure-load-schedule-roster": args.exposure_load_schedule_roster,
            "--exposure-load-availability-capture": (
                args.exposure_load_availability_capture
            ),
            "--exposure-load-context-decision": args.exposure_load_context_decision,
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-source-context-classification-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = (
            run_exposure_load_source_context_classification_sprint_experiment(
                exposure_events_path=args.exposure_events,
                exposure_participations_path=args.exposure_participations,
                exposure_load_shift_context_path=args.exposure_load_shift_context,
                exposure_load_schedule_roster_path=args.exposure_load_schedule_roster,
                exposure_load_availability_capture_path=(
                    args.exposure_load_availability_capture
                ),
                exposure_load_context_decision_path=args.exposure_load_context_decision,
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
            )
        )
        print(
            "Exposure load source context classification artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_source_resolution_sprint:
        if args.exposure_load_source_context_classification is None:
            parser.error(
                "--exposure-load-source-resolution-sprint requires "
                "--exposure-load-source-context-classification"
            )
        experiment_dir = run_exposure_load_source_resolution_sprint_experiment(
            exposure_load_source_context_classification_path=(
                args.exposure_load_source_context_classification
            ),
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load source resolution artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_source_eligible_calibration_sprint:
        required_paths = {
            "--season-forward-validation-path": args.season_forward_validation_path,
            "--exposure-load-source-resolution-policy": (
                args.exposure_load_source_resolution_policy
            ),
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-source-eligible-calibration-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = (
            run_exposure_load_source_eligible_calibration_sprint_experiment(
                season_forward_validation_path=args.season_forward_validation_path,
                exposure_load_source_resolution_policy_path=(
                    args.exposure_load_source_resolution_policy
                ),
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
            )
        )
        print(
            "Exposure load source-eligible calibration artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_source_eligible_policy_sprint:
        required_paths = {
            "--season-forward-validation-path": args.season_forward_validation_path,
            "--exposure-load-source-eligible-calibration": (
                args.exposure_load_source_eligible_calibration
            ),
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-source-eligible-policy-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = run_exposure_load_source_eligible_policy_sprint_experiment(
            season_forward_validation_path=args.season_forward_validation_path,
            exposure_load_source_eligible_calibration_path=(
                args.exposure_load_source_eligible_calibration
            ),
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(
            "Exposure load source-eligible policy artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_source_eligible_shadow_monitoring_sprint:
        required_paths = {
            "--season-forward-validation-path": args.season_forward_validation_path,
            "--exposure-load-source-eligible-policy": (
                args.exposure_load_source_eligible_policy
            ),
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-source-eligible-shadow-monitoring-sprint "
                "requires "
                + ", ".join(missing)
            )
        experiment_dir = (
            run_exposure_load_source_eligible_shadow_monitoring_sprint_experiment(
                season_forward_validation_path=args.season_forward_validation_path,
                exposure_load_source_eligible_policy_path=(
                    args.exposure_load_source_eligible_policy
                ),
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
            )
        )
        print(
            "Exposure load source-eligible shadow monitoring artifacts written "
            f"to {experiment_dir}"
        )
        return 0

    if args.exposure_load_shadow_channel_lock_sprint:
        if args.exposure_load_source_eligible_shadow_monitoring is None:
            parser.error(
                "--exposure-load-shadow-channel-lock-sprint requires "
                "--exposure-load-source-eligible-shadow-monitoring"
            )
        experiment_dir = run_exposure_load_shadow_channel_lock_sprint_experiment(
            exposure_load_source_eligible_shadow_monitoring_path=(
                args.exposure_load_source_eligible_shadow_monitoring
            ),
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load shadow channel lock artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_shadow_review_protocol_sprint:
        if args.exposure_load_shadow_channel_lock is None:
            parser.error(
                "--exposure-load-shadow-review-protocol-sprint requires "
                "--exposure-load-shadow-channel-lock"
            )
        experiment_dir = run_exposure_load_shadow_review_protocol_sprint_experiment(
            exposure_load_shadow_channel_lock_path=args.exposure_load_shadow_channel_lock,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(
            "Exposure load shadow review protocol artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_shadow_readiness_register_sprint:
        required_paths = {
            "--exposure-load-shadow-channel-lock": (
                args.exposure_load_shadow_channel_lock
            ),
            "--exposure-load-shadow-review-protocol": (
                args.exposure_load_shadow_review_protocol
            ),
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-shadow-readiness-register-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = run_exposure_load_shadow_readiness_register_sprint_experiment(
            exposure_load_shadow_channel_lock_path=(
                args.exposure_load_shadow_channel_lock
            ),
            exposure_load_shadow_review_protocol_path=(
                args.exposure_load_shadow_review_protocol
            ),
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(
            "Exposure load shadow readiness register artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_shadow_replay_sprint:
        required_paths = {
            "--season-forward-validation-path": args.season_forward_validation_path,
            "--exposure-load-shadow-channel-lock": (
                args.exposure_load_shadow_channel_lock
            ),
            "--exposure-load-shadow-review-protocol": (
                args.exposure_load_shadow_review_protocol
            ),
        }
        missing = [flag for flag, path in required_paths.items() if path is None]
        if missing:
            parser.error(
                "--exposure-load-shadow-replay-sprint requires "
                + ", ".join(missing)
            )
        experiment_dir = run_exposure_load_shadow_replay_sprint_experiment(
            season_forward_validation_path=args.season_forward_validation_path,
            exposure_load_shadow_channel_lock_path=(
                args.exposure_load_shadow_channel_lock
            ),
            exposure_load_shadow_review_protocol_path=(
                args.exposure_load_shadow_review_protocol
            ),
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(f"Exposure load shadow replay artifacts written to {experiment_dir}")
        return 0

    if args.exposure_load_shadow_adjudication_sprint:
        if args.exposure_load_shadow_replay is None:
            parser.error(
                "--exposure-load-shadow-adjudication-sprint requires "
                "--exposure-load-shadow-replay"
            )
        experiment_dir = run_exposure_load_shadow_adjudication_sprint_experiment(
            exposure_load_shadow_replay_path=args.exposure_load_shadow_replay,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(
            "Exposure load shadow adjudication artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_shadow_adjudication_summary_sprint:
        if args.exposure_load_shadow_adjudication is None:
            parser.error(
                "--exposure-load-shadow-adjudication-summary-sprint requires "
                "--exposure-load-shadow-adjudication"
            )
        experiment_dir = (
            run_exposure_load_shadow_adjudication_summary_sprint_experiment(
                exposure_load_shadow_adjudication_path=(
                    args.exposure_load_shadow_adjudication
                ),
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
            )
        )
        print(
            "Exposure load shadow adjudication summary artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_shadow_adjudication_decision_sprint:
        if args.exposure_load_shadow_adjudication_summary is None:
            parser.error(
                "--exposure-load-shadow-adjudication-decision-sprint requires "
                "--exposure-load-shadow-adjudication-summary"
            )
        experiment_dir = (
            run_exposure_load_shadow_adjudication_decision_sprint_experiment(
                exposure_load_shadow_adjudication_summary_path=(
                    args.exposure_load_shadow_adjudication_summary
                ),
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
            )
        )
        print(
            "Exposure load shadow adjudication decision artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_shadow_monitoring_plan_sprint:
        if args.exposure_load_shadow_adjudication_decision is None:
            parser.error(
                "--exposure-load-shadow-monitoring-plan-sprint requires "
                "--exposure-load-shadow-adjudication-decision"
            )
        experiment_dir = run_exposure_load_shadow_monitoring_plan_sprint_experiment(
            exposure_load_shadow_adjudication_decision_path=(
                args.exposure_load_shadow_adjudication_decision
            ),
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
        print(
            "Exposure load shadow monitoring plan artifacts written to "
            f"{experiment_dir}"
        )
        return 0

    if args.exposure_load_shadow_collection_template_sprint:
        if args.exposure_load_shadow_monitoring_plan is None:
            parser.error(
                "--exposure-load-shadow-collection-template-sprint requires "
                "--exposure-load-shadow-monitoring-plan"
            )
        experiment_dir = (
            run_exposure_load_shadow_collection_template_sprint_experiment(
                exposure_load_shadow_monitoring_plan_path=(
                    args.exposure_load_shadow_monitoring_plan
                ),
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
            )
        )
        print(
            "Exposure load shadow collection template artifacts written to "
            f"{experiment_dir}"
        )
        return 0

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

    if args.season_drift_diagnostic:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--season-drift-diagnostic requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_season_drift_diagnostic_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Season drift diagnostic artifacts written to {experiment_dir}")
        return 0
    if args.coverage_stratified_evaluation:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--coverage-stratified-evaluation requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_coverage_stratified_evaluation_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Coverage-stratified evaluation written to {experiment_dir}")
        return 0
    if args.coverage_normalized_policy_sprint:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--coverage-normalized-policy-sprint requires live-source "
                    "detailed injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_coverage_normalized_policy_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Coverage-normalized policy artifacts written to {experiment_dir}")
        return 0
    if args.coverage_source_aware_model_sprint:
        experiment_dir = run_coverage_source_aware_model_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
        )
        print(f"Coverage/source-aware model artifacts written to {experiment_dir}")
        return 0
    if args.exposure_load_feature_sprint:
        exposure_participations = _resolve_exposure_participations(
            parser,
            args,
            "--exposure-load-feature-sprint",
        )
        experiment_dir = run_exposure_load_feature_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            exposure_participations_path=exposure_participations,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
        )
        print(f"Exposure load feature artifacts written to {experiment_dir}")
        return 0
    if args.exposure_load_season_forward_validation:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--exposure-load-season-forward-validation requires "
                    "live-source detailed injury events or a sibling "
                    "injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        exposure_participations = _resolve_exposure_participations(
            parser,
            args,
            "--exposure-load-season-forward-validation",
        )
        experiment_dir = (
            run_exposure_load_season_forward_validation_sprint_experiment(
                measurements_path=measurements_path,
                injuries_path=injuries_path,
                detailed_injuries_path=detailed_injuries_path,
                exposure_participations_path=exposure_participations,
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
                graph_window_size=args.graph_window_size,
                model_variant=args.model_variant,
            )
        )
        print(
            "Exposure load season-forward validation artifacts written to "
            f"{experiment_dir}"
        )
        return 0
    if args.coverage_adjusted_threshold_sprint:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--coverage-adjusted-threshold-sprint requires live-source "
                    "detailed injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_coverage_adjusted_threshold_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Coverage-adjusted threshold artifacts written to {experiment_dir}")
        return 0
    if args.season_forward_validation:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--season-forward-validation requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_season_forward_validation_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
        )
        print(f"Season-forward validation artifacts written to {experiment_dir}")
        return 0
    if args.forward_case_review_sprint:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--forward-case-review-sprint requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_forward_case_review_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Forward case review artifacts written to {experiment_dir}")
        return 0
    if args.case_diagnostic_requirements_sprint:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--case-diagnostic-requirements-sprint requires live-source "
                    "detailed injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_case_diagnostic_requirements_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Case diagnostic requirements artifacts written to {experiment_dir}")
        return 0
    if args.injury_history_feature_sprint:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--injury-history-feature-sprint requires live-source "
                    "detailed injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_injury_history_feature_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
        )
        print(f"Injury history feature artifacts written to {experiment_dir}")
        return 0
    if args.injury_history_season_forward_validation:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--injury-history-season-forward-validation requires "
                    "live-source detailed injury events or a sibling "
                    "injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = (
            run_injury_history_season_forward_validation_sprint_experiment(
                measurements_path=measurements_path,
                injuries_path=injuries_path,
                detailed_injuries_path=detailed_injuries_path,
                output_dir=args.output_dir,
                experiment_id=args.experiment_id,
                graph_window_size=args.graph_window_size,
                model_variant=args.model_variant,
            )
        )
        print(
            "Injury history season-forward validation artifacts written to "
            f"{experiment_dir}"
        )
        return 0
    if args.injury_history_forward_diagnostic_sprint:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--injury-history-forward-diagnostic-sprint requires "
                    "live-source detailed injury events or a sibling "
                    "injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_injury_history_forward_diagnostic_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_size=args.graph_window_size,
            model_variant=args.model_variant,
        )
        print(
            "Injury history forward diagnostic artifacts written to "
            f"{experiment_dir}"
        )
        return 0
    elif args.shadow_mode_stability:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--shadow-mode-stability requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_shadow_mode_stability_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Shadow-mode stability artifacts written to {experiment_dir}")
    elif args.policy_decision_sprint:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--policy-decision-sprint requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_policy_decision_sprint_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            graph_window_sizes=tuple(args.policy_window_sizes),
            model_variant=args.model_variant,
        )
        print(f"Policy decision sprint artifacts written to {experiment_dir}")
    elif args.outcome_policy_model_comparison:
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
