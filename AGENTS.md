# Risk Stratification Engine Agent Notes

## Project Direction

This project is a research prototype for an athlete risk stratification platform. The first milestone is not an operator dashboard. It is a reproducible modeling workbench that can test whether Peterson's longitudinal, graph-based injury forecasting philosophy holds in this environment.

Future work may add a dashboard performance tab inspired by the Malum/SPEAR materials, but the research pipeline comes first.

## Core Modeling Philosophy

- Treat each athlete-season as an evolving physiological system.
- Do not treat athlete-days as independent primary examples.
- Preserve the athlete's longitudinal history when preparing model inputs.
- Prefer time-to-event forecasting with censoring over daily injury classification.
- Use dynamic graphs to represent changing relationships among monitoring variables.
- Keep athlete-specific explanations central: elevated risk should be tied to changing variables, edges, and time windows.
- Use baseline tabular models only as secondary benchmarks, not as the north star.

## Source Material Anchors

- Peterson's dissertation is the conceptual anchor: intra-individual dynamic graph construction, longitudinal time-to-event forecasting, and athlete-specific temporal explanations.
- Malum/SPEAR materials inform eventual product concepts: readiness dashboard, injury-free probability horizons, risk timelines, influential variables, simulation, and suggested intervention targets.
- The current folder does not contain a separate Spear document. SPEAR appears within the Malum production UI slide.

## Prototype Priorities

1. Normalize historical athlete monitoring exports into a consistent measurement format.
2. Build athlete-season trajectories from those measurements.
3. Estimate athlete-specific dynamic graph snapshots.
4. Train and evaluate longitudinal time-to-event risk models from graph trajectories.
5. Generate reproducible research outputs: metrics, graph artifacts, risk timelines, and explanation reports.

## Data Handling Principles

- Canonical raw measurement fields should include `athlete_id`, `date`, `season_id`, `source`, `metric_name`, and `metric_value`.
- Modeling artifacts should preserve `athlete_id`, `season_id`, `time_index`, `graph_snapshot`, `event_time`, `event_observed`, `censor_time`, and `injury_type`.
- `event_observed = false` means the athlete was event-free through the observed window, not that the athlete is permanently healthy.
- Use athlete-level and time-aware validation splits. Avoid random daily-row splits.
- Real/local source data should remain in canonical upstream project locations or ignored raw-data folders, not under `src/`.
- Use `config/paths.example.yaml` as the committed template and `config/paths.local.yaml` as the ignored machine-specific file for live source paths.
- The current required live-source keys are `forceplate_db`, `gps_db`, `bodyweight_csv`, `perch_db`, and `injury_csv`; optional `exposure_dir` points to the Baylor exposure export folder for cleaning/audit runs.
- As of 2026-04-25, the local `config/paths.local.yaml` resolves all five live-source keys successfully.
- Local injury exports are in `data/raw/` as ignored `injuries-summary-export-*.csv` files; keep raw injury data ignored.
- When running against live source files, record path metadata, file existence, schemas, and row counts in experiment/data-quality artifacts for reproducibility.
- As of 2026-04-27, live-source ingestion is available through `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id <id>`.
- Live-source ingestion writes ignored canonical inputs to `outputs/live_inputs/<experiment-id>/`, uses stable hashed athlete IDs from normalized names, starts seasons on July 1, and uses the earliest injury issue date per athlete-season with censoring at the last measurement date.
- If `injury_csv` points to a file named `injuries-summary-export-*.csv`, live-source ingestion loads all sibling files with that pattern, de-duplicates exact raw rows, and writes `injury_events_detailed.csv` with one de-identified row per raw injury event. The detailed artifact preserves issue/resolved dates, duration and time-loss fields, recurrence, unavailability, activity, classification/pathology, body area, tissue type, side, participation level, training/game context, source file, and source row number while leaving `canonical_injuries.csv` as the current first-event-per-athlete-season modeling label file.
- Live-source ingestion also writes `data_quality_audit.json` with hashed source-overlap checks, sparse athlete-season flags, large within-season date gaps, duplicate same-day metric rows, and injury events without nearby measurements.
- Live-source name normalization reconciles common `Last, First` export names with `First Last` names before hashing, and duplicate same-day metric rows are aggregated by mean value before modeling with counts recorded in `prep_metadata.json`.
- `data_quality_audit.json` includes privacy-preserving review context for remaining single-source hashed identities and observed injury events outside the nearby-measurement window.
- Observed live-source injury events are labeled by nearest same-season measurement distance: `modelable` at 14 days or less, `low_confidence` at 15-30 days, and `out_of_window` beyond 30 days; downstream modeling should prefer `primary_model_event = true` until calibration work says otherwise.
- Exposure-source cleaning is available through `risk-engine --exposure-cleaning-audit --exposure-dir <Baylor_Exposure_Data> --output-dir outputs --experiment-id <id>`. This cleaning-only pass writes `exposure_events.csv`, `exposure_participations.csv`, and `exposure_cleaning_audit.json` under `outputs/exposure_inputs/<experiment-id>/`; it must precede any exposure/load feature sprint or season-forward validation.
- Exposure cleaning filters football events by `ExternalSquadId == 94`, maps athlete identities from `FirstName + " " + LastName` through `stable_athlete_id(...)`, preserves `ExternalAthleteId` as source metadata, and treats only athletes whose pipe/comma/semicolon-delimited `ExternalSquadIds` include `94` as matched football athletes. Do not use football-looking `Position` text alone because Zandbox athletes can carry football positions.
- Exposure cleaning excludes API/performance-source sessions containing Perch/Perks, ForceDecks, VALD/Vault/Nordbord/GroinBar, SmartSpeed, or Catapult, including `Perch - Weight Room`; plain human-entered `Weight Room` remains eligible.
- Exposure feature requirements sprints are available through `risk-engine --exposure-feature-requirements-sprint --exposure-events <exposure_events.csv> --exposure-participations <exposure_participations.csv> --exposure-audit <exposure_cleaning_audit.json> --output-dir outputs --experiment-id <id>`. The sprint writes `exposure_category_summary.csv`, `exposure_duration_summary.csv`, `exposure_feature_requirements.csv`, `exposure_feature_requirements.json`, and `exposure_feature_requirements_report.md`; it should run after exposure cleaning and before any time-safe exposure/load model feature sprint.
- Exposure load feature sprints are available through `risk-engine --from-live-sources --paths-config config/paths.local.yaml --exposure-load-feature-sprint --exposure-participations <exposure_participations.csv> --output-dir outputs --experiment-id <id> --model-variant l2 --graph-window-size 4`. The sprint writes `exposure_load_features.csv`, `exposure_load_model_comparison.csv`, `exposure_load_model_comparison.json`, and `exposure_load_model_comparison_report.md`; it attaches only participation context before each graph snapshot and compares `graph_plus_coverage_source` against `graph_plus_coverage_exposure_load`.
- Exposure-load season-forward validation sprints are available through `risk-engine --exposure-load-season-forward-validation --exposure-participations <exposure_participations.csv>`. They write `exposure_load_features.csv`, `exposure_load_season_forward_validation.csv`, `exposure_load_season_forward_validation.json`, and `exposure_load_season_forward_validation_report.md`; the sprint trains on earlier complete athlete-season trajectories, evaluates later seasons, compares `graph_plus_coverage_source` against `graph_plus_coverage_exposure_load`, and scores fixed alert channels with the exposure-load feature set.
- Exposure-load forward diagnostic sprints are available through `risk-engine --exposure-load-forward-diagnostic-sprint --season-forward-validation-path <exposure_load_season_forward_validation.csv> --output-dir outputs --experiment-id <id>`. They write `exposure_load_season_forward_validation.csv`, `exposure_load_calibration_diagnostics.csv`, `exposure_load_forward_diagnostic_cases.csv`, `exposure_load_forward_diagnostic.json`, and `exposure_load_forward_diagnostic_report.md`; the sprint diagnoses season/horizon slices where exposure-load improves ranking or triage while damaging calibration or widening the mean-predicted-risk versus observed-rate gap.
- Exposure-load failure-mode sprints are available through `risk-engine --exposure-load-failure-mode-sprint --exposure-load-features <exposure_load_features.csv> --exposure-load-diagnostics <exposure_load_calibration_diagnostics.csv> --output-dir outputs --experiment-id <id>`. They write `exposure_load_failure_mode_features.csv`, `exposure_load_failure_mode_domains.csv`, `exposure_load_failure_modes.json`, and `exposure_load_failure_mode_report.md`; the sprint compares failed forward seasons against calibration-supported comparator seasons by time-safe exposure feature distribution and domain.
- Exposure-load guardrail policy sprints are available through `risk-engine --exposure-load-guardrail-policy-sprint --exposure-load-failure-modes <exposure_load_failure_modes.json> --exposure-load-diagnostics <exposure_load_calibration_diagnostics.csv> --output-dir outputs --experiment-id <id>`. They write `exposure_load_guardrail_policy.csv`, `exposure_load_guardrail_policy.json`, and `exposure_load_guardrail_policy_report.md`; the sprint converts exposure-load failure diagnostics into research operating guardrails before probability-facing, minute-load, pilot, or dashboard escalation.
- Exposure-load shift context sprints are available through `risk-engine --exposure-load-shift-context-sprint --exposure-events <exposure_events.csv> --exposure-participations <exposure_participations.csv> --exposure-load-features <exposure_load_features.csv> --exposure-load-diagnostics <exposure_load_calibration_diagnostics.csv> --exposure-load-failure-modes <exposure_load_failure_modes.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_shift_context.csv`, `exposure_load_shift_context_drivers.csv`, `exposure_load_shift_context_cases.csv`, `exposure_load_shift_context.json`, and `exposure_load_shift_context_report.md`; the sprint joins shifted exposure-load failure domains back to schedule, roster, availability, and managed-risk context before probability-facing, minute-load, pilot, or dashboard escalation.
- Exposure-load schedule/roster sprints are available through `risk-engine --exposure-load-schedule-roster-sprint --exposure-events <exposure_events.csv> --exposure-participations <exposure_participations.csv> --exposure-load-shift-context <exposure_load_shift_context.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_schedule_roster_context.csv`, `exposure_load_schedule_roster_drivers.csv`, `exposure_load_schedule_roster_context.json`, and `exposure_load_schedule_roster_report.md`; the sprint tests whether the failed exposure-load season differs in schedule density, event category mix, active roster size, or participation density.
- Exposure-load availability capture sprints are available through `risk-engine --exposure-load-availability-capture-sprint --exposure-participations <exposure_participations.csv> --exposure-load-shift-context <exposure_load_shift_context.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_availability_capture.csv`, `exposure_load_availability_capture_drivers.csv`, `exposure_load_availability_capture.json`, and `exposure_load_availability_capture_report.md`; the sprint audits modified/no-participation flagging, issue linkage, and duration/RPE/workload capture before treating exposure-load probability as calibrated.
- Exposure-load context decision sprints are available through `risk-engine --exposure-load-context-decision-sprint --exposure-load-shift-context <exposure_load_shift_context.json> --exposure-load-schedule-roster <exposure_load_schedule_roster_context.json> --exposure-load-availability-capture <exposure_load_availability_capture.json> --exposure-load-guardrail-policy <exposure_load_guardrail_policy.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_context_decision.csv`, `exposure_load_context_decision.json`, and `exposure_load_context_decision_report.md`; the sprint converts schedule/roster and availability-capture diagnostics into operating decisions for probability calibration, minute-load expansion, shadow ranking, and model expansion.
- Exposure-load source context classification sprints are available through `risk-engine --exposure-load-source-context-classification-sprint --exposure-events <exposure_events.csv> --exposure-participations <exposure_participations.csv> --exposure-load-shift-context <exposure_load_shift_context.json> --exposure-load-schedule-roster <exposure_load_schedule_roster_context.json> --exposure-load-availability-capture <exposure_load_availability_capture.json> --exposure-load-context-decision <exposure_load_context_decision.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_source_context_classification.csv`, `exposure_load_source_context_evidence.csv`, `exposure_load_source_context_classification.json`, and `exposure_load_source_context_classification_report.md`; the sprint classifies the failed season as true managed-risk context, schedule/roster shift, exposure-capture/documentation shift, or mixed before any model expansion.
- Exposure-load source resolution sprints are available through `risk-engine --exposure-load-source-resolution-sprint --exposure-load-source-context-classification <exposure_load_source_context_classification.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_source_resolution.csv`, `exposure_load_source_resolution_actions.csv`, `exposure_load_source_resolution_policy.json`, and `exposure_load_source_resolution_report.md`; the sprint converts source-context classification into season eligibility and operating policy for probability calibration, shadow ranking, model expansion, and minute-load expansion while preserving complete athlete-season trajectories.
- Exposure-load source-eligible calibration sprints are available through `risk-engine --exposure-load-source-eligible-calibration-sprint --season-forward-validation-path <exposure_load_season_forward_validation.csv> --exposure-load-source-resolution-policy <exposure_load_source_resolution_policy.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_source_eligible_calibration.csv`, `exposure_load_source_eligible_calibration_diagnostics.csv`, `exposure_load_source_eligible_calibration.json`, and `exposure_load_source_eligible_calibration_report.md`; the sprint applies the source-resolution eligibility policy to season-forward validation rows and compares all-season versus source-eligible exposure-load calibration behavior.
- Exposure-load source-eligible policy sprints are available through `risk-engine --exposure-load-source-eligible-policy-sprint --season-forward-validation-path <exposure_load_season_forward_validation.csv> --exposure-load-source-eligible-calibration <exposure_load_source_eligible_calibration.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_source_eligible_policy.csv`, `exposure_load_source_eligible_thresholds.csv`, `exposure_load_source_eligible_policy.json`, and `exposure_load_source_eligible_policy_report.md`; the sprint freezes source-eligible shadow-mode threshold candidates after calibration eligibility is satisfied while keeping product readiness blocked.
- Exposure-load source-eligible shadow-monitoring sprints are available through `risk-engine --exposure-load-source-eligible-shadow-monitoring-sprint --season-forward-validation-path <exposure_load_season_forward_validation.csv> --exposure-load-source-eligible-policy <exposure_load_source_eligible_policy.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_source_eligible_shadow_monitoring.csv`, `exposure_load_source_eligible_shadow_monitoring_seasons.csv`, `exposure_load_source_eligible_shadow_monitoring.json`, and `exposure_load_source_eligible_shadow_monitoring_report.md`; the sprint reviews frozen source-eligible threshold channels against complete athlete-season validation rows and identifies which channels can proceed to prospective shadow review while keeping pilot/dashboard and autonomous intervention blocked.
- Exposure-load shadow launch chain sprints are available as three downstream artifact passes: `risk-engine --exposure-load-shadow-channel-lock-sprint --exposure-load-source-eligible-shadow-monitoring <exposure_load_source_eligible_shadow_monitoring.json> --output-dir outputs --experiment-id <id>`, `risk-engine --exposure-load-shadow-review-protocol-sprint --exposure-load-shadow-channel-lock <exposure_load_shadow_channel_lock.json> --output-dir outputs --experiment-id <id>`, and `risk-engine --exposure-load-shadow-readiness-register-sprint --exposure-load-shadow-channel-lock <exposure_load_shadow_channel_lock.json> --exposure-load-shadow-review-protocol <exposure_load_shadow_review_protocol.json> --output-dir outputs --experiment-id <id>`. They write channel-lock, protocol, and readiness-register CSV/JSON/report artifacts; the chain launches prospective research shadow-monitoring preparation only and keeps pilot/dashboard, probability-facing deployment, minute-load expansion, and autonomous intervention blocked.
- Exposure-load historical shadow replay sprints are available through `risk-engine --exposure-load-shadow-replay-sprint --season-forward-validation-path <exposure_load_season_forward_validation.csv> --exposure-load-shadow-channel-lock <exposure_load_shadow_channel_lock.json> --exposure-load-shadow-review-protocol <exposure_load_shadow_review_protocol.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_shadow_replay_log.csv`, `exposure_load_shadow_review_packets.csv`, `exposure_load_shadow_stop_rules.csv`, `exposure_load_shadow_replay.json`, and `exposure_load_shadow_replay_report.md`; the sprint builds historical replay logs, source-eligibility and burden stop-rule tracking, and adjudication-ready review packets for locked channels without claiming prospective performance.
- Exposure-load shadow adjudication package sprints are available through `risk-engine --exposure-load-shadow-adjudication-sprint --exposure-load-shadow-replay <exposure_load_shadow_replay.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_shadow_adjudication_schema.csv`, `exposure_load_shadow_adjudication_template.csv`, `exposure_load_shadow_adjudication_completion.csv`, `exposure_load_shadow_adjudication.json`, and `exposure_load_shadow_adjudication_report.md`; the sprint converts replay review packets into prospective reviewer collection rows with required usefulness, outcome, source-context, action, and notes fields while keeping product readiness blocked.
- Exposure-load shadow adjudication summary sprints are available through `risk-engine --exposure-load-shadow-adjudication-summary-sprint --exposure-load-shadow-adjudication <completed_adjudication.csv> --output-dir outputs --experiment-id <id>`. They write `exposure_load_shadow_adjudication_validation.csv`, `exposure_load_shadow_adjudication_channel_summary.csv`, `exposure_load_shadow_adjudication_summary.json`, and `exposure_load_shadow_adjudication_summary_report.md`; the sprint validates reviewer completion, summarizes useful/source-trustworthy/actionable packets by channel, and remains blocked until reviewer fields are complete.
- Exposure-load shadow adjudication decision sprints are available through `risk-engine --exposure-load-shadow-adjudication-decision-sprint --exposure-load-shadow-adjudication-summary <exposure_load_shadow_adjudication_summary.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_shadow_adjudication_channel_decisions.csv`, `exposure_load_shadow_adjudication_decision.json`, and `exposure_load_shadow_adjudication_decision_report.md`; the sprint converts completed adjudication evidence into channel-level continue/pause/revise decisions while keeping product readiness blocked.
- Exposure-load shadow monitoring plan sprints are available through `risk-engine --exposure-load-shadow-monitoring-plan-sprint --exposure-load-shadow-adjudication-decision <exposure_load_shadow_adjudication_decision.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_shadow_monitoring_plan.csv`, `exposure_load_shadow_monitoring_paused_channels.csv`, `exposure_load_shadow_monitoring_evidence_gates.csv`, `exposure_load_shadow_monitoring_plan.json`, and `exposure_load_shadow_monitoring_plan_report.md`; the sprint turns retained-channel decisions into a prospective shadow collection plan and explicit calibration/pilot gates.
- Exposure-load shadow collection template sprints are available through `risk-engine --exposure-load-shadow-collection-template-sprint --exposure-load-shadow-monitoring-plan <exposure_load_shadow_monitoring_plan.json> --output-dir outputs --experiment-id <id>`. They write `exposure_load_shadow_collection_schema.csv`, `exposure_load_shadow_collection_template.csv`, `exposure_load_shadow_collection_completion.csv`, `exposure_load_shadow_collection_template.json`, and `exposure_load_shadow_collection_template_report.md`; the sprint creates pending prospective collection rows for retained channels only.
- As of 2026-04-27, `run_research_experiment(...)` trains a discrete-time logistic baseline at the 7, 14, and 30 day horizons over 13 graph snapshot-time features: `time_index`, `node_count`, `edge_count`, `mean_abs_correlation`, `edge_density`, `delta_edge_count`, `delta_mean_abs_correlation`, `delta_edge_density`, `graph_instability`, `z_mean_abs_correlation`, `z_edge_density`, `z_edge_count`, and `z_graph_instability`.
- The temporal delta features (`delta_*`) are computed per athlete-season in chronological order and are zero at each athlete's first snapshot; they capture change from one snapshot to the next. `edge_density` normalizes edge count by the maximum possible edges for the observed node count. `graph_instability` is the rolling population standard deviation of `mean_abs_correlation` over the most recent three snapshots, zero when fewer than two snapshots are available.
- The intra-individual z-score features compare each athlete-season snapshot against that athlete-season's own strictly prior rolling baseline using the graph `window_size`. They require at least two prior snapshots, use population standard deviation, fall back to `0.0` when the prior standard deviation is zero, and are clipped to `[-10.0, 10.0]`.
- The baseline writes `model_summary.json` with the event policy, feature columns, deterministic athlete-level 20% holdout split, per-horizon model kind, positive rates, and holdout Brier scores. If a training horizon has only one class, it records a prevalence fallback instead of fitting an unstable classifier.
- `model_evaluation.json` compares each horizon's holdout predictions against the training-prevalence baseline and reports Brier score, Brier skill score, AUROC, average precision, and top-decile lift when the holdout labels support those metrics.
- `feature_attribution.json` and `feature_ablation_report.md` compare the current 13-feature model against `original_9` and `z_score_only` feature sets using the same deterministic athlete-level holdout split. The artifact includes per-horizon holdout metrics and standardized logistic coefficients for feature attribution.
- Window sensitivity runs are available through `risk-engine --window-sensitivity-sizes <sizes...>` and write `window_sensitivity.json` plus `window_sensitivity_report.md`. The runner reuses one canonical input set, loops graph `window_size` values, and compares the same holdout metrics across windows.
- Model robustness sprints are available through `risk-engine --model-robustness-sprint --stability-splits <n>` and write `model_robustness.json` plus `model_robustness_report.md`. The sprint compares `baseline`, `l2`, `l1`, and `elasticnet` logistic variants across deterministic rotating athlete-level holdout splits, then reports decision-mode winners for ranking, calibration, and triage metrics.
- Combined window/model robustness runs are available by pairing `--model-robustness-sprint` with `--window-sensitivity-sizes <sizes...>`. They write `window_model_robustness.json` plus `window_model_robustness_report.md`, comparing model variants across each requested graph window with the same rotating athlete-level split policy.
- Outcome-policy model comparison runs are available through `risk-engine --outcome-policy-model-comparison` and write `context_policy_model_comparison.csv`, `context_policy_model_comparison.json`, and `context_policy_model_comparison_report.md`. The runner relabels canonical athlete-seasons against selected detailed injury policies, retrains the same graph model for each target, and compares model metrics plus alert episode quality at top-5% and top-10%.
- Policy decision sprint runs are available through `risk-engine --policy-decision-sprint --policy-window-sizes 2 4 7` and write `two_channel_alert_policy.json`, `two_channel_alert_policy_report.md`, `policy_window_sensitivity.csv`, `policy_window_sensitivity.json`, `policy_window_sensitivity_report.md`, `operational_policy_package.json`, and `operational_policy_package_report.md`. The sprint converts policy-comparison evidence into a two-channel research policy, window-sensitivity recommendations, and a shadow-mode operating package.
- Shadow-mode stability runs are available through `risk-engine --shadow-mode-stability` and write `shadow_mode_stability.csv`, `shadow_mode_stability.json`, and `shadow_mode_stability_report.md`. The runner evaluates the fixed policy package by season using season-local percentile thresholds, then marks channel stability from capture-rate variability and alert burden.
- Season drift diagnostic runs are available through `risk-engine --season-drift-diagnostic` and write `season_drift_diagnostics.csv`, `season_drift_diagnostics.json`, and `season_drift_diagnostic_report.md`. The runner reuses the fixed shadow-mode season-local channel rows and joins them to measurement coverage, source mix, detailed injury mix, and simple drift flags.
- Coverage-stratified evaluation runs are available through `risk-engine --coverage-stratified-evaluation` and write `coverage_tiers.csv`, `coverage_stratified_evaluation.csv`, `coverage_stratified_evaluation.json`, and `coverage_stratified_evaluation_report.md`. The runner assigns population-wide coverage tiers (low/medium/high tertile of measurement days per athlete-season), evaluates each fixed shadow-mode channel's capture rate and burden by tier using a population-wide risk threshold, and reports a `coverage_flag` (`coverage_confounded` / `coverage_independent` / `mixed`) indicating whether coverage tier is a major driver of policy performance.
- Coverage-normalized policy sprints are available through `risk-engine --coverage-normalized-policy-sprint` and write `coverage_normalized_policy.csv`, `coverage_normalized_policy.json`, and `coverage_normalized_policy_report.md`. The sprint keeps the fixed shadow-mode channel package, applies coverage eligibility scopes to complete athlete-season trajectories (`all`, `medium_high`, `high_only`), rebuilds season-local alert episodes within each eligible cohort, and reports whether any channel remains stable after coverage control. This is the current Peterson-aligned guardrail before dashboard work because it preserves longitudinal athlete-season trajectories rather than treating daily rows as independent examples.
- Coverage/source-aware model sprints are available through `risk-engine --coverage-source-aware-model-sprint` and write `coverage_source_features.csv`, `coverage_source_model_comparison.csv`, `coverage_source_model_comparison.json`, and `coverage_source_model_comparison_report.md`. The sprint compares `graph_trajectory` against `graph_plus_coverage_source`, adding only time-safe coverage/source covariates available on or before each graph snapshot while keeping the 13 dynamic graph trajectory features as the core signal.
- Coverage-adjusted threshold sprints are available through `risk-engine --coverage-adjusted-threshold-sprint` and write `coverage_adjusted_threshold_policy.csv`, `coverage_adjusted_threshold_policy.json`, and `coverage_adjusted_threshold_report.md`. The sprint compares season-local thresholds, coverage-tier-local thresholds, and burden-capped thresholds after complete athlete-season trajectories have been scored.
- Season-forward validation sprints are available through `risk-engine --season-forward-validation` and write `season_forward_validation.csv`, `season_forward_validation.json`, and `season_forward_validation_report.md`. The sprint trains on earlier seasons, evaluates later seasons, compares `graph_trajectory` against `graph_plus_coverage_source`, and checks fixed alert channels under season-local and burden-capped threshold policies.
- Forward case review sprints are available through `risk-engine --forward-case-review-sprint` and write `forward_case_review_cases.csv`, `forward_case_review.json`, and `forward_case_review_report.md`. The sprint targets forward-surviving seasons and channels, then classifies true positives, false positives, missed injuries, and high intra-individual deviation episodes into review diagnostics.
- Case diagnostic requirements sprints are available through `risk-engine --case-diagnostic-requirements-sprint` and write `forward_case_review_cases.csv`, `case_diagnostic_requirements.csv`, `case_diagnostic_requirements.json`, and `case_diagnostic_requirements_report.md`. The sprint translates forward case-review diagnostics into prioritized production-readiness data domains and modeling actions before any pilot escalation.
- Injury-history feature sprints are available through `risk-engine --injury-history-feature-sprint` and write `injury_history_features.csv`, `injury_history_model_comparison.csv`, `injury_history_model_comparison.json`, and `injury_history_model_comparison_report.md`. The sprint derives time-safe prior-injury, baseline/frailty, and mechanism-context features from `injury_events_detailed.csv`, then compares `graph_plus_coverage_source` against `graph_plus_coverage_injury_history`.
- Injury-history season-forward validation sprints are available through `risk-engine --injury-history-season-forward-validation` and write `injury_history_features.csv`, `injury_history_season_forward_validation.csv`, `injury_history_season_forward_validation.json`, and `injury_history_season_forward_validation_report.md`. The sprint trains on earlier seasons, evaluates later seasons, compares `graph_plus_coverage_source` against `graph_plus_coverage_injury_history`, and scores fixed alert channels with the injury-history feature set.
- Injury-history forward diagnostic sprints are available through `risk-engine --injury-history-forward-diagnostic-sprint` and write `injury_history_features.csv`, `injury_history_season_forward_validation.csv`, `injury_history_calibration_diagnostics.csv`, `injury_history_forward_diagnostic_cases.csv`, `injury_history_forward_diagnostic.json`, and `injury_history_forward_diagnostic_report.md`. The sprint diagnoses where time-safe injury-history features improve forward ranking/triage while damaging calibration, using season/horizon diagnostic slices rather than treating daily rows as independent examples.
- Main single-experiment runs accept `--model-variant baseline|l2|l1|elasticnet`, allowing a regularized candidate to be run through the normal artifact path without using the robustness sweep.
- Enriched graph features (`enriched_graph_features_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.730 (+0.008), 14d AUROC 0.735 (+0.007), 30d AUROC 0.735 (+0.007); Brier skill 30d improved from 0.0142 to 0.0168 (+18%).
- Intra-individual deviation features (`intra_individual_deviation_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.723, Brier skill 0.0020, top-decile lift 3.76; 14d AUROC 0.731, Brier skill 0.0057, top-decile lift 3.96; 30d AUROC 0.736, Brier skill 0.0171, top-decile lift 4.48. Versus `enriched_graph_features_v1`, the 30d AUROC, 7d/30d Brier skill, and all top-decile lifts improved, while 7d/14d AUROC declined slightly.
- Feature attribution/ablation (`feature_attribution_ablation_v1` run, 349 athletes, 70 holdout): `full_13` matched the `intra_individual_deviation_v1` metrics. `original_9` remained stronger on 7d/14d AUROC (0.730/0.735) but had lower 7d/30d top-decile lift (3.68/4.34) than `full_13` (3.76/4.48). `z_score_only` had weak AUROC (7d 0.566, 14d 0.553, 30d 0.491) but strong 7d top-decile lift (4.14), suggesting the z-score features are most useful as ranking modifiers inside the combined model rather than as a standalone risk model.
- Window sensitivity (`window_sensitivity_v1` run, windows 2/3/4/5/7, 349 athletes, 70 holdout): window 4 was best for AUROC at all horizons (7d 0.723, 14d 0.731, 30d 0.736); window 7 was best for Brier skill and Brier score at all horizons (7d Brier skill 0.0062, 14d 0.0119, 30d 0.0301); window 2 was best for top-decile lift at all horizons (7d 5.19, 14d 5.04, 30d 5.13). This suggests the default window 4 remains the best ranking/AUROC baseline, while longer windows improve probability sharpness and shorter windows concentrate positives in the highest-risk decile.
- Model robustness sprint (`model_robustness_sprint_v1` run, graph window 4, 5 rotating splits, 349 athletes): all regularized variants improved average AUROC, Brier skill, Brier score, and top-decile lift relative to baseline at all horizons. L2 was the strongest all-around choice, winning calibration at 7/14/30d, triage at 7/14d, and ranking at 14d. Elastic net narrowly won 30d ranking and 30d triage. L1 narrowly won 7d ranking. Regularized variants standardize features internally before fitting and convert coefficients back to raw feature units for attribution.
- Window/model robustness (`window_model_robustness_v1` run, windows 2/4/7, 5 rotating splits, 349 athletes): no single window/variant dominates all operating goals. Window 7 + L2 won calibration at 7d/14d, window 4 + L2 won 30d calibration, window 2 regularized variants won triage lift at all horizons, and ranking split by horizon: window 2 baseline at 7d AUROC 0.731, window 4 L2 at 14d AUROC 0.729, and window 7 L1 at 30d AUROC 0.729. This supports using L2 as the calibration-oriented production candidate while keeping window 2 as a high-alert triage setting and window 7 under review for 30d ranking.

## Latest Completed Step

**Exposure-load shadow collection template sprint** - implemented and verified on 2026-05-13.

**What changed:** Added `exposure_load_shadow_collection.py`, `run_exposure_load_shadow_collection_template_sprint_experiment(...)`, and the `--exposure-load-shadow-collection-template-sprint` CLI mode. The sprint consumes the retained-channel monitoring plan and writes a prospective collection schema, template, and completion checks.

**Verification:** New TDD tests first failed because the collection module, runner, and CLI dispatch did not exist. After implementation, focused collection-template tests passed. The live command `risk-engine --exposure-load-shadow-collection-template-sprint --exposure-load-shadow-monitoring-plan outputs/experiments/exposure_load_shadow_monitoring_plan_v1/exposure_load_shadow_monitoring_plan.json --output-dir outputs --experiment-id exposure_load_shadow_collection_template_v1` completed.

**Live results (`exposure_load_shadow_collection_template_v1`):**
- Overall recommendation: `collect_retained_channel_shadow_packets`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Retained channels: `broad_30d`, `severity_14d`.
- Paused/revision channels: `severity_7d`.
- Collection rows: 8.
- Pending required-field rows: 8.

**Interpretation:** The next progressive step is now operationalized as a fillable collection template. Real progress toward calibration readiness requires completing these new retained-channel packet rows with prospective source-eligible evidence; no probability-facing or dashboard work should resume before those rows are complete and summarized.

## Previous Completed Step

**Exposure-load shadow monitoring plan sprint** - implemented and verified on 2026-05-13.

**What changed:** Added `exposure_load_shadow_monitoring.py`, `run_exposure_load_shadow_monitoring_plan_sprint_experiment(...)`, and the `--exposure-load-shadow-monitoring-plan-sprint` CLI mode. The sprint consumes the completed adjudication decision JSON and writes retained-channel monitoring, paused-channel revision, and evidence-gate artifacts.

**Verification:** New TDD tests first failed because the monitoring module, runner, and CLI dispatch did not exist. After implementation, focused monitoring-plan tests passed. The live command `risk-engine --exposure-load-shadow-monitoring-plan-sprint --exposure-load-shadow-adjudication-decision outputs/experiments/exposure_load_shadow_adjudication_decision_v1/exposure_load_shadow_adjudication_decision.json --output-dir outputs --experiment-id exposure_load_shadow_monitoring_plan_v1` completed.

**Live results (`exposure_load_shadow_monitoring_plan_v1`):**
- Overall recommendation: `launch_retained_channel_shadow_monitoring`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Retained channels: `broad_30d`, `severity_14d`.
- Paused/revision channel: `severity_7d`.
- Evidence gate: collect at least 4 new complete source-eligible review packets for retained channels before revisiting calibration.
- Probability calibration and pilot/dashboard readiness remain blocked.

**Interpretation:** This is the next major milestone package after adjudication: retained-channel prospective shadow monitoring is now specified. The next real evidence step is collecting new review packets for `broad_30d` and `severity_14d`; do not resume probability-facing or dashboard work until those gates are satisfied.

## Previous Completed Step

**Exposure-load shadow adjudication decision sprint** - implemented and verified on 2026-05-13.

**What changed:** Added `build_exposure_load_shadow_adjudication_decision_package(...)`, `write_exposure_load_shadow_adjudication_decision_report(...)`, `run_exposure_load_shadow_adjudication_decision_sprint_experiment(...)`, and the `--exposure-load-shadow-adjudication-decision-sprint` CLI mode. The sprint consumes the completed adjudication summary JSON and writes channel-level shadow-monitoring decisions.

**Verification:** New TDD tests first failed because the decision runner and CLI dispatch did not exist. After implementation, focused decision tests passed. The live command `risk-engine --exposure-load-shadow-adjudication-decision-sprint --exposure-load-shadow-adjudication-summary outputs/experiments/exposure_load_shadow_adjudication_summary_csv_review_v1/exposure_load_shadow_adjudication_summary.json --output-dir outputs --experiment-id exposure_load_shadow_adjudication_decision_v1` completed and wrote the decision artifacts.

**Live results (`exposure_load_shadow_adjudication_decision_v1`):**
- Overall recommendation: `continue_shadow_monitoring_with_channel_revisions`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Continue shadow monitoring: `broad_30d`, `severity_14d`.
- Pause or revise before more collection: `severity_7d`.

**Interpretation:** The project can now move forward in narrowed shadow-monitoring mode. The current evidence supports continued collection for `broad_30d` and `severity_14d`, but not `severity_7d`; probability calibration, pilot/dashboard readiness, and autonomous intervention remain blocked.

## Previous Completed Step

**Exposure-load shadow adjudication summary readiness sprint** - implemented and verified on 2026-05-13.

**What changed:** Added `build_exposure_load_shadow_adjudication_summary(...)`, `write_exposure_load_shadow_adjudication_summary_report(...)`, `run_exposure_load_shadow_adjudication_summary_sprint_experiment(...)`, and the `--exposure-load-shadow-adjudication-summary-sprint` CLI mode. The sprint consumes a filled adjudication CSV, validates missing/invalid reviewer fields, summarizes useful/source-trustworthy/actionable packets by channel, and keeps product readiness blocked.

**Verification:** TDD tests first failed because the summary runner and CLI dispatch did not exist. After implementation and a parser fix for CSV boolean `false` values, focused adjudication summary tests passed. The live pending-template command `risk-engine --exposure-load-shadow-adjudication-summary-sprint --exposure-load-shadow-adjudication outputs/experiments/exposure_load_shadow_adjudication_v1/exposure_load_shadow_adjudication_template.csv --output-dir outputs --experiment-id exposure_load_shadow_adjudication_summary_pending_v1` completed.

**Live pending-template results (`exposure_load_shadow_adjudication_summary_pending_v1`):**
- Overall recommendation: `complete_adjudication_required_before_operational_summary`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Total rows: 12.
- Complete valid rows: 0.
- Pending or invalid rows: 12.
- Useful, source-trustworthy, actionable rows: 0.

**Interpretation:** This is safe supporting infrastructure while the adjudication packet is being completed. It does not create new evidence from the current data. Once the reviewer fields are filled, rerun the summary sprint against the completed CSV to decide whether shadow monitoring should continue or the alert package should be revised.

## Previous Completed Step

**Exposure-load shadow adjudication package sprint** - implemented and verified on 2026-05-13.

**What changed:** Added `exposure_load_shadow_adjudication.py`, `run_exposure_load_shadow_adjudication_sprint_experiment(...)`, and the `--exposure-load-shadow-adjudication-sprint` CLI mode. The sprint consumes `exposure_load_shadow_replay.json` and writes a prospective adjudication schema, collection template, and required-field completion checks for the historical review packets.

**Verification:** New TDD tests first failed because the adjudication module, experiment runner, and CLI dispatch did not exist. After implementation, focused adjudication tests passed. The live command `risk-engine --exposure-load-shadow-adjudication-sprint --exposure-load-shadow-replay outputs/experiments/exposure_load_shadow_replay_v1/exposure_load_shadow_replay.json --output-dir outputs --experiment-id exposure_load_shadow_adjudication_v1` completed and wrote the adjudication artifacts.

**Live results (`exposure_load_shadow_adjudication_v1`):**
- Overall recommendation: `adjudication_template_ready_for_prospective_collection`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Review packet rows: 12.
- Schema fields: 8.
- Required adjudication fields: 6 (`reviewer_id`, `review_date`, `alert_usefulness`, `outcome_confirmed`, `source_context_ok`, and `action_taken`).
- Pending required-field rows: 12.

**Interpretation:** This is the logical stopping point for current-data preparation. The repo can now generate replay logs, review packets, stop-rule checks, and an adjudication template, but the next real evidence step requires prospective reviewer/adjudication values. Pilot/dashboard, probability-facing deployment, calibration updates, and autonomous intervention remain blocked.

## Previous Completed Step

**Exposure-load historical shadow replay sprint** - implemented and verified on 2026-05-13.

**What changed:** Added `exposure_load_shadow_replay.py`, `run_exposure_load_shadow_replay_sprint_experiment(...)`, and the `--exposure-load-shadow-replay-sprint` CLI mode. The sprint consumes the completed exposure-load season-forward validation CSV, the locked shadow-channel JSON, and the review-protocol JSON, then writes historical replay logs, adjudication-ready review packets, and stop-rule tracking for the locked channels.

**Verification:** New TDD tests first failed because the replay module, experiment runner, and CLI dispatch did not exist. After implementation, focused replay tests passed. `python -m compileall -q src/risk_stratification_engine/exposure_load_shadow_replay.py src/risk_stratification_engine/experiments.py src/risk_stratification_engine/cli.py tests/test_exposure_load_shadow_replay.py tests/test_experiments.py tests/test_cli.py` passed, and `python -m pytest` collected and passed 275 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-load-shadow-replay-sprint --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv --exposure-load-shadow-channel-lock outputs/experiments/exposure_load_shadow_channel_lock_v1/exposure_load_shadow_channel_lock.json --exposure-load-shadow-review-protocol outputs/experiments/exposure_load_shadow_review_protocol_v1/exposure_load_shadow_review_protocol.json --output-dir outputs --experiment-id exposure_load_shadow_replay_v1` completed and wrote the replay artifacts.

**Live results (`exposure_load_shadow_replay_v1`):**
- Overall recommendation: `historical_shadow_replay_ready_for_prospective_collection`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Replay rows: 15.
- Review packets: 12.
- Source-ineligible stop rows: 3.
- Burden stop rows: 0.
- Review packet counts: `broad_30d` 4, `severity_14d` 4, and `severity_7d` 4.

**Interpretation:** The current historical data can now generate replay logs and adjudication packets for the locked source-eligible channels. This is the strongest current-data endpoint before prospective collection. It does not establish prospective performance, real-time usefulness, probability calibration readiness, or pilot/dashboard readiness.

## Previous Completed Step

**Exposure-load shadow launch chain** - implemented and verified on 2026-05-13.

**What changed:** Added `exposure_load_shadow_launch.py`, three experiment runners (`run_exposure_load_shadow_channel_lock_sprint_experiment(...)`, `run_exposure_load_shadow_review_protocol_sprint_experiment(...)`, and `run_exposure_load_shadow_readiness_register_sprint_experiment(...)`), and three CLI modes: `--exposure-load-shadow-channel-lock-sprint`, `--exposure-load-shadow-review-protocol-sprint`, and `--exposure-load-shadow-readiness-register-sprint`. The chain consumes the source-eligible shadow-monitoring JSON, locks only channels that passed the burden guardrail, creates a prospective research review protocol, and writes a readiness register without product escalation.

**Verification:** New TDD tests first failed because the launch module, experiment runners, and CLI dispatch did not exist. After implementation, focused launch-chain tests passed. `python -m compileall -q src/risk_stratification_engine/exposure_load_shadow_launch.py src/risk_stratification_engine/experiments.py src/risk_stratification_engine/cli.py tests/test_exposure_load_shadow_launch.py tests/test_experiments.py tests/test_cli.py` passed, and `python -m pytest` collected and passed 272 tests with one existing sklearn convergence warning. Live commands completed for `exposure_load_shadow_channel_lock_v1`, `exposure_load_shadow_review_protocol_v1`, and `exposure_load_shadow_readiness_register_v1`.

**Live results (`exposure_load_shadow_*_v1`):**
- Channel lock recommendation: `lock_source_eligible_burden_capped_channels_for_shadow_review`.
- Locked channels: `broad_30d`, `severity_14d`, and `severity_7d`, all using `burden_capped_percentile`.
- Held channel: `subtype_lower_extremity_soft_tissue_30d`, with `shadow_burden_guardrail_review_needed`.
- Review protocol recommendation: `launch_research_shadow_review_with_locked_channels`.
- Readiness register recommendation: `launch_research_shadow_monitoring_without_product_escalation`.
- Production readiness remains `not_ready_for_probability_or_pilot`.

**Interpretation:** The next operational posture is prospective source-eligible research shadow monitoring for the three burden-capped channels only. Outcome collection must precede calibration updates, pilot escalation, dashboard work, probability-facing deployment, minute-load expansion, or autonomous intervention.

## Previous Completed Step

**Exposure-load source-eligible shadow monitoring sprint** - implemented and verified on 2026-05-13.

**What changed:** Added `exposure_load_source_eligible_shadow_monitoring.py`, `run_exposure_load_source_eligible_shadow_monitoring_sprint_experiment(...)`, and the `--exposure-load-source-eligible-shadow-monitoring-sprint` CLI mode. The sprint consumes the completed exposure-load season-forward validation CSV and the frozen source-eligible policy JSON, excludes source-ineligible test seasons, and reviews source-eligible channels as a prospective-style shadow monitoring record.

**Verification:** New TDD tests first failed because the shadow-monitoring module, experiment runner, and CLI dispatch did not exist. After implementation, `python -m compileall -q src/risk_stratification_engine/exposure_load_source_eligible_shadow_monitoring.py src/risk_stratification_engine/experiments.py src/risk_stratification_engine/cli.py tests/test_exposure_load_source_eligible_shadow_monitoring.py tests/test_experiments.py tests/test_cli.py` passed, focused module/experiment/CLI tests passed, and `python -m pytest` collected and passed 269 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-load-source-eligible-shadow-monitoring-sprint --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv --exposure-load-source-eligible-policy outputs/experiments/exposure_load_source_eligible_policy_v1/exposure_load_source_eligible_policy.json --output-dir outputs --experiment-id exposure_load_source_eligible_shadow_monitoring_v1` completed and wrote the monitoring artifacts.

**Live results (`exposure_load_source_eligible_shadow_monitoring_v1`):**
- Overall recommendation: `proceed_with_prospective_source_eligible_shadow_monitoring`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Excluded test season: 2024-2025.
- `broad_30d`: `ready_for_prospective_shadow_review`, 4 source-eligible seasons, mean capture 0.151, max burden 0.686, mean threshold drift 0.009.
- `severity_14d`: `ready_for_prospective_shadow_review`, mean capture 0.129, max burden 0.919, mean threshold drift 0.019.
- `severity_7d`: `ready_for_prospective_shadow_review`, mean capture 0.050, max burden 0.615, mean threshold drift 0.034.
- `subtype_lower_extremity_soft_tissue_30d`: `shadow_burden_guardrail_review_needed`, mean capture 0.186, max burden 2.488.

**Interpretation:** The three burden-capped channels can move into prospective source-eligible shadow review under the research-only boundary. The subtype review channel remains blocked by alert burden. This is still not production readiness; pilot, dashboard, probability-facing deployment, autonomous intervention, and minute-load expansion remain blocked.

## Previous Completed Step

**Exposure-load source-eligible policy sprint** - implemented and verified on 2026-05-10.

**What changed:** Added `exposure_load_source_eligible_policy.py`, `run_exposure_load_source_eligible_policy_sprint_experiment(...)`, and the `--exposure-load-source-eligible-policy-sprint` CLI mode. The sprint consumes the completed exposure-load season-forward validation CSV and the source-eligible calibration JSON, excludes source-ineligible test seasons, and freezes threshold candidates for research shadow-mode monitoring.

**Verification:** New TDD tests first failed because the source-eligible policy module, experiment runner, and CLI dispatch did not exist. After implementation, focused source-eligible-policy/experiment/CLI tests passed; adjacent source-eligible calibration, source-resolution, experiment, and CLI tests also passed. `python -m compileall -q src/risk_stratification_engine/exposure_load_source_eligible_policy.py src/risk_stratification_engine/experiments.py src/risk_stratification_engine/cli.py tests/test_exposure_load_source_eligible_policy.py tests/test_experiments.py tests/test_cli.py` passed, then `python -m pytest` collected and passed 266 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-load-source-eligible-policy-sprint --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv --exposure-load-source-eligible-calibration outputs/experiments/exposure_load_source_eligible_calibration_v1/exposure_load_source_eligible_calibration.json --output-dir outputs --experiment-id exposure_load_source_eligible_policy_v1` completed and wrote the policy artifacts.

**Live results (`exposure_load_source_eligible_policy_v1`):**
- Overall recommendation: `advance_source_eligible_shadow_mode_threshold_research`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Excluded test season: 2024-2025.
- Burden cap: 1.000 episodes per athlete-season.
- `broad_30d`: recommended `burden_capped_percentile`, mean capture 0.151, mean burden 0.354, mean threshold 0.044, seasons 2021-2022, 2022-2023, 2023-2024, and 2025-2026.
- `severity_14d`: recommended `burden_capped_percentile`, mean capture 0.129, mean burden 0.453, mean threshold 0.087.
- `severity_7d`: recommended `burden_capped_percentile`, mean capture 0.050, mean burden 0.335, mean threshold 0.077.
- `subtype_lower_extremity_soft_tissue_30d`: recommended `season_local_percentile`, mean capture 0.186, mean burden 0.774, mean threshold 0.100.

**Interpretation:** Source-eligible calibration now supports a frozen research threshold package for shadow-mode monitoring. This is model-readiness progress, not production readiness; pilot, dashboard, autonomous intervention, and probability-facing deployment remain blocked until the package survives prospective shadow-mode review.

## Previous Completed Step

**Exposure-load source-eligible calibration sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_source_eligible_calibration.py`, `run_exposure_load_source_eligible_calibration_sprint_experiment(...)`, and the `--exposure-load-source-eligible-calibration-sprint` CLI mode. The sprint consumes the completed exposure-load season-forward validation CSV and the source-resolution policy JSON, excludes source-ineligible test seasons, and compares calibration diagnostics before versus after applying eligibility.

**Verification:** New TDD tests first failed because the source-eligible calibration runner did not exist. After implementation, focused source-eligible-calibration/experiment/CLI tests passed, then `python -m pytest` collected and passed 263 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-load-source-eligible-calibration-sprint --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv --exposure-load-source-resolution-policy outputs/experiments/exposure_load_source_resolution_v1/exposure_load_source_resolution_policy.json --output-dir outputs --experiment-id exposure_load_source_eligible_calibration_v1` completed and wrote the calibration artifacts.

**Live results (`exposure_load_source_eligible_calibration_v1`):**
- Overall recommendation: `probability_research_can_resume_on_source_eligible_seasons`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Excluded test season: 2024-2025.
- All seasons: 15 diagnostic rows, 3 `ranking_triage_gain_calibration_loss` rows, 6 `calibration_supported` rows, mean Brier skill delta -0.106, and mean prediction-gap delta +0.010.
- Source-eligible seasons: 12 diagnostic rows, 0 `ranking_triage_gain_calibration_loss` rows, 6 `calibration_supported` rows, mean Brier skill delta +0.065, and mean prediction-gap delta -0.005.

**Interpretation:** This is the first evidence that the exposure-load calibration problem is largely source-eligibility driven rather than an unavoidable model ceiling. Probability research can resume on source-eligible seasons only, but pilot/dashboard use and general probability-facing claims remain blocked until source eligibility, calibration thresholds, and shadow-mode operating rules are formalized.

## Previous Completed Step

**Exposure-load source resolution sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_source_resolution.py`, `run_exposure_load_source_resolution_sprint_experiment(...)`, and the `--exposure-load-source-resolution-sprint` CLI mode. The sprint consumes `exposure_load_source_context_classification.json` and converts the failed-season source classification into an explicit eligibility and operating policy.

**Verification:** New TDD tests first failed because the source-resolution runner did not exist. After implementation, focused source-resolution/experiment/CLI tests passed, then `python -m pytest` collected and passed 260 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-load-source-resolution-sprint --exposure-load-source-context-classification outputs/experiments/exposure_load_source_context_classification_v1/exposure_load_source_context_classification.json --output-dir outputs --experiment-id exposure_load_source_resolution_v1` completed and wrote the policy artifacts.

**Live results (`exposure_load_source_resolution_v1`):**
- Overall recommendation: `exclude_failed_season_from_probability_calibration_until_source_resolved`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Source context recommendation: `treat_failed_season_as_schedule_roster_plus_capture_shift`.
- Failure season: 2024-2025. Comparator seasons: 2023-2024 and 2025-2026.
- Policy: exclude the failed season from probability calibration until resolved, block probability calibration pending source resolution, allow shadow ranking only with season monitoring, block model expansion, and defer minute-load expansion until capture consistency is proven.
- Resolution actions: resolve the 2024-2025 schedule/roster and capture shift, repair or document modified/no-participation and issue-linkage capture, exclude 2024-2025 from probability calibration datasets until eligibility is resolved, and retain shadow ranking monitoring only.

**Interpretation:** This is real progress but still research validation. Exposure-load has a usable shadow ranking signal, but probability-facing use, pilot/dashboard work, minute-load expansion, and broader model expansion remain blocked until 2024-2025 source eligibility is resolved.

## Previous Completed Step

**Exposure-load source context classification sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_source_context_classification.py`, `run_exposure_load_source_context_classification_sprint_experiment(...)`, and the `--exposure-load-source-context-classification-sprint` CLI mode. The sprint consumes cleaned source exposure artifacts plus the shift-context, schedule/roster, availability-capture, and context-decision JSON artifacts, then classifies whether the failed 2024-2025 exposure-load season is true managed-risk context, schedule/roster shift, exposure-capture/documentation shift, or mixed.

**Verification:** New TDD tests first failed because the classifier module, experiment runner, and CLI dispatch hook did not exist. After implementation, focused source-classification/experiment/CLI tests passed. The live command `risk-engine --exposure-load-source-context-classification-sprint --exposure-events outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_events.csv --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv --exposure-load-shift-context outputs/experiments/exposure_load_shift_context_v1/exposure_load_shift_context.json --exposure-load-schedule-roster outputs/experiments/exposure_load_schedule_roster_v1/exposure_load_schedule_roster_context.json --exposure-load-availability-capture outputs/experiments/exposure_load_availability_capture_v1/exposure_load_availability_capture.json --exposure-load-context-decision outputs/experiments/exposure_load_context_decision_v1/exposure_load_context_decision.json --output-dir outputs --experiment-id exposure_load_source_context_classification_v1` completed and wrote the classification artifacts.

**Live results (`exposure_load_source_context_classification_v1`):**
- Overall recommendation: `treat_failed_season_as_schedule_roster_plus_capture_shift`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Managed-risk support: `not_supported_by_source_flags`, with reduced modified/no-participation flagging and absent issue linkage.
- Schedule/roster context: `supported_schedule_roster_shift`, with elevated game schedule and reduced lift schedule signals.
- Availability-capture context: `supported_capture_or_documentation_shift`.
- Next model action: `do_not_expand_model_features`.

**Interpretation:** The failed 2024-2025 exposure-load season is now classified as schedule/roster plus exposure-capture/documentation shift rather than true managed-risk context. Exposure-load should remain shadow-only; do not add minute-load terms, probability-facing outputs, pilot/dashboard work, or broader exposure-load model features until the source context is resolved.

## Previous Completed Step

**Exposure-load schedule/roster, availability-capture, and context-decision sprints** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_context_review.py`, `run_exposure_load_schedule_roster_sprint_experiment(...)`, `run_exposure_load_availability_capture_sprint_experiment(...)`, `run_exposure_load_context_decision_sprint_experiment(...)`, and the `--exposure-load-schedule-roster-sprint`, `--exposure-load-availability-capture-sprint`, and `--exposure-load-context-decision-sprint` CLI modes. These three sprints extend the shift-context result into schedule/roster diagnostics, availability-capture diagnostics, and an operating decision synthesis.

**Verification:** New TDD tests first failed because the context-review module, experiment runners, and CLI dispatch hooks did not exist. After implementation, focused context-review/experiment/CLI tests passed. Live commands completed for `exposure_load_schedule_roster_v1`, `exposure_load_availability_capture_v1`, and `exposure_load_context_decision_v1`.

**Live results (`exposure_load_schedule_roster_v1`):**
- Overall recommendation: `review_failed_season_schedule_roster_shift`.
- 2024-2025 had reduced lift events versus comparator seasons (41.0 vs 49.0), but elevated training events (203.0 vs 180.5), total retained events (216.0 vs 192.5), participations per active athlete (148.4 vs 133.0), game events (13.0 vs 12.0), and active athletes (158.0 vs 155.5).

**Live results (`exposure_load_availability_capture_v1`):**
- Overall recommendation: `review_failed_season_availability_capture`.
- 2024-2025 had lower modified-participation flagging (0.014 vs 0.021) and lower no-participation flagging (0.050 vs 0.057) than comparator seasons; issue-linked participation rows remained absent.

**Live results (`exposure_load_context_decision_v1`):**
- Overall recommendation: `keep_shadow_ranking_and_resolve_context_before_model_expansion`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Decisions: probability calibration blocked, minute-load expansion blocked, shadow ranking allowed with monitoring, and model expansion blocked pending context review.

**Interpretation:** The next research step is not more model features. The 2024-2025 over-sharpening now has specific schedule/roster and availability-capture drivers that need source-level review before probability calibration, minute-load terms, pilot escalation, dashboard work, or broader exposure-load model expansion.

## Previous Completed Step

**Exposure-load shift context sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_shift_context.py`, `run_exposure_load_shift_context_sprint_experiment(...)`, and the `--exposure-load-shift-context-sprint` CLI mode. The sprint consumes cleaned exposure events/participations, exposure-load features, calibration diagnostics, and failure-mode JSON artifacts, then joins the shifted 2024-2025 exposure-load domains back to schedule, roster, availability, and managed-risk context.

**Verification:** New TDD tests first failed because `risk_stratification_engine.exposure_load_shift_context`, the experiment runner, and CLI dispatch hook did not exist. After implementation, focused shift-context/experiment/CLI tests passed, adjacent exposure-load tests passed, then `python -m pytest` collected and passed 245 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-load-shift-context-sprint --exposure-events outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_events.csv --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv --exposure-load-features outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_features.csv --exposure-load-diagnostics outputs/experiments/exposure_load_forward_diagnostic_v1/exposure_load_calibration_diagnostics.csv --exposure-load-failure-modes outputs/experiments/exposure_load_failure_modes_v1/exposure_load_failure_modes.json --output-dir outputs --experiment-id exposure_load_shift_context_v1` completed and wrote the context artifacts.

**Live results (`exposure_load_shift_context_v1`):**
- Overall recommendation: `review_schedule_roster_availability_context`.
- Failure season: 2024-2025. Comparator seasons: 2023-2024 and 2025-2026.
- Top driver context: reduced `exposure_lift_sessions_28d` in the failure season (2.110 vs 3.180), elevated `exposure_games_prior_count` (5.999 vs 4.840), reduced `exposure_modified_participations_28d` (0.283 vs 0.358), and longer `exposure_days_since_last_modified_or_no_participation` (29.589 vs 24.820).
- Additional shifted signals: longer `exposure_days_since_last_game`, elevated 28d practice exposure, reduced 28d no-participation flags, and slightly elevated 28d game-event exposure.
- Season context: 2024-2025 had 216 retained exposure events, 13 game events, 203 training events, 158 matched athletes, 23,445 matched participation rows, and a lower modified-participation rate than comparators (0.014 vs 0.021).

**Interpretation:** The 2024-2025 exposure-load over-sharpening is now tied to schedule, roster, and availability-context questions rather than a generic model failure. Probability-facing use, pilot escalation, dashboard work, and minute-load expansion remain blocked until these context drivers are reviewed. The next sprint should inspect whether the failed season reflects a true managed-risk context, schedule/roster shift, or exposure-capture/availability-documentation change before adding new model features.

## Previous Completed Step

**Exposure-load failure-mode and guardrail policy sprints** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_failure_modes.py`, `exposure_load_guardrail_policy.py`, `run_exposure_load_failure_mode_sprint_experiment(...)`, `run_exposure_load_guardrail_policy_sprint_experiment(...)`, and the `--exposure-load-failure-mode-sprint` / `--exposure-load-guardrail-policy-sprint` CLI modes. The first sprint compares 2024-2025 exposure feature distributions against calibration-supported comparator seasons; the second converts those diagnostics into operating guardrails.

**Verification:** New TDD tests first failed because the two modules, experiment runners, and CLI dispatch hooks did not exist. After implementation, focused failure-mode/guardrail/experiment/CLI tests passed, then `python -m pytest` collected and passed 242 tests with one existing sklearn convergence warning. Live commands completed: `risk-engine --exposure-load-failure-mode-sprint --exposure-load-features outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_features.csv --exposure-load-diagnostics outputs/experiments/exposure_load_forward_diagnostic_v1/exposure_load_calibration_diagnostics.csv --output-dir outputs --experiment-id exposure_load_failure_modes_v1` and `risk-engine --exposure-load-guardrail-policy-sprint --exposure-load-failure-modes outputs/experiments/exposure_load_failure_modes_v1/exposure_load_failure_modes.json --exposure-load-diagnostics outputs/experiments/exposure_load_forward_diagnostic_v1/exposure_load_calibration_diagnostics.csv --output-dir outputs --experiment-id exposure_load_guardrail_policy_v1`.

**Live results (`exposure_load_failure_modes_v1`):**
- Overall recommendation: `inspect_exposure_feature_shift_drivers`.
- Failure season: 2024-2025. Comparator seasons: 2023-2024 and 2025-2026.
- Top shifted feature/domain drivers were `exposure_lift_sessions_28d` reduced in the failed season (2.110 vs 3.096), `exposure_games_prior_count` elevated (5.999 vs 4.823), `exposure_modified_participations_28d` reduced (0.283 vs 0.361), and `exposure_days_since_last_modified_or_no_participation` elevated (29.589 vs 24.512).
- Top shifted domains were `category_specific_load`, `game_exposure`, and `participation_status`.

**Live results (`exposure_load_guardrail_policy_v1`):**
- Overall recommendation: `use_exposure_load_for_shadow_ranking_only`.
- Production readiness: `not_ready_for_probability_or_pilot`.
- Guardrails: probability calibration is blocked until the failure mode is resolved; ranking/triage is allowed only for shadow research with calibration monitoring; minute-load expansion is deferred; feature-domain review is required before the next model expansion.
- Evidence: 3 over-sharpened high-priority failure rows and 6 calibration-supported comparator rows.

**Interpretation:** Exposure-load remains useful as a shadow ranking/triage signal, but probability-facing use, pilot escalation, dashboard work, and duration/minute-load expansion remain blocked. The next research sprint should inspect the shifted 2024-2025 exposure domains against schedule, roster, availability, and managed-risk context.

## Previous Completed Step

**Exposure-load forward diagnostic sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_forward_diagnostics.py`, `run_exposure_load_forward_diagnostic_sprint_experiment(...)`, and the `--exposure-load-forward-diagnostic-sprint` CLI mode. The sprint consumes a completed `exposure_load_season_forward_validation.csv` artifact directly, compares `graph_plus_coverage_source` against `graph_plus_coverage_exposure_load` by forward season/horizon, and flags slices where ranking or triage gains come with calibration loss or predicted-risk over-sharpening.

**Verification:** New TDD tests first failed because `risk_stratification_engine.exposure_load_forward_diagnostics`, the experiment runner, and CLI dispatch did not exist. After implementation, focused diagnostic/experiment/CLI tests passed, then `python -m pytest` collected and passed 236 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-load-forward-diagnostic-sprint --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv --output-dir outputs --experiment-id exposure_load_forward_diagnostic_v1` completed and wrote the diagnostic artifacts.

**Live results (`exposure_load_forward_diagnostic_v1`):**
- Overall recommendation: `inspect_exposure_load_forward_failure_modes`.
- Diagnostic summary: 3 `ranking_triage_gain_calibration_loss`, 6 `calibration_supported`, and 6 `mixed_or_no_exposure_load_gain` season/horizon rows.
- All high-priority failures were 2024-2025 at 7d/14d/30d.
- In 2024-2025, exposure-load improved AUROC by +0.044/+0.052/+0.058 and top-decile lift by +0.157/+0.571/+0.949 at 7d/14d/30d, but Brier skill worsened by -0.655/-0.663/-0.544.
- 2024-2025 predicted-risk over-sharpening widened by +0.050/+0.067/+0.097 at 7d/14d/30d: exposure-load mean predicted risk was 9.3% / 13.4% / 20.2% versus observed rates of 2.8% / 4.8% / 8.9%.
- 2023-2024 and 2025-2026 provided calibration-supported comparator rows rather than the primary failure mode.

**Interpretation:** Exposure-load context is a real forward ranking/triage signal, but 2024-2025 over-sharpening confirms it is not pilot-ready calibration. The next sprint should inspect why the 2024-2025 exposure context overstates probability before adding duration/minute-load terms or moving toward dashboard work.

## Previous Completed Step

**Exposure-load season-forward validation sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `run_exposure_load_season_forward_validation_sprint_experiment(...)` and the `--exposure-load-season-forward-validation` CLI mode. The sprint reuses the conservative exposure-load feature set, compares `graph_plus_coverage_source` against `graph_plus_coverage_exposure_load` under train-prior-seasons / evaluate-later-seasons validation, and scores fixed alert channels with the exposure-load feature set. `attach_exposure_load_features(...)` now uses grouped date arrays and binary-search window counts so full live season-forward runs complete without row-by-row full-table scans.

**Verification:** New TDD tests first failed because the experiment runner and CLI dispatch did not exist. After implementation, focused exposure-load/season-forward/CLI tests passed, then `python -m pytest` collected and passed 232 tests with one existing sklearn convergence warning. The initial live `--from-live-sources` attempt wrote the canonical live-input snapshot but timed out before artifacts; after optimizing exposure attachment, the snapshot command `risk-engine --measurements outputs/live_inputs/exposure_load_season_forward_validation_v1/canonical_measurements.csv --injuries outputs/live_inputs/exposure_load_season_forward_validation_v1/canonical_injuries.csv --exposure-load-season-forward-validation --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv --output-dir outputs --experiment-id exposure_load_season_forward_validation_v1 --model-variant l2 --graph-window-size 4` completed and wrote the validation artifacts.

**Live results (`exposure_load_season_forward_validation_v1`, L2, window 4):**
- Overall recommendation: `continue_season_forward_research`.
- Evaluated forward seasons 2021-2022 through 2025-2026 with 75,380 graph snapshot rows.
- `graph_plus_coverage_exposure_load` won all selected best forward ranking, calibration, and burden-triage slots.
- 7d: best ranking/calibration came from 2025-2026 with AUROC 0.682 and Brier skill 0.016; best top-decile lift came from 2023-2024 at 1.618.
- 14d: best ranking/calibration came from 2025-2026 with AUROC 0.699 and Brier skill 0.035; best top-decile lift came from 2023-2024 at 2.207.
- 30d: best ranking/calibration came from 2025-2026 with AUROC 0.719 and Brier skill 0.074; best top-decile lift came from 2024-2025 at 1.987.
- Exposure-load improved average forward AUROC and top-decile lift versus coverage/source at all horizons, but calibration was unstable by season. In 2024-2025, Brier skill was materially worse with exposure load: -0.682 / -0.691 / -0.567 at 7d/14d/30d.
- Alert-policy forward checks remain research-only: `broad_30d` burden-capped 16.4% capture / 0.40 burden, `severity_14d` burden-capped 10.7% / 0.50, `severity_7d` burden-capped 4.8% / 0.40, and subtype review season-local 19.0% / 0.74 episodes per athlete-season.

**Interpretation:** Exposure/load survives season-forward validation as a ranking and triage signal and now improves the best 2025-2026 calibration slots, but the 2024-2025 calibration penalty means it is not production/pilot ready. The next sprint should diagnose exposure-load forward calibration failure modes, especially 2024-2025 over-sharpening, before adding duration/minute-load terms or escalating to dashboard work.

## Previous Completed Step

**Exposure load feature sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_load_features.py`, `exposure_load_modeling.py`, the `run_exposure_load_feature_sprint_experiment(...)` runner, and the `--exposure-load-feature-sprint` CLI mode. The sprint attaches conservative time-safe exposure features to graph snapshots from cleaned participation rows, then compares `graph_plus_coverage_source` against `graph_plus_coverage_exposure_load`.

**Verification:** New TDD tests first failed because `risk_stratification_engine.exposure_load_features` and the CLI runner hook did not exist. After implementation, focused exposure/CLI tests passed, then `python -m pytest` collected and passed 230 tests with one existing sklearn convergence warning. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --exposure-load-feature-sprint --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv --output-dir outputs --experiment-id exposure_load_feature_v1 --model-variant l2 --graph-window-size 4` completed and wrote the live inputs plus exposure-load artifacts.

**Live results (`exposure_load_feature_v1`, L2, window 4):**
- Overall recommendation: `continue_exposure_load_research`.
- Production readiness: `not_ready_research_validation_required`.
- Live inputs contained 913,973 measurement rows, 1,027 canonical injury rows, 638 detailed injury rows, and 331 observed events.
- The feature table contained 75,380 graph snapshots.
- Nonzero exposure context appeared in 48,984 snapshots for 7d training-session count, 50,470 for 28d training-session count, 34,837 for prior game count, 5,500 for 28d modified participation count, 16,319 for 28d no-participation count, and 21,723 for 28d game exposure count.
- `graph_plus_coverage_exposure_load` beat `graph_plus_coverage_source` at all horizons and decision modes in the standard holdout comparison.
- AUROC improved from 0.728 to 0.802 at 7d, 0.732 to 0.810 at 14d, and 0.743 to 0.819 at 30d.
- Brier skill improved from 0.017 to 0.037 at 7d, 0.031 to 0.064 at 14d, and 0.059 to 0.119 at 30d.
- Top-decile lift improved from 1.96 to 3.20 at 7d, 1.97 to 3.00 at 14d, and 1.99 to 2.78 at 30d.

**Interpretation:** Exposure/load context is now a strong first-pass modeling signal when attached before each snapshot. This is still research validation, not dashboard or pilot clearance. The next sprint should run season-forward validation with the exposure-load feature set and inspect whether the gains survive forward-season splits, especially before adding duration/minute-load terms.

## Previous Completed Step

**Exposure feature requirements sprint** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_feature_requirements.py`, the `run_exposure_feature_requirements_sprint_experiment(...)` runner, and the `--exposure-feature-requirements-sprint` CLI mode. The sprint consumes cleaned `exposure_events.csv`, `exposure_participations.csv`, and `exposure_cleaning_audit.json` artifacts, then writes `exposure_category_summary.csv`, `exposure_duration_summary.csv`, `exposure_feature_requirements.csv`, `exposure_feature_requirements.json`, and `exposure_feature_requirements_report.md`.

**Verification:** New TDD tests first failed because `risk_stratification_engine.exposure_feature_requirements`, the experiment runner, and CLI dispatch did not exist. After implementation, focused exposure/CLI tests passed and `python -m pytest` collected and passed 226 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-feature-requirements-sprint --exposure-events outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_events.csv --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv --exposure-audit outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_cleaning_audit.json --output-dir outputs --experiment-id exposure_feature_requirements_v1` completed and wrote the requirements artifacts.

**Live results (`exposure_feature_requirements_v1`):**
- Overall recommendation: `proceed_with_count_and_status_features_first`.
- Production readiness: `not_ready_feature_design_required`.
- Reviewed 1,055 retained exposure events and 108,587 participation rows.
- Readiness summary: 3 ready domains and 2 caution domains.
- Ready domains: `session_count_load`, `participation_status`, and `category_specific_load`.
- Caution domains: `duration_load` and `game_exposure`, driven mainly by game participation duration missingness.
- Training participation duration missingness was 8,703 of 102,964 rows (8.45%); game participation duration missingness was 2,648 of 5,623 rows (47.09%).
- Athlete matching remained strong: 102,935 of 102,964 retained training rows and 5,621 of 5,623 game rows matched.
- Retained training categories covered 17 usable categories with no unclassified session types.

**Interpretation:** The next major modeling sprint should attach time-safe count, recency, participation-status, and coarse-category exposure features to graph snapshots before using minute-load terms as primary features. Use game counts/recency first and treat game minutes as secondary until duration semantics are reviewed. This is still research-readiness work, not dashboard or pilot clearance.

## Previous Completed Step

**Exposure-data cleaning/audit layer** - implemented and verified on 2026-05-09.

**What changed:** Added `exposure_sources.py`, optional `exposure_dir` path config support, and the `--exposure-cleaning-audit` CLI mode. The sprint intentionally stops at cleaned exposure artifacts and audit summaries before model features. It writes `exposure_events.csv`, `exposure_participations.csv`, and `exposure_cleaning_audit.json` under `outputs/exposure_inputs/<experiment-id>/`.

**Verification:** New TDD tests first failed because optional exposure path config, `risk_stratification_engine.exposure_sources`, and the `--exposure-cleaning-audit` dispatch did not exist. After implementation, focused CLI/config/exposure tests passed and `python -m pytest` collected and passed 221 tests with one existing sklearn convergence warning. The live command `risk-engine --exposure-cleaning-audit --exposure-dir C:\Users\eric_rash\Desktop\DEV\Football\Baylor_Exposure_Data --output-dir outputs --experiment-id exposure_cleaning_audit_v1` completed and wrote the exposure audit artifacts.

**Live results (`exposure_cleaning_audit_v1`):**
- Football scope matched the handoff: 2,997 training events / 170,006 training participation rows and 64 games / 5,623 game participation rows.
- Clean retained artifacts contain 991 human-entered football training events, 64 football games, 102,964 retained training participation rows, and 5,623 game participation rows.
- Excluded football training events were 2,006 API/performance-source sessions; no football session types remained unclassified after keeping the `Speed-Power + Cond + Weight Room` and `Speed-Power + Conditioning` variants as human-entered exposure.
- Athlete matching uses football squad membership rather than position text alone: retained training rows had 102,935 matched and 29 unmatched rows; game rows had 5,621 matched and 2 unmatched rows.
- No duplicate retained athlete-event keys were found in training or game participations.

**Interpretation:** Exposure/load data now has a reproducible cleaning layer and audit surface. The next sprint should review retained/excluded exposure categories, missing participation duration, and candidate feature definitions before attaching prior-window exposure features to graph snapshots. Do not run season-forward validation or pilot/dashboard work until the exposure/load feature design is reviewed.

## Previous Completed Step

**Repaired-snapshot season-forward and injury-history diagnostic review** - verified on 2026-05-08.

**What changed:** No new data domains were added. The repaired ForcePlate live-input snapshot from `coverage_stratified_after_forceplate_repair_v1` was reused to run `season_forward_after_forceplate_repair_v1` and `injury_history_forward_diagnostic_after_forceplate_repair_v1`, preserving the same canonical measurement/injury inputs while updating season-forward and injury-history calibration evidence after the ForcePlate repair.

**Verification:** `risk-engine --measurements outputs/live_inputs/coverage_stratified_after_forceplate_repair_v1/canonical_measurements.csv --injuries outputs/live_inputs/coverage_stratified_after_forceplate_repair_v1/canonical_injuries.csv --output-dir outputs --experiment-id season_forward_after_forceplate_repair_v1 --season-forward-validation --model-variant l2 --graph-window-size 4` completed and wrote the season-forward artifacts. `risk-engine --measurements outputs/live_inputs/coverage_stratified_after_forceplate_repair_v1/canonical_measurements.csv --injuries outputs/live_inputs/coverage_stratified_after_forceplate_repair_v1/canonical_injuries.csv --output-dir outputs --experiment-id injury_history_forward_diagnostic_after_forceplate_repair_v1 --injury-history-forward-diagnostic-sprint --model-variant l2 --graph-window-size 4` completed and wrote the injury-history diagnostic artifacts. No `risk-engine`, `run_pipeline.py`, or Python worker processes remained active after verification.

**Live results (`season_forward_after_forceplate_repair_v1`, L2, window 4):**
- Overall recommendation: `continue_season_forward_research`.
- Best 7d ranking/calibration/triage all came from `graph_plus_coverage_source` in 2025-2026: AUROC 0.648, Brier skill 0.014, top-decile lift 1.556.
- Best 14d ranking/calibration came from 2025-2026 with AUROC 0.650 and Brier skill 0.023; best 14d triage remained 2023-2024 with top-decile lift 1.852.
- Best 30d ranking/triage remained 2023-2024 with AUROC 0.649 and top-decile lift 1.818; best 30d calibration remained 2025-2026 with Brier skill 0.035.
- Alert-policy forward checks stayed research-only: `broad_30d` burden-capped 10.7% capture / 0.47 burden, `severity_14d` season-local 12.7% / 0.86, `severity_7d` burden-capped 5.9% / 0.40, and subtype review season-local 16.4% / 0.78 episodes per athlete-season.

**Live results (`injury_history_forward_diagnostic_after_forceplate_repair_v1`, L2, window 4):**
- Overall recommendation: `inspect_injury_history_forward_failure_modes`.
- 2024-2025 remains the dominant high-priority failure slice across 7d/14d/30d.
- In 2024-2025, adding injury history improved AUROC by +0.065/+0.071/+0.061 and top-decile lift by +1.536/+1.824/+1.838 at 7d/14d/30d, but Brier skill worsened by -0.508/-0.484/-0.407.
- The 2024-2025 injury-history model over-predicted average risk relative to observed positive rates: 7d 8.8% mean predicted vs 2.8% observed, 14d 12.7% vs 4.8%, and 30d 19.1% vs 8.9%.
- 2025-2026 was more mixed: 7d and 30d were `calibration_supported`, while 14d still showed a small calibration loss with improved ranking/triage.
- Injury-history context is now common in later seasons: prior-injury context appears in 9,949 of 15,898 repaired-snapshot 2024-2025 feature rows and 8,504 of 14,427 2025-2026 rows.

**Interpretation:** The ForcePlate repair improved coverage independence and reduced the injury-history calibration penalty compared with the earlier diagnostic, but it did not remove the 2024-2025 failure mode. Injury history is concentrating events into higher ranked groups while over-sharpening probability estimates. Do not proceed to new season-forward validation or pilot/dashboard work until the user's additional exposure/load, mechanism, availability/intervention, and frailty-related data for the next feature sprint has been reviewed.

## Previous Completed Step

**Post-ForcePlate repair coverage-stratified refresh** - verified on 2026-05-08.

**What changed:** No code changes were needed for this refresh. The ForcePlate source DB had been repaired/backfilled upstream, so the Risk Engine live-source inputs were regenerated from `config/paths.local.yaml` and the coverage-stratified evaluation was rerun as `coverage_stratified_after_forceplate_repair_v1`.

**Verification:** `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id coverage_stratified_after_forceplate_repair_v1 --coverage-stratified-evaluation --model-variant l2 --graph-window-size 4` wrote the live-input snapshot but hit the shell timeout before experiment artifacts were created. The canonical snapshot was then reused directly with `risk-engine --measurements outputs/live_inputs/coverage_stratified_after_forceplate_repair_v1/canonical_measurements.csv --injuries outputs/live_inputs/coverage_stratified_after_forceplate_repair_v1/canonical_injuries.csv --output-dir outputs --experiment-id coverage_stratified_after_forceplate_repair_v1 --coverage-stratified-evaluation --model-variant l2`, which completed and wrote `coverage_tiers.csv`, `coverage_stratified_evaluation.csv`, `coverage_stratified_evaluation.json`, `coverage_stratified_evaluation_report.md`, and `config.json`.

**Live results (`coverage_stratified_after_forceplate_repair_v1`, L2):**
- Overall coverage flag changed to `coverage_independent`.
- This supersedes the earlier `coverage_stratified_eval_v1` result of `coverage_confounded` and the interim `forceplate_2023_backfill_coverage_v1` result of `mixed`.
- Live canonical inputs contained 913,973 measurement rows, 1,027 canonical injury rows, 638 detailed injury rows, and 331 observed events.
- Measurement source rows were: GPS 637,013, ForcePlate 225,384, bodyweight 40,943, and Perch 10,633.
- Coverage tiers were balanced: low 350 athlete-seasons, medium 335, and high 342.
- Broad 30d capture by coverage tier was low 38.7%, medium 15.4%, high 28.0%.
- Severity 7d capture was low 27.8%, medium 19.1%, high 27.5%.
- Severity 14d capture was low 38.9%, medium 19.1%, high 35.2%.
- Lower-extremity soft-tissue 30d capture was low 68.8%, medium 52.8%, high 50.6%.

**Interpretation:** The earlier coverage-confounding concern appears materially improved after the ForcePlate repair. This supports continuing shadow-mode research with the repaired source snapshot, but it is not standalone pilot clearance because injury-history calibration failures and missing exposure/load, mechanism, availability/intervention, and frailty context remain open production-readiness blockers.

## Previous Completed Step

**Injury-history forward diagnostic sprint** - implemented and verified on 2026-05-08.

**What changed:** Added `injury_history_forward_diagnostics.py` with calibration-diagnostic summary/report helpers, plus `run_injury_history_forward_diagnostic_sprint_experiment(...)` and the `--injury-history-forward-diagnostic-sprint` CLI mode. The new sprint compares forward-season `graph_plus_coverage_source` vs `graph_plus_coverage_injury_history` rows, creates season/horizon diagnostic cases, and writes `injury_history_features.csv`, `injury_history_season_forward_validation.csv`, `injury_history_calibration_diagnostics.csv`, `injury_history_forward_diagnostic_cases.csv`, `injury_history_forward_diagnostic.json`, `injury_history_forward_diagnostic_report.md`, and `config.json`.

**Peterson guardrail:** The sprint diagnoses injury-history behavior after complete athlete-season trajectories have been scored in a season-forward setup. It does not use future injury details and does not convert daily measurement rows into independent injury-classification examples.

**Verification:** New TDD tests first failed because `risk_stratification_engine.injury_history_forward_diagnostics`, `run_injury_history_forward_diagnostic_sprint_experiment(...)`, and the `--injury-history-forward-diagnostic-sprint` CLI dispatch did not exist. After implementation, focused tests passed and `python -m pytest` collected and passed 217 tests with one sklearn convergence warning in a fixture sprint. A raw live-source refresh attempt was blocked by the active ForcePlate DuckDB writer, so live diagnostic artifacts were generated from the already completed `injury_history_season_forward_validation_v1` season-forward output.

**Live results (`injury_history_forward_diagnostic_v1`, L2, window 4 source validation):**
- Overall recommendation: `inspect_injury_history_forward_failure_modes`.
- Diagnosed 15 season/horizon comparisons and generated 6 season/horizon diagnostic cases.
- Six rows were flagged `ranking_triage_gain_calibration_loss`: all 7d/14d/30d rows in 2024-2025 and 2025-2026.
- 2024-2025 had the largest tradeoff: AUROC improved by +0.049/+0.045/+0.043 and top-decile lift improved by +1.599/+1.952/+1.863 at 7d/14d/30d, while Brier skill dropped by -1.054/-0.839/-0.584.
- 2025-2026 showed smaller but still unfavorable calibration tradeoffs: top-decile lift improved by +0.646/+0.484/+0.517 at 7d/14d/30d, while Brier skill dropped by -0.008/-0.020/-0.016.
- 2021-2022 had no evaluable deltas, and 2022-2023 plus 2023-2024 were effectively unchanged.

**Interpretation:** Injury history is not a calibration-ready production feature yet. It appears to concentrate injuries into higher-risk ranks in later seasons, especially 2024-2025, but at the cost of materially worse probability calibration. The next sprint should inspect why prior-injury signal is over-sharpening probabilities: start with feature attribution/calibration bins for the 2024-2025 high-lift rows, then test whether exposure/load, mechanism, availability/intervention, or frailty features can absorb the calibration failure before any dashboard or pilot work.

## Previous Completed Step

**Injury-history season-forward validation sprint** - implemented and verified on 2026-05-08.

**What changed:** Added `run_injury_history_season_forward_validation_sprint_experiment(...)` and the `--injury-history-season-forward-validation` CLI mode. Generalized the season-forward validation internals so a caller can provide a custom feature-set comparison and alert-policy scoring feature set. The new runner compares `graph_plus_coverage_source` with `graph_plus_coverage_injury_history`, scores fixed alert channels with injury-history features, and writes `injury_history_features.csv`, `injury_history_season_forward_validation.csv`, `injury_history_season_forward_validation.json`, `injury_history_season_forward_validation_report.md`, and `config.json`.

**Peterson guardrail:** The sprint still trains on earlier complete athlete-season trajectories and evaluates later complete athlete-season trajectories. Injury-history features remain time-safe because they are derived only from detailed injury events before each graph snapshot date; the sprint does not use future injury details or convert daily rows into independent injury-classification examples.

**Verification:** New TDD tests first failed because `build_season_forward_validation_summary(...)` did not accept a custom experiment type, `run_injury_history_season_forward_validation_sprint_experiment(...)` did not exist, and the `--injury-history-season-forward-validation` CLI dispatch did not exist. After implementation, focused tests passed and `python -m pytest` collected and passed 213 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id injury_history_season_forward_validation_v1 --injury-history-season-forward-validation --model-variant l2 --graph-window-size 4` completed and wrote the sprint artifacts.

**Live results (`injury_history_season_forward_validation_v1`, L2, window 4):**
- Overall recommendation: `continue_season_forward_research`.
- Evaluated forward seasons 2021-2022 through 2025-2026 with 74,860 graph snapshot rows.
- `graph_plus_coverage_injury_history` won selected forward ranking/triage slots, but did not dominate calibration.
- 7d: best ranking was injury-history in 2025-2026 (AUROC 0.662); best calibration remained coverage/source in 2025-2026 (Brier skill 0.016); best top-decile lift was injury-history in 2024-2025 (2.477).
- 14d: best ranking and calibration remained coverage/source in 2025-2026 (AUROC 0.677; Brier skill 0.028); best top-decile lift was injury-history in 2024-2025 (2.928).
- 30d: best ranking was injury-history in 2023-2024 (AUROC 0.689); best calibration remained coverage/source in 2025-2026 (Brier skill 0.052); best top-decile lift was injury-history in 2024-2025 (3.063).
- Injury-history alert-policy forward checks selected low-burden policies, but mean event capture stayed low: `broad_30d` burden-capped 5.0% capture / 0.14 episodes per athlete-season, `severity_14d` burden-capped 7.9% / 0.26, `severity_7d` burden-capped 4.8% / 0.29, and subtype review season-local 8.2% / 0.37.
- Nonzero prior-injury context appeared in 30,482 snapshots; prior lower-extremity soft-tissue context appeared in 13,637 snapshots.

**Interpretation:** Injury history survives forward validation as a ranking and triage signal in later seasons, but it is not a clean calibration improvement and alert capture remains too low for pilot clearance. The next sprint should inspect injury-history forward case behavior and calibration failure modes, especially 2024-2025 high-lift/poor-calibration rows and 2025-2026 calibration comparisons, before any dashboard or pilot work.

## Previous Completed Step

**Injury-history feature sprint** - implemented and verified on 2026-05-08.

**What changed:** Added `injury_history_features.py` with `INJURY_HISTORY_FEATURE_COLUMNS` and `attach_injury_history_features(...)`, plus `injury_history_modeling.py` with summary/report helpers. Added `run_injury_history_feature_sprint_experiment(...)` and the `--injury-history-feature-sprint` CLI mode. The runner compares `graph_plus_coverage_source` with `graph_plus_coverage_injury_history` and writes `injury_history_features.csv`, `injury_history_model_comparison.csv`, `injury_history_model_comparison.json`, `injury_history_model_comparison_report.md`, and `config.json`.

**Peterson guardrail:** Injury-history features are attached only from detailed injury events strictly before each graph snapshot date. The sprint adds prior-injury context to scored athlete-season trajectories and does not use future injury details or convert daily rows into independent examples.

**Verification:** New TDD tests first failed because `risk_stratification_engine.injury_history_features`, `risk_stratification_engine.injury_history_modeling`, `run_injury_history_feature_sprint_experiment`, and the `--injury-history-feature-sprint` CLI dispatch did not exist. After implementation, focused tests passed and `python -m pytest` collected and passed 210 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id injury_history_feature_v1 --injury-history-feature-sprint --model-variant l2 --graph-window-size 4` completed and wrote the sprint artifacts.

**Live results (`injury_history_feature_v1`, L2, window 4):**
- Overall recommendation: `continue_injury_history_research`.
- Added 12 prior-injury features across 74,860 graph snapshot rows.
- `graph_plus_coverage_injury_history` beat `graph_plus_coverage_source` at every horizon and decision mode in this holdout comparison.
- 7d: AUROC improved 0.721 -> 0.755, Brier skill 0.015 -> 0.028, top-decile lift 1.85 -> 3.73.
- 14d: AUROC improved 0.726 -> 0.759, Brier skill 0.026 -> 0.052, top-decile lift 1.84 -> 3.51.
- 30d: AUROC improved 0.743 -> 0.774, Brier skill 0.054 -> 0.111, top-decile lift 2.13 -> 3.46.
- Nonzero prior-injury context appeared in 30,482 snapshots; prior lower-extremity soft-tissue context appeared in 13,637 snapshots.

**Interpretation:** The uploaded injury data contains real usable signal for baseline/frailty and mechanism-context modeling. This is the strongest model-readiness improvement since coverage/source features, but it is still not pilot clearance because this sprint used the standard holdout comparison rather than season-forward validation. The next sprint should run season-forward validation with the injury-history feature set, then review whether alert policies improve without unacceptable burden.

## Previous Completed Step

**Case diagnostic requirements sprint** - implemented and verified on 2026-05-08.

**What changed:** Added `case_diagnostic_requirements.py` with requirement-domain mapping, summary helpers, and a Markdown report writer. Added `run_case_diagnostic_requirements_sprint_experiment(...)` and the `--case-diagnostic-requirements-sprint` CLI mode. The runner reuses the forward case-review cases and writes `forward_case_review_cases.csv`, `case_diagnostic_requirements.csv`, `case_diagnostic_requirements.json`, `case_diagnostic_requirements_report.md`, and `config.json`.

**Peterson guardrail:** The sprint does not introduce daily-row classification. It converts reviewed cases from scored complete athlete-season trajectories into data/model requirements, preserving the longitudinal athlete-season unit.

**Verification:** New TDD tests first failed because `risk_stratification_engine.case_diagnostic_requirements`, `run_case_diagnostic_requirements_sprint_experiment`, and the `--case-diagnostic-requirements-sprint` CLI dispatch did not exist. After implementation, focused tests passed and `python -m pytest` collected and passed 204 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id case_diagnostic_requirements_v1 --case-diagnostic-requirements-sprint --model-variant l2` completed and wrote the requirements artifacts.

**Live results (`case_diagnostic_requirements_v1`, L2):**
- Overall recommendation: `prioritize_data_acquisition_before_production`.
- Production readiness: `not_ready_missing_context`.
- Reviewed 44 forward case-review cases and generated 5 requirement domains.
- Critical domains: `exposure_load` (24 cases, 54.5%), `baseline_frailty` (20 cases, 45.5%), `injury_mechanism` (20 cases, 45.5%), and `intervention_availability` (12 cases, 27.3%).
- High-priority domain: `explanation_fidelity` (8 cases, 18.2%).
- Key missing fields include session participation, minutes exposed, practice intensity, acute/chronic load, game exposure, availability status, modified training status, treatment/rehab flags, prior injury count, chronic condition flags, athlete baseline state, injury mechanism, contact/non-contact context, activity context, body-area detail, and graph node/edge change traces.

**Interpretation:** The model is not blocked by only threshold tuning. The strongest current production blockers are missing exposure/load, athlete baseline/frailty, injury mechanism, and intervention/availability context. The next major sprint should acquire or derive those fields, add time-safe feature builders for them, and rerun season-forward validation before any dashboard or pilot work.

## Previous Completed Step

**Forward case review sprint** - implemented and verified on 2026-05-08.

**What changed:** Added `forward_case_review.py` with summary/report helpers, plus `run_forward_case_review_sprint_experiment(...)` and the `--forward-case-review-sprint` CLI mode. The runner targets the forward-surviving channel set, scores complete athlete-season trajectories with coverage/source-aware graph features, rebuilds alert episodes, and writes deterministic case-review artifacts.

**Peterson guardrail:** The sprint reviews cases only after complete athlete-season trajectories are scored. It targets season/channel behavior that survived season-forward validation instead of converting daily measurement rows into independent classification examples.

**Verification:** New TDD tests first failed because `risk_stratification_engine.forward_case_review`, `run_forward_case_review_sprint_experiment`, and the `--forward-case-review-sprint` CLI dispatch did not exist. After implementation, focused tests passed and `python -m pytest` collected and passed 200 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id forward_case_review_v1 --forward-case-review-sprint --model-variant l2` completed and wrote the review artifacts.

**Live results (`forward_case_review_v1`, L2):**
- Reviewed 44 deterministic cases across the targeted forward seasons 2023-2024 and 2025-2026.
- Diagnostic labels split into 14 `model_signal_supported`, 12 `model_miss`, 12 `missing_context_or_managed_risk`, and 6 `explanation_gap`.
- Case types split into 12 true-positive episodes, 12 false-positive episodes, 12 missed injuries, and 8 high intra-individual deviation episodes.
- `broad_30d` produced the strongest supported-case balance with 6 of 16 cases labeled `model_signal_supported`.
- `severity_14d` was dominated by explanation gaps, and `subtype_lower_extremity_soft_tissue_30d` was dominated by missing-context or managed-risk labels.

**Interpretation:** The forward case evidence is mixed and still research-only. The model has supported examples in the broad 30d channel, but severity and subtype channels expose explanation gaps, misses, and missing exposure/intervention/baseline/mechanism context. The next sprint should translate those case diagnostics into explicit data requirements and targeted model improvements rather than moving toward a dashboard.

## Previous Completed Step

**Season-forward validation sprint** — implemented and verified on 2026-05-08.

**What changed:** Added `season_forward_validation.py` with summary/report helpers, plus `run_season_forward_validation_sprint_experiment(...)` and the `--season-forward-validation` CLI mode. The runner builds graph and coverage/source features, trains row-based temporal logistic models on prior seasons, evaluates later seasons, and checks fixed shadow-mode channels under season-local and burden-capped thresholds.

**Peterson guardrail:** The sprint trains on earlier complete athlete-season trajectories and evaluates later complete athlete-season trajectories. It does not randomize daily rows or treat measurements as independent injury-classification examples.

**Verification:** New TDD tests first failed because `risk_stratification_engine.season_forward_validation`, `run_season_forward_validation_sprint_experiment`, and the `--season-forward-validation` CLI dispatch did not exist. After implementation, focused season-forward tests passed and `python -m pytest` collected and passed 196 tests. A direct run against the latest completed prepared live-input snapshot (`coverage_source_aware_model_v1`) wrote `season_forward_validation_v1`; the raw `--from-live-sources` refresh path remained blocked by an active ForcePlate DuckDB writer, so the upstream-source refresh itself was not rerun for this sprint.

**Live snapshot results (`season_forward_validation_v1`, L2, window 4):**
- Evaluated forward seasons 2021-2022 through 2025-2026.
- `graph_plus_coverage_source` won the best observed ranking slots at all horizons in 2023-2024: AUROC 0.667 at 7d, 0.673 at 14d, and 0.703 at 30d.
- `graph_plus_coverage_source` also won the best calibration slots in 2025-2026: Brier skill 0.014 at 7d, 0.026 at 14d, and 0.048 at 30d.
- Early forward seasons remain weak: 2021-2022 had no evaluable discrimination metrics, and 2022-2023 was effectively prevalence-like at AUROC 0.500.
- Alert-policy forward checks stayed research-only. Recommended mean capture/burden pairs were: `broad_30d` burden-capped 11.6% / 0.41, `severity_14d` season-local 15.0% / 0.86, `severity_7d` burden-capped 6.4% / 0.47, and subtype review season-local 19.5% / 0.85 episodes per athlete-season.

**Interpretation:** Coverage/source-aware graph models show real forward-season signal in later, richer seasons, but the evidence is not stable enough for dashboard or pilot escalation. The most likely limitation is still missing context around exposure, training intent, interventions, athlete frailty/baseline, and injury mechanism. The next sprint should be an episode quality/case review focused only on the forward-surviving windows and channels, especially 2023-2024 ranking and 2025-2026 calibration behavior.

## Previous Completed Step

**Coverage-adjusted threshold sprint** — implemented and verified on 2026-05-08.

**What changed:** Added `coverage_adjusted_thresholds.py` with threshold-policy row builders, summary helpers, and a Markdown report writer. Added `run_coverage_adjusted_threshold_sprint_experiment(...)` and the `--coverage-adjusted-threshold-sprint` CLI mode. The runner retrains the fixed shadow-mode channels, merges coverage tiers onto alert timelines, then compares `season_local_percentile`, `coverage_tier_local_percentile`, and `burden_capped_percentile` policies.

**Peterson guardrail:** Thresholds are adjusted only after complete athlete-season trajectories are scored. The sprint does not relabel or resample daily measurement rows as independent injury-classification examples.

**Verification:** New TDD tests first failed because `risk_stratification_engine.coverage_adjusted_thresholds`, `run_coverage_adjusted_threshold_sprint_experiment`, and the `--coverage-adjusted-threshold-sprint` CLI dispatch did not exist. After implementation, the focused sprint tests passed and `python -m pytest` collected and passed 192 tests. A direct run against the latest completed prepared live-input snapshot (`coverage_source_aware_model_v1`) wrote `coverage_adjusted_threshold_v1`; the raw `--from-live-sources` refresh path was blocked by an active ForcePlate DuckDB writer, so the upstream-source refresh itself was not rerun for this sprint.

**Live snapshot results (`coverage_adjusted_threshold_v1`, L2):**
- Overall recommendation: `continue_threshold_research`.
- All four channels selected `burden_capped_percentile` when enforcing the 1.0 episode per athlete-season cap.
- Burden-capped policies reduced mean burden below the cap but materially reduced mean capture: `broad_30d` 10.4% capture / 0.65 burden, `severity_14d` 8.8% / 0.75, `severity_7d` 5.5% / 0.81, and `subtype_lower_extremity_soft_tissue_30d` 9.7% / 0.63.
- Coverage-tier-local thresholds preserved more capture than burden caps but stayed too burdensome: `severity_14d` mean burden 2.48, `severity_7d` 2.69, and subtype review 2.58 episodes per athlete-season.
- Season-local thresholds remained higher capture but high burden, especially `severity_14d` at 33.8% capture / 2.58 burden and subtype review at 46.2% / 2.66.

**Interpretation:** Coverage-tier-local thresholds do not solve the burden problem, and burden caps solve burden mostly by erasing event capture. The fixed shadow-mode package should remain research-only. The next sprint should be season-forward validation: train on earlier seasons, evaluate on later seasons, and only then send any surviving channel into episode quality/case review.

## Previous Completed Step

**Coverage/source-aware model sprint** — implemented and verified on 2026-05-08.

**What changed:** Added `coverage_source_features.py` with `COVERAGE_SOURCE_FEATURE_COLUMNS` and `attach_coverage_source_features(...)`, plus `coverage_source_modeling.py` with comparison summary/report helpers. Added `run_coverage_source_aware_model_sprint_experiment(...)` and the `--coverage-source-aware-model-sprint` CLI mode. The runner builds graph snapshots, attaches time-safe coverage/source features, labels the same athlete-season trajectories, and compares `graph_trajectory` against `graph_plus_coverage_source`.

**Peterson guardrail:** Coverage/source context is added as explicit covariates available at each snapshot date, but the dynamic graph trajectory features remain the core signal. The sprint still models athlete-season trajectories and does not treat daily rows as independent examples.

**Verification:** New TDD tests first failed because `risk_stratification_engine.coverage_source_features`, `risk_stratification_engine.coverage_source_modeling`, `run_coverage_source_aware_model_sprint_experiment`, and the `--coverage-source-aware-model-sprint` CLI dispatch did not exist. After implementation, the focused coverage/source tests passed and `python -m pytest` collected and passed 186 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id coverage_source_aware_model_v1 --coverage-source-aware-model-sprint --model-variant l2 --graph-window-size 4` completed and wrote the comparison artifacts.

**Live results (`coverage_source_aware_model_v1`, L2, window 4):**
- Overall recommendation: `continue_coverage_source_research`.
- `graph_plus_coverage_source` improved AUROC at all horizons: 7d 0.686 to 0.723, 14d 0.676 to 0.723, and 30d 0.689 to 0.734.
- Brier skill improved at all horizons: 7d -0.000 to 0.014, 14d -0.001 to 0.024, and 30d 0.007 to 0.048.
- Top-decile lift was mixed: unchanged at 7d (1.906), lower at 14d (1.652 to 1.555), and higher at 30d (1.686 to 1.869).

**Interpretation:** Coverage/source covariates improve ranking and calibration-oriented metrics enough to continue controlled validation, but they do not clear the model for dashboard or pilot escalation. The next practical sprint should test coverage-adjusted thresholds and burden-capped alert policies, then season-forward validation.

## Previous Completed Step

**Coverage-normalized policy sprint** — implemented and verified on 2026-05-08.

**What changed:** Added `coverage_policy.py` with `COVERAGE_ELIGIBILITY_SCOPES`, `COVERAGE_SCOPE_TIERS`, `build_coverage_normalized_policy_summary`, and `write_coverage_normalized_policy_report`. Added `run_coverage_normalized_policy_sprint_experiment(...)` and the `--coverage-normalized-policy-sprint` CLI mode. The runner computes athlete-season coverage tiers, retrains the fixed shadow-mode channel models, merges coverage tier onto each alert timeline, filters complete athlete-season trajectories by `all`, `medium_high`, and `high_only` scopes, rebuilds season-local alert episodes, and writes `coverage_normalized_policy.csv`, `coverage_normalized_policy.json`, and `coverage_normalized_policy_report.md`.

**Peterson guardrail:** Coverage controls are applied to complete athlete-season trajectories before season-local alert episodes are rebuilt. The sprint explicitly avoids treating daily measurement rows as independent injury-classification examples.

**Verification:** New TDD tests first failed because `risk_stratification_engine.coverage_policy`, `run_coverage_normalized_policy_sprint_experiment`, and the `--coverage-normalized-policy-sprint` CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 180 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id coverage_normalized_policy_v1 --coverage-normalized-policy-sprint --model-variant l2` completed and wrote the policy artifacts.

**Live results (`coverage_normalized_policy_v1`, L2):**
- Overall recommendation: `continue_research_shadow_mode`.
- No fixed channel remained stable across all three coverage eligibility scopes.
- `severity_14d` was stable only in the full-population scope (`all`: mean capture 33.8%, range 9.1%, mean burden 2.58 episodes per athlete-season), but became unstable under `medium_high` (range 14.0%, burden 3.56) and `high_only` (range 13.0%, burden 4.40).
- `broad_30d`, `severity_7d`, and `subtype_lower_extremity_soft_tissue_30d` were unstable under `all`, `medium_high`, and `high_only`.
- Alert burden increased sharply as eligibility moved toward higher coverage: for example, `severity_7d` mean burden rose from 2.80 (`all`) to 3.91 (`medium_high`) to 4.80 (`high_only`) episodes per athlete-season.

**Interpretation:** The fixed policy package is still not ready for dashboard or pilot escalation. The earlier `severity_14d` improvement is not robust to coverage eligibility controls, so the next research step should test source/coverage-aware modeling or coverage-adjusted thresholds rather than productizing the current channels.

## Previous Completed Step

**Coverage-stratified evaluation** — implemented and verified on 2026-04-29.

**What changed:** Added `coverage_analysis.py` with `build_coverage_tiers`, `build_coverage_stratified_evaluation`, `build_coverage_flag`, and `write_coverage_stratified_evaluation_report`. Added `run_coverage_stratified_evaluation_experiment(...)` to `experiments.py` and the `--coverage-stratified-evaluation` CLI mode. The runner computes population-wide coverage tiers (low/medium/high tertile of measurement days per athlete-season), loops the four fixed shadow-mode channels, joins tiers onto each per-channel model timeline, evaluates alert capture rate and burden by tier and tier×season using a population-wide risk threshold, and writes `coverage_tiers.csv`, `coverage_stratified_evaluation.csv`, `coverage_stratified_evaluation.json`, and `coverage_stratified_evaluation_report.md`.

**Verification:** New TDD tests first failed because `risk_stratification_engine.coverage_analysis`, `run_coverage_stratified_evaluation_experiment`, and the `--coverage-stratified-evaluation` CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 175 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id coverage_stratified_eval_v1 --coverage-stratified-evaluation --model-variant l2` completed and wrote the evaluation artifacts.

**Live results (`coverage_stratified_eval_v1`, L2):**
- Evaluated 954 athlete-seasons; tier distribution: 348 low, 293 medium, 313 high.
- Overall `coverage_flag`: `coverage_confounded`.
- High-coverage tier captured at substantially higher rates than low-coverage tier across all four channels:
  - broad_30d: low 7.9%, medium 11.1%, high 23.9% (pop. threshold 0.1268)
  - severity_7d: low 13.6%, medium 10.0%, high 35.2% (pop. threshold 0.0177)
  - severity_14d: low 13.6%, medium 10.0%, high 38.5% (pop. threshold 0.0318)
  - subtype_lower_extremity_soft_tissue_30d: low 12.0%, medium 10.6%, high 34.4% (pop. threshold 0.0671)
- Mean high-low capture rate difference across channels: ~21 pp, well above the 15 pp confounded threshold.

**Interpretation:** Coverage tier is a major driver of shadow-mode channel performance. Athletes with more measurement days are flagged at 2–3× the rate of low-coverage athletes. This is consistent with the season drift finding: 2025-2026's dominance is substantially explained by measurement density, not pure model signal. The next sprint should test whether model signal survives after controlling for or normalizing by coverage — for example, restricting training/evaluation to athletes above a minimum measurement-day threshold, or adding coverage features as explicit model inputs.

## Previous Completed Step

**Season drift diagnostic** — implemented and verified on 2026-04-29.

**What changed:** Added `season_drift.py`, `run_season_drift_diagnostic_experiment(...)`, and the `--season-drift-diagnostic` CLI mode. The runner reuses the fixed shadow-mode policy package and season-local thresholds, then joins each season's channel capture/burden rows with measurement coverage, source mix, canonical injury counts, detailed injury target counts, time-loss buckets, and simple drift flags. It writes `season_drift_diagnostics.csv`, `season_drift_diagnostics.json`, and `season_drift_diagnostic_report.md`.

**Verification:** New TDD tests first failed because `risk_stratification_engine.season_drift`, `run_season_drift_diagnostic_experiment`, and the `--season-drift-diagnostic` CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 157 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id season_drift_diagnostic_v1 --season-drift-diagnostic --model-variant l2` completed and wrote the drift artifacts.

**Live results (`season_drift_diagnostic_v1`, L2):**
- Evaluated 6 season slices; latest season is 2025-2026.
- 2025-2026 was the highest-capture season for every fixed channel: broad 30d 33.3%, severity 7d 25.5%, severity 14d 36.2%, and lower-extremity soft-tissue 30d 63.3%.
- 2025-2026 also had the strongest coverage: 223 athletes, 186,363 measurement rows, 179 measurement dates, and 4 sources.
- Earlier event seasons had far less coverage: 2022-2023 had 7,484 measurement rows from 1 source, 2023-2024 had 7,295 from 1 source, and 2024-2025 had 30,020 from 2 sources.
- Injury volumes were comparable enough that coverage/source mix is a plausible drift driver: observed event counts were 75 in 2022-2023, 88 in 2023-2024, 74 in 2024-2025, and 85 in 2025-2026.
- 2020-2021 was flagged for low measurement coverage and had no observed/detailed injury events in the live prepared data.

**Interpretation:** The shadow-mode instability is not safe to interpret as pure model failure. The strongest season has much richer and broader monitoring coverage, so the next performance sprint should test coverage/source-aware modeling or source-normalized evaluation before a shadow pilot. The model remains research-only.

## Previous Completed Step

**Shadow-mode policy stability audit** — implemented and verified on 2026-04-29.

**What changed:** Added `shadow_mode.py`, `run_shadow_mode_stability_experiment(...)`, and the `--shadow-mode-stability` CLI mode. The audit freezes the current policy package rather than reselecting targets: `broad_30d` (`exclude_concussion`, window 4, 30d top-5%), `severity_7d` and `severity_14d` (`model_safe_time_loss`, window 4, top-10%), and `subtype_lower_extremity_soft_tissue_30d` (window 2, 30d top-10%). It evaluates each channel by season using season-local percentile thresholds and writes `shadow_mode_stability.csv`, `shadow_mode_stability.json`, and `shadow_mode_stability_report.md`.

**Verification:** New TDD tests first failed because `risk_stratification_engine.shadow_mode`, `run_shadow_mode_stability_experiment`, and the `--shadow-mode-stability` CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 153 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id shadow_mode_stability_v1 --shadow-mode-stability --model-variant l2` completed and wrote the stability artifacts.

**Live results (`shadow_mode_stability_v1`, L2):**
- Evaluated 4 fixed channels across 6 season slices.
- Overall recommendation: `review_before_shadow_pilot`.
- `broad_30d`: unstable; mean capture 14.8%, range 6.1%-33.3% across event seasons, 44 total captured events, mean burden 0.97 episodes per athlete-season.
- `severity_7d`: unstable; mean capture 10.5%, range 0.0%-25.5%, 19 total captured events, mean burden 1.27 episodes per athlete-season.
- `severity_14d`: unstable; mean capture 13.1%, range 0.0%-36.2%, 24 total captured events, mean burden 1.20 episodes per athlete-season.
- `subtype_lower_extremity_soft_tissue_30d`: unstable; mean capture 24.5%, range 4.3%-63.3%, 42 total captured events, mean burden 0.96 episodes per athlete-season.

**Interpretation:** The policy package is not stable enough for a dashboard or pilot escalation yet. Capture is much stronger in 2025-2026 than in earlier event seasons even when thresholds are computed within each season. The next major sprint should explain this season drift: source coverage/freshness, roster overlap, measurement density, injury mix, or true monitoring-pattern change.

## Previous Completed Step

**Three policy decision sprints** — implemented and verified on 2026-04-29.

**What changed:** Added `policy_sprints.py`, `run_policy_decision_sprint_experiment(...)`, and the `--policy-decision-sprint` CLI mode. This executes three linked research sprints: a two-channel alert policy artifact, a policy/window sensitivity artifact, and an operational policy package. The run compares selected target policies across graph windows, writes strict JSON plus Markdown reports, and keeps the final recommendation explicitly in `research_shadow_mode`.

**Verification:** New TDD tests first failed because `risk_stratification_engine.policy_sprints`, `run_policy_decision_sprint_experiment`, and the `--policy-decision-sprint` CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 149 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id policy_decision_sprint_v1 --policy-decision-sprint --policy-window-sizes 2 4 7 --model-variant l2` completed and wrote the three sprint artifacts.

**Live results (`policy_decision_sprint_v1`, L2, windows 2/4/7):**
- Wrote 90 policy/window comparison rows across `any_injury`, `exclude_concussion`, `model_safe_time_loss`, `lower_extremity_soft_tissue`, and `severe_time_loss`.
- Broad 30d early-warning recommendation: `exclude_concussion`, window 4, top-5%; Brier skill 0.0168, top-decile lift 2.46, 44/292 unique observed events captured, median start lead 12 days, and 0.49 episodes per athlete-season.
- Severity short-horizon recommendation: `model_safe_time_loss`, window 4, top-10%; at 7d it captured 32/173 unique events (18.5%) with 3.28 lift and median lead 3 days; at 14d it captured 42/173 (24.3%) with 2.58 lift and median lead 7 days.
- Subtype-review recommendation: `lower_extremity_soft_tissue`, window 2, 30d top-10%; captured 40/168 unique events (23.8%) but with 1.08 episodes per athlete-season and weak Brier skill, so it should remain a review view rather than a primary channel.
- `severe_time_loss` and `concussion_only` are explicitly not recommended as primary targets.

**Interpretation:** The next deployable research posture is not one universal injury alarm. It is a shadow-mode, two-channel policy: window 4 + `exclude_concussion` for broad 30d early warning, and window 4 + `model_safe_time_loss` for 7d/14d severity triage. Lower-extremity soft-tissue can be monitored as a high-burden subtype-review channel. The next sprint should audit shadow-mode stability over refreshes and seasons before any dashboard work.

## Previous Completed Step

**Outcome-policy model comparison sprint** — implemented and verified on 2026-04-29.

**What changed:** Added `build_policy_injury_events(...)`, `policy_event_count(...)`, `run_outcome_policy_model_comparison_experiment(...)`, and a new `--outcome-policy-model-comparison` CLI mode. The runner consumes canonical live measurements/injuries plus `injury_events_detailed.csv`, relabels each athlete-season under candidate injury target policies, retrains the same graph model for each target, evaluates the holdout model metrics, builds top-5%/top-10% alert episodes, and writes `context_policy_model_comparison.csv`, `context_policy_model_comparison.json`, and `context_policy_model_comparison_report.md`. CSV key fields are now normalized as text on load so numeric-looking season IDs do not break joins.

**Verification:** New TDD tests first failed because policy relabeling, the outcome-policy comparison runner, and the CLI dispatch did not exist. A focused test then exposed the season-ID dtype mismatch, which is now covered by canonical text normalization. After implementation, `python -m pytest` collected and passed 144 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id outcome_policy_model_comparison_v1 --outcome-policy-model-comparison --model-variant l2 --graph-window-size 4` completed and wrote the comparison artifacts.

**Live results (`outcome_policy_model_comparison_v1`, L2, window 4):**
- Compared 7 target policies: `any_injury`, `model_safe_time_loss`, `moderate_plus_time_loss`, `severe_time_loss`, `lower_extremity_soft_tissue`, `concussion_only`, and `exclude_concussion`.
- Policy event volumes: any injury 638, model-safe time-loss 256, moderate-plus time-loss 177, severe time-loss 69, lower-extremity soft-tissue 241, concussion-only 79, and exclude-concussion 559.
- At 7d top-10%, `model_safe_time_loss` had the strongest unique-event capture rate: 32/173 observed athlete-season events captured (18.5%), with AUROC 0.667 and top-decile lift 3.28.
- At 14d top-10%, `model_safe_time_loss` again led unique-event capture: 42/173 captured (24.3%), with AUROC 0.662 and top-decile lift 2.58.
- At 30d top-5%, `lower_extremity_soft_tissue` led capture rate at 33/168 (19.6%), but with higher alert burden; `exclude_concussion` and `any_injury` retained the strongest Brier skill around 0.017.
- At 30d top-10%, capture rates clustered tightly: severe time-loss 13/59 (22.0%), lower-extremity soft-tissue 37/168 (22.0%), model-safe time-loss 37/173 (21.4%), and any injury 60/308 (19.5%).

**Interpretation:** This was not a simple "just use severity" win. Cleaner time-loss targets improve 7d/14d alert capture and lift, especially `model_safe_time_loss`, but they reduce event volume and do not improve 30d calibration. `severe_time_loss` is too sparse and noisy as a primary target. The strongest next sprint is an ensemble/policy sprint: keep `any_injury` or `exclude_concussion` as the broad 30d early-warning target, test `model_safe_time_loss` as a secondary severity-oriented alert channel, and audit whether lower-extremity soft-tissue alerts are operationally coherent enough to justify a subtype-specific view.

## Previous Completed Step

**Injury outcome policy and severity semantics audit** — implemented and verified on 2026-04-29.

**What changed:** Added `injury_outcomes.py`, a new `--injury-outcome-policies` CLI mode, and `run_injury_outcome_policy_experiment(...)`. The run consumes `injury_events_detailed.csv` from live-source preparation and writes `injury_severity_audit.csv`, `injury_severity_audit.json`, `injury_severity_audit_report.md`, `outcome_policy_table.csv`, `outcome_policy_summary.json`, and `outcome_policy_report.md`. The severity audit checks missing/negative/extreme time-loss values, duration/resolved-date consistency, time-loss buckets, and event-level severity semantics flags. The policy table defines candidate model targets including any injury, time-loss-only, model-safe time-loss, moderate-plus time-loss, severe time-loss, caused-unavailability, recurrent, lower-extremity, soft-tissue, lower-extremity soft-tissue, concussion-only, and exclude-concussion.

**Verification:** New TDD tests first failed because `risk_stratification_engine.injury_outcomes`, `run_injury_outcome_policy_experiment`, and the `--injury-outcome-policies` CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 141 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id injury_outcome_policy_v1 --injury-outcome-policies` completed and wrote the severity audit plus policy artifacts.

**Live results (`injury_outcome_policy_v1`):**
- Detailed injury events audited: 638.
- Severity semantics flags: 625 usable, 13 `review_extreme_time_loss`, 0 missing time-loss, 0 negative time-loss, and 0 duration/resolved-date mismatches.
- Time-loss buckets: 369 `0d`, 79 `1-7d`, 108 `8-28d`, 69 `29-365d`, and 13 `extreme_366d+`.
- Candidate target volumes: time-loss-only 269 events, model-safe time-loss 256, moderate-plus time-loss 177, severe time-loss 69, caused-unavailability 272, lower-extremity 312, soft-tissue 367, lower-extremity soft-tissue 241, concussion-only 79, and exclude-concussion 559.

**Interpretation:** The severity fields are usable enough to support the next modeling iteration, but the 13 extreme time-loss events should be excluded or audited before training severity-weighted models. The next sprint should run model and alert comparisons across the strongest candidate targets: model-safe time-loss, moderate-plus time-loss, severe time-loss, lower-extremity soft-tissue, concussion-only, and exclude-concussion.

## Previous Completed Step

**Injury context outcome artifacts** — implemented and verified on 2026-04-29.

**What changed:** Added `injury_context.py` and extended the existing `--alert-episodes` runner to consume `injury_events_detailed.csv` when it is available beside live canonical inputs. The alert run now writes `injury_event_context_profiles.csv`, `injury_context_outcomes.csv`, `injury_context_outcomes.json`, and `injury_context_outcome_report.md`. Event profiles compare each detailed injury event against each horizon/threshold alert policy, marking whether an alert episode started, peaked, or ended within the horizon before the injury. Context rows roll those profiles up by injury type, pathology, classification, body area, tissue type, side, recurrence, unavailability, activity group/type, and time-loss bucket.

**Verification:** New TDD tests first failed because `risk_stratification_engine.injury_context` and the alert-run context artifacts did not exist. After implementation, `python -m pytest` collected and passed 137 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id injury_context_outcomes_v1 --alert-episodes --model-variant l2 --graph-window-size 4` completed and wrote the injury-context outcome artifacts.

**Live results (`injury_context_outcomes_v1`, L2, window 4):**
- Event profiles: 3,828 rows for 638 detailed injury events across 7d/14d/30d and top-5%/top-10% policies.
- Grouped context rows: 2,046.
- At 30d top-5%, time-loss bucket capture rates remained low: `0d` 27/369 events (7.3%), `1-7d` 12/79 (15.2%), `8-28d` 9/108 (8.3%), and `29d+` 7/82 (8.5%).
- Low-capture 30d top-5% body-area contexts included lower leg, elbow, neck, thigh, and groin/hip. Low-capture activity contexts included S&C, practice, game, and other groups.
- Low-capture subtype contexts included tendinopathy, bone stress injury, bone contusion, bursitis, arthritis, hamstring strain/tear, and several isolated high time-loss fracture/ligament cases.
- Some time-loss values are extremely large, so severity fields need a semantics/data-quality audit before being used directly as model targets.

**Interpretation:** The context artifact confirms the next performance lever is not just more graph tuning. The current alert policy misses recognizable injury categories and severity buckets, including some high time-loss events. The next sprint should audit time-loss semantics and then test context-aware event targets/features: subtype-specific outcomes, time-loss-weighted events, recurrence/unavailability indicators, and training/game/S&C context.

## Previous Completed Step

**Detailed injury event enrichment ingestion** — implemented and verified on 2026-04-29.

**What changed:** Live-source injury ingestion now discovers all sibling `injuries-summary-export-*.csv` files when `injury_csv` points at one of those exports, de-duplicates exact raw rows, and writes `injury_events_detailed.csv` beside `canonical_measurements.csv` and `canonical_injuries.csv`. The detailed artifact keeps one de-identified row per injury event with hashed `athlete_id`, stable `injury_event_id`, season, issue/entry/resolved dates, injury type, pathology, classification, body area, tissue type, side, recurrence, unavailability, activity context, participation/training/game context, duration/time-loss/availability-day fields, ICD/code fields, source file, and source row number. The existing `canonical_injuries.csv` modeling contract is unchanged and still uses the earliest injury issue date per athlete-season.

**Verification:** New TDD tests first failed because `build_detailed_injury_event_rows`, `LiveSourcePreparationResult.detailed_injuries_path`, and the sibling-export preparation behavior did not exist. After implementation, `python -m pytest` collected and passed 136 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id injury_event_enrichment_v1 --alert-episodes --model-variant l2 --graph-window-size 4` completed and wrote the enriched live inputs plus alert episode artifacts.

**Live results (`injury_event_enrichment_v1`, L2, window 4):**
- Injury source files loaded: `injuries-summary-export-306ff9.csv`, `injuries-summary-export-3ad17d.csv`, and `injuries-summary-export-e433b3.csv`.
- Detailed injury events: 638 rows, spanning 2022-09-01 through 2026-04-23.
- Canonical athlete-season injury label rows: 954, with 322 observed events and 300 primary modelable events.
- Event-window quality counts: 632 censored, 300 modelable, 8 low-confidence, 14 out-of-window, and 0 no-measurements.
- Rich injury field coverage was high for model-relevant context: pathology 638/638, time-loss days 638/638, recurrence 638/638, caused unavailability 638/638, activity 637/638, classification 600/638, body area 599/638, and duration 583/638.
- Alert episode output shifted with the expanded injury history: 4,246 total episodes; at 30d top-5%, 392 episodes captured 76 events after episode start, 87 after peak, and 100 after episode end.

**Interpretation:** The data richness bottleneck is now less about missing injury column headings and more about using the preserved context in modeling. The next sprint should build severity/subtype/context-aware outcome artifacts from `injury_events_detailed.csv`, then test whether time-loss, recurrence, body area, activity context, and injury subtype explain noisy alerts or missed events better than graph dynamics alone.

## Previous Completed Step

**Model Improvement Diagnostic Table v1** — implemented and verified on 2026-04-28.

**What changed:** Added `model_diagnostics.py` and extended the existing `--alert-episodes` runner with `model_improvement_diagnostics.csv`, `model_improvement_diagnostics.json`, and `model_improvement_diagnostic_report.md`. The diagnostic table compares useful alerts, noisy alerts, and missed observed injury events for every horizon/threshold pair, then reports risk summaries, pre-event risk summaries for missed events, elevated z-score rates, top feature counts, event-window quality counts, measurement-gap summaries, and a concise recommended next action.

**Verification:** New TDD tests first failed because `risk_stratification_engine.model_diagnostics` did not exist. The alert episode experiment test then failed because the runner did not write model-improvement artifacts. After implementation, targeted tests passed, and `python -m pytest` collected and passed 134 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id model_improvement_diagnostics_v1 --alert-episodes --model-variant l2 --graph-window-size 4` completed and wrote the diagnostic artifacts.

**Live results (`model_improvement_diagnostics_v1`, L2, window 4):**
- 18 diagnostic rows were written: true-positive episodes, false-positive episodes, and missed events for each 7d/14d/30d horizon and top-5%/top-10% threshold.
- Recommended action rows: 6 `retain_policy_signal`, 6 `add_context_features`, and 6 `review_threshold_policy`.
- At the current headline policy, 30d top-5%, the table reported 130 true-positive episodes, 490 false-positive episodes, and 133 missed events.
- True-positive and false-positive 30d top-5% episodes had nearly identical median peak risk (0.136 vs 0.137) and similarly high elevated z-feature rates (82.3% vs 79.4%), reinforcing that current graph-risk features alone do not separate useful warnings from noisy alerts.
- Missed 30d top-5% events were mostly modelable (129 of 133), with median pre-event snapshot count 7, median pre-event risk 0.034, and maximum pre-event risk 0.701. This suggests a mixed issue: many missed events stay low-risk, but some high-risk missed events need threshold or episode-policy review.

**Interpretation:** The performance ceiling is not just a data problem. Better injury labels and measurement coverage still matter, but the strongest next improvement lever is adding context that separates managed-risk/adaptation from harmful risk and adding event-specific features for injuries whose pre-event graph profile stays low. Keep L2 + window 4 + 30d top-5% as the primary early-warning policy while the next sprint tests context features.

## Previous Completed Step

**Qualitative Case Review + Data Diagnostic v1** — implemented and verified on 2026-04-28.

**What changed:** Added `case_review.py` and extended the existing `--alert-episodes` runner with `qualitative_case_review.json` and `qualitative_case_review_report.md`. The case-review artifact samples deterministic true-positive, false-positive, missed-injury, and high intra-individual deviation cases for each horizon/threshold pair, then attaches compact athlete-season timeline context, model drivers, elevated z-score features, event-window quality, nearest-measurement gap, and a simple diagnostic label.

**Verification:** New TDD tests first failed because `risk_stratification_engine.case_review` did not exist. The experiment integration test then failed because the alert episode runner did not write case-review artifacts. A live-run regression exposed date-format mismatch between missed-injury case references and timeline event dates; the regression now verifies that `YYYY-MM-DD` and `YYYY-MM-DD 00:00:00` match. After implementation, `python -m pytest` collected and passed 130 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id qualitative_case_review_v1 --alert-episodes --model-variant l2 --graph-window-size 4` completed and wrote the case-review artifacts.

**Live results (`qualitative_case_review_v1`, L2, window 4):**
- 24 deterministic review cases were written across the 7d/14d/30d horizons and top-5%/top-10% thresholds.
- Diagnostic counts: 6 `model_signal_supported`, 6 `model_miss`, 6 `missing_context_or_managed_risk`, and 6 `explanation_gap`.
- The model-supported true-positive cases had modelable event-window quality and near-event measurement coverage, confirming the pipeline can surface coherent early-warning examples.
- The missed-injury examples included a modelable fracture case that was not captured by the selected alert policy, so current performance limits are not only a data-quality problem.
- The false-positive and high-deviation examples reinforce the missing-context/explanation gap: high-risk physiology-like patterns are visible, but current artifacts cannot tell whether they were managed-risk periods, benign adaptation, or genuine noise.

**Interpretation:** Improving performance likely needs both data/context work and model refinement. Better injury labels and richer exposure/availability context remain high-value, but the case review also shows true model misses on modelable events. The next sprint should turn case-review findings into a model-improvement diagnostic table: missed-event feature profiles, false-positive context requirements, and candidate features for exposure, availability, injury subtype, and event-severity stratification.

## Previous Completed Step

**Episode Quality Audit v1** — implemented and verified on 2026-04-28.

**What changed:** Added `episode_quality.py` and extended the existing `--alert-episodes` runner with second-pass audit artifacts: `alert_episode_quality.csv`, `alert_episode_quality.json`, and `alert_episode_quality_report.md`. The audit keeps alert episode construction separate from quality analysis, then reports start-based true-positive episodes, false-positive episodes, unique observed injury events captured, missed observed events, alert burden per athlete-season, median lead times, top-5%/top-10% overlap, true-positive vs false-positive explanation summaries, and deterministic representative cases.

**Verification:** New TDD tests first failed because `risk_stratification_engine.episode_quality` did not exist. The experiment integration test then failed because the alert episode runner did not write quality artifacts. After implementation, `python -m pytest` collected and passed 127 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id alert_episode_quality_v1 --alert-episodes --model-variant l2 --graph-window-size 4` completed and wrote the new quality artifacts.

**Live results (`alert_episode_quality_v1`, L2, window 4, 349 athletes, 902 athlete-seasons, 39,189 snapshots, 188 unique observed events):**
- 30d top-5% remains the strongest policy: 620 episodes, 130 start-based true-positive episodes, 490 false-positive episodes, 55/188 unique observed events captured (29.3%), 133 missed observed events, 0.69 episodes per athlete-season, and median start lead time of 11 days.
- 30d top-10% increased burden without improving unique-event capture: 689 episodes, 106 true-positive episodes, 583 false-positive episodes, 54/188 unique events captured (28.7%), 134 missed events, 0.76 episodes per athlete-season, and median start lead time of 12.5 days.
- 14d top-5% captured 47/188 unique events (25.0%) with 657 episodes and 76 true-positive episodes; 7d top-5% captured 33/188 unique events (17.6%) with 634 episodes and 38 true-positive episodes.
- Top-5% and top-10% episode overlap was low by exact episode identity: 13.9% of 30d top-5% episodes overlapped top-10%, 16.0% at 14d, and 21.0% at 7d. This means threshold choice changes episode boundaries materially, not only alert volume.
- True-positive and false-positive episodes had similar median peak risk and similar elevated z-feature rates. At 30d top-5%, TP median peak risk was 0.136 vs FP 0.137, and elevated z-feature rates were 82.3% vs 79.4%.

**Interpretation:** The audit supports 30d top-5% as the current default early-warning policy because it captures slightly more unique observed events with less alert burden than 30d top-10%. It also exposes the main limitation: peak risk and current z-score flags do not cleanly separate useful warnings from noisy episodes. The next Peterson-aligned step should be qualitative case review and explanation refinement, not dashboard build-out.

## Previous Completed Step

**Alert episode validation artifacts** — implemented and verified on 2026-04-28.

**What changed:** Added `alert_episodes.py` and a new CLI run mode, `--alert-episodes`, for converting snapshot-level risk scores into contiguous athlete-season alert episodes. The runner trains the selected model variant, adds model contribution and intra-individual z-score explanation context, applies top-5% and top-10% percentile thresholds at the 7d/14d/30d horizons, collapses contiguous alert snapshots, and writes `alert_episodes.csv`, `alert_episodes.json`, `alert_episode_summary.json`, and `alert_episode_report.md`. Censored athlete-seasons now keep event timing fields empty instead of treating censoring dates as injuries.

**Verification:** New TDD tests first failed because `risk_stratification_engine.alert_episodes`, `run_alert_episode_experiment`, and the `--alert-episodes` CLI dispatch did not exist. A regression test also first failed because censored episodes exposed censoring distance as `days_from_*_to_event`. After implementation, `python -m pytest` collected and passed 124 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id alert_episode_validation_v1 --alert-episodes --model-variant l2 --graph-window-size 4` completed and wrote the alert episode artifacts.

**Live results (`alert_episode_validation_v1`, L2, window 4, 349 athletes, 39,189 snapshots):**
- Total episodes across horizons and top-5%/top-10% thresholds: 4,268.
- 7d: top-5% produced 634 episodes, with 38 start-captured, 47 peak-captured, and 57 end-captured events; top-10% produced 844 episodes, with 38/59/77 captured at start/peak/end.
- 14d: top-5% produced 657 episodes, with 76/83/90 captured at start/peak/end; top-10% produced 824 episodes, with 71/94/108 captured.
- 30d: top-5% produced 620 episodes, with 130/137/150 captured at start/peak/end; top-10% produced 689 episodes, with 106/126/145 captured.
- Median episode lengths remained short: top-5% episodes were 2 snapshots across horizons, while top-10% episodes were 3 snapshots at 7d/14d and 4 snapshots at 30d.

**Interpretation:** The model is producing coherent short alert episodes rather than only isolated snapshot spikes. The strongest episode-level evidence is at 30d top-5%, where roughly one in five episodes begins within the forecast horizon of an observed event and roughly one in four ends within that horizon. The 7d episode signal is weaker, so the current candidate looks more defensible as an early-warning workbench than as a same-week injury alarm.

## Previous Completed Step

**Explicit intra-individual deviation explanations** — implemented and verified on 2026-04-28.

**What changed:** `athlete_explanations.json` now surfaces Peterson-style own-baseline departures directly. Each snapshot includes an `intra_individual_deviations` block for `z_mean_abs_correlation`, `z_edge_density`, `z_edge_count`, and `z_graph_instability`, with the current z-score value, `elevated = abs(value) > 2.0`, and signed risk contributions for the 7d, 14d, and 30d horizons. Each athlete-season also includes `peak_intra_individual_deviation`, the snapshot with the highest combined absolute z-score signal, its flagged features, and its ranked deviation details.

**Verification:** 2 new tests first failed because `intra_individual_deviations` and `peak_intra_individual_deviation` did not exist. After implementation, `python -m pytest` collected and passed 117 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id intra_individual_explanations_v1 --model-variant l2 --graph-window-size 4` completed and wrote the updated `athlete_explanations.json`.

**Live results (`intra_individual_explanations_v1`, L2, window 4, 902 athlete-seasons, 39,189 snapshots):**
- 3,529 snapshots had at least one elevated intra-individual z-score feature (`abs(z) > 2.0`).
- Elevated snapshot counts by feature: `z_graph_instability` 2,092; `z_mean_abs_correlation` 2,028; `z_edge_density` 1,989; `z_edge_count` 1,974.
- Season-level peak deviation summaries flagged all four z-score features across the cohort: `z_mean_abs_correlation` 175 athlete-seasons, `z_edge_density` 179, `z_edge_count` 179, and `z_graph_instability` 163.

**Interpretation:** The explanation layer now separates two ideas that were previously blended: population-relative model contribution and athlete-relative departure from that athlete's own rolling graph baseline. This makes the artifact more Peterson-true and easier to inspect for high-sensitivity cases where the model's top overall driver is still a population-level graph feature, but the athlete is also showing a meaningful intra-individual deviation.

## Previous Completed Step

**Per-athlete feature contribution explanations** — implemented and verified on 2026-04-28.

**What changed:** Replaced the hard-coded `_primary_signal` heuristic (based on `mean_abs_correlation >= 0.7`) with model-informed per-snapshot feature contributions. New `_compute_snapshot_contributions(row, feature_attribution, feature_columns)` computes `contribution_k = standardized_coefficient_k × (value_k − train_mean_k) / train_std_k` for each feature. `_explanation_summary` now produces `top_feature_{h}d` and `top_contribution_{h}d` columns per horizon. New `_athlete_explanations(timeline, model_summary)` produces `athlete_explanations.json`: per-athlete-season peak risk per horizon, top-3 dominant features averaged over the season, and per-snapshot top-3 signed feature contributions. The `primary_signal` column is removed.

**Verification:** 6 new tests first failed because `_compute_snapshot_contributions` and the new artifact structure did not exist. After implementation, `python -m pytest` collected and passed 115 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id athlete_explanations_v1 --model-variant l2 --graph-window-size 4` completed and wrote `explanation_summary.csv` and `athlete_explanations.json`.

**Live results (`athlete_explanations_v1`, L2, window 4, 902 athlete-seasons, 39,189 snapshots):**
- `mean_abs_correlation` is the dominant risk driver in 99% of snapshots at 7d (38,760/39,189), followed by `delta_mean_abs_correlation` and `graph_instability`. This is consistent with the model-level standardized coefficient rankings from feature attribution.
- Dominant features over entire athlete-seasons at both 7d and 30d: `mean_abs_correlation`, `edge_density`, `edge_count`. The z-score features appear infrequently as the top contributor (confirming they are useful ranking modifiers rather than primary signal sources).
- Contributions are signed: `mean_abs_correlation` typically contributes positively (+0.6), `edge_density` negatively (-0.4) for high-risk snapshots — reflecting that high correlation and low edge density together characterize elevated risk.

**Interpretation:** This is the first artifact that directly answers Peterson's core question: "why is this athlete high-risk right now?" The contribution decomposition ties each snapshot's risk score back to specific graph structural changes relative to the population baseline. The z-score features (which capture intra-individual departure from each athlete's own baseline) appear as top drivers for a small but meaningful subset of snapshots — exactly the high-sensitivity cases Peterson's methodology is designed to detect. The next Peterson-true step is to surface intra-individual z-score signals more directly in the explanation artifact, and eventually to move from population-relative to athlete-relative baselines in the explanation layer.

## Previous Completed Step

**Calibration and threshold tables** — implemented and verified on 2026-04-28.

**What changed:** New `calibration.py` module provides `build_calibration_bins` (equal-count quantile binning) and `build_threshold_table` (percentile and fixed-probability thresholds). `run_calibration_threshold_experiment(...)` uses rotating out-of-fold splits so every athlete-season snapshot is scored by a model that did not train on that athlete. CLI: `--calibration-thresholds` (reuses `--model-variant`, `--graph-window-size`, `--stability-splits`). Artifacts: `calibration_summary.json` (OOF stats, calibration bins, and threshold rows per horizon), `threshold_table.csv`, and `calibration_report.md`.

**Verification:** 16 new tests first failed because `calibration.py`, `run_calibration_threshold_experiment`, and the CLI flag did not exist. After implementation, `python -m pytest` collected and passed 109 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id calibration_threshold_v1 --calibration-thresholds --model-variant l2 --graph-window-size 4 --stability-splits 5` completed and wrote calibration artifacts.

**Live results (`calibration_threshold_v1`, L2, window 4, 5 OOF splits, 39,189 OOF snapshots):**
- 7d: 601 positives (1.5%), Brier 0.015, Brier skill 0.003. Top 10%: 3,919 alerts, 36% recall, 5.5% PPV, 3.6x lift. Top 5%: 1,959 alerts, 16% recall, 5.1% PPV, 3.3x lift.
- 14d: 1,046 positives (2.7%), Brier 0.026, Brier skill 0.006. Top 10%: 3,919 alerts, 38% recall, 10.3% PPV, 3.8x lift. Top 5%: 1,959 alerts, 17% recall, 8.9% PPV, 3.3x lift.
- 30d: 1,932 positives (4.9%), Brier 0.046, Brier skill 0.019. Top 10%: 3,919 alerts, 38% recall, 18.6% PPV, 3.8x lift. Top 5%: 1,959 alerts, 18% recall, 17.3% PPV, 3.5x lift.

**Interpretation:** Fixed probability thresholds (0.10/0.20/0.30) are nearly useless at 7d because L2 predictions concentrate in a narrow low-risk band, leaving almost no snapshots above any fixed threshold. Percentile-based thresholds are the correct operating mode at short horizons: top-10% reliably delivers 3.5-4x lift. At 30d the model is more spread, and a probability threshold of 0.10 gives comparable lift. Calibration bins confirm the model separates risk well in the upper two deciles but is flat across the lower eight, making the score most useful as a ranking signal. Brier skill is modest (0.003-0.019); the 3-4x top-decile lift is the operationally relevant metric.

**Window 7 comparison (`calibration_threshold_w7_v1`, L2, window 7, 5 OOF splits, 39,189 OOF snapshots):**
- 7d: Brier skill 0.003, top-10% lift 3.2x (vs W4: 3.6x)
- 14d: Brier skill 0.008, top-10% lift 3.3x (vs W4: 0.006, 3.8x)
- 30d: Brier skill 0.018, top-10% lift 3.6x (vs W4: 0.019, 3.8x)
- Fixed probability thresholds even more useless at W7: prob 0.20 at 7d gives 2 alerts (vs W4's 63). W7 is more prediction-concentrated, not less.
- W7 beats W4 only at 14d Brier skill (0.008 vs 0.006). W4 wins all lift metrics at all horizons.
- Conclusion: the robustness sprint's Brier calibration win for W7 does not translate into a better operational threshold profile. W4+L2 is the right primary candidate.

## Previous Completed Step

**Window/model robustness sprint** — implemented and verified on 2026-04-27.

**What changed:** Main research runs now accept a `model_variant` parameter, and the CLI exposes it as `--model-variant`. Pairing `--model-robustness-sprint` with `--window-sensitivity-sizes` now runs a combined window/model robustness sprint across graph windows and model variants, writing `window_model_robustness.json` plus `window_model_robustness_report.md`.

**Verification:** New tests first failed because the main-path model variant, combined runner, and CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 93 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id window_model_robustness_v1 --model-robustness-sprint --window-sensitivity-sizes 2 4 7 --stability-splits 5` completed and wrote the combined robustness artifacts.

**Interpretation:** L2 remains the best production-candidate default for calibrated probabilities because it won the calibration decision mode at all horizons when paired with its best window. The shorter window 2 remains best for high-alert triage because it concentrates positives in the top decile.

## Earlier Completed Step

**Model robustness sprint** — implemented and verified on 2026-04-27.

**What changed:** `train_discrete_time_risk_model(...)` now supports explicit athlete holdout IDs and model variants: `baseline`, `l2`, `l1`, and `elasticnet`. Non-baseline regularized variants standardize features for fitting. `run_model_robustness_experiment(...)` compares variants across deterministic rotating athlete-level holdout splits and writes `model_robustness.json` plus `model_robustness_report.md`. The CLI supports this through `--model-robustness-sprint`.

**Verification:** New tests first failed because explicit split control, model variants, the robustness runner, and CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 90 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id model_robustness_sprint_v1 --model-robustness-sprint --graph-window-size 4 --stability-splits 5` completed and wrote the robustness artifacts.

**Interpretation:** Regularization is now the clearest next modeling upgrade. L2 looks like the pragmatic default candidate because it improved all average metrics versus baseline and won most decision-mode comparisons. Elastic net remains worth tracking for 30d ranking/triage. The next sprint should promote a chosen regularized variant into the primary experiment path or run the same robustness sprint across windows 2/4/7 to separate regularization effects from window effects.

## Previous Completed Step

**Graph window-size sensitivity artifacts** — implemented and verified on 2026-04-27.

**What changed:** `run_window_sensitivity_experiment(...)` now compares multiple graph window sizes over the same prepared inputs and writes `window_sensitivity.json` plus `window_sensitivity_report.md`. The CLI supports this with `--window-sensitivity-sizes`, including live-source runs.

**Verification:** New tests first failed because the window-sensitivity runner and CLI dispatch did not exist. After implementation, `python -m pytest` collected and passed 86 tests. The live command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id window_sensitivity_v1 --window-sensitivity-sizes 2 3 4 5 7` completed and wrote the sensitivity artifacts.

**Interpretation:** No single window dominates every criterion. Keep window 4 as the headline AUROC/ranking baseline, investigate window 7 for better Brier skill/calibration, and treat window 2 as a possible high-alert triage setting because it produces the strongest top-decile lift while hurting Brier skill.

## Previous Completed Step

**Feature attribution and ablation artifacts** — implemented and verified on 2026-04-27.

**What changed:** `train_discrete_time_risk_model(...)` now accepts explicit feature subsets and records standardized coefficient attribution for each horizon. `run_research_experiment(...)` now writes `feature_attribution.json` and `feature_ablation_report.md` for three feature sets: `full_13`, `original_9`, and `z_score_only`.

**Verification:** New tests first failed against the old behavior because feature subset training, coefficient attribution, and the artifact files did not exist. After implementation, `python -m pytest` collected and passed 84 tests. The live experiment command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id feature_attribution_ablation_v1` completed and wrote the new attribution/ablation artifacts.

**Interpretation:** `edge_count` remains the dominant standardized coefficient in the combined and original feature sets across horizons. The z-score-only model is not enough by itself, especially at 30d, but z-score features improve lift in the combined model. The logical next research step is window-size sensitivity or regularized feature selection, not adding more raw features yet.

## Earlier Completed Step

**Intra-individual z-score deviation features** — implemented and verified on 2026-04-27.

Full spec: `docs/superpowers/specs/2026-04-27-intra-individual-deviation-design.md`

**Why:** The prior 9-feature model used only absolute graph values (population-level). Peterson's methodology is fundamentally intra-individual: risk emerges from departure of an athlete's own dynamic baseline, not from population position. Z-score features are the direct implementation of that philosophy.

**Added features (4 new features, 13 total):**

| New column | Source feature | Logic |
|---|---|---|
| `z_mean_abs_correlation` | `mean_abs_correlation` | z-score vs. athlete's own rolling baseline |
| `z_edge_density` | `edge_density` | same |
| `z_edge_count` | `edge_count` | same |
| `z_graph_instability` | `graph_instability` | same |

**Computation rules (per athlete-season, chronological):**
- Baseline window: `group_rows[max(0, i − window_size + 1) : i]` — strictly prior, no current row
- Minimum 2 prior snapshots required; else z-score = 0.0
- std = 0 → z-score = 0.0
- Clip to `[-10.0, 10.0]`, round to 6 dp
- Use population std (ddof=0), same as `graph_instability`

**Changed files:** `graphs.py` (OUTPUT_COLUMNS + `_add_temporal_features`), `models.py` (GRAPH_SNAPSHOT_FEATURE_COLUMNS), `tests/test_graphs.py` (6 new TDD tests), `tests/test_models.py` (fixture columns), `tests/test_experiments.py` (feature_columns assertion), `AGENTS.md`, and `README.md`.

**Verification:** The 6 required graph tests were written first and failed against the old schema. After implementation, `python -m pytest` collected and passed 82 tests. The live experiment command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id intra_individual_deviation_v1` completed and wrote `outputs/experiments/intra_individual_deviation_v1/model_evaluation.json`.

**6 added tests:**
1. `test_build_graph_snapshots_includes_z_score_feature_columns` — all 4 columns present
2. `test_build_graph_snapshots_z_scores_are_zero_at_first_snapshot` — no prior history
3. `test_build_graph_snapshots_z_scores_are_zero_at_second_snapshot` — only 1 prior (below minimum-2 threshold)
4. `test_build_graph_snapshots_z_score_nonzero_once_baseline_has_two_prior_snapshots` — known z-score from 3-row synthetic fixture
5. `test_build_graph_snapshots_z_score_is_zero_when_baseline_std_is_zero` — std-zero fallback
6. `test_build_graph_snapshots_z_score_clips_extreme_departures` — clip to ±10.0

**Live comparison:** `intra_individual_deviation_v1` improved 30d AUROC from 0.735 to 0.736, 30d Brier skill from 0.0168 to 0.0171, and top-decile lift from 4.34 to 4.48. It also improved 7d Brier skill from 0.0017 to 0.0020 and top-decile lifts at all horizons. The 7d and 14d AUROC values were slightly lower than `enriched_graph_features_v1`.

## Engineering Preferences

- Keep the first implementation modular and research-friendly.
- Favor explicit experiment configuration and reproducible artifacts over hidden notebook state.
- Add dashboard-facing outputs later, after the research pipeline can produce reliable risk timelines and explanations.
- After every major change, commit the intended repo changes and push them to GitHub.
