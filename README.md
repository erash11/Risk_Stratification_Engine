# Risk Stratification Engine

Research prototype for a Peterson-inspired athlete risk stratification pipeline.

The first implementation models athlete-seasons as longitudinal trajectories, builds athlete-specific graph snapshots, assembles time-to-event labels with censoring, trains a discrete-time logistic baseline, and writes reproducible experiment artifacts.

## Quickstart

```bash
python -m pip install -e ".[dev]"
pytest
```

## Run Fixture Experiment

```bash
risk-engine \
  --measurements tests/fixtures/measurements.csv \
  --injuries tests/fixtures/injuries.csv \
  --output-dir outputs \
  --experiment-id fixture_run \
  --graph-window-size 2
```

## Local Data Paths

Real data sources should stay in their canonical project locations. Copy
`config/paths.example.yaml` to `config/paths.local.yaml`, update it for your
machine, and keep the local file uncommitted.

The expected sources are:

- `forceplate_db`
- `gps_db`
- `bodyweight_csv`
- `perch_db`
- `injury_csv`
- `exposure_dir` (optional, for cleaning football exposure exports)

When `injury_csv` points at a file named `injuries-summary-export-*.csv`, live
preparation loads all sibling injury summary exports with that pattern from the
same folder. This supports period-sliced injury exports without manually
concatenating them.

## Run Exposure Cleaning Audit

The exposure-data integration starts with a cleaning/audit pass, not a model
validation sprint. The source folder should contain `athletes.csv`, `squads.csv`,
`training_sessions.csv`, `training_session_participations.csv`, `games.csv`, and
`game_participations.csv`.

```bash
risk-engine \
  --exposure-cleaning-audit \
  --exposure-dir C:/Users/eric_rash/Desktop/DEV/Football/Baylor_Exposure_Data \
  --output-dir outputs \
  --experiment-id exposure_cleaning_audit_v1
```

This writes `exposure_events.csv`, `exposure_participations.csv`, and
`exposure_cleaning_audit.json` under
`outputs/exposure_inputs/<experiment-id>/`. The cleaner filters events to
football with `ExternalSquadId == 94`, maps athlete identities through
`stable_athlete_id(FirstName + " " + LastName)`, excludes API/performance-source
sessions such as Perch, ForceDecks, VALD, SmartSpeed, and Catapult, and keeps
plain human-entered football training, practice, weight-room, conditioning,
scrimmage, walkthrough, RTP, and game exposure rows for audit review.

## Run Exposure Feature Requirements Sprint

Before exposure/load features are attached to graph snapshots, run the feature
requirements sprint against the cleaned exposure artifacts:

```bash
risk-engine \
  --exposure-feature-requirements-sprint \
  --exposure-events outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_events.csv \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --exposure-audit outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_cleaning_audit.json \
  --output-dir outputs \
  --experiment-id exposure_feature_requirements_v1
```

This writes `exposure_category_summary.csv`, `exposure_duration_summary.csv`,
`exposure_feature_requirements.csv`, `exposure_feature_requirements.json`, and
`exposure_feature_requirements_report.md` under
`outputs/experiments/<experiment-id>/`. The sprint separates count/status
features that are ready for first-pass time-safe modeling from minute-load
features that need duration-completeness review.

The live `exposure_feature_requirements_v1` run recommended
`proceed_with_count_and_status_features_first`. It reviewed 1,055 retained
events and 108,587 participation rows. Session counts, participation status, and
coarse category features were ready; duration-load and game-exposure minutes
were caution domains because game participation duration was missing in 47.09%
of rows. The next exposure modeling sprint should attach time-safe count,
recency, participation-status, and coarse-category features first.

## Run Exposure Load Feature Sprint

The exposure load feature sprint attaches conservative, time-safe exposure
context to graph snapshots and compares it against the coverage/source reference
model:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --exposure-load-feature-sprint \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --output-dir outputs \
  --experiment-id exposure_load_feature_v1 \
  --model-variant l2 \
  --graph-window-size 4
```

This writes `exposure_load_features.csv`,
`exposure_load_model_comparison.csv`,
`exposure_load_model_comparison.json`, and
`exposure_load_model_comparison_report.md` under
`outputs/experiments/<experiment-id>/`. The first-pass feature set includes
prior 7/14/28 day training-session counts, season-to-date game count and game
recency, prior 28 day full/modified/no-participation counts, days since the last
modified or no-participation session, and coarse 28 day practice/lift/
conditioning/RTP/game counts. Minute-load terms remain out of the first-pass
model until duration semantics are reviewed.

The live `exposure_load_feature_v1` run recommended
`continue_exposure_load_research` while keeping production readiness at
`not_ready_research_validation_required`. The exposure-load feature set beat the
coverage/source reference at 7d, 14d, and 30d: AUROC improved from
0.728/0.732/0.743 to 0.802/0.810/0.819, Brier skill improved from
0.017/0.031/0.059 to 0.037/0.064/0.119, and top-decile lift improved from
1.96/1.97/1.99 to 3.20/3.00/2.78. The next validation step is a
season-forward exposure-load sprint before adding duration/minute-load terms.

## Run Exposure Load Season-Forward Validation Sprint

The exposure-load season-forward validation sprint tests whether the conservative
count, recency, participation-status, and category-count features survive the
train-prior-seasons / evaluate-later-seasons setup:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --exposure-load-season-forward-validation \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --output-dir outputs \
  --experiment-id exposure_load_season_forward_validation_v1 \
  --model-variant l2 \
  --graph-window-size 4
```

This writes `exposure_load_features.csv`,
`exposure_load_season_forward_validation.csv`,
`exposure_load_season_forward_validation.json`, and
`exposure_load_season_forward_validation_report.md`.

The live `exposure_load_season_forward_validation_v1` run recommended
`continue_season_forward_research`. `graph_plus_coverage_exposure_load` won all
selected best forward ranking, calibration, and burden-triage slots, with
2025-2026 best AUROC/Brier skill of 0.682/0.016 at 7d, 0.699/0.035 at 14d, and
0.719/0.074 at 30d. The main caution is calibration instability: in 2024-2025,
Brier skill worsened to -0.682/-0.691/-0.567 at 7d/14d/30d. This remains
research-only; the next step is an exposure-load forward diagnostic before
adding duration or minute-load terms.

## Run Exposure Load Forward Diagnostic Sprint

The exposure-load forward diagnostic sprint consumes a completed
`exposure_load_season_forward_validation.csv` artifact and diagnoses which
season/horizon rows show exposure-load ranking or triage gains with calibration
loss:

```bash
risk-engine \
  --exposure-load-forward-diagnostic-sprint \
  --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv \
  --output-dir outputs \
  --experiment-id exposure_load_forward_diagnostic_v1
```

This writes `exposure_load_season_forward_validation.csv`,
`exposure_load_calibration_diagnostics.csv`,
`exposure_load_forward_diagnostic_cases.csv`,
`exposure_load_forward_diagnostic.json`, and
`exposure_load_forward_diagnostic_report.md`.

The live `exposure_load_forward_diagnostic_v1` run recommended
`inspect_exposure_load_forward_failure_modes`. The diagnostic found three
high-priority `ranking_triage_gain_calibration_loss` rows, all in 2024-2025.
Exposure-load improved AUROC by +0.044/+0.052/+0.058 and top-decile lift by
+0.157/+0.571/+0.949 at 7d/14d/30d, but Brier skill worsened by
-0.655/-0.663/-0.544. Mean predicted risk was 9.3% / 13.4% / 20.2% against
observed positive rates of 2.8% / 4.8% / 8.9%, so the next modeling work should
inspect 2024-2025 exposure-load over-sharpening before adding duration or
minute-load terms.

## Run Exposure Load Failure Mode Sprint

The exposure-load failure-mode sprint compares the failed forward season against
calibration-supported comparator seasons by exposure feature/domain:

```bash
risk-engine \
  --exposure-load-failure-mode-sprint \
  --exposure-load-features outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_features.csv \
  --exposure-load-diagnostics outputs/experiments/exposure_load_forward_diagnostic_v1/exposure_load_calibration_diagnostics.csv \
  --output-dir outputs \
  --experiment-id exposure_load_failure_modes_v1
```

This writes `exposure_load_failure_mode_features.csv`,
`exposure_load_failure_mode_domains.csv`, `exposure_load_failure_modes.json`,
and `exposure_load_failure_mode_report.md`.

The live `exposure_load_failure_modes_v1` run recommended
`inspect_exposure_feature_shift_drivers`. The failure season was 2024-2025, with
2023-2024 and 2025-2026 as calibration-supported comparators. The largest shifted
drivers were reduced 28d lift sessions, elevated prior game count, reduced 28d
modified-participation count, and more days since last modified/no-participation
session. The top shifted domains were `category_specific_load`, `game_exposure`,
and `participation_status`.

## Run Exposure Load Guardrail Policy Sprint

The exposure-load guardrail policy sprint converts the forward diagnostic and
failure-mode artifacts into research operating guardrails:

```bash
risk-engine \
  --exposure-load-guardrail-policy-sprint \
  --exposure-load-failure-modes outputs/experiments/exposure_load_failure_modes_v1/exposure_load_failure_modes.json \
  --exposure-load-diagnostics outputs/experiments/exposure_load_forward_diagnostic_v1/exposure_load_calibration_diagnostics.csv \
  --output-dir outputs \
  --experiment-id exposure_load_guardrail_policy_v1
```

This writes `exposure_load_guardrail_policy.csv`,
`exposure_load_guardrail_policy.json`, and
`exposure_load_guardrail_policy_report.md`.

The live `exposure_load_guardrail_policy_v1` run recommended
`use_exposure_load_for_shadow_ranking_only` with production readiness
`not_ready_for_probability_or_pilot`. Probability calibration remains blocked
until the 2024-2025 failure mode is resolved, ranking/triage use is limited to
shadow research with calibration monitoring, minute-load expansion is deferred,
and shifted feature-domain review is required before the next model expansion.

## Run Exposure Load Shift Context Sprint

The exposure-load shift context sprint joins the failure-mode driver artifacts
back to cleaned exposure events and participations. It reviews whether the
failed season's shifted exposure domains reflect schedule density, roster
participation, availability flagging, or managed-risk documentation before any
probability-facing, pilot, dashboard, or minute-load escalation:

```bash
risk-engine \
  --exposure-load-shift-context-sprint \
  --exposure-events outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_events.csv \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --exposure-load-features outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_features.csv \
  --exposure-load-diagnostics outputs/experiments/exposure_load_forward_diagnostic_v1/exposure_load_calibration_diagnostics.csv \
  --exposure-load-failure-modes outputs/experiments/exposure_load_failure_modes_v1/exposure_load_failure_modes.json \
  --output-dir outputs \
  --experiment-id exposure_load_shift_context_v1
```

This writes `exposure_load_shift_context.csv`,
`exposure_load_shift_context_drivers.csv`,
`exposure_load_shift_context_cases.csv`, `exposure_load_shift_context.json`, and
`exposure_load_shift_context_report.md`.

The live `exposure_load_shift_context_v1` run recommended
`review_schedule_roster_availability_context`. The failed season remained
2024-2025, with 2023-2024 and 2025-2026 as comparators. The main context
signals were reduced 28d lift-session exposure, elevated prior game count,
reduced modified-participation flagging, longer gaps since modified/no
participation and game exposure, elevated 28d practice exposure, and slightly
elevated 28d game-event exposure. Probability-facing use, pilot escalation,
dashboard work, and minute-load expansion remain blocked until this schedule,
roster, availability, and managed-risk context is reviewed.

## Run Exposure Load Schedule/Roster Sprint

```bash
risk-engine \
  --exposure-load-schedule-roster-sprint \
  --exposure-events outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_events.csv \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --exposure-load-shift-context outputs/experiments/exposure_load_shift_context_v1/exposure_load_shift_context.json \
  --output-dir outputs \
  --experiment-id exposure_load_schedule_roster_v1
```

This writes `exposure_load_schedule_roster_context.csv`,
`exposure_load_schedule_roster_drivers.csv`,
`exposure_load_schedule_roster_context.json`, and
`exposure_load_schedule_roster_report.md`.

The live `exposure_load_schedule_roster_v1` run recommended
`review_failed_season_schedule_roster_shift`. In 2024-2025, the failed season
had fewer lift events than comparators (41.0 vs 49.0), but more training events
(203.0 vs 180.5), more total retained events (216.0 vs 192.5), more
participations per active athlete (148.4 vs 133.0), and slightly more game
events (13.0 vs 12.0).

## Run Exposure Load Availability Capture Sprint

```bash
risk-engine \
  --exposure-load-availability-capture-sprint \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --exposure-load-shift-context outputs/experiments/exposure_load_shift_context_v1/exposure_load_shift_context.json \
  --output-dir outputs \
  --experiment-id exposure_load_availability_capture_v1
```

This writes `exposure_load_availability_capture.csv`,
`exposure_load_availability_capture_drivers.csv`,
`exposure_load_availability_capture.json`, and
`exposure_load_availability_capture_report.md`.

The live `exposure_load_availability_capture_v1` run recommended
`review_failed_season_availability_capture`. The failed 2024-2025 season had a
lower modified-participation rate (0.014 vs 0.021) and lower no-participation
rate (0.050 vs 0.057) than comparator seasons, while issue-linked participation
rows remained absent.

## Run Exposure Load Context Decision Sprint

```bash
risk-engine \
  --exposure-load-context-decision-sprint \
  --exposure-load-shift-context outputs/experiments/exposure_load_shift_context_v1/exposure_load_shift_context.json \
  --exposure-load-schedule-roster outputs/experiments/exposure_load_schedule_roster_v1/exposure_load_schedule_roster_context.json \
  --exposure-load-availability-capture outputs/experiments/exposure_load_availability_capture_v1/exposure_load_availability_capture.json \
  --exposure-load-guardrail-policy outputs/experiments/exposure_load_guardrail_policy_v1/exposure_load_guardrail_policy.json \
  --output-dir outputs \
  --experiment-id exposure_load_context_decision_v1
```

This writes `exposure_load_context_decision.csv`,
`exposure_load_context_decision.json`, and
`exposure_load_context_decision_report.md`.

The live `exposure_load_context_decision_v1` run recommended
`keep_shadow_ranking_and_resolve_context_before_model_expansion`, with
production readiness still `not_ready_for_probability_or_pilot`. Probability
calibration and minute-load expansion remain blocked, shadow ranking is allowed
with monitoring, and model expansion stays blocked until the failed season is
classified as true managed-risk context, schedule/roster shift, or
exposure-capture change.

## Run Exposure Load Source Context Classification Sprint

```bash
risk-engine \
  --exposure-load-source-context-classification-sprint \
  --exposure-events outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_events.csv \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --exposure-load-shift-context outputs/experiments/exposure_load_shift_context_v1/exposure_load_shift_context.json \
  --exposure-load-schedule-roster outputs/experiments/exposure_load_schedule_roster_v1/exposure_load_schedule_roster_context.json \
  --exposure-load-availability-capture outputs/experiments/exposure_load_availability_capture_v1/exposure_load_availability_capture.json \
  --exposure-load-context-decision outputs/experiments/exposure_load_context_decision_v1/exposure_load_context_decision.json \
  --output-dir outputs \
  --experiment-id exposure_load_source_context_classification_v1
```

This writes `exposure_load_source_context_classification.csv`,
`exposure_load_source_context_evidence.csv`,
`exposure_load_source_context_classification.json`, and
`exposure_load_source_context_classification_report.md`.

The live `exposure_load_source_context_classification_v1` run recommended
`treat_failed_season_as_schedule_roster_plus_capture_shift`. It classified
true managed-risk support as `not_supported_by_source_flags`, schedule/roster
context as `supported_schedule_roster_shift`, availability context as
`supported_capture_or_documentation_shift`, and the next model action as
`do_not_expand_model_features`. Exposure-load remains shadow-only until this
source context is resolved.

## Run Exposure Load Source Resolution Sprint

```bash
risk-engine \
  --exposure-load-source-resolution-sprint \
  --exposure-load-source-context-classification outputs/experiments/exposure_load_source_context_classification_v1/exposure_load_source_context_classification.json \
  --output-dir outputs \
  --experiment-id exposure_load_source_resolution_v1
```

This writes `exposure_load_source_resolution.csv`,
`exposure_load_source_resolution_actions.csv`,
`exposure_load_source_resolution_policy.json`, and
`exposure_load_source_resolution_report.md`.

The live `exposure_load_source_resolution_v1` run recommended
`exclude_failed_season_from_probability_calibration_until_source_resolved`.
The policy excludes the failed 2024-2025 season from probability calibration
until source eligibility is resolved, blocks probability calibration and broader
model expansion, defers minute-load expansion, and keeps exposure-load available
only for shadow ranking with season-level monitoring.

## Run Exposure Load Source-Eligible Calibration Sprint

```bash
risk-engine \
  --exposure-load-source-eligible-calibration-sprint \
  --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv \
  --exposure-load-source-resolution-policy outputs/experiments/exposure_load_source_resolution_v1/exposure_load_source_resolution_policy.json \
  --output-dir outputs \
  --experiment-id exposure_load_source_eligible_calibration_v1
```

This writes `exposure_load_source_eligible_calibration.csv`,
`exposure_load_source_eligible_calibration_diagnostics.csv`,
`exposure_load_source_eligible_calibration.json`, and
`exposure_load_source_eligible_calibration_report.md`.

The live `exposure_load_source_eligible_calibration_v1` run recommended
`probability_research_can_resume_on_source_eligible_seasons`. All seasons still
showed 3 calibration-loss rows with mean Brier skill delta -0.106, but after
excluding source-ineligible 2024-2025 the source-eligible scope had 0
calibration-loss rows, 6 calibration-supported rows, mean Brier skill delta
+0.065, and mean prediction-gap delta -0.005. This reopens probability research
only for source-eligible seasons; it is still not pilot or dashboard clearance.

## Run Exposure Load Source-Eligible Policy Sprint

```bash
risk-engine \
  --exposure-load-source-eligible-policy-sprint \
  --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv \
  --exposure-load-source-eligible-calibration outputs/experiments/exposure_load_source_eligible_calibration_v1/exposure_load_source_eligible_calibration.json \
  --output-dir outputs \
  --experiment-id exposure_load_source_eligible_policy_v1
```

This writes `exposure_load_source_eligible_policy.csv`,
`exposure_load_source_eligible_thresholds.csv`,
`exposure_load_source_eligible_policy.json`, and
`exposure_load_source_eligible_policy_report.md`.

The live `exposure_load_source_eligible_policy_v1` run recommended
`advance_source_eligible_shadow_mode_threshold_research` with production
readiness still `not_ready_for_probability_or_pilot`. It excludes 2024-2025 and
freezes research shadow-mode candidates under a 1.0 episode per athlete-season
burden cap: `broad_30d` mean capture 0.151 / burden 0.354, `severity_14d`
0.129 / 0.453, `severity_7d` 0.050 / 0.335, and
`subtype_lower_extremity_soft_tissue_30d` 0.186 / 0.774. This is model-readiness
progress for prospective shadow monitoring, not pilot or dashboard clearance.

## Run Exposure Load Source-Eligible Shadow Monitoring Sprint

```bash
risk-engine \
  --exposure-load-source-eligible-shadow-monitoring-sprint \
  --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv \
  --exposure-load-source-eligible-policy outputs/experiments/exposure_load_source_eligible_policy_v1/exposure_load_source_eligible_policy.json \
  --output-dir outputs \
  --experiment-id exposure_load_source_eligible_shadow_monitoring_v1
```

This writes `exposure_load_source_eligible_shadow_monitoring.csv`,
`exposure_load_source_eligible_shadow_monitoring_seasons.csv`,
`exposure_load_source_eligible_shadow_monitoring.json`, and
`exposure_load_source_eligible_shadow_monitoring_report.md`.

The sprint reviews the frozen source-eligible threshold package against complete
athlete-season validation rows, excludes source-ineligible seasons, and reports
which channels are ready for prospective shadow review under the research-only
deployment boundary. It is not pilot or dashboard clearance.

The live `exposure_load_source_eligible_shadow_monitoring_v1` run recommended
`proceed_with_prospective_source_eligible_shadow_monitoring` with production
readiness still `not_ready_for_probability_or_pilot`. It excluded 2024-2025.
The three burden-capped channels were ready for prospective shadow review:
`broad_30d` mean capture 0.151 / max burden 0.686, `severity_14d` 0.129 /
0.919, and `severity_7d` 0.050 / 0.615. The subtype channel
`subtype_lower_extremity_soft_tissue_30d` remained blocked for burden guardrail
review because max burden reached 2.488 episodes per athlete-season. This
supports prospective shadow review for the burden-capped channels only, not
pilot/dashboard or autonomous intervention.

## Run Exposure Load Shadow Launch Chain

After source-eligible shadow monitoring, run the three launch-preparation
sprints consecutively:

```bash
risk-engine \
  --exposure-load-shadow-channel-lock-sprint \
  --exposure-load-source-eligible-shadow-monitoring outputs/experiments/exposure_load_source_eligible_shadow_monitoring_v1/exposure_load_source_eligible_shadow_monitoring.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_channel_lock_v1

risk-engine \
  --exposure-load-shadow-review-protocol-sprint \
  --exposure-load-shadow-channel-lock outputs/experiments/exposure_load_shadow_channel_lock_v1/exposure_load_shadow_channel_lock.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_review_protocol_v1

risk-engine \
  --exposure-load-shadow-readiness-register-sprint \
  --exposure-load-shadow-channel-lock outputs/experiments/exposure_load_shadow_channel_lock_v1/exposure_load_shadow_channel_lock.json \
  --exposure-load-shadow-review-protocol outputs/experiments/exposure_load_shadow_review_protocol_v1/exposure_load_shadow_review_protocol.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_readiness_register_v1
```

These write channel-lock, review-protocol, and readiness-register CSV/JSON/report
artifacts. The chain locks only channels that passed source-eligible burden
guardrails, prepares the prospective research review protocol, and records the
launch boundary. This launches research shadow monitoring preparation; it still
does not clear pilot/dashboard work, probability-facing deployment, or
autonomous intervention.

The live `exposure_load_shadow_channel_lock_v1`,
`exposure_load_shadow_review_protocol_v1`, and
`exposure_load_shadow_readiness_register_v1` runs launched the research shadow
preparation chain. Locked channels are `broad_30d`, `severity_14d`, and
`severity_7d`, all with `burden_capped_percentile` thresholds. The subtype
channel remains held because `subtype_lower_extremity_soft_tissue_30d` still
needs burden guardrail review. The readiness register recommends
`launch_research_shadow_monitoring_without_product_escalation`; outcome
collection must precede calibration updates, pilot escalation, dashboard work,
or autonomous intervention.

## Run Exposure Load Historical Shadow Replay Sprint

```bash
risk-engine \
  --exposure-load-shadow-replay-sprint \
  --season-forward-validation-path outputs/experiments/exposure_load_season_forward_validation_v1/exposure_load_season_forward_validation.csv \
  --exposure-load-shadow-channel-lock outputs/experiments/exposure_load_shadow_channel_lock_v1/exposure_load_shadow_channel_lock.json \
  --exposure-load-shadow-review-protocol outputs/experiments/exposure_load_shadow_review_protocol_v1/exposure_load_shadow_review_protocol.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_replay_v1
```

This writes `exposure_load_shadow_replay_log.csv`,
`exposure_load_shadow_review_packets.csv`,
`exposure_load_shadow_stop_rules.csv`, `exposure_load_shadow_replay.json`, and
`exposure_load_shadow_replay_report.md`. The sprint replays the locked channels
against the completed season-forward validation rows, marks source-ineligible
seasons and burden stop-rule rows, and generates review packets for historical
source-eligible channel-season rows. This is the current-data bridge into
prospective outcome collection; it does not prove prospective performance.

The live `exposure_load_shadow_replay_v1` run recommended
`historical_shadow_replay_ready_for_prospective_collection`. It produced 15
historical replay rows, 12 source-eligible review packets, 3 source-ineligible
stop rows, and 0 burden-stop rows. Review packets were evenly distributed across
the locked channels: 4 each for `broad_30d`, `severity_14d`, and `severity_7d`.
This is the strongest endpoint available from the current historical data: it
prepares the prospective collection workflow but does not itself establish
prospective performance, real-time usefulness, probability calibration readiness,
or pilot/dashboard readiness.

## Run Exposure Load Shadow Adjudication Package Sprint

```bash
risk-engine \
  --exposure-load-shadow-adjudication-sprint \
  --exposure-load-shadow-replay outputs/experiments/exposure_load_shadow_replay_v1/exposure_load_shadow_replay.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_adjudication_v1
```

This writes `exposure_load_shadow_adjudication_schema.csv`,
`exposure_load_shadow_adjudication_template.csv`,
`exposure_load_shadow_adjudication_completion.csv`,
`exposure_load_shadow_adjudication.json`, and
`exposure_load_shadow_adjudication_report.md`. The sprint turns the replay
review packets into a prospective collection template with required reviewer,
date, usefulness, outcome, source-context, action, and notes fields. This is
collection infrastructure only; completing the template with prospective
reviewer data is the next evidence step.

The live `exposure_load_shadow_adjudication_v1` run recommended
`adjudication_template_ready_for_prospective_collection`. It produced 12
prospective collection rows from the historical review packets, 8 schema fields,
and completion checks showing all 12 rows still pending the six required
reviewer fields: `reviewer_id`, `review_date`, `alert_usefulness`,
`outcome_confirmed`, `source_context_ok`, and `action_taken`. This is the
logical stopping point for current-data preparation: the next progress requires
actual reviewer/adjudication values from prospective collection.

See `docs/shadow_adjudication_guide.md` for the manual review workflow, field
definitions, and conservative decision rules for completing the template.

## Run Exposure Load Shadow Adjudication Summary Sprint

After the adjudication template has been filled, validate and summarize it with:

```bash
risk-engine \
  --exposure-load-shadow-adjudication-summary-sprint \
  --exposure-load-shadow-adjudication outputs/experiments/exposure_load_shadow_adjudication_v1/exposure_load_shadow_adjudication_template.csv \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_adjudication_summary_v1
```

This writes `exposure_load_shadow_adjudication_validation.csv`,
`exposure_load_shadow_adjudication_channel_summary.csv`,
`exposure_load_shadow_adjudication_summary.json`, and
`exposure_load_shadow_adjudication_summary_report.md`. The sprint validates
missing and invalid reviewer fields, counts completed rows, summarizes useful,
source-trustworthy, and actionable packets by channel, and keeps product
readiness blocked.

The current unfilled template was checked with
`exposure_load_shadow_adjudication_summary_pending_v1`; it correctly reported
12 total rows, 0 complete valid rows, 12 pending or invalid rows, and
`complete_adjudication_required_before_operational_summary`.

## Run Exposure Load Shadow Adjudication Decision Sprint

After the completed adjudication summary exists, convert it into channel-level
shadow-monitoring decisions with:

```bash
risk-engine \
  --exposure-load-shadow-adjudication-decision-sprint \
  --exposure-load-shadow-adjudication-summary outputs/experiments/exposure_load_shadow_adjudication_summary_csv_review_v1/exposure_load_shadow_adjudication_summary.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_adjudication_decision_v1
```

This writes `exposure_load_shadow_adjudication_channel_decisions.csv`,
`exposure_load_shadow_adjudication_decision.json`, and
`exposure_load_shadow_adjudication_decision_report.md`. The live
`exposure_load_shadow_adjudication_decision_v1` run recommended
`continue_shadow_monitoring_with_channel_revisions`: continue `broad_30d` and
`severity_14d` shadow monitoring, pause or revise `severity_7d`, and keep
probability calibration, pilot, and dashboard readiness blocked.

## Run Exposure Load Shadow Monitoring Plan Sprint

After channel decisions exist, create the retained-channel monitoring plan with:

```bash
risk-engine \
  --exposure-load-shadow-monitoring-plan-sprint \
  --exposure-load-shadow-adjudication-decision outputs/experiments/exposure_load_shadow_adjudication_decision_v1/exposure_load_shadow_adjudication_decision.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_monitoring_plan_v1
```

This writes `exposure_load_shadow_monitoring_plan.csv`,
`exposure_load_shadow_monitoring_paused_channels.csv`,
`exposure_load_shadow_monitoring_evidence_gates.csv`,
`exposure_load_shadow_monitoring_plan.json`, and
`exposure_load_shadow_monitoring_plan_report.md`. The live
`exposure_load_shadow_monitoring_plan_v1` run recommended
`launch_retained_channel_shadow_monitoring`: collect at least 4 new complete
source-eligible review packets for `broad_30d` and `severity_14d`; keep
`severity_7d` paused for threshold/channel revision; keep probability
calibration and pilot/dashboard readiness blocked.

## Run Exposure Load Shadow Collection Template Sprint

After the retained-channel monitoring plan exists, create the prospective
collection template with:

```bash
risk-engine \
  --exposure-load-shadow-collection-template-sprint \
  --exposure-load-shadow-monitoring-plan outputs/experiments/exposure_load_shadow_monitoring_plan_v1/exposure_load_shadow_monitoring_plan.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_collection_template_v1
```

This writes `exposure_load_shadow_collection_schema.csv`,
`exposure_load_shadow_collection_template.csv`,
`exposure_load_shadow_collection_completion.csv`,
`exposure_load_shadow_collection_template.json`, and
`exposure_load_shadow_collection_template_report.md`. The live
`exposure_load_shadow_collection_template_v1` run created 8 prospective
collection rows: 4 for `broad_30d` and 4 for `severity_14d`. All rows start
pending required fields; `severity_7d` remains excluded until revised.

## Run Exposure Load Shadow Collection Packet Workflow Sprint

Before reviewer evidence is entered, generate reviewer packet materials and an
audit-trail seed with:

```bash
risk-engine \
  --exposure-load-shadow-collection-packet-workflow-sprint \
  --exposure-load-shadow-collection outputs/experiments/exposure_load_shadow_collection_template_v1/exposure_load_shadow_collection_template.csv \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_collection_packet_workflow_v1
```

This writes `exposure_load_shadow_collection_packet_manifest.csv`,
`exposure_load_shadow_collection_packet_checklist.csv`,
`exposure_load_shadow_collection_packet_audit_trail.csv`,
`exposure_load_shadow_collection_reviewer_instructions.md`,
`exposure_load_shadow_collection_packet_workflow.json`,
`exposure_load_shadow_collection_packet_workflow_report.md`, and one
de-identified markdown file per packet under `review_packets/`. The live
`exposure_load_shadow_collection_packet_workflow_v1` run created 8 reviewer
packets, 56 checklist rows, and 8 audit-trail seed rows without completing
evidence or changing readiness status.

For the exact reviewer workflow and user responsibilities, see
`docs/shadow_collection_reviewer_guide.md`.

## Run Exposure Load Shadow Collection Evidence Prefill Sprint

To avoid asking reviewers to manually recover fields that are already present in
the shadow replay artifacts, prefill retained-channel collection rows with:

```bash
risk-engine \
  --exposure-load-shadow-collection-evidence-prefill-sprint \
  --exposure-load-shadow-review-packets outputs/experiments/exposure_load_shadow_replay_v1/exposure_load_shadow_review_packets.csv \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_collection_evidence_prefill_v1
```

This writes `exposure_load_shadow_collection_prefilled.csv`,
`exposure_load_shadow_collection_prefill_validation.csv`,
`exposure_load_shadow_collection_prefill_excluded.csv`,
`exposure_load_shadow_collection_evidence_prefill.json`, and
`exposure_load_shadow_collection_evidence_prefill_report.md`. The live
`exposure_load_shadow_collection_evidence_prefill_v1` run produced 8
retained-channel rows with replay-derived season, source-eligibility, episode,
observed-event, and captured-event fields prefilled. Reviewer judgment fields
remain blank.

The reviewer process is preserved in
`docs/shadow_collection_reviewer_process_reference.md`. The current local
prefilled file has been completed by carrying forward the existing CSV-only
adjudication judgments; this remains research evidence and is not independent
clinical/practitioner adjudication.

## Run Exposure Load Shadow Collection Summary Sprint

After retained-channel prefilled collection rows are reviewed, validate and
summarize them with:

```bash
risk-engine \
  --exposure-load-shadow-collection-summary-sprint \
  --exposure-load-shadow-collection outputs/experiments/exposure_load_shadow_collection_evidence_prefill_v1/exposure_load_shadow_collection_prefilled.csv \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_collection_summary_v1
```

This writes `exposure_load_shadow_collection_validation.csv`,
`exposure_load_shadow_collection_channel_summary.csv`,
`exposure_load_shadow_collection_summary.json`, and
`exposure_load_shadow_collection_summary_report.md`. When run before reviewer
fields are complete, the summary reports the remaining missing reviewer fields
and keeps `not_ready_for_calibration_claims`; probability, pilot, and dashboard
readiness remain blocked until the retained-channel collection evidence is
reviewed and summarized.

The completed local run
`exposure_load_shadow_collection_summary_completed_v1` reports 8 complete valid
rows, 0 pending/invalid rows, 8 complete source-eligible rows, and 4
useful/source-trustworthy/actionable rows. Its recommendation is
`revisit_calibration_readiness_with_prospective_shadow_evidence`; this means the
next step is a calibration-readiness review, not calibration claims or product
deployment.

After independent practitioner/source-context review, preserve a separate
reviewed input instead of overwriting the CSV-only prefill:
`outputs/experiments/exposure_load_shadow_collection_practitioner_review_v1/exposure_load_shadow_collection_practitioner_review.csv`.
The practitioner-reviewed local run
`exposure_load_shadow_collection_summary_practitioner_v1` reports 8 complete
valid rows, 0 pending/invalid rows, 8 complete source-eligible rows, 5
useful/source-trustworthy/actionable rows, 8 practitioner-adjudicated rows, and
0 CSV-only review rows. Its independent practitioner adjudication status is
`satisfied`.

## Run Exposure Load Shadow Calibration Readiness Sprint

After the retained-channel collection summary is complete, convert it into a
bounded calibration-readiness decision package with:

```bash
risk-engine \
  --exposure-load-shadow-calibration-readiness-sprint \
  --exposure-load-shadow-collection-summary outputs/experiments/exposure_load_shadow_collection_summary_completed_v1/exposure_load_shadow_collection_summary.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_calibration_readiness_v1
```

This writes `exposure_load_shadow_calibration_readiness_channels.csv`,
`exposure_load_shadow_calibration_readiness_gaps.csv`,
`exposure_load_shadow_calibration_readiness.json`, and
`exposure_load_shadow_calibration_readiness_report.md`. The live
`exposure_load_shadow_calibration_readiness_v1` run recommends
`defer_calibration_claims_pending_independent_practitioner_adjudication`.
Both retained channels are calibration research candidates only after
independent practitioner/source-context adjudication. Probability-facing output,
pilot/dashboard readiness, calibration claims, and autonomous intervention
remain blocked.

After the practitioner-reviewed summary is available, rerun this sprint against
`exposure_load_shadow_collection_summary_practitioner_v1`. The live
`exposure_load_shadow_calibration_readiness_practitioner_v1` run recommends
`advance_to_bounded_calibration_research_not_claims`, with both retained
channels marked `calibration_research_candidate_practitioner_adjudicated`.
Probability-facing output, pilot/dashboard readiness, calibration claims, and
autonomous intervention remain blocked.

## Run Exposure Load Shadow Event Crosswalk Sprint

Before independent practitioner adjudication, generate the retained-channel
captured/missed injury-event crosswalk with:

```bash
risk-engine \
  --measurements outputs/live_inputs/exposure_load_season_forward_validation_v1/canonical_measurements.csv \
  --injuries outputs/live_inputs/exposure_load_season_forward_validation_v1/canonical_injuries.csv \
  --exposure-participations outputs/exposure_inputs/exposure_cleaning_audit_v1/exposure_participations.csv \
  --exposure-load-shadow-replay outputs/experiments/exposure_load_shadow_replay_v1/exposure_load_shadow_replay.json \
  --exposure-load-shadow-collection outputs/experiments/exposure_load_shadow_collection_evidence_prefill_v1/exposure_load_shadow_collection_prefilled.csv \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_event_crosswalk_v1 \
  --graph-window-size 4 \
  --model-variant l2 \
  --exposure-load-shadow-event-crosswalk-sprint
```

This writes `exposure_load_shadow_event_crosswalk.csv`,
`exposure_load_shadow_event_crosswalk_summary.csv`,
`exposure_load_shadow_event_crosswalk.json`, and
`exposure_load_shadow_event_crosswalk_report.md`. The live
`exposure_load_shadow_event_crosswalk_v1` run is filtered to the 8 retained
collection rows only: 363 event rows, 55 captured events, and 308 missed events,
with every packet matching the replay aggregate counts. Use this artifact to
inspect which de-identified injury events were captured or missed before
changing the practitioner judgment fields. It is not probability calibration,
pilot/dashboard clearance, or intervention evidence.

## Run Live-Source Experiment

When `config/paths.local.yaml` points to available local sources, the CLI can
prepare canonical inputs and run the research engine in one step:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id actual_data_probe_run \
  --graph-window-size 4
```

To compare graph construction windows over the same prepared input set:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id window_sensitivity_v1 \
  --window-sensitivity-sizes 2 3 4 5 7
```

To compare regularized model variants across rotating athlete-level holdout
splits:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id model_robustness_sprint_v1 \
  --model-robustness-sprint \
  --graph-window-size 4 \
  --stability-splits 5
```

To compare regularized model variants across multiple graph windows in the same
sprint:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id window_model_robustness_v1 \
  --model-robustness-sprint \
  --window-sensitivity-sizes 2 4 7 \
  --stability-splits 5
```

Live-source preparation writes ignored canonical CSVs under
`outputs/live_inputs/<experiment-id>/` and records preparation metadata plus a
`data_quality_audit.json` beside them. It writes `canonical_measurements.csv`,
`canonical_injuries.csv`, and `injury_events_detailed.csv`. The detailed injury
event file preserves one de-identified row per raw injury event with richer
context such as issue/resolved dates, duration, time-loss days, modified
availability days, recurrence, unavailability, activity, classification,
pathology, body area, tissue type, side, participation level, training/game
context, source file, and source row number.

The audit reports hashed identity overlap across sources, sparse
athlete-seasons, large within-season measurement gaps, duplicate same-day metric
rows, and observed injury events without nearby measurements. It also includes
review context for remaining single-source hashed identities and injury events
outside the nearby-measurement window. Athlete identities are stable hashes of
normalized names, seasons start on July 1, and the current modeling label policy
still uses the earliest injury issue date per athlete-season while censoring
event-free athlete-seasons at their last measurement date. Name normalization
reconciles common `Last, First` export style with `First Last` names before
hashing. Duplicate same-day metric rows are aggregated by mean `metric_value` per
athlete, season, date, source, and metric before modeling; the aggregation counts
are recorded in `prep_metadata.json`. Observed injury events are labeled by
nearest same-season measurement distance: `modelable` at 14 days or less,
`low_confidence` at 15-30 days, and `out_of_window` beyond 30 days. When
`primary_model_event` is available, the discrete-time baseline uses it as the
positive-event policy so low-confidence and out-of-window observed events do not
become training positives.

## Modeling Baseline

The experiment runner writes `model_summary.json` and `model_evaluation.json`
alongside `model_metrics.json`. The current risk model is a discrete-time
logistic baseline trained separately for the 7, 14, and 30 day horizons over
13 graph snapshot features: `time_index`, `node_count`, `edge_count`,
`mean_abs_correlation`, `edge_density`, `delta_edge_count`,
`delta_mean_abs_correlation`, `delta_edge_density`, `graph_instability`,
`z_mean_abs_correlation`, `z_edge_density`, `z_edge_count`, and
`z_graph_instability`.
The temporal delta features are computed per athlete-season in chronological
order and capture change from one snapshot to the next. `edge_density`
normalizes edge count by the maximum possible edges. `graph_instability` is a
rolling population standard deviation of `mean_abs_correlation` over the most
recent three snapshots. The z-score features compare each snapshot to that
athlete-season's own strictly prior rolling baseline, use population standard
deviation, require at least two prior snapshots, and clip extreme departures to
`[-10.0, 10.0]`. The split is deterministic and athlete-level, with a
sorted 20% holdout. If a training fold has only one class at a horizon, the
runner records a prevalence fallback for that horizon instead of fitting an
unstable classifier.

`model_evaluation.json` compares holdout predictions to the training prevalence
baseline for each horizon. It reports holdout event counts and rates, mean
predicted risk, model and prevalence Brier scores, Brier skill score, AUROC,
average precision, and top-decile lift when those metrics are defined by the
holdout labels.

The experiment runner also writes `feature_attribution.json` and
`feature_ablation_report.md`. These compare the full 13-feature model with the
prior `original_9` feature set and a `z_score_only` feature set using the same
athlete-level holdout split. Each horizon includes the same holdout metrics plus
standardized logistic coefficients for feature attribution.

The `explanations/` subdirectory contains two artifacts that answer Peterson's
core question — "why is this athlete high-risk right now?" — at the snapshot
level. `explanation_summary.csv` adds `top_feature_{h}d` and
`top_contribution_{h}d` columns to the per-snapshot table, where contribution is
`standardized_coefficient_k × (value_k − train_mean_k) / train_std_k`. This is
directly comparable across features in log-odds units: a positive contribution
means the feature pushed risk above the population average, negative means below.
`athlete_explanations.json` structures the same information per athlete-season:
peak risk dates per horizon, the top-3 dominant features averaged over the
season, and per-snapshot top-3 signed feature contributions.

Window-sensitivity runs write `window_sensitivity.json` and
`window_sensitivity_report.md`, comparing multiple graph `window_size` values
with the same holdout policy and evaluation metrics.

Model robustness sprints write `model_robustness.json` and
`model_robustness_report.md`, comparing `baseline`, `l2`, `l1`, and
`elasticnet` logistic variants across deterministic rotating athlete-level
holdout splits. Regularized variants standardize features internally for model
fitting and convert coefficients back to raw feature units for attribution.
Pairing `--model-robustness-sprint` with `--window-sensitivity-sizes` writes
`window_model_robustness.json` and `window_model_robustness_report.md`, which
compare those variants across each requested graph window. Single research runs
also accept `--model-variant baseline|l2|l1|elasticnet`.

Calibration and threshold runs use out-of-fold predictions across rotating
athlete-level splits so every snapshot is scored by a model that did not train on
that athlete:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id calibration_threshold_v1 \
  --calibration-thresholds \
  --model-variant l2 \
  --graph-window-size 4 \
  --stability-splits 5
```

These runs write `calibration_summary.json` (OOF stats, calibration bins, and
threshold rows per horizon), `threshold_table.csv`, and `calibration_report.md`.
The threshold table reports alert count, event capture (recall), precision (PPV),
and lift for both percentile-based (top 5/10/20%) and fixed-probability
(0.10/0.20/0.30/0.50) thresholds at each horizon.

Alert episode validation turns snapshot-level percentile alerts into contiguous
athlete-season risk episodes:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id alert_episode_validation_v1 \
  --alert-episodes \
  --model-variant l2 \
  --graph-window-size 4
```

These runs write `alert_episodes.csv`, `alert_episodes.json`,
`alert_episode_summary.json`, and `alert_episode_report.md`. They also write
the Episode Quality Audit artifacts: `alert_episode_quality.csv`,
`alert_episode_quality.json`, and `alert_episode_quality_report.md`, plus the
qualitative case-review artifacts: `qualitative_case_review.json` and
`qualitative_case_review_report.md`, plus the model-improvement diagnostic
artifacts: `model_improvement_diagnostics.csv`,
`model_improvement_diagnostics.json`, and
`model_improvement_diagnostic_report.md`. When
`injury_events_detailed.csv` is available beside the canonical live inputs, the
same run also writes injury-context outcome artifacts:
`injury_event_context_profiles.csv`, `injury_context_outcomes.csv`,
`injury_context_outcomes.json`, and `injury_context_outcome_report.md`.
Episodes use the top-5% and top-10% percentile thresholds, collapse contiguous
alert snapshots, record start/peak/end event timing without treating censoring
dates as injuries, and roll up model contribution and intra-individual z-score
flags. The quality audit adds
start-based true-positive counts, false-positive counts, unique injury-event
capture, missed observed events, alert burden per athlete-season, median lead
time, threshold overlap, and representative cases. The case-review artifact adds
timeline context and simple diagnostic labels for useful warnings, noisy alerts,
missed injuries, and high own-baseline-deviation cases. The model-improvement
diagnostic table compares useful alerts, noisy alerts, and missed observed
injuries side by side so the next modeling sprint can target missing context,
threshold policy, or event-specific feature gaps. The injury-context outcome
artifact profiles each detailed injury event against each horizon/threshold and
rolls capture rates up by injury type, pathology, classification, body area,
tissue type, side, recurrence, unavailability, activity context, and time-loss
bucket.

Injury outcome policy runs audit severity semantics and define candidate target
policies before changing the model target:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id injury_outcome_policy_v1 \
  --injury-outcome-policies
```

These runs write `injury_severity_audit.csv`,
`injury_severity_audit.json`, `injury_severity_audit_report.md`,
`outcome_policy_table.csv`, `outcome_policy_summary.json`, and
`outcome_policy_report.md`. The severity audit checks time-loss availability,
negative values, extreme values above 365 days, and consistency between
duration and issue/resolved dates. The policy table defines candidate targets
such as time-loss-only, moderate-plus time-loss, severe time-loss,
lower-extremity, soft-tissue, lower-extremity soft-tissue, concussion-only, and
exclude-concussion.

Outcome-policy model comparisons relabel athlete-seasons under detailed injury
target policies and retrain the same graph model for each target:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id outcome_policy_model_comparison_v1 \
  --outcome-policy-model-comparison \
  --model-variant l2 \
  --graph-window-size 4
```

These runs write `context_policy_model_comparison.csv`,
`context_policy_model_comparison.json`, and
`context_policy_model_comparison_report.md`. The comparison table reports
holdout AUROC, Brier skill, top-decile lift, alert episodes, unique event
capture, missed events, alert burden, and median lead time for each target
policy, horizon, and top-5%/top-10% threshold.

Policy decision sprint runs execute the next three research iterations in one
reproducible pass: two-channel policy selection, policy/window sensitivity, and
an operational shadow-mode package.

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id policy_decision_sprint_v1 \
  --policy-decision-sprint \
  --policy-window-sizes 2 4 7 \
  --model-variant l2
```

These runs write `two_channel_alert_policy.json`,
`two_channel_alert_policy_report.md`, `policy_window_sensitivity.csv`,
`policy_window_sensitivity.json`, `policy_window_sensitivity_report.md`,
`operational_policy_package.json`, and
`operational_policy_package_report.md`.

Shadow-mode stability runs freeze the selected policy package and evaluate
whether it holds up across season slices:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id shadow_mode_stability_v1 \
  --shadow-mode-stability \
  --model-variant l2
```

These runs write `shadow_mode_stability.csv`,
`shadow_mode_stability.json`, and `shadow_mode_stability_report.md`. The audit
uses season-local percentile thresholds so each season is evaluated like a
shadow-mode cohort, then summarizes capture-rate range and alert burden for each
fixed channel.

Season drift diagnostic runs explain the season-to-season instability by joining
the fixed shadow-mode channels with measurement coverage, source mix, and injury
mix:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id season_drift_diagnostic_v1 \
  --season-drift-diagnostic \
  --model-variant l2
```

These runs write `season_drift_diagnostics.csv`,
`season_drift_diagnostics.json`, and `season_drift_diagnostic_report.md`. The
diagnostic flags low-coverage seasons, lists the highest-capture season per
channel, and keeps policy performance tied to coverage and injury context.

Coverage-normalized policy sprints test whether the fixed shadow-mode channels
remain stable after coverage eligibility controls are applied to complete
athlete-season trajectories:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id coverage_normalized_policy_v1 \
  --coverage-normalized-policy-sprint \
  --model-variant l2
```

These runs write `coverage_normalized_policy.csv`,
`coverage_normalized_policy.json`, and `coverage_normalized_policy_report.md`.
The sprint evaluates the fixed channels under `all`, `medium_high`, and
`high_only` coverage scopes, rebuilding season-local alert episodes after
filtering whole athlete-seasons. This preserves the Peterson-style longitudinal
unit of analysis and avoids treating daily rows as independent examples.

Coverage/source-aware model sprints test whether explicit measurement-density
and source-mix covariates improve the current graph trajectory model without
replacing the graph features as the core signal:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id coverage_source_aware_model_v1 \
  --coverage-source-aware-model-sprint \
  --model-variant l2 \
  --graph-window-size 4
```

These runs write `coverage_source_features.csv`,
`coverage_source_model_comparison.csv`,
`coverage_source_model_comparison.json`, and
`coverage_source_model_comparison_report.md`. The added features are computed
only from measurements available on or before each graph snapshot date:
measurement days to date, measurement rows to date, source count to date, days
since the previous measurement, and source-seen flags for bodyweight,
forceplate, GPS, and Perch.

Coverage-adjusted threshold sprints test whether coverage-tier-local or
burden-capped alert policies can control shadow-mode alert load without
collapsing event capture:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id coverage_adjusted_threshold_v1 \
  --coverage-adjusted-threshold-sprint \
  --model-variant l2
```

These runs write `coverage_adjusted_threshold_policy.csv`,
`coverage_adjusted_threshold_policy.json`, and
`coverage_adjusted_threshold_report.md`. Threshold policies are applied only
after complete athlete-season trajectories are scored; the sprint does not
turn daily measurement rows into independent examples.

Season-forward validation sprints test whether the current signal generalizes
from earlier seasons into later seasons:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id season_forward_validation_v1 \
  --season-forward-validation \
  --model-variant l2 \
  --graph-window-size 4
```

These runs write `season_forward_validation.csv`,
`season_forward_validation.json`, and `season_forward_validation_report.md`.
The sprint trains on earlier complete athlete-season trajectories, evaluates
later complete athlete-season trajectories, compares `graph_trajectory` against
`graph_plus_coverage_source`, and checks fixed alert channels under season-local
and burden-capped thresholds.

Forward case review sprints inspect the channels and seasons that still show
forward-season signal:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id forward_case_review_v1 \
  --forward-case-review-sprint \
  --model-variant l2
```

These runs write `forward_case_review_cases.csv`,
`forward_case_review.json`, and `forward_case_review_report.md`. The sprint
scores complete athlete-season trajectories, targets forward-surviving
seasons/channels, rebuilds alert episodes, and classifies true positives, false
positives, missed injuries, and high intra-individual deviation episodes into
case-review diagnostics.

Case diagnostic requirements sprints convert those reviewed cases into
production-readiness data requirements and modeling actions:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id case_diagnostic_requirements_v1 \
  --case-diagnostic-requirements-sprint \
  --model-variant l2
```

These runs write `forward_case_review_cases.csv`,
`case_diagnostic_requirements.csv`, `case_diagnostic_requirements.json`, and
`case_diagnostic_requirements_report.md`. The sprint does not resample daily
rows; it turns case diagnostics from complete athlete-season trajectories into
prioritized missing-data domains and model-improvement actions.

Injury-history feature sprints derive time-safe prior-injury context from the
uploaded detailed injury data and compare it against the coverage/source
reference model:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id injury_history_feature_v1 \
  --injury-history-feature-sprint \
  --model-variant l2 \
  --graph-window-size 4
```

These runs write `injury_history_features.csv`,
`injury_history_model_comparison.csv`,
`injury_history_model_comparison.json`, and
`injury_history_model_comparison_report.md`. The sprint uses only detailed
injury events before each graph snapshot date to derive prior-injury count,
same-season prior injury count, days since last injury, prior time-loss load,
lower-extremity and soft-tissue history, activity-context history, and prior
unavailability history.

Injury-history season-forward validation sprints test whether those prior-injury
features survive the stricter train-prior-seasons / evaluate-later-seasons
setup:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id injury_history_season_forward_validation_v1 \
  --injury-history-season-forward-validation \
  --model-variant l2 \
  --graph-window-size 4
```

These runs write `injury_history_features.csv`,
`injury_history_season_forward_validation.csv`,
`injury_history_season_forward_validation.json`, and
`injury_history_season_forward_validation_report.md`. The sprint compares
`graph_plus_coverage_source` against `graph_plus_coverage_injury_history` under
season-forward validation and scores fixed alert channels with the
injury-history feature set.

Latest live-source comparison (`intra_individual_deviation_v1`, 349 athletes,
70 holdout): 7d AUROC 0.723, Brier skill 0.0020; 14d AUROC 0.731, Brier skill
0.0057; 30d AUROC 0.736, Brier skill 0.0171. Compared with
`enriched_graph_features_v1`, the 30d AUROC, 7d/30d Brier skill, and all
top-decile lifts improved, while 7d/14d AUROC declined slightly. The current
feature attribution run (`feature_attribution_ablation_v1`) showed that
`original_9` remains stronger on 7d/14d AUROC, while the combined `full_13`
model improves 7d/30d top-decile lift. The `z_score_only` model is weak as a
standalone model but has strong 7d lift, suggesting the z-score features are
most useful as ranking modifiers inside the combined model.

The current window sensitivity run (`window_sensitivity_v1`, graph windows
2/3/4/5/7) showed a useful tradeoff: window 4 was best for AUROC at all
horizons, window 7 was best for Brier skill at all horizons, and window 2 was
best for top-decile lift at all horizons.

The current model robustness sprint (`model_robustness_sprint_v1`, graph window
4, 5 rotating splits) showed that all regularized variants improved average
AUROC, Brier skill, Brier score, and top-decile lift relative to baseline at all
horizons. L2 was the strongest all-around choice, winning calibration at
7/14/30d, triage at 7/14d, and ranking at 14d. Elastic net narrowly won 30d
ranking and 30d triage.

The current window/model robustness sprint (`window_model_robustness_v1`,
windows 2/4/7, 5 rotating splits) showed that operating goal matters. L2 remains
the best calibration candidate, winning 7d/14d calibration with window 7 and 30d
calibration with window 4. Window 2 produced the strongest top-decile lift at all
horizons, making it the clearest high-alert triage setting. Ranking split by
horizon: window 2 baseline won 7d AUROC, window 4 L2 won 14d AUROC, and window 7
L1 narrowly won 30d AUROC.

The calibration and threshold run (`calibration_threshold_v1`, L2, window 4,
5 OOF splits, 39,189 snapshots) confirmed that fixed probability thresholds are
nearly useless at short horizons because L2 predictions concentrate below 0.05 at
7d, leaving almost no snapshots above any fixed cutoff. Percentile-based
thresholds are the correct operating mode: top-10% delivers 3.5-4x lift at all
horizons. At 30d, a 0.10 probability threshold matches the top-10% percentile
threshold in alert volume and lift. Calibration bins are flat across the lower
eight deciles and sharply elevated in the upper two, confirming the score is most
useful as a ranking signal.

A second calibration run at window 7 (`calibration_threshold_w7_v1`) found that
window 7 does not spread predictions more than window 4 — fixed probability
thresholds are even less useful at window 7. Window 4 outperforms window 7 on
lift at every horizon (3.6x vs 3.2x at 7d, 3.8x vs 3.3x at 14d, 3.8x vs 3.6x
at 30d). Window 7 wins only at 14d Brier skill (0.008 vs 0.006). The confirmed
primary candidate is L2 + window 4, with percentile-based (top-N%) thresholds as
the operational interface.

The per-athlete explanation run (`athlete_explanations_v1`, L2, window 4, 902
athlete-seasons) shows `mean_abs_correlation` as the dominant risk driver in 99%
of snapshots at 7d, with positive contributions for elevated correlation
structure and negative `edge_density` contributions for high-risk snapshots.
Z-score features (intra-individual deviations) appear as top drivers for a
targeted subset of snapshots — the high-sensitivity cases Peterson's methodology
is designed to detect.

The follow-up explanation run (`intra_individual_explanations_v1`, L2, window 4)
adds explicit own-baseline departure details to `athlete_explanations.json`.
Every snapshot now reports the four intra-individual z-score feature values,
whether each is elevated (`abs(z) > 2.0`), and its signed contribution at the 7d,
14d, and 30d horizons. Each athlete-season also reports the peak combined
intra-individual deviation snapshot. In the live run, 3,529 of 39,189 snapshots
had at least one elevated z-score feature, led by `z_graph_instability` (2,092
snapshots) and `z_mean_abs_correlation` (2,028).

The alert episode quality run (`alert_episode_quality_v1`, L2, window 4)
produced 4,268 episodes across the 7d/14d/30d horizons and top-5%/top-10%
thresholds. The strongest policy remains 30d top-5%: 620 episodes, 130
start-based true-positive episodes, 490 false-positive episodes, 55 of 188
unique observed events captured, 133 missed events, and median start lead time
of 11 days. Top-10% increased burden without improving unique-event capture
(54 of 188). Peak risk and elevated z-score rates were similar between
true-positive and false-positive episodes, so the next useful sprint is
qualitative case review and explanation refinement rather than dashboard work.

The qualitative case-review run (`qualitative_case_review_v1`, L2, window 4)
wrote 24 deterministic review cases across the same horizons and thresholds: 6
model-supported useful warnings, 6 model misses, 6 missing-context/noisy-alert
cases, and 6 explanation gaps. The missed-injury cases include modelable events,
so performance is not only a data-quality problem. Better injury labels and
exposure/availability context are still likely high-value, but the next
performance sprint should explicitly compare missed-event feature profiles
against false-positive and true-positive profiles.

The model-improvement diagnostic run (`model_improvement_diagnostics_v1`, L2,
window 4) wrote 18 comparison rows: useful alerts, noisy alerts, and missed
events for each 7d/14d/30d horizon and top-5%/top-10% threshold. At the current
headline policy, 30d top-5%, it found 130 true-positive episodes, 490
false-positive episodes, and 133 missed events. True-positive and false-positive
episodes again had very similar median peak risk (0.136 vs 0.137) and high
z-feature rates (82.3% vs 79.4%), reinforcing the need for added context
features. Missed 30d top-5% events were mostly modelable (129 of 133) and had a
maximum pre-event risk of 0.701 but low median pre-event risk (0.034), pointing
to threshold/policy review plus event-specific feature work rather than data
cleanup alone.

The injury-context outcome run (`injury_context_outcomes_v1`, L2, window 4)
used the three uploaded detailed injury exports and wrote 3,828 event profiles
for 638 unique detailed injury events across the six horizon/threshold views,
plus 2,046 grouped context rows. At 30d top-5%, the lowest-capture severity
buckets were `0d` time-loss (369 events, 7.3% start-capture rate), `8-28d` (108
events, 8.3%), and `29d+` (82 events, 8.5%). Low-capture contexts included
lower-leg, elbow, neck, thigh, and groin/hip body-area events, S&C activity
group events, tendinopathy, bone stress injury, and hamstring strain/tear. Some
time-loss values are extremely large, so severity should be audited before it
becomes a model target, but the artifact confirms that injury subtype and
activity context are now visible enough to drive the next modeling sprint. The
injury outcome policy run (`injury_outcome_policy_v1`) audited 638 detailed
events: 625 were marked usable, 13 had extreme time-loss values above 365 days,
and none had missing, negative, or duration/resolved-date mismatch flags. The
candidate target counts were: time-loss-only 269 events, model-safe time-loss
256, moderate-plus time-loss 177, severe time-loss 69, lower-extremity 312,
soft-tissue 367, lower-extremity soft-tissue 241, concussion-only 79, and
exclude-concussion 559.

The outcome-policy model comparison run (`outcome_policy_model_comparison_v1`,
L2, window 4) compared seven target definitions. `model_safe_time_loss` was the
clearest short-horizon improvement: at top-10%, it captured 32/173 observed
athlete-season events at 7d (18.5%) and 42/173 at 14d (24.3%), with top-decile
lift of 3.28 and 2.58. At 30d, cleaner severity targets did not clearly beat the
broad target for calibration: `exclude_concussion` and `any_injury` retained
Brier skill around 0.017. `lower_extremity_soft_tissue` had the best 30d top-5%
capture rate (33/168, 19.6%) but higher alert burden.

The policy decision sprint (`policy_decision_sprint_v1`, L2, windows 2/4/7)
wrote 90 comparison rows across five target policies. The recommended
shadow-mode policy is now two-channel: `exclude_concussion`, window 4, 30d
top-5% for broad early warning; and `model_safe_time_loss`, window 4, top-10%
for 7d/14d severity triage. The broad channel captured 44/292 unique observed
events with Brier skill 0.0168 and median start lead of 12 days. The severity
channel captured 32/173 unique events at 7d and 42/173 at 14d. Window 2 improved
30d lower-extremity soft-tissue capture to 40/168, but with higher alert burden,
so that remains a subtype-review view.

The shadow-mode stability audit (`shadow_mode_stability_v1`, L2) evaluated the
fixed policy package across six season slices. All four channels were unstable.
The broad 30d channel averaged 14.8% capture but ranged from 6.1% to 33.3% in
event seasons. The 7d and 14d severity channels ranged from 0.0% to 25.5% and
0.0% to 36.2%, respectively. The lower-extremity soft-tissue review channel
ranged from 4.3% to 63.3%. The recommendation is
`review_before_shadow_pilot`; the model should stay research-only until the
season drift is explained.

The coverage-normalized policy sprint (`coverage_normalized_policy_v1`, L2)
shows that no fixed channel remained stable across `all`, `medium_high`, and
`high_only` coverage eligibility scopes. `severity_14d` was stable only in the
full-population scope, then became unstable after low-coverage athlete-seasons
were removed. Alert burden also rose materially in higher-coverage cohorts.

The coverage/source-aware model sprint (`coverage_source_aware_model_v1`, L2,
window 4) compared `graph_trajectory` against `graph_plus_coverage_source`.
Coverage/source covariates improved AUROC at all horizons: 7d 0.686 to 0.723,
14d 0.676 to 0.723, and 30d 0.689 to 0.734. Brier skill also improved from
-0.000 to 0.014 at 7d, -0.001 to 0.024 at 14d, and 0.007 to 0.048 at 30d.
Top-decile lift was mixed: unchanged at 7d, lower at 14d, and better at 30d.
The current interpretation is that coverage/source features are worth continued
research validation, but this is not dashboard or pilot clearance.

The coverage-adjusted threshold sprint (`coverage_adjusted_threshold_v1`, L2)
selected burden-capped thresholds for all fixed channels under a 1.0 episode per
athlete-season cap, but the cost was lower mean capture: `broad_30d` 10.4%,
`severity_14d` 8.8%, `severity_7d` 5.5%, and
`subtype_lower_extremity_soft_tissue_30d` 9.7%. Coverage-tier-local thresholds
kept more capture but still produced high mean burden, including 2.48 episodes
per athlete-season for `severity_14d`, 2.69 for `severity_7d`, and 2.58 for the
subtype review channel.

The season-forward validation sprint (`season_forward_validation_v1`, L2, window
4) found that `graph_plus_coverage_source` had the best observed forward ranking
slots in 2023-2024, with AUROC 0.667 at 7d, 0.673 at 14d, and 0.703 at 30d. It
also had the best observed forward calibration slots in 2025-2026, with Brier
skill 0.014 at 7d, 0.026 at 14d, and 0.048 at 30d. Earlier forward seasons were
weak or unevaluable: 2021-2022 had no discrimination metrics and 2022-2023 was
prevalence-like at AUROC 0.500. Alert-policy checks remained modest: recommended
mean capture/burden was 11.6% / 0.41 for `broad_30d`, 15.0% / 0.86 for
`severity_14d`, 6.4% / 0.47 for `severity_7d`, and 19.5% / 0.85 for subtype
review. The fixed policy package remains research-only.

The forward case review sprint (`forward_case_review_v1`, L2) reviewed 44
deterministic cases from 2023-2024 and 2025-2026 across `broad_30d`,
`severity_14d`, and `subtype_lower_extremity_soft_tissue_30d`. Diagnostics split
into 14 `model_signal_supported`, 12 `model_miss`, 12
`missing_context_or_managed_risk`, and 6 `explanation_gap`. `broad_30d` had the
strongest supported-case balance, `severity_14d` was dominated by explanation
gaps, and subtype review was dominated by missing-context or managed-risk
labels. This supports continued research, not production. The next sprint should
turn case labels into data requirements and feature plans around exposure,
intervention, baseline/frailty, and injury-mechanism context.

The case diagnostic requirements sprint (`case_diagnostic_requirements_v1`, L2)
made that production-readiness gap explicit. It reviewed the same 44 forward
case-review cases and generated five requirement domains. Four are critical:
`exposure_load` (24 cases, 54.5%), `baseline_frailty` (20, 45.5%),
`injury_mechanism` (20, 45.5%), and `intervention_availability` (12, 27.3%).
`explanation_fidelity` is high priority (8, 18.2%). The key missing fields are
session participation, minutes exposed, practice intensity, acute/chronic load,
game exposure, availability status, modified training, treatment/rehab status,
prior injury count, chronic condition flags, athlete baseline state, injury
mechanism, contact/non-contact context, activity context, body-area detail, and
graph node/edge change traces. The recommendation is
`prioritize_data_acquisition_before_production`; the model remains
`not_ready_missing_context`.

The injury-history feature sprint (`injury_history_feature_v1`, L2, window 4)
shows that the uploaded injury data does contain real model signal when used as
time-safe prior context. `graph_plus_coverage_injury_history` beat
`graph_plus_coverage_source` at every horizon and decision mode in the standard
holdout comparison. AUROC improved from 0.721 to 0.755 at 7d, 0.726 to 0.759 at
14d, and 0.743 to 0.774 at 30d. Brier skill improved from 0.015 to 0.028 at 7d,
0.026 to 0.052 at 14d, and 0.054 to 0.111 at 30d. Top-decile lift improved from
1.85 to 3.73 at 7d, 1.84 to 3.51 at 14d, and 2.13 to 3.46 at 30d. Prior-injury
context was nonzero in 30,482 of 74,860 graph snapshots, and prior
lower-extremity soft-tissue context was nonzero in 13,637 snapshots. This is
not pilot clearance yet.

The injury-history season-forward validation sprint
(`injury_history_season_forward_validation_v1`, L2, window 4) found that
`graph_plus_coverage_injury_history` survives as a ranking and triage signal but
does not cleanly win calibration. It won 7d ranking in 2025-2026 (AUROC 0.662),
30d ranking in 2023-2024 (AUROC 0.689), and top-decile lift in 2024-2025 at 7d,
14d, and 30d (2.477, 2.928, and 3.063). Calibration winners remained
`graph_plus_coverage_source` in 2025-2026 at all horizons. Injury-history alert
checks had low burden but low mean capture: `broad_30d` 5.0% capture / 0.14
episodes per athlete-season, `severity_14d` 7.9% / 0.26, `severity_7d` 4.8% /
0.29, and subtype review 8.2% / 0.37. The next research step is case/calibration
review of the injury-history forward rows, not pilot escalation. The current
test suite has 213 passing tests.

The reported risk values are baseline model estimates for research comparison,
not calibrated clinical probabilities.

## Philosophy

The athlete-season is the primary modeling unit. Daily measurements are observations inside a trajectory, not independent injury-classification examples.

## Source Materials

The project folder may contain local research PDFs and blueprint documents. They are treated as source references, not package inputs. The research pipeline expects canonical measurement and injury CSV files.

## First Milestone

The first milestone is a reproducible research engine that proves the longitudinal graph/time-to-event data contract. Dashboard performance views come after stable research artifacts exist.
