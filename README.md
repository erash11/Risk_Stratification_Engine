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
