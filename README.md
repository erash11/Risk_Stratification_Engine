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

When `injury_csv` points at a file named `injuries-summary-export-*.csv`, live
preparation loads all sibling injury summary exports with that pattern from the
same folder. This supports period-sliced injury exports without manually
concatenating them.

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
season drift is explained. The current test suite has 153 passing tests.

The reported risk values are baseline model estimates for research comparison,
not calibrated clinical probabilities.

## Philosophy

The athlete-season is the primary modeling unit. Daily measurements are observations inside a trajectory, not independent injury-classification examples.

## Source Materials

The project folder may contain local research PDFs and blueprint documents. They are treated as source references, not package inputs. The research pipeline expects canonical measurement and injury CSV files.

## First Milestone

The first milestone is a reproducible research engine that proves the longitudinal graph/time-to-event data contract. Dashboard performance views come after stable research artifacts exist.
