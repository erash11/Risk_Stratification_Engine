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
`data_quality_audit.json` beside them. The audit reports hashed identity overlap
across sources, sparse athlete-seasons, large within-season measurement gaps,
duplicate same-day metric rows, and observed injury events without nearby
measurements. It also includes review context for remaining single-source hashed
identities and injury events outside the nearby-measurement window. Athlete
identities are stable hashes of normalized names, seasons start on July 1, and
the current injury label policy uses the earliest injury issue date per
athlete-season while censoring event-free athlete-seasons at their last
measurement date. Name normalization reconciles common `Last, First` export style
with `First Last` names before hashing. Duplicate same-day metric rows are
aggregated by mean `metric_value` per athlete, season, date, source, and metric
before modeling; the aggregation counts are recorded in `prep_metadata.json`.
Observed injury events are labeled by nearest same-season measurement distance:
`modelable` at 14 days or less, `low_confidence` at 15-30 days, and
`out_of_window` beyond 30 days. When `primary_model_event` is available, the
discrete-time baseline uses it as the positive-event policy so low-confidence
and out-of-window observed events do not become training positives.

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
is designed to detect. The current test suite has 115 passing tests.

The reported risk values are baseline model estimates for research comparison,
not calibrated clinical probabilities.

## Philosophy

The athlete-season is the primary modeling unit. Daily measurements are observations inside a trajectory, not independent injury-classification examples.

## Source Materials

The project folder may contain local research PDFs and blueprint documents. They are treated as source references, not package inputs. The research pipeline expects canonical measurement and injury CSV files.

## First Milestone

The first milestone is a reproducible research engine that proves the longitudinal graph/time-to-event data contract. Dashboard performance views come after stable research artifacts exist.
