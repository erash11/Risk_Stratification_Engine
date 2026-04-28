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
- The current live-source keys are `forceplate_db`, `gps_db`, `bodyweight_csv`, `perch_db`, and `injury_csv`.
- As of 2026-04-25, the local `config/paths.local.yaml` resolves all five live-source keys successfully.
- The local injury export is in `data/raw/` as `injuries-summary-export-3ad17d.csv`; keep raw injury data ignored.
- When running against live source files, record path metadata, file existence, schemas, and row counts in experiment/data-quality artifacts for reproducibility.
- As of 2026-04-27, live-source ingestion is available through `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id <id>`.
- Live-source ingestion writes ignored canonical inputs to `outputs/live_inputs/<experiment-id>/`, uses stable hashed athlete IDs from normalized names, starts seasons on July 1, and uses the earliest injury issue date per athlete-season with censoring at the last measurement date.
- Live-source ingestion also writes `data_quality_audit.json` with hashed source-overlap checks, sparse athlete-season flags, large within-season date gaps, duplicate same-day metric rows, and injury events without nearby measurements.
- Live-source name normalization reconciles common `Last, First` export names with `First Last` names before hashing, and duplicate same-day metric rows are aggregated by mean value before modeling with counts recorded in `prep_metadata.json`.
- `data_quality_audit.json` includes privacy-preserving review context for remaining single-source hashed identities and observed injury events outside the nearby-measurement window.
- Observed live-source injury events are labeled by nearest same-season measurement distance: `modelable` at 14 days or less, `low_confidence` at 15-30 days, and `out_of_window` beyond 30 days; downstream modeling should prefer `primary_model_event = true` until calibration work says otherwise.
- As of 2026-04-27, `run_research_experiment(...)` trains a discrete-time logistic baseline at the 7, 14, and 30 day horizons over 13 graph snapshot-time features: `time_index`, `node_count`, `edge_count`, `mean_abs_correlation`, `edge_density`, `delta_edge_count`, `delta_mean_abs_correlation`, `delta_edge_density`, `graph_instability`, `z_mean_abs_correlation`, `z_edge_density`, `z_edge_count`, and `z_graph_instability`.
- The temporal delta features (`delta_*`) are computed per athlete-season in chronological order and are zero at each athlete's first snapshot; they capture change from one snapshot to the next. `edge_density` normalizes edge count by the maximum possible edges for the observed node count. `graph_instability` is the rolling population standard deviation of `mean_abs_correlation` over the most recent three snapshots, zero when fewer than two snapshots are available.
- The intra-individual z-score features compare each athlete-season snapshot against that athlete-season's own strictly prior rolling baseline using the graph `window_size`. They require at least two prior snapshots, use population standard deviation, fall back to `0.0` when the prior standard deviation is zero, and are clipped to `[-10.0, 10.0]`.
- The baseline writes `model_summary.json` with the event policy, feature columns, deterministic athlete-level 20% holdout split, per-horizon model kind, positive rates, and holdout Brier scores. If a training horizon has only one class, it records a prevalence fallback instead of fitting an unstable classifier.
- `model_evaluation.json` compares each horizon's holdout predictions against the training-prevalence baseline and reports Brier score, Brier skill score, AUROC, average precision, and top-decile lift when the holdout labels support those metrics.
- `feature_attribution.json` and `feature_ablation_report.md` compare the current 13-feature model against `original_9` and `z_score_only` feature sets using the same deterministic athlete-level holdout split. The artifact includes per-horizon holdout metrics and standardized logistic coefficients for feature attribution.
- Window sensitivity runs are available through `risk-engine --window-sensitivity-sizes <sizes...>` and write `window_sensitivity.json` plus `window_sensitivity_report.md`. The runner reuses one canonical input set, loops graph `window_size` values, and compares the same holdout metrics across windows.
- Model robustness sprints are available through `risk-engine --model-robustness-sprint --stability-splits <n>` and write `model_robustness.json` plus `model_robustness_report.md`. The sprint compares `baseline`, `l2`, `l1`, and `elasticnet` logistic variants across deterministic rotating athlete-level holdout splits, then reports decision-mode winners for ranking, calibration, and triage metrics.
- Combined window/model robustness runs are available by pairing `--model-robustness-sprint` with `--window-sensitivity-sizes <sizes...>`. They write `window_model_robustness.json` plus `window_model_robustness_report.md`, comparing model variants across each requested graph window with the same rotating athlete-level split policy.
- Main single-experiment runs accept `--model-variant baseline|l2|l1|elasticnet`, allowing a regularized candidate to be run through the normal artifact path without using the robustness sweep.
- Enriched graph features (`enriched_graph_features_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.730 (+0.008), 14d AUROC 0.735 (+0.007), 30d AUROC 0.735 (+0.007); Brier skill 30d improved from 0.0142 to 0.0168 (+18%).
- Intra-individual deviation features (`intra_individual_deviation_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.723, Brier skill 0.0020, top-decile lift 3.76; 14d AUROC 0.731, Brier skill 0.0057, top-decile lift 3.96; 30d AUROC 0.736, Brier skill 0.0171, top-decile lift 4.48. Versus `enriched_graph_features_v1`, the 30d AUROC, 7d/30d Brier skill, and all top-decile lifts improved, while 7d/14d AUROC declined slightly.
- Feature attribution/ablation (`feature_attribution_ablation_v1` run, 349 athletes, 70 holdout): `full_13` matched the `intra_individual_deviation_v1` metrics. `original_9` remained stronger on 7d/14d AUROC (0.730/0.735) but had lower 7d/30d top-decile lift (3.68/4.34) than `full_13` (3.76/4.48). `z_score_only` had weak AUROC (7d 0.566, 14d 0.553, 30d 0.491) but strong 7d top-decile lift (4.14), suggesting the z-score features are most useful as ranking modifiers inside the combined model rather than as a standalone risk model.
- Window sensitivity (`window_sensitivity_v1` run, windows 2/3/4/5/7, 349 athletes, 70 holdout): window 4 was best for AUROC at all horizons (7d 0.723, 14d 0.731, 30d 0.736); window 7 was best for Brier skill and Brier score at all horizons (7d Brier skill 0.0062, 14d 0.0119, 30d 0.0301); window 2 was best for top-decile lift at all horizons (7d 5.19, 14d 5.04, 30d 5.13). This suggests the default window 4 remains the best ranking/AUROC baseline, while longer windows improve probability sharpness and shorter windows concentrate positives in the highest-risk decile.
- Model robustness sprint (`model_robustness_sprint_v1` run, graph window 4, 5 rotating splits, 349 athletes): all regularized variants improved average AUROC, Brier skill, Brier score, and top-decile lift relative to baseline at all horizons. L2 was the strongest all-around choice, winning calibration at 7/14/30d, triage at 7/14d, and ranking at 14d. Elastic net narrowly won 30d ranking and 30d triage. L1 narrowly won 7d ranking. Regularized variants standardize features internally before fitting and convert coefficients back to raw feature units for attribution.
- Window/model robustness (`window_model_robustness_v1` run, windows 2/4/7, 5 rotating splits, 349 athletes): no single window/variant dominates all operating goals. Window 7 + L2 won calibration at 7d/14d, window 4 + L2 won 30d calibration, window 2 regularized variants won triage lift at all horizons, and ranking split by horizon: window 2 baseline at 7d AUROC 0.731, window 4 L2 at 14d AUROC 0.729, and window 7 L1 at 30d AUROC 0.729. This supports using L2 as the calibration-oriented production candidate while keeping window 2 as a high-alert triage setting and window 7 under review for 30d ranking.

## Latest Completed Step

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
