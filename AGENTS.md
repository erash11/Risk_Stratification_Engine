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
- Enriched graph features (`enriched_graph_features_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.730 (+0.008), 14d AUROC 0.735 (+0.007), 30d AUROC 0.735 (+0.007); Brier skill 30d improved from 0.0142 to 0.0168 (+18%).
- Intra-individual deviation features (`intra_individual_deviation_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.723, Brier skill 0.0020, top-decile lift 3.76; 14d AUROC 0.731, Brier skill 0.0057, top-decile lift 3.96; 30d AUROC 0.736, Brier skill 0.0171, top-decile lift 4.48. Versus `enriched_graph_features_v1`, the 30d AUROC, 7d/30d Brier skill, and all top-decile lifts improved, while 7d/14d AUROC declined slightly.
- Feature attribution/ablation (`feature_attribution_ablation_v1` run, 349 athletes, 70 holdout): `full_13` matched the `intra_individual_deviation_v1` metrics. `original_9` remained stronger on 7d/14d AUROC (0.730/0.735) but had lower 7d/30d top-decile lift (3.68/4.34) than `full_13` (3.76/4.48). `z_score_only` had weak AUROC (7d 0.566, 14d 0.553, 30d 0.491) but strong 7d top-decile lift (4.14), suggesting the z-score features are most useful as ranking modifiers inside the combined model rather than as a standalone risk model.

## Latest Completed Step

**Feature attribution and ablation artifacts** — implemented and verified on 2026-04-27.

**What changed:** `train_discrete_time_risk_model(...)` now accepts explicit feature subsets and records standardized coefficient attribution for each horizon. `run_research_experiment(...)` now writes `feature_attribution.json` and `feature_ablation_report.md` for three feature sets: `full_13`, `original_9`, and `z_score_only`.

**Verification:** New tests first failed against the old behavior because feature subset training, coefficient attribution, and the artifact files did not exist. After implementation, `python -m pytest` collected and passed 84 tests. The live experiment command `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id feature_attribution_ablation_v1` completed and wrote the new attribution/ablation artifacts.

**Interpretation:** `edge_count` remains the dominant standardized coefficient in the combined and original feature sets across horizons. The z-score-only model is not enough by itself, especially at 30d, but z-score features improve lift in the combined model. The logical next research step is window-size sensitivity or regularized feature selection, not adding more raw features yet.

## Previous Completed Step

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
