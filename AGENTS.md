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
- Local injury exports are in `data/raw/` as ignored `injuries-summary-export-*.csv` files; keep raw injury data ignored.
- When running against live source files, record path metadata, file existence, schemas, and row counts in experiment/data-quality artifacts for reproducibility.
- As of 2026-04-27, live-source ingestion is available through `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id <id>`.
- Live-source ingestion writes ignored canonical inputs to `outputs/live_inputs/<experiment-id>/`, uses stable hashed athlete IDs from normalized names, starts seasons on July 1, and uses the earliest injury issue date per athlete-season with censoring at the last measurement date.
- If `injury_csv` points to a file named `injuries-summary-export-*.csv`, live-source ingestion loads all sibling files with that pattern, de-duplicates exact raw rows, and writes `injury_events_detailed.csv` with one de-identified row per raw injury event. The detailed artifact preserves issue/resolved dates, duration and time-loss fields, recurrence, unavailability, activity, classification/pathology, body area, tissue type, side, participation level, training/game context, source file, and source row number while leaving `canonical_injuries.csv` as the current first-event-per-athlete-season modeling label file.
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
- Outcome-policy model comparison runs are available through `risk-engine --outcome-policy-model-comparison` and write `context_policy_model_comparison.csv`, `context_policy_model_comparison.json`, and `context_policy_model_comparison_report.md`. The runner relabels canonical athlete-seasons against selected detailed injury policies, retrains the same graph model for each target, and compares model metrics plus alert episode quality at top-5% and top-10%.
- Policy decision sprint runs are available through `risk-engine --policy-decision-sprint --policy-window-sizes 2 4 7` and write `two_channel_alert_policy.json`, `two_channel_alert_policy_report.md`, `policy_window_sensitivity.csv`, `policy_window_sensitivity.json`, `policy_window_sensitivity_report.md`, `operational_policy_package.json`, and `operational_policy_package_report.md`. The sprint converts policy-comparison evidence into a two-channel research policy, window-sensitivity recommendations, and a shadow-mode operating package.
- Main single-experiment runs accept `--model-variant baseline|l2|l1|elasticnet`, allowing a regularized candidate to be run through the normal artifact path without using the robustness sweep.
- Enriched graph features (`enriched_graph_features_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.730 (+0.008), 14d AUROC 0.735 (+0.007), 30d AUROC 0.735 (+0.007); Brier skill 30d improved from 0.0142 to 0.0168 (+18%).
- Intra-individual deviation features (`intra_individual_deviation_v1` run, 349 athletes, 70 holdout): 7d AUROC 0.723, Brier skill 0.0020, top-decile lift 3.76; 14d AUROC 0.731, Brier skill 0.0057, top-decile lift 3.96; 30d AUROC 0.736, Brier skill 0.0171, top-decile lift 4.48. Versus `enriched_graph_features_v1`, the 30d AUROC, 7d/30d Brier skill, and all top-decile lifts improved, while 7d/14d AUROC declined slightly.
- Feature attribution/ablation (`feature_attribution_ablation_v1` run, 349 athletes, 70 holdout): `full_13` matched the `intra_individual_deviation_v1` metrics. `original_9` remained stronger on 7d/14d AUROC (0.730/0.735) but had lower 7d/30d top-decile lift (3.68/4.34) than `full_13` (3.76/4.48). `z_score_only` had weak AUROC (7d 0.566, 14d 0.553, 30d 0.491) but strong 7d top-decile lift (4.14), suggesting the z-score features are most useful as ranking modifiers inside the combined model rather than as a standalone risk model.
- Window sensitivity (`window_sensitivity_v1` run, windows 2/3/4/5/7, 349 athletes, 70 holdout): window 4 was best for AUROC at all horizons (7d 0.723, 14d 0.731, 30d 0.736); window 7 was best for Brier skill and Brier score at all horizons (7d Brier skill 0.0062, 14d 0.0119, 30d 0.0301); window 2 was best for top-decile lift at all horizons (7d 5.19, 14d 5.04, 30d 5.13). This suggests the default window 4 remains the best ranking/AUROC baseline, while longer windows improve probability sharpness and shorter windows concentrate positives in the highest-risk decile.
- Model robustness sprint (`model_robustness_sprint_v1` run, graph window 4, 5 rotating splits, 349 athletes): all regularized variants improved average AUROC, Brier skill, Brier score, and top-decile lift relative to baseline at all horizons. L2 was the strongest all-around choice, winning calibration at 7/14/30d, triage at 7/14d, and ranking at 14d. Elastic net narrowly won 30d ranking and 30d triage. L1 narrowly won 7d ranking. Regularized variants standardize features internally before fitting and convert coefficients back to raw feature units for attribution.
- Window/model robustness (`window_model_robustness_v1` run, windows 2/4/7, 5 rotating splits, 349 athletes): no single window/variant dominates all operating goals. Window 7 + L2 won calibration at 7d/14d, window 4 + L2 won 30d calibration, window 2 regularized variants won triage lift at all horizons, and ranking split by horizon: window 2 baseline at 7d AUROC 0.731, window 4 L2 at 14d AUROC 0.729, and window 7 L1 at 30d AUROC 0.729. This supports using L2 as the calibration-oriented production candidate while keeping window 2 as a high-alert triage setting and window 7 under review for 30d ranking.

## Latest Completed Step

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
