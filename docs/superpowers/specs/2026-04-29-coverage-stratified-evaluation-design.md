# Coverage-Stratified Evaluation Design

## Goal

Test whether the existing shadow-mode policy signal is driven by true physiological
patterns or is confounded by measurement coverage differences across athlete-seasons.

The season drift diagnostic showed that 2025-2026 has ~25x more measurement rows and
4 sources vs. 1 source in 2022-2024, while injury event volumes are similar. The open
question is whether high-coverage athletes score higher regardless of season — i.e.,
whether coverage tier predicts channel capture rate more than true risk does.

This sprint produces a diagnostic artifact only. No retraining with coverage features,
no policy changes. If coverage is confounding model scores, that finding shapes the
next sprint.

## Research Question

Do athletes in the high coverage tier get captured by shadow-mode channels at
substantially higher rates than low-coverage athletes, independent of season?

## CLI

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id coverage_stratified_eval_v1 \
  --coverage-stratified-evaluation \
  --model-variant l2
```

## Architecture

```
coverage_analysis.py        new module, follows the shape of season_drift.py
  build_coverage_tiers(measurements)
  build_coverage_stratified_evaluation(alert_timeline_with_tiers, channel)
  write_coverage_stratified_evaluation_report(path, result)

experiments.py              new runner
  run_coverage_stratified_evaluation_experiment(
      measurements_path, injuries_path, detailed_injuries_path,
      output_dir, experiment_id, model_variant
  )

cli.py                      new --coverage-stratified-evaluation flag
```

`coverage_analysis.py` contains data-transformation functions and a report writer
(same pattern as `season_drift.py`). The runner owns all other disk writes and
joins `coverage_tiers` onto the alert timeline before passing it to
`build_coverage_stratified_evaluation`.

## Coverage Tier Definition

`build_coverage_tiers(measurements)` returns one row per `athlete_id × season_id`
with these columns:

- `measurement_days` — unique calendar dates with any measurement (primary sort key)
- `measurement_row_count` — total measurement rows
- `source_count` — distinct sources for that athlete-season
- `median_days_between_measurements` — reuses the computation from `season_drift.py`
- `coverage_tier` — `low` / `medium` / `high`

Tier assignment uses a population-wide tertile of `measurement_days` across all
athlete-seasons. Bottom third → `low`, middle third → `medium`, top third → `high`.
Ties at a boundary fall into the lower tier. Athlete-seasons with zero measurement
days (not expected in canonical data) receive `low`.

`measurement_days` is preferred over raw row count because multiple sources logging
many rows on a single day does not improve temporal coverage. Source count and median
gap are included as supplementary context, not as tier criteria.

## Data Flow in the Runner

1. Load measurements, canonical injuries, detailed injuries.
2. `build_coverage_tiers(measurements)` → `coverage_tiers` DataFrame.
3. For each of the four fixed shadow-mode channels (same channels as
   `run_shadow_mode_stability_experiment`):
   - Build graph snapshots (window-cached across channels).
   - Relabel injuries by channel policy.
   - Train model with `model_variant`.
   - Build alert timeline (same as `_shadow_mode_stability_frame` internals).
   - Join `coverage_tiers` onto the alert timeline by `athlete_id × season_id`.
   - Call `build_coverage_stratified_evaluation(alert_timeline_with_tiers,
     channel)` → per-tier and per-tier×season episode quality stats.
4. Aggregate all channel results into the JSON summary.
5. Write four artifacts.

## Artifacts

| File | Content |
|---|---|
| `coverage_tiers.csv` | One row per athlete-season with coverage metrics and tier |
| `coverage_stratified_evaluation.csv` | One row per channel × tier × season |
| `coverage_stratified_evaluation.json` | Structured summary + coverage_flag |
| `coverage_stratified_evaluation_report.md` | Human-readable tables + interpretation |

### coverage_tiers.csv columns

`athlete_id`, `season_id`, `measurement_days`, `measurement_row_count`,
`source_count`, `median_days_between_measurements`, `coverage_tier`

### coverage_stratified_evaluation.csv columns

`channel_name`, `coverage_tier`, `season_id`, `athlete_season_count`,
`observed_event_count`, `captured_event_count`, `capture_rate`,
`episodes_per_athlete_season`, `mean_measurement_days`

### coverage_stratified_evaluation.json summary fields

- `experiment_type`: `"coverage_stratified_evaluation"`
- `tier_distribution`: count of athlete-seasons per tier
- `channel_results`: per-channel tier summaries collapsed across seasons (mean
  capture rate per tier, mean burden per tier)
- `coverage_flag`: see below

## Coverage Flag Logic

Computed from the mean capture-rate difference between high and low tiers, averaged
across channels:

| Flag | Condition |
|---|---|
| `coverage_confounded` | High-tier mean capture ≥ low-tier mean capture + 15 pp on average |
| `coverage_independent` | Difference < 5 pp consistently across all channels |
| `mixed` | Between 5–15 pp, or inconsistent across channels |

## Tests (TDD — all written before implementation)

1. `build_coverage_tiers` with empty measurements → empty DataFrame with correct
   columns (`athlete_id`, `season_id`, `measurement_days`, `measurement_row_count`,
   `source_count`, `median_days_between_measurements`, `coverage_tier`).
2. `build_coverage_tiers` with varied athlete-seasons → correct tertile split,
   correct `measurement_days` computation (unique dates, not row count).
3. `build_coverage_stratified_evaluation` with a synthetic alert timeline
   (already tier-joined) → correct grouping, correct capture rates per tier.
4. Integration test: `run_coverage_stratified_evaluation_experiment` with fixture
   data → all four artifact files written with correct top-level structure.

## Implementation Notes

- Re-use `_shadow_mode_stability_frame`'s internal channel loop structure rather
  than calling `_shadow_mode_stability_frame` itself, because that function
  aggregates to season level and discards per-athlete-season scores.
- `median_days_between_measurements` computation can be shared with or extracted
  from `season_drift.py`'s `_median_days_between_measurements` (currently private).
  If sharing is clean, extract to a shared helper; if not, duplicate the logic
  in `coverage_analysis.py`.
- Keep `coverage_analysis.py` free of experiment-runner imports. Only pandas and
  standard library are allowed as dependencies.
- The coverage flag is a research interpretation aid, not a hard gate.
