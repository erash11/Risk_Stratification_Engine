# Intra-Individual Deviation Features — Design Spec

**Date:** 2026-04-27
**Status:** Approved, ready for implementation
**Branch:** master

---

## Motivation

Peterson's methodology is fundamentally intra-individual: injury risk emerges from departure of an athlete's own dynamic physiological system from its recent baseline, not from where the athlete sits in the population. The current 9-feature baseline uses only absolute graph values, making it a population-level model. This step adds z-score deviation features that capture "how unusual is this athlete right now relative to their own recent history," implementing the core of Peterson's philosophy.

---

## Feature Definitions

Four new z-score features, one for each graph state feature that reflects physiological system dynamics:

| New column | Source feature |
|---|---|
| `z_mean_abs_correlation` | `mean_abs_correlation` |
| `z_edge_density` | `edge_density` |
| `z_edge_count` | `edge_count` |
| `z_graph_instability` | `graph_instability` |

**Excluded from z-scoring:**
- `node_count` — structural count that does not change meaningfully within a season
- `delta_*` features — already capture change; z-scoring them would produce noisy second-derivatives
- `time_index` — ordinal position, not a physiological state measure

**Computation per snapshot at position i (per athlete-season, chronological order):**

```
baseline_window = group_rows[max(0, i − window_size + 1) : i]   # strictly prior, no current row
```

- If `len(baseline_window) < 2` → z-score = 0.0
- Else: `z = (current_value − mean(baseline)) / std(baseline)`  using population std (ddof=0)
- If `std(baseline) == 0` → z-score = 0.0
- Clip final value to `[-10.0, 10.0]`
- Round to 6 decimal places

**Window:** same `window_size` parameter passed to `build_graph_snapshots` (default 4). This means for the default window, the baseline uses at most 3 prior snapshots.

---

## Architecture

**Files changed:**

### `src/risk_stratification_engine/graphs.py`
- Add 4 columns to `OUTPUT_COLUMNS` (after `graph_instability`):
  `z_mean_abs_correlation`, `z_edge_density`, `z_edge_count`, `z_graph_instability`
- Extend `_add_temporal_features(group_rows, window_size)` to accept `window_size` and compute the 4 z-scores using the baseline window logic above
- Update `build_graph_snapshots` to pass `window_size` to `_add_temporal_features`

### `src/risk_stratification_engine/models.py`
- Add the 4 new z-score columns to `GRAPH_SNAPSHOT_FEATURE_COLUMNS` (13 total)

### `tests/test_graphs.py`
- Add 6 new tests (see Testing section)
- Import `OUTPUT_COLUMNS` from `graphs.py` — already done

### `tests/test_models.py`
- Add 4 new columns to `_labeled_snapshot_frame()` with sensible numeric placeholder values

### `tests/test_experiments.py`
- Update feature_columns assertion to include all 13 columns

**No other files change.**

---

## Testing (TDD — write tests first, watch them fail, then implement)

Six new tests in `test_graphs.py`:

1. **`test_build_graph_snapshots_includes_z_score_feature_columns`**
   Assert all 4 z-score column names are present in `OUTPUT_COLUMNS` and in the snapshots DataFrame.

2. **`test_build_graph_snapshots_z_scores_are_zero_at_first_snapshot`**
   At `time_index == 0` for any athlete, all 4 z-score features equal 0.0 (no prior baseline).

3. **`test_build_graph_snapshots_z_scores_are_zero_at_second_snapshot`**
   At `time_index == 1`, all 4 z-score features equal 0.0 (only 1 prior point, below minimum-2 threshold).

4. **`test_build_graph_snapshots_z_score_nonzero_once_baseline_has_two_prior_snapshots`**
   Build a minimal inline fixture with 3 snapshots where the third snapshot departs from the first two. Assert `z_mean_abs_correlation != 0.0` at `time_index == 2`. Use a 3-row synthetic measurement matrix so the expected z-score is calculable by hand.

5. **`test_build_graph_snapshots_z_score_is_zero_when_baseline_std_is_zero`**
   Build a fixture where the prior snapshots have identical `mean_abs_correlation` values. Assert `z_mean_abs_correlation == 0.0` at the departure snapshot (std-zero fallback, not div-by-zero).

6. **`test_build_graph_snapshots_z_score_clips_extreme_departures`**
   Build a fixture where the current value is extremely far from the baseline mean. Assert the z-score is clamped to 10.0 (or −10.0).

---

## Live Experiment

After implementation, run:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id intra_individual_deviation_v1
```

Compare `outputs/experiments/intra_individual_deviation_v1/model_evaluation.json`
against `outputs/experiments/enriched_graph_features_v1/model_evaluation.json`.

**Baseline to beat (enriched_graph_features_v1):**
- 7d AUROC 0.730, Brier skill 0.0017, top-decile lift 3.68
- 14d AUROC 0.735, Brier skill 0.0058, top-decile lift 3.70
- 30d AUROC 0.735, Brier skill 0.0168, top-decile lift 4.34

---

## Success Criteria

- 76 + 6 = 82 tests pass
- All 4 z-score columns appear in `model_summary.json → feature_columns`
- Live experiment completes without error
- `model_evaluation.json` is written and parseable
- AUROC and Brier skill at one or more horizons improve over `enriched_graph_features_v1`

---

## Implementation Notes

- Use `np.std(arr)` (ddof=0, population std) — consistent with `graph_instability` computation
- The baseline window is **strictly prior** (does not include the current row) — this prevents any data leakage
- `_add_temporal_features` currently takes only `group_rows`; add `window_size: int` as a second parameter and thread it through from `build_graph_snapshots`
- Keep z-score computation in `_add_temporal_features` after the delta/density block so `edge_density` and `graph_instability` are already populated when their z-scores are computed
- After implementation, update `AGENTS.md` with new feature list, test count, and live experiment results
- Commit only: `graphs.py`, `models.py`, `tests/test_graphs.py`, `tests/test_models.py`, `tests/test_experiments.py`, `AGENTS.md`, `README.md`
- Push to `origin master` after verification
