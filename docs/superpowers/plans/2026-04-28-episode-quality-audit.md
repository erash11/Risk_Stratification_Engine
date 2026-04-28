# Episode Quality Audit v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add episode-level quality audit artifacts that separate useful alert episodes from noisy alert episodes.

**Architecture:** Keep alert episode construction in `alert_episodes.py` and add a new `episode_quality.py` module for second-pass analysis. The alert episode experiment will write quality CSV, JSON, and Markdown artifacts after the existing episode files.

**Tech Stack:** Python, pandas, pytest, existing `risk-engine` CLI and JSON/CSV artifact helpers.

---

### Task 1: Quality Metric Builder

**Files:**
- Create: `src/risk_stratification_engine/episode_quality.py`
- Test: `tests/test_episode_quality.py`

- [ ] **Step 1: Write failing tests**

Add tests that construct a small timeline and episode table. Assert that `build_alert_episode_quality(...)` reports true-positive and false-positive counts, unique injury capture, missed injuries, alert burden per athlete-season, median lead time, top feature counts, and elevated z-feature rates.

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_episode_quality.py -v`

Expected: import failure for `risk_stratification_engine.episode_quality`.

- [ ] **Step 3: Implement builder**

Create `build_alert_episode_quality(episodes, timeline)` returning a dictionary with `quality_rows`, `threshold_overlaps`, and `representative_cases`. Include helper functions for event identity, episode identity, JSON-safe values, rates, and feature list parsing.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_episode_quality.py -v`

Expected: all new quality-builder tests pass.

### Task 2: Experiment Artifact Integration

**Files:**
- Modify: `src/risk_stratification_engine/experiments.py`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing experiment test**

Extend `test_run_alert_episode_experiment_writes_episode_artifacts` to assert `alert_episode_quality.csv`, `alert_episode_quality.json`, and `alert_episode_quality_report.md` exist and contain quality rows.

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_experiments.py::test_run_alert_episode_experiment_writes_episode_artifacts -v`

Expected: failure because the new quality artifacts are missing.

- [ ] **Step 3: Implement artifact writes**

Import `build_alert_episode_quality`, call it after building episodes, write a quality CSV from `quality_rows`, write JSON metadata, and add `_write_alert_episode_quality_report(...)`.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_experiments.py::test_run_alert_episode_experiment_writes_episode_artifacts tests/test_episode_quality.py -v`

Expected: targeted tests pass.

### Task 3: Documentation and Live Verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update docs**

Document the new quality artifacts and add the live `alert_episode_quality_v1` result summary after verification.

- [ ] **Step 2: Run full tests**

Run: `python -m pytest`

Expected: full test suite passes.

- [ ] **Step 3: Run live command**

Run:

```bash
risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id alert_episode_quality_v1 --alert-episodes --model-variant l2 --graph-window-size 4
```

Expected: existing alert episode artifacts plus the new quality artifacts are written.

- [ ] **Step 4: Commit and push**

Stage intended source, test, docs, and spec/plan files explicitly. Commit with `feat: add episode quality audit artifacts`, then push `master` to `origin`.
