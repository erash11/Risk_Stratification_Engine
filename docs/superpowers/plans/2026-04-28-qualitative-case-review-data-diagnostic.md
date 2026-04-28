# Qualitative Case Review + Data Diagnostic v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic qualitative case-review artifacts that explain selected true-positive, false-positive, missed-injury, and high-deviation alert cases.

**Architecture:** Add a focused `case_review.py` module that consumes alert episodes, alert timeline rows, and episode-quality output. The alert episode experiment writes JSON and Markdown case-review artifacts after the quality audit artifacts.

**Tech Stack:** Python, pandas, pytest, existing `risk-engine --alert-episodes` workflow.

---

### Task 1: Case Review Builder

**Files:**
- Create: `src/risk_stratification_engine/case_review.py`
- Test: `tests/test_case_review.py`

- [ ] **Step 1: Write failing tests**

Add tests that build a small alert timeline, episodes table, and quality payload. Assert that `build_qualitative_case_review(...)` selects true-positive, false-positive, missed-injury, and high-deviation cases and assigns useful diagnostic labels.

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_case_review.py -v`

Expected: import failure for `risk_stratification_engine.case_review`.

- [ ] **Step 3: Implement builder**

Create `build_qualitative_case_review(episodes, alert_timeline, quality)` returning `case_count`, `cases`, and `diagnostic_summary`. Include helpers for matching representative cases, event identity, timeline context, feature-list parsing, and JSON-safe values.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_case_review.py -v`

Expected: all case-review tests pass.

### Task 2: Experiment Artifact Integration

**Files:**
- Modify: `src/risk_stratification_engine/experiments.py`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing integration test**

Extend the alert episode experiment test to assert `qualitative_case_review.json` and `qualitative_case_review_report.md` are written and include case records.

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_experiments.py::test_run_alert_episode_experiment_writes_episode_artifacts -v`

Expected: failure because qualitative case-review artifacts are missing.

- [ ] **Step 3: Implement artifact writes**

Import `build_qualitative_case_review`, call it after quality audit construction, write JSON, and add `_write_qualitative_case_review_report(...)`.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_case_review.py tests/test_experiments.py::test_run_alert_episode_experiment_writes_episode_artifacts -v`

Expected: targeted tests pass.

### Task 3: Documentation and Live Verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Run full tests**

Run: `python -m pytest`

Expected: full test suite passes.

- [ ] **Step 2: Run live command**

Run:

```bash
risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id qualitative_case_review_v1 --alert-episodes --model-variant l2 --graph-window-size 4
```

Expected: existing alert episode and quality artifacts plus `qualitative_case_review.json` and `qualitative_case_review_report.md`.

- [ ] **Step 3: Update docs**

Summarize the live case-review diagnostic counts and the recommended performance-improvement direction in `README.md` and `AGENTS.md`.

- [ ] **Step 4: Commit and push**

Stage intended files explicitly. Commit with `feat: add qualitative case review diagnostics`, then push `master` to `origin`.
