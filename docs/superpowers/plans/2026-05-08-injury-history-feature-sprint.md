# Injury History Feature Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive time-safe injury-history, frailty, and mechanism-context features from the uploaded detailed injury data and test whether they improve forward model evidence.

**Architecture:** Add an `injury_history_features.py` module that attaches prior-injury context to graph snapshot rows using only detailed injury events before each snapshot date. Add an injury-history model sprint that compares `graph_plus_coverage_source` against `graph_plus_coverage_injury_history`, then writes feature, comparison, JSON, report, and config artifacts.

**Tech Stack:** Python, pandas, pytest, existing `risk-engine` CLI and experiment artifact patterns.

---

### Task 1: Time-Safe Injury History Features

**Files:**
- Create: `src/risk_stratification_engine/injury_history_features.py`
- Test: `tests/test_injury_history_features.py`

- [ ] **Step 1: Write failing tests**

Add tests proving features use only injuries before each snapshot date. Include prior injury count, prior same-season count, days since last injury, prior time-loss sums/max, prior lower-extremity count, prior soft-tissue count, prior lower-extremity soft-tissue count, prior game/practice/S&C injury counts, and prior caused-unavailability count.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_injury_history_features.py -q`

Expected: fail because `risk_stratification_engine.injury_history_features` does not exist.

- [ ] **Step 3: Implement minimal feature module**

Implement `INJURY_HISTORY_FEATURE_COLUMNS` and `attach_injury_history_features(graph_features, detailed_injuries)`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_injury_history_features.py -q`

Expected: pass.

### Task 2: Sprint Summary and CLI Runner

**Files:**
- Create: `src/risk_stratification_engine/injury_history_modeling.py`
- Modify: `src/risk_stratification_engine/experiments.py`
- Modify: `src/risk_stratification_engine/cli.py`
- Test: `tests/test_injury_history_modeling.py`
- Test: `tests/test_experiments.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Add summary/report tests for the new sprint. Add an experiment test asserting `run_injury_history_feature_sprint_experiment(...)` writes `injury_history_features.csv`, `injury_history_model_comparison.csv`, `injury_history_model_comparison.json`, `injury_history_model_comparison_report.md`, and `config.json`. Add a CLI test for `--injury-history-feature-sprint`.

- [ ] **Step 2: Verify RED**

Run the focused test set and confirm failures are missing module/runner/CLI flag failures.

- [ ] **Step 3: Implement minimal runner and CLI dispatch**

Attach coverage/source features, attach injury-history features, compare feature sets, evaluate with the existing holdout path, and write artifacts.

- [ ] **Step 4: Verify GREEN**

Run the focused test set and confirm it passes.

### Task 3: Live Run, Docs, and Publish

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Run full verification**

Run: `python -m pytest`

Expected: all tests pass.

- [ ] **Step 2: Run live sprint**

Run:

```powershell
risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id injury_history_feature_v1 --injury-history-feature-sprint --model-variant l2 --graph-window-size 4
```

Expected: writes injury-history feature and comparison artifacts under `outputs/experiments/injury_history_feature_v1`.

- [ ] **Step 3: Update docs**

Document the CLI command, artifact contract, live result, and interpretation in `README.md` and `AGENTS.md`.

- [ ] **Step 4: Commit and push**

Run final `python -m pytest`, `git diff --check`, stage intended files explicitly, commit, and push `master`.
