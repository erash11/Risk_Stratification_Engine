# Case Diagnostic Requirements Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn forward case-review diagnostics into prioritized data requirements and model-improvement artifacts.

**Architecture:** Add a focused `case_diagnostic_requirements.py` module that maps case diagnostics to requirement domains, priority tiers, missing fields, and modeling actions. Add an experiment runner and CLI flag that reuse the existing forward case-review cases, then write CSV, JSON, Markdown, and config artifacts.

**Tech Stack:** Python, pandas, pytest, existing `risk-engine` CLI and experiment artifact patterns.

---

### Task 1: Requirements Summary Module

**Files:**
- Create: `src/risk_stratification_engine/case_diagnostic_requirements.py`
- Test: `tests/test_case_diagnostic_requirements.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `build_case_diagnostic_requirements(...)` with cases covering `model_miss`, `missing_context_or_managed_risk`, `explanation_gap`, and `model_signal_supported`. Assert the returned rows include `exposure_load`, `intervention_availability`, `baseline_frailty`, `injury_mechanism`, and `explanation_fidelity` domains with priority tiers and recommended modeling actions.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_case_diagnostic_requirements.py -q`

Expected: fail because `risk_stratification_engine.case_diagnostic_requirements` does not exist.

- [ ] **Step 3: Implement minimal module**

Implement:
- `build_case_diagnostic_requirements(cases)`
- `build_case_diagnostic_requirements_summary(requirements, cases)`
- `write_case_diagnostic_requirements_report(path, summary)`

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_case_diagnostic_requirements.py -q`

Expected: pass.

### Task 2: Experiment Runner and CLI

**Files:**
- Modify: `src/risk_stratification_engine/experiments.py`
- Modify: `src/risk_stratification_engine/cli.py`
- Test: `tests/test_experiments.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Add an experiment test for `run_case_diagnostic_requirements_sprint_experiment(...)` using `_write_season_forward_fixture_inputs(...)`. Assert the runner writes `case_diagnostic_requirements.csv`, `case_diagnostic_requirements.json`, `case_diagnostic_requirements_report.md`, `forward_case_review_cases.csv`, and `config.json`.

Add a CLI test for `--case-diagnostic-requirements-sprint` from live sources, asserting live prepared paths and `model_variant` are passed to the runner.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_case_diagnostic_requirements.py tests/test_experiments.py::test_run_case_diagnostic_requirements_sprint_writes_requirement_artifacts tests/test_cli.py::test_cli_runs_case_diagnostic_requirements_sprint_from_live_sources -q`

Expected: fail because the runner and CLI flag do not exist.

- [ ] **Step 3: Implement minimal runner and CLI dispatch**

In `experiments.py`, import the new module, call `_forward_case_review_cases(...)`, build requirements and summary, and write artifacts.

In `cli.py`, add parser flag and detailed-injury guard matching the existing forward case-review sprint.

- [ ] **Step 4: Verify GREEN**

Run the same focused pytest command.

Expected: pass.

### Task 3: Documentation, Live Run, and Publish

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Run full verification**

Run: `python -m pytest`

Expected: all tests pass.

- [ ] **Step 2: Run live sprint**

Run:

```powershell
risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id case_diagnostic_requirements_v1 --case-diagnostic-requirements-sprint --model-variant l2
```

Expected: writes case diagnostic requirement artifacts under `outputs/experiments/case_diagnostic_requirements_v1`.

- [ ] **Step 3: Update docs**

Add the new CLI command, artifact contract, live results, and interpretation to `README.md` and `AGENTS.md`.

- [ ] **Step 4: Final checks and publish**

Run `python -m pytest`, `git diff --check`, stage intended files explicitly, commit, and push `master`.
