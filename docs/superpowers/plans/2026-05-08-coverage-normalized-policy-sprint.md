# Coverage-Normalized Policy Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a coverage-normalized shadow-mode policy sprint that tests whether fixed Peterson-style graph alert channels remain stable after athlete-season coverage eligibility controls.

**Architecture:** Keep the existing longitudinal graph model and fixed shadow-mode channel definitions. Add a small coverage policy module that summarizes stability audits by eligibility scope, then add an experiment runner and CLI flag that filter complete athlete-seasons by coverage tier before recomputing season-local alert episodes.

**Tech Stack:** Python, pandas, pytest, existing `risk_stratification_engine` experiment and CLI patterns.

---

### Task 1: Coverage-Normalized Policy Summary Module

**Files:**
- Create: `src/risk_stratification_engine/coverage_policy.py`
- Test: `tests/test_coverage_policy.py`

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd

from risk_stratification_engine.coverage_policy import (
    COVERAGE_ELIGIBILITY_SCOPES,
    build_coverage_normalized_policy_summary,
    write_coverage_normalized_policy_report,
)


def test_build_coverage_normalized_policy_summary_marks_ready_channel_when_scopes_are_stable():
    audits = {
        "all": {
            "channel_summaries": [
                {"channel_name": "severity_14d", "stability_status": "stable"},
                {"channel_name": "broad_30d", "stability_status": "unstable"},
            ]
        },
        "medium_high": {
            "channel_summaries": [
                {"channel_name": "severity_14d", "stability_status": "stable"},
                {"channel_name": "broad_30d", "stability_status": "unstable"},
            ]
        },
        "high_only": {
            "channel_summaries": [
                {"channel_name": "severity_14d", "stability_status": "stable"},
                {"channel_name": "broad_30d", "stability_status": "unstable"},
            ]
        },
    }

    summary = build_coverage_normalized_policy_summary(audits)

    assert summary["experiment_type"] == "coverage_normalized_policy_sprint"
    assert summary["eligibility_scopes"] == list(COVERAGE_ELIGIBILITY_SCOPES)
    assert summary["channel_recommendations"]["severity_14d"]["recommendation"] == "candidate_after_coverage_control"
    assert summary["channel_recommendations"]["broad_30d"]["recommendation"] == "continue_research_review"


def test_write_coverage_normalized_policy_report_names_peterson_guardrail(tmp_path):
    summary = {
        "experiment_type": "coverage_normalized_policy_sprint",
        "overall_recommendation": "continue_research_shadow_mode",
        "eligibility_scopes": ["all"],
        "channel_recommendations": {
            "severity_14d": {
                "recommendation": "candidate_after_coverage_control",
                "stable_scope_count": 1,
                "evaluated_scope_count": 1,
                "scope_statuses": {"all": "stable"},
            }
        },
    }
    path = tmp_path / "coverage_normalized_policy_report.md"

    write_coverage_normalized_policy_report(path, summary)

    text = path.read_text(encoding="utf-8")
    assert "Coverage-Normalized Policy Sprint" in text
    assert "athlete-season trajectories" in text
    assert "severity_14d" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coverage_policy.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.coverage_policy'`.

- [ ] **Step 3: Implement minimal module**

Create constants for `all`, `medium_high`, and `high_only` eligibility scopes. Implement summary aggregation from existing shadow-mode audit payloads and a Markdown report writer that explicitly states the Peterson guardrail: filtering complete athlete-season trajectories, not independent daily rows.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coverage_policy.py -v`

Expected: PASS.

### Task 2: Experiment Runner

**Files:**
- Modify: `src/risk_stratification_engine/experiments.py`
- Test: `tests/test_experiments.py`

- [ ] **Step 1: Write failing integration test**

Add a test named `test_run_coverage_normalized_policy_sprint_writes_scope_artifacts` using `_write_policy_fixture_inputs`. Assert that the runner writes `coverage_normalized_policy.csv`, `coverage_normalized_policy.json`, `coverage_normalized_policy_report.md`, and `config.json`, and that the JSON payload includes `coverage_eligibility_scopes`, `scope_audits`, and `channel_recommendations`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_experiments.py::test_run_coverage_normalized_policy_sprint_writes_scope_artifacts -v`

Expected: FAIL because `run_coverage_normalized_policy_sprint_experiment` does not exist.

- [ ] **Step 3: Implement runner**

Add `run_coverage_normalized_policy_sprint_experiment(...)`. Build coverage tiers from canonical measurements, build graph timelines per fixed shadow-mode channel, merge coverage tier onto each timeline, filter complete athlete-seasons by each scope, recompute season-local alert episodes, audit each scope with `build_shadow_mode_stability_audit`, and write summary artifacts.

- [ ] **Step 4: Run focused test**

Run: `python -m pytest tests/test_experiments.py::test_run_coverage_normalized_policy_sprint_writes_scope_artifacts -v`

Expected: PASS.

### Task 3: CLI Dispatch

**Files:**
- Modify: `src/risk_stratification_engine/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add a test named `test_cli_runs_coverage_normalized_policy_sprint_from_live_sources`. Mirror the existing coverage-stratified CLI test but assert dispatch to `run_coverage_normalized_policy_sprint_experiment` when `--coverage-normalized-policy-sprint` is passed.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_cli_runs_coverage_normalized_policy_sprint_from_live_sources -v`

Expected: FAIL because the parser/dispatch does not know the flag.

- [ ] **Step 3: Implement CLI flag and dispatch**

Add `--coverage-normalized-policy-sprint` and dispatch before broader shadow-mode/default branches. Require live-source or sibling `injury_events_detailed.csv` exactly like the existing coverage and shadow-mode modes.

- [ ] **Step 4: Run focused test**

Run: `python -m pytest tests/test_cli.py::test_cli_runs_coverage_normalized_policy_sprint_from_live_sources -v`

Expected: PASS.

### Task 4: Verification and Docs

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md` if the CLI command list or latest sprint summary needs updating.

- [ ] **Step 1: Run full tests**

Run: `python -m pytest`

Expected: all tests pass.

- [ ] **Step 2: Run live-source sprint**

Run: `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id coverage_normalized_policy_v1 --coverage-normalized-policy-sprint --model-variant l2`

Expected: artifacts written under `outputs/experiments/coverage_normalized_policy_v1`.

- [ ] **Step 3: Update repo notes**

Record the new CLI mode, artifacts, live result summary, and interpretation in `AGENTS.md`. Update `README.md` only if the user-facing workflow list is stale.

- [ ] **Step 4: Commit and push**

Run: `git status --short`, stage intended files explicitly, commit, and push.
