# Outcome Policy Model Comparison Design

## Goal

Test whether the model becomes more useful when the target is not "any injury"
but a cleaner detailed-injury policy such as model-safe time-loss,
moderate-plus time-loss, severe time-loss, lower-extremity soft-tissue,
concussion-only, or exclude-concussion.

## Implementation

- Add a policy relabeling helper that takes the canonical athlete-season injury
  table plus `injury_events_detailed.csv` and replaces each athlete-season event
  with the earliest detailed event matching the selected policy.
- Preserve censored athlete-seasons when they have no matching policy event.
- Normalize `athlete_id` and `season_id` as text when loading canonical inputs so
  numeric-looking season IDs do not break joins.
- Reuse one graph feature table per run, then train/evaluate a fresh model for
  each target policy using the same `model_variant` and `graph_window_size`.
- Convert each policy timeline into alert episodes with the existing top-5% and
  top-10% percentile thresholds.
- Write a compact comparison artifact with both model metrics and alert-quality
  metrics for each policy/horizon/threshold.

## Artifacts

- `context_policy_model_comparison.csv`
- `context_policy_model_comparison.json`
- `context_policy_model_comparison_report.md`

## CLI

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id outcome_policy_model_comparison_v1 \
  --outcome-policy-model-comparison \
  --model-variant l2 \
  --graph-window-size 4
```

## Live Result Summary

The first live run compared seven policies under L2 + graph window 4. The result
did not show a single replacement target. `model_safe_time_loss` improved
7d/14d alert capture and top-decile lift, while broad `any_injury` and
`exclude_concussion` kept the stronger 30d calibration profile. `severe_time_loss`
was too sparse to use as the primary target. The next iteration should test a
two-channel policy: broad 30d early warning plus a severity-oriented
`model_safe_time_loss` alert view.

## Verification

- Focused TDD tests covered policy relabeling, comparison artifact creation, and
  CLI dispatch.
- `python -m pytest` passed with 144 tests after implementation.
- The live command completed and wrote `outcome_policy_model_comparison_v1`.
