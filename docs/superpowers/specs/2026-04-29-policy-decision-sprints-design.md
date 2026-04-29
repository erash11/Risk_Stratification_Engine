# Policy Decision Sprints Design

## Goal

Execute the next three research-workbench sprints after the outcome-policy
comparison:

1. Two-channel alert policy.
2. Policy/window sensitivity.
3. Operational policy package.

## Sprint 1: Two-Channel Alert Policy

The two-channel artifact turns comparison rows into a practical research policy:

- A broad 30d early-warning channel selected from `any_injury` and
  `exclude_concussion` by Brier skill at top-5%.
- A short-horizon severity channel using `model_safe_time_loss`, selected by
  unique event capture at 7d and 14d.
- A lower-extremity soft-tissue subtype-review channel selected by unique event
  capture at 30d.

Artifacts:

- `two_channel_alert_policy.json`
- `two_channel_alert_policy_report.md`

## Sprint 2: Policy Window Sensitivity

The window sensitivity artifact retrains selected target policies across graph
windows 2/4/7 and keeps the same top-5%/top-10% alert episode evaluation. This
checks whether the preferred policy changes once the target definition changes.

Artifacts:

- `policy_window_sensitivity.csv`
- `policy_window_sensitivity.json`
- `policy_window_sensitivity_report.md`

## Sprint 3: Operational Policy Package

The package artifact converts the two-channel and window sensitivity outputs
into a concise research recommendation. It explicitly marks the system as
`research_shadow_mode`, identifies targets that should not be primary operating
targets yet, and names the next sprint.

Artifacts:

- `operational_policy_package.json`
- `operational_policy_package_report.md`

## CLI

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id policy_decision_sprint_v1 \
  --policy-decision-sprint \
  --policy-window-sizes 2 4 7 \
  --model-variant l2
```

## Live Result Summary

The first live run wrote 90 comparison rows across five target policies and
three graph windows. Window 4 remained the recommendation for the broad 30d
early-warning channel and for the 7d/14d `model_safe_time_loss` severity
channel. Window 2 produced the strongest 30d lower-extremity soft-tissue capture,
but with higher alert burden, so that channel remains a subtype-review view
rather than a primary policy.

## Verification

- Focused TDD tests covered pure policy selection, the sprint runner, and CLI
  dispatch.
- `python -m pytest` passed with 149 tests after implementation.
- The live command completed and wrote `policy_decision_sprint_v1`.
