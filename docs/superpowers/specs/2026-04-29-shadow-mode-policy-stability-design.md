# Shadow-Mode Policy Stability Design

## Goal

Evaluate whether the current shadow-mode policy package is stable enough to
support a monitored pilot. The sprint does not reselect policies. It tests the
fixed channels selected by `policy_decision_sprint_v1`.

## Channels

- `broad_30d`: `exclude_concussion`, graph window 4, 30d, top-5%.
- `severity_7d`: `model_safe_time_loss`, graph window 4, 7d, top-10%.
- `severity_14d`: `model_safe_time_loss`, graph window 4, 14d, top-10%.
- `subtype_lower_extremity_soft_tissue_30d`: lower-extremity soft-tissue,
  graph window 2, 30d, top-10%.

## Method

For each channel, train the selected L2 graph model and then evaluate alert
episode quality separately by season. Percentile thresholds are computed within
each season slice, not globally across all history, because shadow-mode review
would operate against the currently observed cohort rather than against a pooled
multi-year threshold.

The audit reports per-season:

- observed events
- captured events
- unique event capture rate
- alert episode count
- alert burden per athlete-season
- median lead time

The channel summary marks a channel as stable only when it has at least two
season slices with event evidence and the capture-rate range is at most 0.10.

## Artifacts

- `shadow_mode_stability.csv`
- `shadow_mode_stability.json`
- `shadow_mode_stability_report.md`

## CLI

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id shadow_mode_stability_v1 \
  --shadow-mode-stability \
  --model-variant l2
```

## Live Result Summary

The first live run found all four channels unstable across season slices. The
latest season, 2025-2026, carried much stronger capture than earlier event
seasons. The broad 30d channel averaged 14.8% capture but ranged from 6.1% to
33.3%. The severity channels had similar instability, and the lower-extremity
soft-tissue review channel ranged from 4.3% to 63.3%.

## Decision

The model should remain in research shadow mode. The next sprint should focus on
why older seasons have weaker capture: source freshness, measurement coverage,
season context, roster/source overlap, or a real shift in injury/monitoring
patterns.
