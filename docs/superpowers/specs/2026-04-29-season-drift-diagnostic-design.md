# Season Drift Diagnostic Design

## Goal

Explain why the fixed shadow-mode policy package varies by season before moving
toward dashboard or pilot work.

The diagnostic should not reselect the policy. It should reuse the current
shadow-mode channels and season-local threshold logic, then attach season-level
context that can explain drift:

- measurement coverage and source mix
- injury target volume and detailed injury mix
- channel capture, burden, and captured-event counts
- simple flags for low-capture event seasons and low measurement coverage

## CLI

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id season_drift_diagnostic_v1 \
  --season-drift-diagnostic \
  --model-variant l2
```

## Artifacts

- `season_drift_diagnostics.csv`
- `season_drift_diagnostics.json`
- `season_drift_diagnostic_report.md`

## Implementation Notes

- Share the shadow-mode season-local channel computation with
  `run_shadow_mode_stability_experiment(...)`.
- Keep the diagnostic layer separate from model fitting so the artifact remains
  explainable.
- Treat the output as a research audit, not a production readiness gate.
