# Injury Outcome Policy Design

## Purpose

Audit severity semantics and define candidate injury outcome policies before
changing the model target. This is the first step in the broader
context-aware outcome modeling sprint.

## Design

The new `--injury-outcome-policies` CLI mode uses the detailed injury event
table produced by live-source preparation. It writes two evidence layers:

- `injury_severity_audit.*`: event-level severity semantics checks.
- `outcome_policy_*`: candidate target definitions and counts.

The severity audit compares injury issue/resolved dates with exported duration,
checks missing/negative/extreme time-loss values, buckets time loss, and marks
events as usable or needing review. Extreme time-loss is currently defined as
more than 365 days.

The outcome policy summary defines candidate labels without retraining the
model:

- any injury
- any time-loss injury
- model-safe time-loss injury
- moderate-plus time-loss injury
- severe time-loss injury
- caused-unavailability injury
- recurrent injury
- lower-extremity injury
- soft-tissue injury
- lower-extremity soft-tissue injury
- concussion-only
- exclude-concussion

## Testing

Unit tests cover severity flags, time-loss buckets, and policy counts. The
experiment integration test verifies that the runner writes all CSV, JSON, and
Markdown artifacts. The CLI test verifies that live-source preparation passes
the generated `injury_events_detailed.csv` into the new mode.
