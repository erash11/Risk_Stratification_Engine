# Injury Context Outcomes Design

## Purpose

Use the enriched detailed injury event table to identify which injury contexts
the current alert policy captures or misses. This sprint does not change the
risk model. It adds evidence artifacts for deciding whether the next model
iteration should use injury subtype, severity, recurrence, unavailability, or
activity context.

## Design

The alert episode experiment accepts an optional detailed injury event table. In
live-source runs, the CLI passes the prepared `injury_events_detailed.csv`
automatically. If a canonical injury file has a sibling
`injury_events_detailed.csv`, the runner can also discover it without a new CLI
flag.

The new context builder creates two tables:

- `injury_event_context_profiles.csv`: one row per detailed injury event per
  horizon/threshold policy. It records event metadata, time-loss bucket, nearest
  prior episode start/peak/end gaps, and whether each timing anchor captured the
  event within the horizon.
- `injury_context_outcomes.csv`: grouped context summaries by injury type,
  pathology, classification, body area, tissue type, side, recurrence,
  unavailability, activity group/type, and time-loss bucket. Each row reports
  event counts, captured/missed counts, capture rate, median time-loss,
  recurrence count, unavailability count, and a recommended next action.

The JSON and Markdown report mirror those tables and highlight low-capture
contexts plus high time-loss missed contexts.

## Testing

The unit test covers the core matching logic: an alert episode that starts before
an injury and falls within the horizon marks that injury as captured, while a
later injury remains missed. The experiment integration test verifies that
passing a detailed injury file writes the profile CSV, grouped context CSV, JSON,
and Markdown report. Full verification requires `python -m pytest` and a live
alert episode run.
