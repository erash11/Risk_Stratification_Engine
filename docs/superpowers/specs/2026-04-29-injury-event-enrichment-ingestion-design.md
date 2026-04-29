# Injury Event Enrichment Ingestion Design

## Purpose

Preserve the richer injury context from the uploaded injury summary exports
without changing the current first-event-per-athlete-season modeling label
contract. The immediate goal is to make injury severity, subtype, activity
context, and availability fields available for analysis before promoting any of
them into model features or event policies.

## Design

When `config/paths.local.yaml` points `injury_csv` at an
`injuries-summary-export-*.csv` file, live-source preparation loads every sibling
CSV matching that pattern from the same folder and de-duplicates exact raw rows.
This supports period-sliced exports while keeping the existing single-path local
configuration.

Live-source preparation now writes three input artifacts:

- `canonical_measurements.csv`: unchanged measurement input table.
- `canonical_injuries.csv`: unchanged athlete-season modeling label table using
  the earliest injury issue date per athlete-season.
- `injury_events_detailed.csv`: new de-identified one-row-per-injury-event table.

The detailed table includes hashed athlete IDs, stable injury event IDs,
season/date fields, injury type, pathology/classification, body area, tissue
type, side, recurrence, unavailability, activity context, participation context,
duration, time-loss and availability-day fields, ICD/code fields, source file,
and source row number. Direct athlete names and DOB are not written.

## Testing

The sprint follows TDD with focused coverage for:

- detailed injury event rows preserving rich context while excluding identifying
  name/DOB fields;
- live-source preparation discovering sibling injury exports and writing the new
  detailed artifact while preserving the canonical injury label contract.

Full verification requires `python -m pytest` plus a live-source run that writes
the detailed artifact from the local uploaded injury files.
