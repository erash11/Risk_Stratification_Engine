# Prospective Collection Packet Crosswalk

This document maps the current prospective collection packet IDs to the
reviewer files, worksheet fields, allowed values, ingest checks, and follow-on
validation path.

It is a reviewer-facing reference. It is not evidence that any packet has been
completed, validated, calibrated, or cleared for product use.

Current package:

`outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/`

Current status:

- 8 packet rows are ready for de-identified practitioner completion.
- 0 packet rows are complete.
- `broad_30d` and `severity_14d` each require 4 completed packet rows before
  bounded retest can be considered.
- Calibration claims, probability-facing output, pilot/dashboard readiness,
  autonomous intervention, and load modification remain blocked.

## Packet Crosswalk

| Packet ID | Channel | Target type | Packet file | Required action | Completion status |
|---|---|---|---|---|---|
| `broad_30d__prospective_collection_001` | `broad_30d` | `monitoring_context_packet` | `review_packets/broad_30d__prospective_collection_001.md` | Capture practitioner monitoring context and outcome follow-up. | Pending practitioner completion |
| `broad_30d__prospective_collection_002` | `broad_30d` | `monitoring_context_packet` | `review_packets/broad_30d__prospective_collection_002.md` | Capture practitioner monitoring context and outcome follow-up. | Pending practitioner completion |
| `broad_30d__prospective_collection_003` | `broad_30d` | `missed_only_error_packet` | `review_packets/broad_30d__prospective_collection_003.md` | Adjudicate missed-only error context. | Pending practitioner completion |
| `broad_30d__prospective_collection_004` | `broad_30d` | `outcome_context_packet` | `review_packets/broad_30d__prospective_collection_004.md` | Capture outcome context or mark unavailable. | Pending practitioner completion |
| `severity_14d__prospective_collection_001` | `severity_14d` | `monitoring_context_packet` | `review_packets/severity_14d__prospective_collection_001.md` | Capture practitioner monitoring context and outcome follow-up. | Pending practitioner completion |
| `severity_14d__prospective_collection_002` | `severity_14d` | `monitoring_context_packet` | `review_packets/severity_14d__prospective_collection_002.md` | Capture practitioner monitoring context and outcome follow-up. | Pending practitioner completion |
| `severity_14d__prospective_collection_003` | `severity_14d` | `missed_only_error_packet` | `review_packets/severity_14d__prospective_collection_003.md` | Adjudicate missed-only error context. | Pending practitioner completion |
| `severity_14d__prospective_collection_004` | `severity_14d` | `outcome_context_packet` | `review_packets/severity_14d__prospective_collection_004.md` | Capture outcome context or mark unavailable. | Pending practitioner completion |

## Target Type Meaning

| Target type | Reviewer question | Conservative completion rule |
|---|---|---|
| `monitoring_context_packet` | Did the packet provide useful, source-trustworthy monitoring context that could support a reasonable staff-facing response? | Mark useful only when the alert context is coherent, source context is trustworthy, and a defensible response exists. |
| `missed_only_error_packet` | Why were events missed, and does the channel remain research-useful under that error mode? | Do not call the packet useful just because events are observable; document whether misses reflect source limits, channel limits, or unavailable context. |
| `outcome_context_packet` | Is outcome or managed-risk context available and defensible for the packet window? | Mark unavailable or unclear when the outcome context cannot be confirmed without speculation. |

## Worksheet Fields

Complete the worksheet copied from:

`outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/exposure_load_shadow_prospective_collection_worksheet.csv`

Generated metadata fields. Preserve these unless there is a real source error:

- `collection_packet_id`
- `channel_name`
- `packet_sequence`
- `target_type`
- `required_action`
- `target_captured_events_needed`
- `maximum_allowed_missed_event_rate`

Required completion fields:

- `collection_season_id`
- `packet_start_date`
- `packet_end_date`
- `source_eligible`
- `episode_count`
- `unique_observed_event_count`
- `unique_captured_event_count`
- `unique_missed_event_count`
- `missed_event_rate`
- `alert_usefulness`
- `outcome_confirmed`
- `source_context_ok`
- `action_taken`
- `reviewer_id`
- `review_date`
- `collection_status`

Recommended rationale field:

- `notes`

## Allowed Values

Use exactly these values where applicable:

| Field | Allowed values |
|---|---|
| `source_eligible` | `true`, `false` |
| `alert_usefulness` | `useful`, `not_useful`, `unclear` |
| `outcome_confirmed` | `true`, `false` |
| `source_context_ok` | `true`, `false` |
| `action_taken` | `monitor`, `none`, `modified_followup`, `other` |
| `collection_status` | `complete_practitioner_adjudication` for completed rows; `pending_prospective_collection` for pending rows |

Use dates in `YYYY-MM-DD` format. Use nonnegative integers for counts. Use a
number from `0` to `1` for `missed_event_rate`.

## De-Identification Guardrails

Do not enter athlete names, source IDs, medical record IDs, or source athlete
IDs in the worksheet, notes, packet files, or any added columns.

The ingest sprint rejects nonblank values in identifier fields such as:

- `athlete_name`
- `first_name`
- `last_name`
- `full_name`
- `athlete_id`
- `external_athlete_id`
- `source_athlete_id`

Use stable reviewer codes or initials for `reviewer_id`; avoid full names if
the file will be shared.

## Completion-to-Validation Flow

1. Complete a working copy of the worksheet.
2. Run prospective collection ingest against the completed worksheet.
3. Review ingest validation for duplicate packet IDs, unknown packet IDs,
   pending rows, or de-identification violations.
4. If ingest succeeds, run completion validation against the ingested operations
   JSON.
5. Stop at completion validation. Do not move to bounded retest unless the
   completion sprint says `ready_for_bounded_retest_not_claims`.

Ingest command:

```bash
risk-engine \
  --exposure-load-shadow-prospective-collection-ingest-sprint \
  --exposure-load-shadow-prospective-collection-operations outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/exposure_load_shadow_prospective_collection_operations.json \
  --completed-prospective-collection <path_to_completed_prospective_collection.csv> \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_prospective_collection_ingest_completed_v1
```

Completion validation command:

```bash
risk-engine \
  --exposure-load-shadow-prospective-collection-completion-sprint \
  --exposure-load-shadow-prospective-collection-operations outputs/experiments/exposure_load_shadow_prospective_collection_ingest_completed_v1/exposure_load_shadow_prospective_collection_ingested_operations.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_prospective_collection_completion_after_ingest_v1
```

## Stop Conditions

Stop and repair the worksheet before validation if ingest reports:

- unknown packet IDs
- duplicate packet IDs
- de-identification violations
- completed rows with missing required evidence
- pending rows that were expected to be complete

Stop and request practitioner/user clarification if:

- source eligibility cannot be defended
- event counts cannot be reconciled
- outcome context is unavailable but required for a useful judgment
- notes would require identifiable athlete details

## Boundary

This crosswalk is a completion aid. It does not establish prospective
performance, calibration readiness, probability-facing output readiness,
pilot/dashboard readiness, autonomous intervention readiness, or load
modification readiness.
