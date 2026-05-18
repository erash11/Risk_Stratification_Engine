# Prospective Collection Completion Guide

This guide explains what the user/practitioner needs to do next for the
retained-channel prospective collection packets.

Current status:

- The retained channels are `broad_30d` and `severity_14d`.
- Each channel has 4 prospective collection packets, for 8 total packet rows.
- The current operations worksheet is blank by design.
- The ingest path is ready and validates completed de-identified worksheet rows.
- The live ingest run found 8 pending rows, 0 completed practitioner rows, and
  0 ingest errors.
- Calibration, pilot, dashboard, probability-facing output, autonomous
  intervention, and load modification remain blocked.

This is still research collection. The goal is to collect de-identified
practitioner evidence so the project can decide whether a bounded retest is
justified. Completing this worksheet does not create calibration claims or
product readiness.

## Your Required Involvement

You are needed for practitioner packet completion. The tooling can prepare the
packet files, ingest the completed worksheet, validate completion, and summarize
the result. It cannot decide whether a prospective alert packet was useful,
source-trustworthy, or action-relevant.

For each packet, you or the practitioner need to provide:

- the de-identified season/window represented by the packet
- whether the packet remains source eligible
- alert episode count
- observed, captured, and missed event counts
- missed event rate
- whether the packet was useful, not useful, or unclear
- whether outcome context was confirmed
- whether source context was trustworthy
- what action, if any, was reasonable
- reviewer code/date and a short de-identified rationale

Expected involvement is moderate. There are only 8 rows, but each row should be
reviewed carefully. If the packet context is already available, expect roughly
10-20 minutes per packet. If source, outcome, roster, or operations context must
be gathered first, the work may take longer.

## What Codex Can Do For You

Codex can handle the mechanical work:

- inspect packet manifests and checklists
- regenerate or summarize packet files
- run the ingest sprint against your completed worksheet
- run completion validation after ingest
- report exactly which rows are incomplete, invalid, duplicated, or not
  de-identified
- update docs and repo state
- keep unrelated files out of commits

Codex should not invent reviewer evidence. If a value depends on practitioner,
clinical, operations, roster, source-quality, or outcome context, it needs your
judgment or an approved source you provide.

## Files To Use

The operations package is:

`outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/`

Key files:

- `exposure_load_shadow_prospective_collection_worksheet.csv`: the file to copy
  and complete.
- `exposure_load_shadow_prospective_collection_packet_manifest.csv`: packet IDs,
  target types, and packet file names.
- `exposure_load_shadow_prospective_collection_checklist.csv`: checklist items
  for every packet.
- `exposure_load_shadow_prospective_collection_reviewer_instructions.md`:
  concise reviewer instructions.
- `review_packets/*.md`: one de-identified packet worksheet per packet.
- `exposure_load_shadow_prospective_collection_operations.json`: machine-readable
  operations package used by ingest and validation.

The packet crosswalk is:

`docs/prospective_collection_packet_crosswalk.md`

Use the crosswalk to match packet IDs to channels, packet types, packet files,
allowed values, ingest checks, and the follow-on validation path.

The current ingest package is:

`outputs/experiments/exposure_load_shadow_prospective_collection_ingest_v1/`

It contains:

- `exposure_load_shadow_prospective_collection_ingest_validation.csv`: row-level
  ingest status.
- `exposure_load_shadow_prospective_collection_ingest_summary.csv`: channel-level
  ingest status.
- `exposure_load_shadow_prospective_collection_ingested_worksheet.csv`: the
  worksheet after ingest.
- `exposure_load_shadow_prospective_collection_ingested_operations.json`: the
  operations package after valid completed rows are merged.
- `exposure_load_shadow_prospective_collection_ingest_report.md`: summary of the
  current ingest result.

The live ingest result is expected for an uncompleted worksheet: all 8 rows are
pending and 0 completed practitioner rows are ingested.

## Exact Work To Do

1. Make a working copy of:

   `outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/exposure_load_shadow_prospective_collection_worksheet.csv`

   Save the completed copy outside the generated artifact folder or under a new
   reviewed-artifact folder. Use a clear name such as
   `completed_prospective_collection.csv`.

2. Complete these 8 packet rows:

   - `broad_30d__prospective_collection_001`
   - `broad_30d__prospective_collection_002`
   - `broad_30d__prospective_collection_003`
   - `broad_30d__prospective_collection_004`
   - `severity_14d__prospective_collection_001`
   - `severity_14d__prospective_collection_002`
   - `severity_14d__prospective_collection_003`
   - `severity_14d__prospective_collection_004`

3. Use the packet target type to guide the review:

| Target type | Required review focus |
|---|---|
| `monitoring_context_packet` | Capture practitioner monitoring context, outcome follow-up, alert usefulness, and action relevance. |
| `missed_only_error_packet` | Adjudicate why missed events were not captured and whether the retained channel remains research-useful under that error mode. |
| `outcome_context_packet` | Capture outcome context or mark it unavailable; do not infer an outcome that cannot be defended. |

4. Preserve generated metadata fields:

   `collection_packet_id`, `channel_name`, `packet_sequence`, `target_type`,
   `required_action`, `target_captured_events_needed`, and
   `maximum_allowed_missed_event_rate`.

5. Do not add identifiable athlete columns or values. The ingest sprint flags
   nonblank values in identifier fields such as `athlete_name`, `first_name`,
   `last_name`, `full_name`, `athlete_id`, `external_athlete_id`, or
   `source_athlete_id`.

## Fields To Complete

Fill these required fields for every completed row:

| Field | What to enter |
|---|---|
| `collection_season_id` | De-identified season/window label, for example `2026-2027`. |
| `packet_start_date` | Packet start date in `YYYY-MM-DD` format. |
| `packet_end_date` | Packet end date in `YYYY-MM-DD` format. |
| `source_eligible` | `true` only if the packet is source eligible and interpretable; otherwise leave pending until resolved. |
| `episode_count` | Nonnegative integer count of alert episodes reviewed. |
| `unique_observed_event_count` | Nonnegative integer count of observed events in the review window. |
| `unique_captured_event_count` | Nonnegative integer count of observed events captured by packet alerts. |
| `unique_missed_event_count` | Nonnegative integer count of observed events missed by packet alerts. |
| `missed_event_rate` | Number from `0` to `1`; usually missed events divided by observed events. |
| `alert_usefulness` | One of `useful`, `not_useful`, or `unclear`. |
| `outcome_confirmed` | `true` or `false`. |
| `source_context_ok` | `true` or `false`. |
| `action_taken` | One of `monitor`, `none`, `modified_followup`, or `other`. |
| `reviewer_id` | Stable reviewer code or initials. Avoid full names if not needed. |
| `review_date` | Review date in `YYYY-MM-DD` format. |
| `collection_status` | Set to `complete_practitioner_adjudication` only when the row is complete. |

Fill `notes` with a short de-identified rationale. Do not include athlete names,
medical record identifiers, source athlete IDs, or enough contextual detail to
re-identify the athlete.

Rows can remain pending while under review. A pending row should keep
`collection_status=pending_prospective_collection` and should not be expected to
advance bounded retest readiness.

## Decision Rules

Use conservative judgments.

Set `source_eligible=true` only when:

- the packet represents the intended prospective review unit
- source coverage is trustworthy enough for interpretation
- event and alert counts are defensible
- roster, availability, or documentation context does not invalidate the packet

Use `alert_usefulness=useful` only when:

- the alert packet points to a coherent monitoring concern
- the context would have supported a reasonable practitioner response
- the packet is not explained away by source gaps or known documentation shifts

Use `alert_usefulness=not_useful` when the packet is interpretable but does not
support a meaningful monitoring decision. Use `unclear` when available context
is insufficient.

Use `action_taken=none` when no practical response is identifiable. Use
`monitor`, `modified_followup`, or `other` only when the packet supports that
level of response.

The packet should not support escalation when:

- source context is not trustworthy
- event counts cannot be defended
- usefulness is `not_useful` or `unclear`
- no reasonable action is identifiable
- the judgment depends on identifiable or unavailable context

## Example Completed Rows

The worksheet starts like this:

```csv
collection_packet_id,channel_name,packet_sequence,target_type,required_action,collection_season_id,packet_start_date,packet_end_date,source_eligible,episode_count,unique_observed_event_count,unique_captured_event_count,unique_missed_event_count,missed_event_rate,alert_usefulness,outcome_confirmed,source_context_ok,action_taken,reviewer_id,review_date,notes,target_captured_events_needed,maximum_allowed_missed_event_rate,collection_status
broad_30d__prospective_collection_001,broad_30d,1,monitoring_context_packet,capture_practitioner_monitoring_context_and_outcome_followup,,,,,,,,,,,,,,,,,8,0.75,pending_prospective_collection
```

A completed useful row could look like this:

```csv
broad_30d__prospective_collection_001,broad_30d,1,monitoring_context_packet,capture_practitioner_monitoring_context_and_outcome_followup,2026-2027,2026-08-01,2026-12-01,true,2,10,2,2,0.2,useful,true,true,monitor,ER1,2027-01-15,"De-identified review: packet aligned with a plausible managed-risk period and supported closer monitoring.",8,0.75,complete_practitioner_adjudication
```

A conservative unclear row could look like this:

```csv
broad_30d__prospective_collection_003,broad_30d,3,missed_only_error_packet,adjudicate_missed_only_error_context,2026-2027,2026-08-01,2026-12-01,true,1,8,0,8,1.0,unclear,false,true,none,ER1,2027-01-15,"De-identified review: event counts were observable, but context was insufficient to call the alert useful.",8,0.75,complete_practitioner_adjudication
```

## Ingest Command

After the worksheet is completed, run ingest against the completed CSV:

```bash
risk-engine \
  --exposure-load-shadow-prospective-collection-ingest-sprint \
  --exposure-load-shadow-prospective-collection-operations outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/exposure_load_shadow_prospective_collection_operations.json \
  --completed-prospective-collection <path_to_completed_prospective_collection.csv> \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_prospective_collection_ingest_completed_v1
```

This command can be run before all rows are complete. It will report pending
rows, unknown packet IDs, duplicate packet IDs, and de-identification problems.

## Completion Validation Command

If ingest succeeds and writes completed rows into the ingested operations
package, run completion validation against the ingested operations JSON:

```bash
risk-engine \
  --exposure-load-shadow-prospective-collection-completion-sprint \
  --exposure-load-shadow-prospective-collection-operations outputs/experiments/exposure_load_shadow_prospective_collection_ingest_completed_v1/exposure_load_shadow_prospective_collection_ingested_operations.json \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_prospective_collection_completion_after_ingest_v1
```

Completion validation will report exactly which packet rows are complete,
pending, invalid, or blocked. The bounded retest gate should not be considered
ready until the completion sprint says `ready_for_bounded_retest_not_claims`.

## What Good Progress Looks Like

Good progress is not a dashboard or probability output. Good progress is a
completed de-identified worksheet with:

- 8 known packet IDs and no duplicate packet IDs
- no identifiable athlete fields or values
- complete required fields for every row
- defensible source-eligibility decisions
- captured, missed, and observed event counts
- reviewer notes that explain uncertainty without identifying athletes
- successful ingest with nonzero completed practitioner rows
- successful completion validation, or a clear list of rows to repair

Only after completion validation passes should the project move to a bounded
retest sprint. That future sprint should remain research-only and should not
claim calibrated probabilities.

## Current Boundary

The project is blocked on user/practitioner completion of these 8 de-identified
packet rows. The code-side ingest path is ready, but the blank worksheet cannot
advance bounded retest readiness.

The right next move is completing the worksheet, then running ingest and
completion validation. Do not move to bounded retest, calibration claims,
pilot/dashboard deployment, autonomous intervention, load modification, or
probability-facing output until the required rows are complete and revalidated.
