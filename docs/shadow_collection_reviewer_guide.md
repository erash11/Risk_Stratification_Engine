# Shadow Collection Reviewer Guide

This guide explains what the user/practitioner needs to do next for retained
channel prospective shadow monitoring.

Current status:

- The retained channels are `broad_30d` and `severity_14d`.
- `severity_7d` remains paused until revision.
- The collection template has 8 pending rows.
- Reviewer packet materials have been generated for those 8 rows.
- Calibration, pilot, dashboard, probability-facing output, and autonomous
  intervention remain blocked.

This is still research collection. The goal is to collect prospective
source-eligible reviewer evidence, not to decide that the model is ready for
product use.

## Your Required Involvement

You are needed for the evidence judgment. The tooling can prepare packet files,
validate CSV completion, and summarize results, but it cannot decide whether an
alert was operationally useful or source-trustworthy.

You need to provide or supervise these judgments for each packet:

- Was the packet source-eligible?
- What complete source-eligible season/window does this packet represent?
- How many alert episodes and observed/captured events belong to the packet?
- Was the alert useful, noisy, misleading, or unclear?
- Was the outcome or managed-risk context confirmed?
- Was the source context trustworthy enough for interpretation?
- What action, if any, would have been reasonable?
- What short de-identified rationale should be recorded?

Expected involvement is moderate. There are only 8 rows, but each row should be
reviewed carefully. If the needed internal context is already available, expect
roughly 10-20 minutes per packet. If source context or outcome context must be
gathered first, the work may take longer.

## What Codex Can Do For You

Codex can handle the mechanical work:

- regenerate reviewer packets
- inspect schemas and missing fields
- rerun validation and summary sprints
- summarize completed evidence
- update docs and repo state
- keep unrelated files out of commits

Codex should not invent reviewer evidence. If a value depends on clinical,
operations, roster, source-quality, or practitioner context, it needs your
judgment or an approved source you provide.

## Files To Use

The collection template to complete is:

`outputs/experiments/exposure_load_shadow_collection_template_v1/exposure_load_shadow_collection_template.csv`

The reviewer workflow package is:

`outputs/experiments/exposure_load_shadow_collection_packet_workflow_v1/`

Key files:

- `exposure_load_shadow_collection_reviewer_instructions.md`: concise reviewer instructions.
- `exposure_load_shadow_collection_packet_manifest.csv`: one row per packet.
- `exposure_load_shadow_collection_packet_checklist.csv`: packet checklist items.
- `exposure_load_shadow_collection_packet_audit_trail.csv`: seed audit rows.
- `review_packets/*.md`: one de-identified packet worksheet per collection row.
- `exposure_load_shadow_collection_packet_workflow_report.md`: package summary.

The current validation summary is:

`outputs/experiments/exposure_load_shadow_collection_summary_v1/`

It currently reports 8 pending/invalid rows and 0 complete valid rows. That is
expected until the collection template is filled.

## Exact Work To Do

For each of the 8 rows in
`exposure_load_shadow_collection_template.csv`, open the matching packet file
under `review_packets/` and complete the blank fields in the CSV.

The packet IDs are:

- `broad_30d__prospective_001`
- `broad_30d__prospective_002`
- `broad_30d__prospective_003`
- `broad_30d__prospective_004`
- `severity_14d__prospective_001`
- `severity_14d__prospective_002`
- `severity_14d__prospective_003`
- `severity_14d__prospective_004`

Use one row per complete source-eligible athlete-season review unit. If a
packet fails the source rule, record `source_eligible=false` and explain the
de-identified reason in `notes`.

## Fields To Complete

Fill these required fields:

| Field | What to enter |
|---|---|
| `collection_season_id` | De-identified season label, such as `2026-2027`. |
| `packet_start_date` | Packet review window start in `YYYY-MM-DD` format. |
| `packet_end_date` | Packet review window end in `YYYY-MM-DD` format. |
| `source_eligible` | `true` if the packet passes the source rule; otherwise `false`. |
| `episode_count` | Non-negative count of alert episodes in the packet. |
| `unique_observed_event_count` | Non-negative count of observed relevant events. |
| `unique_captured_event_count` | Non-negative count of observed events captured by the channel. This cannot exceed observed events. |
| `alert_usefulness` | One of `useful`, `noisy`, `misleading`, or `unclear`. |
| `reviewer_id` | Stable reviewer code or initials. Avoid full names if not needed. |
| `review_date` | Review date in `YYYY-MM-DD` format. |

Fill these recommended fields when the information is available:

| Field | What to enter |
|---|---|
| `outcome_confirmed` | `true` if the packet aligns with a real injury, limitation, clinical context, or meaningful managed-risk case; otherwise `false`. |
| `source_context_ok` | `true` if the source context is trustworthy enough for interpretation; otherwise `false`. |
| `action_taken` | One of `none`, `monitor`, `communicate`, `modify_load`, `clinical_review`, or `other`. |
| `notes` | Short de-identified rationale. Do not include athlete names or identifiable details. |

Leave generated packet identity fields unchanged, including
`collection_packet_id`, `channel_name`, `packet_sequence`, `collection_unit`,
`evidence_gate`, and `source_rule`.

## Decision Rules

Use conservative judgments.

Mark `source_eligible=true` only when:

- the packet represents a complete source-eligible review unit
- the source rule is satisfied
- the relevant monitoring and exposure context is interpretable

Use `alert_usefulness=useful` only when:

- the alert points to a coherent operational concern
- the source context is not the main explanation
- the packet would have supported a reasonable staff-facing response

Use `alert_usefulness=noisy` when the signal is interpretable but not useful.
Use `misleading` when it would likely push a practitioner in the wrong
direction. Use `unclear` when the available context is insufficient.

Use `action_taken=none` when no practical response is identifiable. Use
`monitor`, `communicate`, `modify_load`, or `clinical_review` only when that
response would have been reasonable from the packet context.

## Example Completed Row

The template starts like this:

```csv
collection_packet_id,channel_name,packet_sequence,collection_unit,evidence_gate,source_rule,collection_season_id,packet_start_date,packet_end_date,source_eligible,episode_count,unique_observed_event_count,unique_captured_event_count,alert_usefulness,outcome_confirmed,source_context_ok,action_taken,reviewer_id,review_date,notes,collection_status
broad_30d__prospective_001,broad_30d,1,complete source-eligible athlete-season,prospective_shadow_review_before_calibration,stop if source eligibility fails or alert burden exceeds policy cap,,,,,,,,,,,,,,,pending_collection
```

A completed source-eligible row could look like this:

```csv
broad_30d__prospective_001,broad_30d,1,complete source-eligible athlete-season,prospective_shadow_review_before_calibration,stop if source eligibility fails or alert burden exceeds policy cap,2026-2027,2026-08-01,2026-12-01,true,3,1,1,useful,true,true,monitor,ER1,2026-12-15,"De-identified review: packet aligned with a plausible managed-risk period and would have supported closer monitoring.",complete
```

A conservative source-failed row could look like this:

```csv
broad_30d__prospective_002,broad_30d,2,complete source-eligible athlete-season,prospective_shadow_review_before_calibration,stop if source eligibility fails or alert burden exceeds policy cap,2026-2027,2026-12-02,2027-02-01,false,0,0,0,unclear,false,false,none,ER1,2027-02-15,"De-identified review: packet failed source eligibility because context was incomplete.",complete
```

## Validation Command

After you fill the CSV, run:

```bash
risk-engine \
  --exposure-load-shadow-collection-summary-sprint \
  --exposure-load-shadow-collection outputs/experiments/exposure_load_shadow_collection_template_v1/exposure_load_shadow_collection_template.csv \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_collection_summary_completed_v1
```

This produces:

- `exposure_load_shadow_collection_validation.csv`
- `exposure_load_shadow_collection_channel_summary.csv`
- `exposure_load_shadow_collection_summary.json`
- `exposure_load_shadow_collection_summary_report.md`

The command can be run before all rows are complete. It will report exactly
which rows are still pending or invalid.

## What Good Progress Looks Like

Good progress is not a dashboard or probability output. Good progress is a
completed collection CSV with:

- 8 complete valid rows, or a documented reason a row failed source eligibility
- clear source-eligibility decisions
- useful/noisy/misleading/unclear classifications
- de-identified reviewer notes
- channel-level counts for useful, source-trustworthy, actionable packets

Only after this evidence exists should the project move to a calibration
readiness decision sprint. That future sprint should ask whether calibration
research is justified. It should not claim calibrated probabilities.

## Current Boundary

The project has reached a human-evidence bottleneck. The code can prepare,
validate, and summarize the review workflow, but the next meaningful step
requires practitioner/source-context judgment. Until the 8 retained-channel
rows are complete and summarized, do not move to probability-facing outputs,
calibration claims, pilot/dashboard deployment, or autonomous intervention.
