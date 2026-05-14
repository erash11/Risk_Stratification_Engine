# Shadow Collection Reviewer Guide

This guide explains what the user/practitioner needs to do next for retained
channel prospective shadow monitoring.

Current status:

- The retained channels are `broad_30d` and `severity_14d`.
- `severity_7d` remains paused until revision.
- The evidence-prefilled collection template has 8 completed CSV-only reviewer
  rows carried forward from the completed adjudication review.
- Reviewer packet materials have been generated for those 8 rows.
- The completed local summary reports 8 complete valid rows, 0 pending/invalid
  rows, and 4 useful/source-trustworthy/actionable rows.
- Calibration, pilot, dashboard, probability-facing output, and autonomous
  intervention remain blocked.

This is still research collection. The completed rows are CSV-only evidence
review, not independent clinical/practitioner adjudication. The goal is to
decide whether a calibration-readiness review is justified, not to decide that
the model is ready for product use.

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

The evidence-prefilled collection template to complete is:

`outputs/experiments/exposure_load_shadow_collection_evidence_prefill_v1/exposure_load_shadow_collection_prefilled.csv`

This is the preferred file for review. It now carries forward the completed
CSV-only adjudication judgments for the retained channels and already fills the
replay-derived fields that the project knows from existing injury and shadow
replay artifacts. Use the older blank template only if you are intentionally
creating a new prospective collection cycle from scratch.

The preserved reviewer process reference is:

`docs/shadow_collection_reviewer_process_reference.md`

The original blank collection template is:

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

The current evidence prefill package is:

`outputs/experiments/exposure_load_shadow_collection_evidence_prefill_v1/`

It contains:

- `exposure_load_shadow_collection_prefilled.csv`: the file to complete.
- `exposure_load_shadow_collection_prefill_validation.csv`: current missing-field status.
- `exposure_load_shadow_collection_prefill_excluded.csv`: rows excluded from retained-channel review.
- `exposure_load_shadow_collection_evidence_prefill_report.md`: prefill summary.

The original pending validation summary is:

`outputs/experiments/exposure_load_shadow_collection_summary_v1/`

It currently reports 8 pending/invalid rows and 0 complete valid rows. That is
expected for the unreviewed prefill state.

The completed local validation summary is:

`outputs/experiments/exposure_load_shadow_collection_summary_completed_v1/`

It reports 8 complete valid rows, 0 pending/invalid rows, 8 complete
source-eligible rows, and 4 useful/source-trustworthy/actionable rows. Its
recommendation is `revisit_calibration_readiness_with_prospective_shadow_evidence`,
with calibration readiness limited to
`ready_for_calibration_readiness_review_not_calibration_claim`.

## Exact Work To Do

For each of the 8 rows in
`exposure_load_shadow_collection_prefilled.csv`, review the row and complete the
remaining blank reviewer fields. The quantitative replay fields are already
filled. In the current local artifact, those reviewer fields have been completed
from the existing CSV-only adjudication review.

The packet IDs are:

- `broad_30d__2021-2022`
- `broad_30d__2022-2023`
- `broad_30d__2023-2024`
- `broad_30d__2025-2026`
- `severity_14d__2021-2022`
- `severity_14d__2022-2023`
- `severity_14d__2023-2024`
- `severity_14d__2025-2026`

These rows come from existing shadow replay evidence. `severity_7d` is excluded
because it is paused, and `2024-2025` is excluded because the source-resolution
policy marked it source-ineligible.

## Data Source Rule

You do not need to source new injury data to complete the current CSV-only
review. The injury/event counts, alert episode counts, captured events, source
eligibility, retained-channel status, and excluded-channel decisions all come
from project artifacts already generated from the data you provided.

Use the current project artifacts for these fields:

| Field group | Source to use |
|---|---|
| Season/window fields | Existing shadow replay and prefill artifacts. |
| Source eligibility | Existing source-resolution, shadow replay, and prefill artifacts. |
| Alert episode counts | Existing shadow replay packets. |
| Observed/captured event counts | Existing shadow replay packets derived from provided injury data. |
| Retained vs paused channel | Existing adjudication decision and monitoring plan artifacts. |

Do not look for public, external, or new injury data to answer those fields.
If a field is already prefilled, leave it alone unless you find a specific
source error.

The only information not fully contained in the project artifacts is true
practitioner/source-context judgment. That includes whether an alert would have
been useful to staff, whether a managed-risk context was clinically meaningful,
whether source coverage was trustworthy in practice, and what action would have
been reasonable. If you do not have that context, do not invent it. Mark the row
as CSV-only evidence review and use conservative values.

Use this evidence hierarchy:

| Evidence available | Authentic reviewer action |
|---|---|
| Existing replay/adjudication artifacts only | Complete as CSV-only evidence review; notes must say the judgment is based on replay/adjudication artifacts only. |
| De-identified internal practitioner, clinical, operations, roster, or source-quality context is available | Use that context to refine `alert_usefulness`, `outcome_confirmed`, `source_context_ok`, `action_taken`, and `notes`. |
| Context is unavailable or ambiguous | Use `alert_usefulness=unclear`, `outcome_confirmed=false`, `action_taken=none`, and explain the limitation in `notes`. |

So the answer to "do I need data other than what I already provided?" is:
not for the current CSV-only completion. Additional internal context is optional
and only needed if you want to make stronger practitioner-adjudicated judgments.
Without that context, the authentic completion must stay explicitly
artifact-limited.

## Fields To Complete

These fields are already prefilled from project artifacts:

| Field | What to enter |
|---|---|
| `collection_season_id` | Copied from shadow replay `test_season_id`. |
| `packet_start_date` | Derived from the season label using the July 1 season start rule. |
| `packet_end_date` | Derived from the season label using the June 30 season end rule. |
| `source_eligible` | Derived from ready source-eligible replay packet status. |
| `episode_count` | Copied from shadow replay packet evidence. |
| `unique_observed_event_count` | Copied from shadow replay packet evidence. |
| `unique_captured_event_count` | Copied from shadow replay packet evidence. |

Fill these required reviewer fields:

| Field | What to enter |
|---|---|
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

Leave generated and prefilled evidence fields unchanged unless you find a
specific source error. These include
`collection_packet_id`, `channel_name`, `packet_sequence`, `collection_unit`,
`evidence_gate`, `source_rule`, `collection_season_id`, `packet_start_date`,
`packet_end_date`, `source_eligible`, `episode_count`,
`unique_observed_event_count`, and `unique_captured_event_count`.

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

The prefilled template starts like this:

```csv
collection_packet_id,channel_name,packet_sequence,collection_unit,evidence_gate,source_rule,collection_season_id,packet_start_date,packet_end_date,source_eligible,episode_count,unique_observed_event_count,unique_captured_event_count,alert_usefulness,outcome_confirmed,source_context_ok,action_taken,reviewer_id,review_date,notes,collection_status,evidence_source_packet_id,evidence_source
broad_30d__2021-2022,broad_30d,1,complete source-eligible athlete-season,historical_replay_evidence_prefill_before_reviewer_judgment,source_eligible=true and replay_status=ready_for_research_adjudication,2021-2022,2021-07-01,2022-06-30,True,8,0,0,,,,,,,,pending_reviewer_judgment,broad_30d__2021-2022,exposure_load_shadow_review_packets
```

A completed reviewer row could look like this:

```csv
broad_30d__2021-2022,broad_30d,1,complete source-eligible athlete-season,historical_replay_evidence_prefill_before_reviewer_judgment,source_eligible=true and replay_status=ready_for_research_adjudication,2021-2022,2021-07-01,2022-06-30,True,8,0,0,useful,true,true,monitor,ER1,2026-05-14,"De-identified review: packet aligned with a plausible managed-risk period and would have supported closer monitoring.",complete,broad_30d__2021-2022,exposure_load_shadow_review_packets
```

A conservative uncertain row could look like this:

```csv
broad_30d__2022-2023,broad_30d,2,complete source-eligible athlete-season,historical_replay_evidence_prefill_before_reviewer_judgment,source_eligible=true and replay_status=ready_for_research_adjudication,2022-2023,2022-07-01,2023-06-30,True,9,71,0,unclear,false,true,none,ER1,2026-05-14,"De-identified review: replay evidence is quantitative, but reviewer context was insufficient to call it useful.",complete,broad_30d__2022-2023,exposure_load_shadow_review_packets
```

## Validation Command

After you fill the CSV, run:

```bash
risk-engine \
  --exposure-load-shadow-collection-summary-sprint \
  --exposure-load-shadow-collection outputs/experiments/exposure_load_shadow_collection_evidence_prefill_v1/exposure_load_shadow_collection_prefilled.csv \
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

The 8 retained-channel rows are now complete and summarized in the local
CSV-only evidence-review artifact. The next meaningful step is a
calibration-readiness review sprint, not probability-facing output. Do not move
to calibration claims, pilot/dashboard deployment, autonomous intervention, or
probability-facing outputs unless a separate readiness sprint explicitly
supports that escalation.
