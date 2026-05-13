# Shadow Adjudication Guide

This guide explains the next required step after the exposure-load shadow replay
and adjudication package sprints.

Current status:

- The historical replay is ready for prospective review.
- The adjudication template has 12 review packet rows.
- Product readiness remains blocked.
- The next evidence step is human adjudication of those 12 rows.

This is still research collection. Do not use these outputs for a dashboard,
probability-facing display, pilot deployment, or autonomous intervention.

## Files To Use

The current adjudication package is:

`outputs/experiments/exposure_load_shadow_adjudication_v1/`

Key files:

- `exposure_load_shadow_adjudication_report.md`: summary of the package.
- `exposure_load_shadow_adjudication_schema.csv`: field definitions and allowed values.
- `exposure_load_shadow_adjudication_template.csv`: the file to complete during review.
- `exposure_load_shadow_adjudication_completion.csv`: current missing-field status.
- `exposure_load_shadow_adjudication.json`: full machine-readable package.

The template starts with blank adjudication fields by design. A blank row means
the alert packet has not yet been reviewed.

## Review Goal

For each shadow alert packet, answer one practical question:

Did this alert identify a real, reviewable athlete-context signal that would
have been useful to a practitioner at the time?

The reviewer is not judging whether the model is production ready. The reviewer
is judging whether each alert packet is useful enough to support further shadow
monitoring.

## Required Fields

Fill these fields for every row in `exposure_load_shadow_adjudication_template.csv`.

| Field | What to enter |
|---|---|
| `reviewer_id` | A stable reviewer code or initials. Avoid full names if this file will be shared. |
| `review_date` | Review date in `YYYY-MM-DD` format. |
| `alert_usefulness` | One of `useful`, `noisy`, `misleading`, or `unclear`. |
| `outcome_confirmed` | `true` if the alert aligns with a real injury, limitation, clinical context, or meaningful managed-risk case; otherwise `false`. |
| `source_context_ok` | `true` if the data context looks trustworthy enough for interpretation; otherwise `false`. |
| `action_taken` | One of `none`, `monitor`, `communicate`, `modify_load`, `clinical_review`, or `other`. |
| `notes` | Optional short rationale. Use de-identified language. |

Leave the generated context columns unchanged. These include fields such as
`review_packet_id`, `channel_name`, `test_season_id`, event counts, burden, and
`review_packet_status`. The reviewer only fills the blank adjudication fields.

## Example Completed Row

The template starts like this:

```csv
review_packet_id,channel_name,test_season_id,minimum_review_unit,required_evidence,episode_count,unique_observed_event_count,unique_captured_event_count,missed_event_count,episodes_per_athlete_season,review_packet_status,reviewer_id,review_date,alert_usefulness,outcome_confirmed,source_context_ok,action_taken,notes,adjudication_status
broad_30d__2021-2022,broad_30d,2021-2022,complete source-eligible athlete-season,"frozen alert episodes, source eligibility, exposure capture status, outcome adjudication, and alert burden",8,0,0,0,0.05,ready_for_research_adjudication,,,,,,,,pending_review
```

An example completed row could look like this:

```csv
review_packet_id,channel_name,test_season_id,minimum_review_unit,required_evidence,episode_count,unique_observed_event_count,unique_captured_event_count,missed_event_count,episodes_per_athlete_season,review_packet_status,reviewer_id,review_date,alert_usefulness,outcome_confirmed,source_context_ok,action_taken,notes,adjudication_status
broad_30d__2021-2022,broad_30d,2021-2022,complete source-eligible athlete-season,"frozen alert episodes, source eligibility, exposure capture status, outcome adjudication, and alert burden",8,0,0,0,0.05,ready_for_research_adjudication,ER1,2026-05-13,useful,true,true,monitor,"De-identified review: alert aligned with a plausible managed-risk period and would have supported closer monitoring.",pending_review
```

The same decision in plain language:

| Field | Example value | Why |
|---|---|---|
| `reviewer_id` | `ER1` | Stable reviewer code, not a full name. |
| `review_date` | `2026-05-13` | ISO date format. |
| `alert_usefulness` | `useful` | The packet pointed to a coherent operational concern. |
| `outcome_confirmed` | `true` | Review found a matching injury, limitation, clinical context, or managed-risk case. |
| `source_context_ok` | `true` | The packet was not explained away by source gaps or roster/capture shifts. |
| `action_taken` | `monitor` | A reasonable staff-facing response would have been closer monitoring. |
| `notes` | Short de-identified rationale | Explains the call without naming the athlete. |

If the same packet had weak context, a conservative completion could instead be:

```csv
broad_30d__2021-2022,broad_30d,2021-2022,complete source-eligible athlete-season,"frozen alert episodes, source eligibility, exposure capture status, outcome adjudication, and alert burden",8,0,0,0,0.05,ready_for_research_adjudication,ER1,2026-05-13,unclear,false,false,none,"De-identified review: source context was insufficient, so the alert should not support escalation.",pending_review
```

## How To Adjudicate Each Row

Use the row context from the template and any approved internal review sources.
Keep the review time-safe: evaluate whether the packet would have been useful
based on information available around that historical alert window, not later
knowledge that would not have been available operationally.

For each row:

1. Confirm the packet identity.
   Check `review_packet_id`, season, alert channel, horizon, and the associated
   shadow replay context.

2. Check source trust.
   If the row appears driven by missing exposure data, roster-documentation
   changes, season eligibility problems, or a known source shift, mark
   `source_context_ok` as `false`.

3. Judge usefulness.
   Use `alert_usefulness = useful` only when the alert points to a coherent
   operational concern. Use `noisy` when it is interpretable but not useful.
   Use `misleading` when it would likely push a practitioner in the wrong
   direction. Use `unclear` when the available context is insufficient.

4. Confirm whether there was a real outcome or managed-risk context.
   Set `outcome_confirmed = true` only if the alert aligns with a meaningful
   clinical, participation, limitation, or managed-risk signal.

5. Record the action that would have been reasonable.
   Use `none` when no action would have been taken. Use `monitor`,
   `communicate`, `modify_load`, or `clinical_review` only when the packet
   supports that level of practical response.

6. Add a short note.
   The note should explain the decision without identifying the athlete.

## Decision Rules

Treat the review as conservative.

An alert packet should be considered promising only when:

- `source_context_ok = true`
- `alert_usefulness = useful`
- `outcome_confirmed = true`, or the notes support a credible managed-risk case
- `action_taken` is stronger than `none`

An alert packet should not support escalation when:

- source context is not trustworthy
- usefulness is `noisy`, `misleading`, or `unclear`
- no reasonable staff-facing action is identifiable
- the judgment depends on information that would not have been available at the time

## What Good Progress Looks Like

After the 12 rows are reviewed, we should be able to summarize:

- How many alerts were useful versus noisy, misleading, or unclear.
- Which locked channels produced useful packets.
- Whether useful alerts came from trustworthy source contexts.
- Whether any packet supported a practical action.
- Whether the evidence is strong enough to continue prospective shadow monitoring.

This still would not clear the project for product use. It would tell us whether
the shadow alert package has enough operational signal to justify collecting
more prospective review data.

## What To Do After The Template Is Filled

Once the template is complete, run the adjudication summary sprint:

```bash
risk-engine \
  --exposure-load-shadow-adjudication-summary-sprint \
  --exposure-load-shadow-adjudication outputs/experiments/exposure_load_shadow_adjudication_v1/exposure_load_shadow_adjudication_template.csv \
  --output-dir outputs \
  --experiment-id exposure_load_shadow_adjudication_summary_v1
```

Use the path to your completed adjudication CSV if you save a copy under a
different name.

The sprint ingests the filled adjudication file and produces:

- packet-level completion status
- usefulness counts by channel and horizon
- source-context pass/fail counts
- actionability counts
- reviewer-note audit fields
- recommendation for continued shadow monitoring or rollback

The output files are:

- `exposure_load_shadow_adjudication_validation.csv`
- `exposure_load_shadow_adjudication_channel_summary.csv`
- `exposure_load_shadow_adjudication_summary.json`
- `exposure_load_shadow_adjudication_summary_report.md`

This sprint can run before the packet is complete as a validation check, but it
will return `complete_adjudication_required_before_operational_summary` until
all required fields are complete and valid. It cannot produce real operational
usefulness evidence until the required fields are filled.

## Current Boundary

With the current data alone, the project has reached a preparation limit. The
modeling pipeline can package alerts and templates, but it cannot determine
true operational usefulness without human adjudication.

The right next move is therefore not another model expansion. It is completing
the review template and then evaluating the completed adjudication evidence.
