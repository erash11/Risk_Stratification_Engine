# Prospective Collection Completion Process Reference

This reference preserves the exact process for completing the retained-channel
prospective collection worksheet. Read it alongside
`docs/prospective_collection_completion_guide.md`.

Use this file as the starting template:

`outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/exposure_load_shadow_prospective_collection_worksheet.csv`

Save your completed working copy under a new name, for example:

`completed_prospective_collection.csv`

The exact process is:

1. Open one packet row at a time.
   Treat each row as one de-identified prospective collection packet for a
   retained shadow channel.

2. Confirm the packet identity and target type.
   The 8 valid packet IDs are:

   - `broad_30d__prospective_collection_001`
   - `broad_30d__prospective_collection_002`
   - `broad_30d__prospective_collection_003`
   - `broad_30d__prospective_collection_004`
   - `severity_14d__prospective_collection_001`
   - `severity_14d__prospective_collection_002`
   - `severity_14d__prospective_collection_003`
   - `severity_14d__prospective_collection_004`

3. Do not change generated metadata unless you find a real source error:
   `collection_packet_id`, `channel_name`, `packet_sequence`, `target_type`,
   `required_action`, `target_captured_events_needed`, and
   `maximum_allowed_missed_event_rate`.

4. Do not enter identifiable athlete information.
   Do not add or populate fields such as `athlete_name`, `first_name`,
   `last_name`, `full_name`, `athlete_id`, `external_athlete_id`, or
   `source_athlete_id`. The ingest sprint flags nonblank values in those fields
   as de-identification violations.

5. Fill the required evidence fields:
   - `collection_season_id`
   - `packet_start_date`
   - `packet_end_date`
   - `source_eligible`
   - `episode_count`
   - `unique_observed_event_count`
   - `unique_captured_event_count`
   - `unique_missed_event_count`
   - `missed_event_rate`

6. Fill the practitioner judgment fields:
   - `alert_usefulness`: `useful`, `not_useful`, or `unclear`
   - `outcome_confirmed`: `true` or `false`
   - `source_context_ok`: `true` or `false`
   - `action_taken`: `monitor`, `none`, `modified_followup`, or `other`
   - `reviewer_id`: stable reviewer code or initials
   - `review_date`: `YYYY-MM-DD`
   - `notes`: short de-identified rationale

7. Set `collection_status` to `complete_practitioner_adjudication` only when
   the row is complete and defensible.

The most important authenticity rule: if the packet does not give you enough
context, do not force a positive answer. Use `alert_usefulness=unclear`,
`outcome_confirmed=false`, `action_taken=none`, the best
`source_context_ok` judgment you can defend, and explain the limitation in
`notes`.

A defensible uncertain row is better than an overconfident useful row. These
rows are meant to decide whether retained-channel shadow monitoring deserves a
bounded retest, not to create calibration or deployment evidence prematurely.

After the worksheet is filled, run ingest:

```powershell
risk-engine --exposure-load-shadow-prospective-collection-ingest-sprint --exposure-load-shadow-prospective-collection-operations outputs/experiments/exposure_load_shadow_prospective_collection_operations_v1/exposure_load_shadow_prospective_collection_operations.json --completed-prospective-collection <path_to_completed_prospective_collection.csv> --output-dir outputs --experiment-id exposure_load_shadow_prospective_collection_ingest_completed_v1
```

If ingest succeeds, run completion validation on the ingested operations JSON:

```powershell
risk-engine --exposure-load-shadow-prospective-collection-completion-sprint --exposure-load-shadow-prospective-collection-operations outputs/experiments/exposure_load_shadow_prospective_collection_ingest_completed_v1/exposure_load_shadow_prospective_collection_ingested_operations.json --output-dir outputs --experiment-id exposure_load_shadow_prospective_collection_completion_after_ingest_v1
```

Stop at the validation result. Do not proceed to bounded retest until completion
validation says the retained channels are ready for bounded retest, and do not
make calibration, probability-facing, pilot/dashboard, autonomous intervention,
or load-modification claims.
