# Shadow Collection Reviewer Process Reference

This reference preserves the exact reviewer process used for completing the
retained-channel shadow collection template. It should be read alongside
`docs/shadow_collection_reviewer_guide.md`.

Use this file, not the older blank template:

`outputs/experiments/exposure_load_shadow_collection_evidence_prefill_v1/exposure_load_shadow_collection_prefilled.csv`

The exact reviewer process is:

1. Open one row at a time.
   Treat each row as one retained-channel shadow packet for a complete
   source-eligible season/window.

2. Do not change the prefilled evidence fields unless you find a real source
   error:
   `collection_packet_id`, `channel_name`, `collection_season_id`,
   `packet_start_date`, `packet_end_date`, `source_eligible`, `episode_count`,
   `unique_observed_event_count`, `unique_captured_event_count`,
   `evidence_source_packet_id`, `evidence_source`.

3. Review the packet evidence and any internal context you have.
   Your job is not to prove the model is right. Your job is to judge whether
   this packet would have been interpretable and operationally meaningful in
   shadow mode.

4. Fill `alert_usefulness` conservatively:
   - `useful`: coherent concern, source context trustworthy, would support
     reasonable staff attention.
   - `noisy`: interpretable signal, but not practically useful.
   - `misleading`: would likely point staff in the wrong direction.
   - `unclear`: not enough context to judge.

5. Fill the remaining judgment fields:
   - `outcome_confirmed`: `true` only if the packet aligns with real injury,
     limitation, clinical concern, or managed-risk context; otherwise `false`.
   - `source_context_ok`: `true` if the data context is trustworthy enough to
     interpret; otherwise `false`.
   - `action_taken`: one of `none`, `monitor`, `communicate`, `modify_load`,
     `clinical_review`, `other`.
   - `reviewer_id`: stable initials/code.
   - `review_date`: `YYYY-MM-DD`.
   - `notes`: short de-identified rationale.

6. Set `collection_status` to `complete` only after you have made the reviewer
   judgment for that row.

The most important authenticity rule: if the packet does not give you enough
context, do not infer or force a positive answer. Use
`alert_usefulness=unclear`, `outcome_confirmed=false`, the best
`source_context_ok` judgment you can defend, `action_taken=none`, and explain
the limitation in `notes`.

A defensible uncertain row is better than an overconfident useful row. These
rows are meant to test whether retained-channel shadow monitoring deserves
more research, not to create calibration or deployment evidence prematurely.

After the 8 rows are filled, run the summary validation:

```powershell
risk-engine --exposure-load-shadow-collection-summary-sprint --exposure-load-shadow-collection outputs/experiments/exposure_load_shadow_collection_evidence_prefill_v1/exposure_load_shadow_collection_prefilled.csv --output-dir outputs --experiment-id exposure_load_shadow_collection_summary_completed_v1
```
