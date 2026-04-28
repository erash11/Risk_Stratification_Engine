# Model Improvement Diagnostic Table Design

## Purpose

The qualitative case review showed that current performance limits are not only a data problem: some modelable injuries are still missed, while some high-risk alerts look physiologically plausible but lack enough context to know whether they are useful. This sprint turns that finding into a cohort-level diagnostic artifact.

## Scope

Add a deterministic model-improvement diagnostic table to the existing `--alert-episodes` workflow. The artifact compares three groups for each horizon and threshold:

- `true_positive_episode`: alert episodes that begin within the forecast horizon before an observed event.
- `false_positive_episode`: alert episodes with no observed event within the forecast horizon after episode start.
- `missed_event`: observed injury events not captured by the selected alert policy.

The table should not change model fitting, thresholds, labels, or alert episode construction. It is an analysis layer that explains what to improve next.

## Outputs

The alert episode runner will write:

- `model_improvement_diagnostics.csv`
- `model_improvement_diagnostics.json`
- `model_improvement_diagnostic_report.md`

Each row will include counts, risk summaries, elevated intra-individual z-score rates, top model feature counts, event-window quality counts, nearest-measurement gap summaries, available pre-event snapshot counts for missed events, and a concise `recommended_next_action`.

## Diagnostic Logic

For true-positive and false-positive alert groups, use the existing alert episode rows. Summaries should include episode count, median/mean peak risk, median duration, elevated z-feature episode rate, and top model feature counts.

For missed events, identify observed events in the modeled timeline that are not represented by a start-based true-positive episode for that horizon/threshold. For each missed event, summarize the model behavior in the pre-event lead window where `0 <= days_to_event <= horizon_days`: maximum risk, median risk, snapshot count, elevated z-feature rate, top feature counts, event-window quality, and nearest-measurement gap. If no pre-event window rows exist, record zero snapshots and null risk summaries.

## Recommendations

The `recommended_next_action` field should make the table actionable:

- `retain_policy_signal`: true-positive alert group with modelable event coverage.
- `add_context_features`: false-positive alert group, because the current feature set cannot separate managed-risk/adaptation from harmful risk.
- `improve_data_linkage`: missed events dominated by non-modelable or sparse measurement context.
- `add_event_specific_features`: modelable missed events whose pre-event risk stayed low.
- `review_threshold_policy`: modelable missed events whose pre-event risk was high but did not cross the selected threshold.

## Testing

Use TDD. Unit tests should prove:

- the table includes all three group types for a threshold with true-positive, false-positive, and missed-event examples;
- missed events use the pre-event lead window instead of censoring dates or post-event rows;
- recommendation labels distinguish low-risk model misses from near-threshold policy misses;
- empty groups produce stable rows with null summaries instead of crashing.

Integration tests should prove the alert episode runner writes all three new artifacts.
