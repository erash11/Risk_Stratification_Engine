# Qualitative Case Review + Data Diagnostic v1 Design

## Goal

Add a small, deterministic review artifact that explains why selected alert episodes are worth human inspection and classifies each case as likely model signal, label/data issue, missing context, or explanation gap.

## Scope

This sprint extends the existing `--alert-episodes` workflow. It does not add a dashboard, a manual annotation UI, or a new modeling algorithm. The artifact should help decide whether the next model-performance improvement should come from better labels, richer context data, feature work, or model changes.

## Artifacts

The alert episode run will write:

- `qualitative_case_review.json`: structured case records and summary counts.
- `qualitative_case_review_report.md`: a short human-readable review report.

These artifacts are written beside the existing alert episode and episode quality artifacts.

## Case Types

The review includes one deterministic case per horizon/threshold when available:

- `true_positive_episode`: highest peak-risk episode whose start is within the forecast horizon before an observed event.
- `false_positive_episode`: highest peak-risk episode whose start is not within the forecast horizon before an observed event.
- `missed_injury`: earliest observed injury event not captured by a true-positive episode for that horizon/threshold.
- `high_intra_individual_deviation_episode`: highest peak-risk episode with at least one elevated z-score feature.

The first operational target remains 30d top-5%, but the builder should work for every horizon/threshold pair already produced by the episode quality audit.

## Case Record

Each case includes:

- case type, review label, and suggested diagnosis.
- athlete ID and season ID, using existing hashed IDs only.
- horizon and threshold.
- episode dates and event lead-time fields when the case is episode-based.
- event date and injury type when the case is injury-based.
- event-window quality, nearest measurement gap, and primary-model-event flag when available.
- top model features and elevated z-score features.
- compact timeline context for the same athlete-season, including snapshot date, time index, risk columns, top feature columns, elevated z-score features, event timing fields, and data-quality label fields.

## Diagnostic Labels

The diagnostic labels are intentionally simple:

- `model_signal_supported`: true-positive episode with modelable event-window quality.
- `missing_context_or_managed_risk`: false-positive episode, especially when risk and z-score signals are high but no event is observed.
- `possible_label_or_measurement_gap`: missed injury with low-confidence, out-of-window, or large nearest-measurement gap.
- `model_miss`: missed injury that appears modelable but was not preceded by an alert.
- `explanation_gap`: high-deviation episode that is not a true-positive episode.

These labels are not final clinical conclusions. They are triage notes for deciding where the next sprint should focus.

## Integration

Create `risk_stratification_engine.case_review` so case-review logic stays separate from episode construction and quality metrics. `run_alert_episode_experiment(...)` will call the case-review builder after `build_alert_episode_quality(...)`, then write JSON and Markdown artifacts.

## Testing

Use strict TDD. First add failing tests for:

- deterministic true-positive, false-positive, missed-injury, and high-deviation case selection.
- diagnostic labels for model-supported, missing-context, label/data-gap, model-miss, and explanation-gap cases.
- timeline context inclusion without exposing unhashed identifiers.
- experiment-level artifact writes.

Then implement the minimal code needed, run targeted tests, the full suite, and a live-source `qualitative_case_review_v1` command.
