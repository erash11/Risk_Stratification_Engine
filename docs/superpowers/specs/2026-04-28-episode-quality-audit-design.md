# Episode Quality Audit v1 Design

## Goal

Add a durable audit layer to alert episode validation so the research workbench can distinguish useful early warnings from noisy high-risk episodes before any dashboard work begins.

## Scope

Episode Quality Audit v1 extends the existing `--alert-episodes` run path. It does not add a separate dashboard, new model family, or manual review workflow. The audit uses the same trained model, graph window, horizons, and percentile alert thresholds already used by alert episode validation.

## Artifacts

The alert episode run will write three additional artifacts beside the current episode files:

- `alert_episode_quality.csv`: one row per horizon and threshold.
- `alert_episode_quality.json`: experiment metadata plus the same per-threshold quality rows and representative case records.
- `alert_episode_quality_report.md`: a compact human-readable report for the sprint handoff.

The existing `alert_episodes.csv`, `alert_episodes.json`, `alert_episode_summary.json`, and `alert_episode_report.md` remain backward-compatible.

## Definitions

An episode is a true-positive episode when `event_within_horizon_after_start` is true. It is a false-positive episode when that field is false. This start-based definition is the default because it measures whether the alert began early enough to be actionable.

A unique observed injury event is identified by `athlete_id`, `season_id`, `event_date`, and `injury_type` from timeline rows where `event_observed` is true. If an event date is unavailable, the audit still groups by athlete-season and injury type.

A missed injury for a horizon and threshold is an observed event that has no true-positive episode at that same horizon and threshold.

Lead time is `days_from_start_to_event` for true-positive episodes. The audit also preserves median peak and end lead time when available.

Top-5% and top-10% overlap is computed within each horizon by matching episode identity fields: `athlete_id`, `season_id`, `start_time_index`, `end_time_index`, and `horizon_days`.

## Metrics

Each horizon/threshold quality row includes:

- episode count
- true-positive episode count and rate
- false-positive episode count and rate
- unique observed injury count
- unique injury events captured count and rate
- missed injury count
- alert episodes per athlete-season
- median lead days from start, peak, and end
- median duration days and snapshot count
- true-positive and false-positive median peak risk
- true-positive and false-positive median duration days
- true-positive and false-positive elevated z-feature episode rates
- true-positive and false-positive top model feature counts

## Representative Cases

The JSON report includes a small deterministic case set per horizon/threshold:

- `true_positive_episode`: highest peak-risk true-positive episode.
- `false_positive_episode`: highest peak-risk false-positive episode.
- `missed_injury`: earliest observed injury without a prior true-positive episode.
- `high_intra_individual_deviation_episode`: highest peak-risk episode with at least one elevated z-score feature.

Case records include only hashed athlete IDs, season IDs, horizon, threshold, dates, lead-time fields, peak/mean risk, injury type, top model features, and elevated z-score features.

## Integration

Add a focused module, `risk_stratification_engine.episode_quality`, so `alert_episodes.py` remains responsible for episode construction. `run_alert_episode_experiment(...)` will call the quality builder after episodes are created and write the new artifacts.

## Testing

Use strict TDD. First add failing tests for:

- unique injury capture, missed injury count, and false-positive rate
- alert burden per athlete-season and top-5/top-10 overlap
- representative true-positive, false-positive, missed-injury, and high-deviation cases
- experiment-level artifact writes

Then implement the minimum code needed to pass, run the targeted tests, the full pytest suite, and a live-source `alert_episode_quality_v1` command.
