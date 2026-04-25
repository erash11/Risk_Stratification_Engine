# Risk Stratification Research Prototype Design

## Summary

Build a Peterson-inspired longitudinal risk stratification research prototype. The system will model each athlete-season as an evolving physiological system, not as disconnected daily observations. The prototype will ingest historical athlete monitoring data, construct athlete-specific dynamic graph trajectories, forecast injury risk as a longitudinal time-to-event problem, and produce athlete-specific explanations for elevated risk.

The first milestone is a research engine. A coach-facing dashboard or performance tab can come later after the modeling pipeline demonstrates useful behavior.

## Source Review

The current project folder contains:

- `Peterson et al. - 2021 - Longitudinal time-to-event graph mining pipeline for musculoskeletal injury forecasting.pdf`
- `Malum.pdf`
- `Local_AI_Analytics_Blueprint.md`
- `Local_AI_Analytics_Blueprint-v2.md`

The Peterson dissertation is the primary conceptual source. Its central thesis is that injury forecasting should be formulated around complex systems, idiographic athlete trajectories, dynamic graphs, longitudinal time-to-event modeling, and local temporal explanations.

The Malum deck contributes product direction. It emphasizes readiness, musculoskeletal injury forecasting, dynamic risk horizons, what-if simulation, influential variables, suggested interventions, and future digital-twin-style planning. A separate Spear document was not present in the folder; SPEAR appears inside the Malum production UI slide.

## Design Goals

- Preserve athlete history instead of treating each day as independent.
- Model injury as a future event with censoring.
- Estimate athlete-specific dynamic graph trajectories from monitoring variables.
- Evaluate whether risk forecasts improve as more athlete history accumulates.
- Explain high-risk forecasts through variable relationships and relevant time windows.
- Produce reproducible research artifacts that can later power a dashboard.

## Non-Goals

- Build the final dashboard in the first milestone.
- Treat the first prototype as a daily binary injury classifier.
- Optimize for real-time data streaming before the historical research pipeline works.
- Claim clinical or operational validity before retrospective validation is complete.

## Core Architecture

The system follows three Peterson-inspired stages.

### Stage 1: Intra-Athlete Dynamic Graph Construction

Each athlete-season becomes a sequence of graph snapshots. Nodes represent monitoring variables, such as force plate metrics, HR or recovery measures, wellness scores, GPS load variables, strength or asymmetry metrics, body composition, or motion capture features. Edges represent evolving relationships among those variables.

The first prototype should start with the densest and most reliable streams. Force plate and internal-load or recovery metrics are preferred starting points because they align closely with Peterson's applied example. Additional streams can be added once the pipeline is stable.

Initial graph snapshots should be weekly unless the available data density suggests a different interval.

### Stage 2: Longitudinal Time-To-Event Forecasting

The forecasting model consumes graph trajectories and estimates future injury risk over defined horizons. Injured athletes contribute event times. Athletes who remain injury-free contribute censored trajectories. Censoring must be handled explicitly because it preserves useful information from athletes who did not experience the event during observation.

Initial horizons should include `+7`, `+14`, and `+30` days.

The first implementation may use graph-level trajectory features or graph embeddings as a practical bridge toward Peterson's full dynamic graph embedding plus recurrent time-to-event architecture. The implementation should preserve the same data contract so more advanced models can replace simpler ones without changing the rest of the pipeline.

### Stage 3: Athlete-Specific Explanation

For each risk forecast, especially elevated-risk forecasts, the system should identify which variables, graph relationships, and time windows contributed most. These explanations are research artifacts first, but they will later become the foundation for a dashboard's influential variables, suggested changes, and simulation views.

## Data Model

### Canonical Measurement Table

Raw exports should normalize into a long measurement format:

```text
athlete_id
date
season_id
source
metric_name
metric_value
session_id
team
```

`session_id` and `team` are optional but useful when available.

### Athlete-Season Trajectory

The modeling dataset centers on athlete-season trajectories:

```text
athlete_id
season_id
time_index
timestamp
measurement_matrix
graph_snapshot
event_time
event_observed
censor_time
injury_type
```

`event_observed = false` means the athlete was event-free through the observed window. It does not mean the athlete is permanently healthy.

## Experiment Design

The first research experiments should answer:

1. Can the available data reconstruct meaningful athlete-specific dynamic graphs?
2. Does forecast performance improve as more athlete history accumulates?
3. Which forecast horizons are most useful: `+7`, `+14`, or `+30` days?
4. Are athlete-specific explanations stable enough to make sense to practitioners?
5. Which data streams add signal, and which add noise or missingness burden?

Validation must use athlete-level and time-aware splits. Random daily-row splits are inappropriate because they leak athlete history and conflict with the longitudinal premise.

## Evaluation

The research prototype should report:

- Time-dependent concordance index for discrimination.
- Time-dependent Brier score for calibration.
- Performance by forecast horizon.
- Performance by season phase or accumulated history length.
- Sensitivity by injury type where sample size allows.
- Explanation stability across adjacent graph snapshots.
- Missingness and data-density diagnostics by source.

## Research Outputs

Each experiment should write reproducible artifacts:

```text
outputs/
  experiments/
    <experiment_id>/
      config.json
      model_metrics.json
      experiment_report.md
      athlete_risk_timeline.csv
      graph_snapshots/
      explanations/
```

The outputs should be machine-readable enough for future dashboard use and human-readable enough for research review.

## Risks And Constraints

- Small samples or sparse monitoring histories may make full dynamic graph estimation unstable.
- Graph estimation can be sensitive to node count, measurement frequency, missingness, and smoothing assumptions.
- The full Peterson architecture may require staged implementation if data density is limited.
- Explanations must be treated as decision-support hypotheses, not causal proof.
- Data privacy and athlete health sensitivity should shape storage, sharing, and later dashboard access controls.

## Future Dashboard Direction

Once the research engine produces credible artifacts, a dashboard performance tab can consume the same outputs. Future dashboard concepts may include:

- Athlete risk horizon cards.
- Injury-free probability timelines.
- Influential variables and graph relationships.
- Cohort comparisons.
- What-if simulation controls.
- Suggested intervention targets.

Dashboard work should not begin until the research pipeline can generate stable risk timelines and explanation reports.

## Approval Status

The user approved the Peterson-inspired research prototype direction in conversation. The next step is for the user to review this written spec before implementation planning begins.
