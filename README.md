# Risk Stratification Engine

Research prototype for a Peterson-inspired athlete risk stratification pipeline.

The first implementation models athlete-seasons as longitudinal trajectories, builds athlete-specific graph snapshots, assembles time-to-event labels with censoring, trains a discrete-time logistic baseline, and writes reproducible experiment artifacts.

## Quickstart

```bash
python -m pip install -e ".[dev]"
pytest
```

## Run Fixture Experiment

```bash
risk-engine \
  --measurements tests/fixtures/measurements.csv \
  --injuries tests/fixtures/injuries.csv \
  --output-dir outputs \
  --experiment-id fixture_run \
  --graph-window-size 2
```

## Local Data Paths

Real data sources should stay in their canonical project locations. Copy
`config/paths.example.yaml` to `config/paths.local.yaml`, update it for your
machine, and keep the local file uncommitted.

The expected sources are:

- `forceplate_db`
- `gps_db`
- `bodyweight_csv`
- `perch_db`
- `injury_csv`

## Run Live-Source Experiment

When `config/paths.local.yaml` points to available local sources, the CLI can
prepare canonical inputs and run the research engine in one step:

```bash
risk-engine \
  --from-live-sources \
  --paths-config config/paths.local.yaml \
  --output-dir outputs \
  --experiment-id actual_data_probe_run \
  --graph-window-size 4
```

Live-source preparation writes ignored canonical CSVs under
`outputs/live_inputs/<experiment-id>/` and records preparation metadata plus a
`data_quality_audit.json` beside them. The audit reports hashed identity overlap
across sources, sparse athlete-seasons, large within-season measurement gaps,
duplicate same-day metric rows, and observed injury events without nearby
measurements. It also includes review context for remaining single-source hashed
identities and injury events outside the nearby-measurement window. Athlete
identities are stable hashes of normalized names, seasons start on July 1, and
the current injury label policy uses the earliest injury issue date per
athlete-season while censoring event-free athlete-seasons at their last
measurement date. Name normalization reconciles common `Last, First` export style
with `First Last` names before hashing. Duplicate same-day metric rows are
aggregated by mean `metric_value` per athlete, season, date, source, and metric
before modeling; the aggregation counts are recorded in `prep_metadata.json`.
Observed injury events are labeled by nearest same-season measurement distance:
`modelable` at 14 days or less, `low_confidence` at 15-30 days, and
`out_of_window` beyond 30 days. When `primary_model_event` is available, the
discrete-time baseline uses it as the positive-event policy so low-confidence
and out-of-window observed events do not become training positives.

## Modeling Baseline

The experiment runner now writes `model_summary.json` alongside
`model_metrics.json`. The first risk model is a discrete-time logistic baseline
trained separately for the 7, 14, and 30 day horizons over graph snapshot
features only: `time_index`, `node_count`, `edge_count`, and
`mean_abs_correlation`. The split is deterministic and athlete-level, with a
sorted 20% holdout. If a training fold has only one class at a horizon, the
runner records a prevalence fallback for that horizon instead of fitting an
unstable classifier.

The reported risk values are baseline model estimates for research comparison,
not calibrated clinical probabilities.

## Philosophy

The athlete-season is the primary modeling unit. Daily measurements are observations inside a trajectory, not independent injury-classification examples.

## Source Materials

The project folder may contain local research PDFs and blueprint documents. They are treated as source references, not package inputs. The research pipeline expects canonical measurement and injury CSV files.

## First Milestone

The first milestone is a reproducible research engine that proves the longitudinal graph/time-to-event data contract. Dashboard performance views come after stable research artifacts exist.
