# Risk Stratification Engine

Research prototype for a Peterson-inspired athlete risk stratification pipeline.

The first implementation models athlete-seasons as longitudinal trajectories, builds athlete-specific graph snapshots, assembles time-to-event labels with censoring, and writes reproducible experiment artifacts.

## Quickstart

```bash
python -m pip install -e ".[dev]"
pytest
```

## Philosophy

The athlete-season is the primary modeling unit. Daily measurements are observations inside a trajectory, not independent injury-classification examples.
