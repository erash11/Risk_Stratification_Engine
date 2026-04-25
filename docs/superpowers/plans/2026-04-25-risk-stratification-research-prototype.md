# Risk Stratification Research Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working Peterson-inspired research pipeline skeleton for athlete-season trajectory construction, dynamic graph snapshots, time-to-event dataset assembly, and reproducible experiment artifacts.

**Architecture:** Create a small Python package with explicit domain contracts and pure pipeline stages. The first version uses deterministic graph-level features from athlete-specific measurement trajectories, preserving the contract needed for later dynamic graph embeddings and recurrent time-to-event models.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, pytest, pyarrow optional for parquet outputs.

---

## File Structure

- Create `pyproject.toml`: package metadata, dependencies, pytest configuration, console script.
- Create `README.md`: quickstart and research pipeline summary.
- Create `src/risk_stratification_engine/__init__.py`: package marker and version.
- Create `src/risk_stratification_engine/domain.py`: schema constants and validation helpers for canonical tables.
- Create `src/risk_stratification_engine/io.py`: CSV loading, schema checks, and output writing helpers.
- Create `src/risk_stratification_engine/trajectories.py`: canonical measurement pivoting and athlete-season trajectory construction.
- Create `src/risk_stratification_engine/graphs.py`: weekly athlete-specific graph snapshot estimation and graph feature extraction.
- Create `src/risk_stratification_engine/events.py`: injury event and censoring label assembly.
- Create `src/risk_stratification_engine/experiments.py`: end-to-end experiment runner that writes reproducible artifacts.
- Create `src/risk_stratification_engine/cli.py`: command-line entry point for running an experiment from CSVs.
- Create `tests/fixtures/measurements.csv`: small canonical measurement fixture.
- Create `tests/fixtures/injuries.csv`: small injury/censor fixture.
- Create `tests/test_domain.py`: schema and validation tests.
- Create `tests/test_trajectories.py`: athlete-season trajectory tests.
- Create `tests/test_graphs.py`: graph snapshot and feature tests.
- Create `tests/test_events.py`: event/censoring tests.
- Create `tests/test_experiments.py`: full artifact-writing experiment test.

## Scope Notes

This plan implements the first testable research engine skeleton. It does not implement Peterson's full dynamic graph embedding plus recurrent neural time-to-event architecture. Instead, it preserves the Peterson-aligned interfaces:

- athlete-season trajectories as primary units,
- graph snapshots as intermediate artifacts,
- event time plus censoring as labels,
- horizon-based risk timelines and experiment reports as outputs.

Future plans can replace the deterministic graph features with Bayesian dynamic graph estimation and learned graph embeddings while retaining the same input and output contracts.

### Task 1: Python Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/risk_stratification_engine/__init__.py`
- Test: `tests/test_domain.py`

- [ ] **Step 1: Write the package import smoke test**

Create `tests/test_domain.py`:

```python
from risk_stratification_engine import __version__


def test_package_imports():
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
pytest tests/test_domain.py::test_package_imports -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine'`.

- [ ] **Step 3: Add project metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "risk-stratification-engine"
version = "0.1.0"
description = "Peterson-inspired athlete risk stratification research prototype"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pandas>=2.1",
  "scikit-learn>=1.3",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4",
  "pyarrow>=14",
]

[project.scripts]
risk-engine = "risk_stratification_engine.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Add the package version**

Create `src/risk_stratification_engine/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Add quickstart documentation**

Create `README.md`:

````markdown
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
````

- [ ] **Step 6: Run the smoke test**

Run:

```bash
pytest tests/test_domain.py::test_package_imports -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml README.md src/risk_stratification_engine/__init__.py tests/test_domain.py
git commit -m "chore: scaffold research prototype package"
```

### Task 2: Domain Contracts

**Files:**
- Create: `src/risk_stratification_engine/domain.py`
- Modify: `tests/test_domain.py`

- [ ] **Step 1: Add schema validation tests**

Append to `tests/test_domain.py`:

```python
import pandas as pd
import pytest

from risk_stratification_engine.domain import (
    CANONICAL_MEASUREMENT_COLUMNS,
    INJURY_EVENT_COLUMNS,
    require_columns,
)


def test_canonical_measurement_columns_match_spec():
    assert CANONICAL_MEASUREMENT_COLUMNS == (
        "athlete_id",
        "date",
        "season_id",
        "source",
        "metric_name",
        "metric_value",
    )


def test_injury_event_columns_include_censoring_contract():
    assert INJURY_EVENT_COLUMNS == (
        "athlete_id",
        "season_id",
        "injury_date",
        "injury_type",
        "event_observed",
        "censor_date",
    )


def test_require_columns_accepts_complete_dataframe():
    frame = pd.DataFrame(
        {
            "athlete_id": ["a1"],
            "date": ["2026-01-01"],
            "season_id": ["2026"],
            "source": ["force_plate"],
            "metric_name": ["jump_height"],
            "metric_value": [42.0],
        }
    )

    require_columns(frame, CANONICAL_MEASUREMENT_COLUMNS, "measurements")


def test_require_columns_reports_missing_columns():
    frame = pd.DataFrame({"athlete_id": ["a1"]})

    with pytest.raises(ValueError) as exc:
        require_columns(frame, CANONICAL_MEASUREMENT_COLUMNS, "measurements")

    assert "measurements missing required columns" in str(exc.value)
    assert "metric_value" in str(exc.value)
```

- [ ] **Step 2: Run schema tests to verify failure**

Run:

```bash
pytest tests/test_domain.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.domain'`.

- [ ] **Step 3: Implement domain contracts**

Create `src/risk_stratification_engine/domain.py`:

```python
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


CANONICAL_MEASUREMENT_COLUMNS = (
    "athlete_id",
    "date",
    "season_id",
    "source",
    "metric_name",
    "metric_value",
)

INJURY_EVENT_COLUMNS = (
    "athlete_id",
    "season_id",
    "injury_date",
    "injury_type",
    "event_observed",
    "censor_date",
)


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns: {', '.join(sorted(missing))}"
        )
```

- [ ] **Step 4: Run schema tests**

Run:

```bash
pytest tests/test_domain.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/risk_stratification_engine/domain.py tests/test_domain.py
git commit -m "feat: add canonical research data contracts"
```

### Task 3: CSV IO Helpers

**Files:**
- Create: `src/risk_stratification_engine/io.py`
- Create: `tests/fixtures/measurements.csv`
- Create: `tests/fixtures/injuries.csv`
- Create: `tests/test_io.py`

- [ ] **Step 1: Add fixture CSVs**

Create `tests/fixtures/measurements.csv`:

```csv
athlete_id,date,season_id,source,metric_name,metric_value
a1,2026-01-01,2026,force_plate,jump_height,42.0
a1,2026-01-01,2026,force_plate,eccentric_peak_force_asymmetry,8.0
a1,2026-01-08,2026,force_plate,jump_height,39.0
a1,2026-01-08,2026,force_plate,eccentric_peak_force_asymmetry,14.0
a2,2026-01-01,2026,force_plate,jump_height,35.0
a2,2026-01-01,2026,force_plate,eccentric_peak_force_asymmetry,5.0
a2,2026-01-08,2026,force_plate,jump_height,36.0
a2,2026-01-08,2026,force_plate,eccentric_peak_force_asymmetry,6.0
```

Create `tests/fixtures/injuries.csv`:

```csv
athlete_id,season_id,injury_date,injury_type,event_observed,censor_date
a1,2026,2026-01-20,lower_extremity_soft_tissue,true,2026-01-20
a2,2026,,none,false,2026-02-01
```

- [ ] **Step 2: Write IO tests**

Create `tests/test_io.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from risk_stratification_engine.io import load_injury_events, load_measurements, write_frame


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_measurements_parses_dates_and_values():
    frame = load_measurements(FIXTURES / "measurements.csv")

    assert str(frame["date"].dtype).startswith("datetime64")
    assert frame["metric_value"].dtype.kind == "f"
    assert list(frame.columns) == [
        "athlete_id",
        "date",
        "season_id",
        "source",
        "metric_name",
        "metric_value",
    ]


def test_load_injury_events_parses_event_observed_and_dates():
    frame = load_injury_events(FIXTURES / "injuries.csv")

    assert str(frame["injury_date"].dtype).startswith("datetime64")
    assert str(frame["censor_date"].dtype).startswith("datetime64")
    assert frame.loc[frame["athlete_id"] == "a1", "event_observed"].item() is True
    assert frame.loc[frame["athlete_id"] == "a2", "event_observed"].item() is False


def test_load_measurements_rejects_bad_schema(tmp_path):
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("athlete_id,date\nx,2026-01-01\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_measurements(bad_path)

    assert "measurements missing required columns" in str(exc.value)


def test_write_frame_writes_csv(tmp_path):
    output = tmp_path / "frame.csv"
    frame = pd.DataFrame({"athlete_id": ["a1"], "risk": [0.25]})

    write_frame(frame, output)

    assert output.exists()
    loaded = pd.read_csv(output)
    assert loaded.to_dict("records") == [{"athlete_id": "a1", "risk": 0.25}]
```

- [ ] **Step 3: Run IO tests to verify failure**

Run:

```bash
pytest tests/test_io.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.io'`.

- [ ] **Step 4: Implement IO helpers**

Create `src/risk_stratification_engine/io.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_stratification_engine.domain import (
    CANONICAL_MEASUREMENT_COLUMNS,
    INJURY_EVENT_COLUMNS,
    require_columns,
)


def load_measurements(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, CANONICAL_MEASUREMENT_COLUMNS, "measurements")
    frame = frame.loc[:, list(CANONICAL_MEASUREMENT_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("measurements contains unparseable date values")
    if frame["metric_value"].isna().any():
        raise ValueError("measurements contains non-numeric metric_value values")
    return frame


def load_injury_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, INJURY_EVENT_COLUMNS, "injury_events")
    frame = frame.loc[:, list(INJURY_EVENT_COLUMNS)].copy()
    frame["injury_date"] = pd.to_datetime(frame["injury_date"], errors="coerce")
    frame["censor_date"] = pd.to_datetime(frame["censor_date"], errors="coerce")
    frame["event_observed"] = frame["event_observed"].map(_parse_bool)
    if frame["censor_date"].isna().any():
        raise ValueError("injury_events contains unparseable censor_date values")
    if frame.loc[frame["event_observed"], "injury_date"].isna().any():
        raise ValueError("observed injury events require injury_date")
    return frame


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value}")
```

- [ ] **Step 5: Run IO tests**

Run:

```bash
pytest tests/test_io.py -v
```

Expected: PASS.

- [ ] **Step 6: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/risk_stratification_engine/io.py tests/fixtures/measurements.csv tests/fixtures/injuries.csv tests/test_io.py
git commit -m "feat: add canonical csv loading"
```

### Task 4: Athlete-Season Trajectory Builder

**Files:**
- Create: `src/risk_stratification_engine/trajectories.py`
- Create: `tests/test_trajectories.py`

- [ ] **Step 1: Write trajectory tests**

Create `tests/test_trajectories.py`:

```python
from pathlib import Path

from risk_stratification_engine.io import load_measurements
from risk_stratification_engine.trajectories import build_measurement_matrix


FIXTURES = Path(__file__).parent / "fixtures"


def test_build_measurement_matrix_pivots_metrics_by_athlete_season_date():
    measurements = load_measurements(FIXTURES / "measurements.csv")

    matrix = build_measurement_matrix(measurements)

    assert list(matrix.columns) == [
        "athlete_id",
        "season_id",
        "date",
        "time_index",
        "eccentric_peak_force_asymmetry",
        "jump_height",
    ]
    assert matrix.shape[0] == 4
    first = matrix.iloc[0].to_dict()
    assert first["athlete_id"] == "a1"
    assert first["time_index"] == 0
    assert first["jump_height"] == 42.0
    assert first["eccentric_peak_force_asymmetry"] == 8.0


def test_build_measurement_matrix_time_index_resets_by_athlete_season():
    measurements = load_measurements(FIXTURES / "measurements.csv")

    matrix = build_measurement_matrix(measurements)

    indices = matrix.groupby(["athlete_id", "season_id"])["time_index"].apply(list)
    assert indices.loc[("a1", 2026)] == [0, 1]
    assert indices.loc[("a2", 2026)] == [0, 1]
```

- [ ] **Step 2: Run trajectory tests to verify failure**

Run:

```bash
pytest tests/test_trajectories.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.trajectories'`.

- [ ] **Step 3: Implement trajectory builder**

Create `src/risk_stratification_engine/trajectories.py`:

```python
from __future__ import annotations

import pandas as pd

from risk_stratification_engine.domain import (
    CANONICAL_MEASUREMENT_COLUMNS,
    require_columns,
)


def build_measurement_matrix(measurements: pd.DataFrame) -> pd.DataFrame:
    require_columns(measurements, CANONICAL_MEASUREMENT_COLUMNS, "measurements")
    matrix = (
        measurements.pivot_table(
            index=["athlete_id", "season_id", "date"],
            columns="metric_name",
            values="metric_value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["athlete_id", "season_id", "date"])
    )
    matrix.columns.name = None
    metric_columns = sorted(
        column
        for column in matrix.columns
        if column not in {"athlete_id", "season_id", "date"}
    )
    matrix = matrix[["athlete_id", "season_id", "date", *metric_columns]]
    matrix.insert(
        3,
        "time_index",
        matrix.groupby(["athlete_id", "season_id"]).cumcount(),
    )
    return matrix
```

- [ ] **Step 4: Run trajectory tests**

Run:

```bash
pytest tests/test_trajectories.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/risk_stratification_engine/trajectories.py tests/test_trajectories.py
git commit -m "feat: build athlete season measurement matrices"
```

### Task 5: Dynamic Graph Snapshot Features

**Files:**
- Create: `src/risk_stratification_engine/graphs.py`
- Create: `tests/test_graphs.py`

- [ ] **Step 1: Write graph snapshot tests**

Create `tests/test_graphs.py`:

```python
from pathlib import Path

from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.io import load_measurements
from risk_stratification_engine.trajectories import build_measurement_matrix


FIXTURES = Path(__file__).parent / "fixtures"


def test_build_graph_snapshots_returns_athlete_specific_snapshots():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)

    assert list(snapshots.columns) == [
        "athlete_id",
        "season_id",
        "time_index",
        "snapshot_date",
        "node_count",
        "edge_count",
        "mean_abs_correlation",
    ]
    assert snapshots.shape[0] == 4
    assert set(snapshots["athlete_id"]) == {"a1", "a2"}


def test_build_graph_snapshots_preserves_early_history_with_zero_edges():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    first = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 0)
    ].iloc[0]

    assert first["node_count"] == 2
    assert first["edge_count"] == 0
    assert first["mean_abs_correlation"] == 0.0


def test_build_graph_snapshots_detects_relationship_after_history_accumulates():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    matrix = build_measurement_matrix(measurements)

    snapshots = build_graph_snapshots(matrix, window_size=2)
    second = snapshots.loc[
        (snapshots["athlete_id"] == "a1") & (snapshots["time_index"] == 1)
    ].iloc[0]

    assert second["node_count"] == 2
    assert second["edge_count"] == 1
    assert second["mean_abs_correlation"] == 1.0
```

- [ ] **Step 2: Run graph tests to verify failure**

Run:

```bash
pytest tests/test_graphs.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.graphs'`.

- [ ] **Step 3: Implement graph snapshot features**

Create `src/risk_stratification_engine/graphs.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {"athlete_id", "season_id", "date", "time_index"}


def build_graph_snapshots(
    measurement_matrix: pd.DataFrame,
    window_size: int = 4,
    correlation_threshold: float = 0.3,
) -> pd.DataFrame:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    metric_columns = [
        column for column in measurement_matrix.columns if column not in IDENTIFIER_COLUMNS
    ]
    rows: list[dict[str, object]] = []
    grouped = measurement_matrix.sort_values(
        ["athlete_id", "season_id", "date"]
    ).groupby(["athlete_id", "season_id"], sort=False)

    for (athlete_id, season_id), group in grouped:
        for row_position, row in enumerate(group.itertuples(index=False)):
            history = group.iloc[max(0, row_position - window_size + 1) : row_position + 1]
            features = _graph_features(history[metric_columns], correlation_threshold)
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "season_id": season_id,
                    "time_index": int(getattr(row, "time_index")),
                    "snapshot_date": getattr(row, "date"),
                    **features,
                }
            )
    return pd.DataFrame(rows)


def _graph_features(
    history: pd.DataFrame,
    correlation_threshold: float,
) -> dict[str, float | int]:
    node_count = len(history.columns)
    if len(history) < 2 or node_count < 2:
        return {
            "node_count": node_count,
            "edge_count": 0,
            "mean_abs_correlation": 0.0,
        }

    corr = history.corr(numeric_only=True).fillna(0.0).abs().to_numpy()
    upper = corr[np.triu_indices(node_count, k=1)]
    if len(upper) == 0:
        edge_count = 0
        mean_abs_correlation = 0.0
    else:
        edge_count = int((upper >= correlation_threshold).sum())
        mean_abs_correlation = float(round(upper.mean(), 6))
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "mean_abs_correlation": mean_abs_correlation,
    }
```

- [ ] **Step 4: Run graph tests**

Run:

```bash
pytest tests/test_graphs.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/risk_stratification_engine/graphs.py tests/test_graphs.py
git commit -m "feat: derive athlete graph snapshot features"
```

### Task 6: Time-To-Event Labels

**Files:**
- Create: `src/risk_stratification_engine/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: Write event label tests**

Create `tests/test_events.py`:

```python
from pathlib import Path

from risk_stratification_engine.events import attach_time_to_event_labels
from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.io import load_injury_events, load_measurements
from risk_stratification_engine.trajectories import build_measurement_matrix


FIXTURES = Path(__file__).parent / "fixtures"


def test_attach_time_to_event_labels_adds_observed_event_horizons():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)

    labeled = attach_time_to_event_labels(snapshots, injuries)

    a1_first = labeled.loc[
        (labeled["athlete_id"] == "a1") & (labeled["time_index"] == 0)
    ].iloc[0]
    assert a1_first["event_observed"] is True
    assert a1_first["days_to_event"] == 19
    assert a1_first["injury_type"] == "lower_extremity_soft_tissue"
    assert a1_first["event_within_7d"] is False
    assert a1_first["event_within_14d"] is False
    assert a1_first["event_within_30d"] is True


def test_attach_time_to_event_labels_preserves_censored_athletes():
    measurements = load_measurements(FIXTURES / "measurements.csv")
    injuries = load_injury_events(FIXTURES / "injuries.csv")
    snapshots = build_graph_snapshots(build_measurement_matrix(measurements), window_size=2)

    labeled = attach_time_to_event_labels(snapshots, injuries)

    a2_first = labeled.loc[
        (labeled["athlete_id"] == "a2") & (labeled["time_index"] == 0)
    ].iloc[0]
    assert a2_first["event_observed"] is False
    assert a2_first["days_to_event"] == 31
    assert a2_first["injury_type"] == "none"
    assert a2_first["event_within_30d"] is False
```

- [ ] **Step 2: Run event tests to verify failure**

Run:

```bash
pytest tests/test_events.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.events'`.

- [ ] **Step 3: Implement time-to-event labels**

Create `src/risk_stratification_engine/events.py`:

```python
from __future__ import annotations

import pandas as pd


DEFAULT_HORIZONS = (7, 14, 30)


def attach_time_to_event_labels(
    snapshots: pd.DataFrame,
    injury_events: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    labeled = snapshots.merge(
        injury_events,
        on=["athlete_id", "season_id"],
        how="left",
        validate="many_to_one",
    )
    labeled["event_date"] = labeled["injury_date"].where(
        labeled["event_observed"], labeled["censor_date"]
    )
    labeled["days_to_event"] = (
        labeled["event_date"] - labeled["snapshot_date"]
    ).dt.days
    labeled = labeled.loc[labeled["days_to_event"] >= 0].copy()
    for horizon in horizons:
        labeled[f"event_within_{horizon}d"] = (
            labeled["event_observed"] & (labeled["days_to_event"] <= horizon)
        )
    boolean_columns = ["event_observed", *(f"event_within_{horizon}d" for horizon in horizons)]
    for column in boolean_columns:
        labeled[column] = labeled[column].astype(object)
    return labeled
```

- [ ] **Step 4: Run event tests**

Run:

```bash
pytest tests/test_events.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/risk_stratification_engine/events.py tests/test_events.py
git commit -m "feat: attach time to event labels"
```

### Task 7: Experiment Artifact Writer

**Files:**
- Create: `src/risk_stratification_engine/experiments.py`
- Create: `tests/test_experiments.py`

- [ ] **Step 1: Write experiment artifact test**

Create `tests/test_experiments.py`:

```python
from pathlib import Path

import json
import pandas as pd

from risk_stratification_engine.experiments import run_research_experiment


FIXTURES = Path(__file__).parent / "fixtures"


def test_run_research_experiment_writes_artifacts(tmp_path):
    result = run_research_experiment(
        measurements_path=FIXTURES / "measurements.csv",
        injuries_path=FIXTURES / "injuries.csv",
        output_dir=tmp_path,
        experiment_id="fixture_research_run",
        graph_window_size=2,
    )

    experiment_dir = tmp_path / "experiments" / "fixture_research_run"
    assert result == experiment_dir
    assert (experiment_dir / "config.json").exists()
    assert (experiment_dir / "model_metrics.json").exists()
    assert (experiment_dir / "experiment_report.md").exists()
    assert (experiment_dir / "athlete_risk_timeline.csv").exists()
    assert (experiment_dir / "graph_snapshots" / "graph_features.csv").exists()
    assert (experiment_dir / "explanations" / "explanation_summary.csv").exists()

    metrics = json.loads((experiment_dir / "model_metrics.json").read_text())
    assert metrics["athlete_count"] == 2
    assert metrics["snapshot_count"] == 4
    assert metrics["observed_event_count"] == 2

    timeline = pd.read_csv(experiment_dir / "athlete_risk_timeline.csv")
    assert {"risk_7d", "risk_14d", "risk_30d"}.issubset(timeline.columns)
```

- [ ] **Step 2: Run experiment test to verify failure**

Run:

```bash
pytest tests/test_experiments.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.experiments'`.

- [ ] **Step 3: Implement experiment runner**

Create `src/risk_stratification_engine/experiments.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from risk_stratification_engine.events import DEFAULT_HORIZONS, attach_time_to_event_labels
from risk_stratification_engine.graphs import build_graph_snapshots
from risk_stratification_engine.io import load_injury_events, load_measurements, write_frame
from risk_stratification_engine.trajectories import build_measurement_matrix


def run_research_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    graph_window_size: int = 4,
) -> Path:
    measurements = load_measurements(measurements_path)
    injuries = load_injury_events(injuries_path)
    matrix = build_measurement_matrix(measurements)
    graph_features = build_graph_snapshots(matrix, window_size=graph_window_size)
    labeled = attach_time_to_event_labels(graph_features, injuries)
    timeline = _risk_timeline(labeled)
    explanations = _explanation_summary(timeline)

    experiment_dir = Path(output_dir) / "experiments" / experiment_id
    graph_dir = experiment_dir / "graph_snapshots"
    explanation_dir = experiment_dir / "explanations"
    graph_dir.mkdir(parents=True, exist_ok=True)
    explanation_dir.mkdir(parents=True, exist_ok=True)

    write_frame(graph_features, graph_dir / "graph_features.csv")
    write_frame(timeline, experiment_dir / "athlete_risk_timeline.csv")
    write_frame(explanations, explanation_dir / "explanation_summary.csv")
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "graph_window_size": graph_window_size,
            "horizons": list(DEFAULT_HORIZONS),
        },
    )
    _write_json(
        experiment_dir / "model_metrics.json",
        {
            "athlete_count": int(labeled["athlete_id"].nunique()),
            "snapshot_count": int(len(labeled)),
            "observed_event_count": int(labeled["event_observed"].sum()),
            "mean_risk_7d": float(timeline["risk_7d"].mean()),
            "mean_risk_14d": float(timeline["risk_14d"].mean()),
            "mean_risk_30d": float(timeline["risk_30d"].mean()),
        },
    )
    _write_report(experiment_dir / "experiment_report.md", timeline)
    return experiment_dir


def _risk_timeline(labeled: pd.DataFrame) -> pd.DataFrame:
    timeline = labeled.copy()
    denominator = timeline["days_to_event"].clip(lower=1)
    for horizon in DEFAULT_HORIZONS:
        temporal_pressure = (horizon / denominator).clip(upper=1.0)
        graph_pressure = timeline["mean_abs_correlation"].clip(lower=0.0, upper=1.0)
        observed_pressure = timeline[f"event_within_{horizon}d"].astype(float)
        timeline[f"risk_{horizon}d"] = (
            0.5 * observed_pressure + 0.3 * temporal_pressure + 0.2 * graph_pressure
        ).round(6)
    return timeline


def _explanation_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "athlete_id",
        "season_id",
        "time_index",
        "snapshot_date",
        "mean_abs_correlation",
        "edge_count",
        "risk_7d",
        "risk_14d",
        "risk_30d",
    ]
    explanations = timeline.loc[:, columns].copy()
    explanations["primary_signal"] = explanations.apply(_primary_signal, axis=1)
    return explanations


def _primary_signal(row: pd.Series) -> str:
    if row["edge_count"] == 0:
        return "insufficient_history"
    if row["mean_abs_correlation"] >= 0.7:
        return "strong_metric_relationship_shift"
    return "moderate_metric_relationship_shift"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_report(path: Path, timeline: pd.DataFrame) -> None:
    lines = [
        "# Experiment Report",
        "",
        f"Snapshots: {len(timeline)}",
        f"Athletes: {timeline['athlete_id'].nunique()}",
        f"Mean +7 day risk: {timeline['risk_7d'].mean():.3f}",
        f"Mean +14 day risk: {timeline['risk_14d'].mean():.3f}",
        f"Mean +30 day risk: {timeline['risk_30d'].mean():.3f}",
        "",
        "This first artifact is a deterministic research baseline over graph snapshot features. It preserves the longitudinal time-to-event contract for later model replacement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run experiment test**

Run:

```bash
pytest tests/test_experiments.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/risk_stratification_engine/experiments.py tests/test_experiments.py
git commit -m "feat: write reproducible research artifacts"
```

### Task 8: CLI Experiment Runner

**Files:**
- Create: `src/risk_stratification_engine/cli.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write CLI test**

Create `tests/test_cli.py`:

```python
from pathlib import Path

from risk_stratification_engine.cli import main


FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_runs_fixture_experiment(tmp_path):
    exit_code = main(
        [
            "--measurements",
            str(FIXTURES / "measurements.csv"),
            "--injuries",
            str(FIXTURES / "injuries.csv"),
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "cli_fixture_run",
            "--graph-window-size",
            "2",
        ]
    )

    assert exit_code == 0
    assert (
        tmp_path
        / "experiments"
        / "cli_fixture_run"
        / "athlete_risk_timeline.csv"
    ).exists()
```

- [ ] **Step 2: Run CLI test to verify failure**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_stratification_engine.cli'`.

- [ ] **Step 3: Implement CLI**

Create `src/risk_stratification_engine/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from risk_stratification_engine.experiments import run_research_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Peterson-inspired risk stratification research pipeline."
    )
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--injuries", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--graph-window-size", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = run_research_experiment(
        measurements_path=args.measurements,
        injuries_path=args.injuries,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        graph_window_size=args.graph_window_size,
    )
    print(f"Experiment artifacts written to {experiment_dir}")
    return 0
```

- [ ] **Step 4: Update README quickstart**

Modify `README.md` to include:

````markdown
## Run Fixture Experiment

```bash
risk-engine \
  --measurements tests/fixtures/measurements.csv \
  --injuries tests/fixtures/injuries.csv \
  --output-dir outputs \
  --experiment-id fixture_run \
  --graph-window-size 2
```
````

- [ ] **Step 5: Run CLI test**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/risk_stratification_engine/cli.py tests/test_cli.py README.md
git commit -m "feat: add research experiment cli"
```

### Task 9: Documentation And Repository Hygiene

**Files:**
- Create: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Add hygiene expectations**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
build/
dist/
*.egg-info/
outputs/
```

- [ ] **Step 2: Add source material note to README**

Append to `README.md`:

````markdown
## Source Materials

The project folder may contain local research PDFs and blueprint documents. They are treated as source references, not package inputs. The research pipeline expects canonical measurement and injury CSV files.

## First Milestone

The first milestone is a reproducible research engine that proves the longitudinal graph/time-to-event data contract. Dashboard performance views come after stable research artifacts exist.
````

- [ ] **Step 3: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 4: Check status**

Run:

```bash
git status --short
```

Expected: `.gitignore` and `README.md` are modified or untracked; source PDFs and previously extracted Malum UI images may remain untracked.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md
git commit -m "docs: document research prototype usage"
```

### Task 10: Final Verification

**Files:**
- No file changes expected.

- [ ] **Step 1: Run full test suite**

Run:

```bash
pytest -v
```

Expected: PASS for all tests.

- [ ] **Step 2: Run fixture experiment through CLI**

Run:

```bash
risk-engine --measurements tests/fixtures/measurements.csv --injuries tests/fixtures/injuries.csv --output-dir outputs --experiment-id fixture_run --graph-window-size 2
```

Expected: command prints `Experiment artifacts written to outputs/experiments/fixture_run`.

- [ ] **Step 3: Inspect generated artifacts**

Run:

```bash
python -c "from pathlib import Path; root=Path('outputs/experiments/fixture_run'); print(sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()))"
```

Expected output includes:

```text
['athlete_risk_timeline.csv', 'config.json', 'experiment_report.md', 'explanations/explanation_summary.csv', 'graph_snapshots/graph_features.csv', 'model_metrics.json']
```

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only local source materials remain untracked unless the user chooses to add them.

## Plan Self-Review

Spec coverage:

- Research prototype first: Tasks 1, 7, 8, and 9.
- Canonical measurement and injury data contracts: Tasks 2 and 3.
- Athlete-season trajectory construction: Task 4.
- Dynamic graph snapshots: Task 5.
- Time-to-event labels and censoring: Task 6.
- Reproducible research artifacts: Tasks 7, 8, and 10.
- Dashboard deferred: Task 9 documents this and no dashboard files are created.

Unresolved-marker scan:

- The plan defines exact files, commands, and expected outcomes.

Type consistency:

- `build_measurement_matrix`, `build_graph_snapshots`, `attach_time_to_event_labels`, `run_research_experiment`, and `main` are introduced before use or in the same task that uses them.
