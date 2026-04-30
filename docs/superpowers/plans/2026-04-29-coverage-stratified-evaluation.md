# Coverage-Stratified Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `coverage_analysis.py` module and `--coverage-stratified-evaluation` CLI mode that assigns population-wide coverage tiers to athlete-seasons and evaluates whether shadow-mode channel capture rates track those tiers.

**Architecture:** New standalone `coverage_analysis.py` module (pure data functions + report writer, no experiment logic). New runner `run_coverage_stratified_evaluation_experiment` in `experiments.py` that loops the four fixed shadow-mode channels, joins coverage tiers to each per-channel model timeline, and calls `build_coverage_stratified_evaluation`. New `--coverage-stratified-evaluation` CLI flag dispatches to the runner.

**Tech Stack:** Python 3.11+, pandas, pytest. No new dependencies.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/risk_stratification_engine/coverage_analysis.py` | Coverage tier assignment, stratified evaluation, report writer |
| Create | `tests/test_coverage_analysis.py` | Unit tests for all three public functions |
| Modify | `src/risk_stratification_engine/experiments.py` | Add `run_coverage_stratified_evaluation_experiment` and `_coverage_flag` helper |
| Modify | `src/risk_stratification_engine/cli.py` | Add `--coverage-stratified-evaluation` flag and dispatch |
| Modify | `tests/test_experiments.py` | Add integration test for the new runner |
| Modify | `tests/test_cli.py` | Add CLI dispatch test for the new flag |

---

## Task 1: build_coverage_tiers — empty input

**Files:**
- Create: `src/risk_stratification_engine/coverage_analysis.py`
- Create: `tests/test_coverage_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage_analysis.py
import pandas as pd
from risk_stratification_engine.coverage_analysis import build_coverage_tiers


def test_build_coverage_tiers_empty_measurements_returns_correct_columns():
    empty = pd.DataFrame(
        columns=["athlete_id", "date", "season_id", "source", "metric_name", "metric_value"]
    )
    result = build_coverage_tiers(empty)
    assert list(result.columns) == [
        "athlete_id",
        "season_id",
        "measurement_days",
        "measurement_row_count",
        "source_count",
        "median_days_between_measurements",
        "coverage_tier",
    ]
    assert len(result) == 0
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_coverage_analysis.py::test_build_coverage_tiers_empty_measurements_returns_correct_columns -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `coverage_analysis` does not exist yet.

- [ ] **Step 3: Create coverage_analysis.py with the empty-input implementation**

```python
# src/risk_stratification_engine/coverage_analysis.py
from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


COVERAGE_TIER_LABELS = ("low", "medium", "high")

_TIER_COLUMNS = [
    "athlete_id",
    "season_id",
    "measurement_days",
    "measurement_row_count",
    "source_count",
    "median_days_between_measurements",
    "coverage_tier",
]


def build_coverage_tiers(measurements: pd.DataFrame) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame(columns=_TIER_COLUMNS)
    frame = measurements.copy()
    frame["season_id"] = frame["season_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    rows = []
    for (athlete_id, season_id), group in frame.groupby(
        ["athlete_id", "season_id"], sort=True
    ):
        rows.append(
            {
                "athlete_id": athlete_id,
                "season_id": str(season_id),
                "measurement_days": int(group["date"].nunique()),
                "measurement_row_count": int(len(group)),
                "source_count": int(group["source"].nunique()),
                "median_days_between_measurements": _median_days(group),
            }
        )
    tier_frame = pd.DataFrame(rows)
    tier_frame["coverage_tier"] = _assign_tiers(tier_frame["measurement_days"])
    return tier_frame[_TIER_COLUMNS]


def _assign_tiers(measurement_days: pd.Series) -> pd.Series:
    n = len(measurement_days)
    if n == 0:
        return pd.Series(dtype=str)
    if n < 3:
        return pd.Series(["low"] * n, index=measurement_days.index)
    try:
        result = pd.qcut(
            measurement_days,
            q=3,
            labels=list(COVERAGE_TIER_LABELS),
            duplicates="drop",
        )
        return result.astype(str).fillna("low")
    except ValueError:
        return pd.Series(["low"] * n, index=measurement_days.index)


def _median_days(group: pd.DataFrame) -> float | None:
    dates = group["date"].dropna().drop_duplicates().sort_values()
    if len(dates) < 2:
        return None
    deltas = dates.diff().dropna().dt.days.tolist()
    if not deltas:
        return None
    return round(float(pd.Series(deltas).median()), 3)


def build_coverage_stratified_evaluation(
    timeline_with_tiers: pd.DataFrame,
    channel: dict,
) -> dict:
    raise NotImplementedError


def build_coverage_flag(channel_results: list[dict]) -> str:
    raise NotImplementedError


def write_coverage_stratified_evaluation_report(
    path: Path,
    result: dict,
) -> None:
    raise NotImplementedError


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if not isinstance(value, str) and pd.isna(value):
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_coverage_analysis.py::test_build_coverage_tiers_empty_measurements_returns_correct_columns -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/risk_stratification_engine/coverage_analysis.py tests/test_coverage_analysis.py
git commit -m "feat: stub coverage_analysis module with build_coverage_tiers empty-input"
```

---

## Task 2: build_coverage_tiers — tertile assignment

**Files:**
- Modify: `tests/test_coverage_analysis.py`
- Modify: `src/risk_stratification_engine/coverage_analysis.py` (implementation already complete from Task 1)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_coverage_analysis.py`:

```python
def test_build_coverage_tiers_assigns_population_wide_tertile_tiers():
    # 3 athlete-seasons with clearly different measurement days
    rows = []
    # a_low: 1 unique date
    rows.append(("a_low", "2026-01-01", "s1", "fp", "m", 1.0))
    # a_med: 5 unique dates
    for d in range(1, 6):
        rows.append(("a_med", f"2026-01-{d:02d}", "s1", "fp", "m", 1.0))
    # a_high: 10 unique dates
    for d in range(1, 11):
        rows.append(("a_high", f"2026-01-{d:02d}", "s1", "fp", "m", 1.0))
    measurements = pd.DataFrame(
        rows, columns=["athlete_id", "date", "season_id", "source", "metric_name", "metric_value"]
    )
    result = build_coverage_tiers(measurements).set_index("athlete_id")

    assert result.loc["a_low", "coverage_tier"] == "low"
    assert result.loc["a_med", "coverage_tier"] == "medium"
    assert result.loc["a_high", "coverage_tier"] == "high"
    assert result.loc["a_low", "measurement_days"] == 1
    assert result.loc["a_high", "measurement_days"] == 10
    assert result.loc["a_med", "measurement_row_count"] == 5


def test_build_coverage_tiers_measurement_days_counts_unique_dates_not_rows():
    # Two rows on the same date should count as 1 measurement_day
    measurements = pd.DataFrame(
        [
            ("a1", "2026-01-01", "s1", "fp", "jump_height", 40.0),
            ("a1", "2026-01-01", "s1", "gps", "distance", 5000.0),  # same date, different source
        ],
        columns=["athlete_id", "date", "season_id", "source", "metric_name", "metric_value"],
    )
    result = build_coverage_tiers(measurements)
    assert result.loc[0, "measurement_days"] == 1
    assert result.loc[0, "measurement_row_count"] == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_coverage_analysis.py -v -k "tertile or unique_dates"
```

Expected: Both tests FAIL (function may not even exist in full form yet if NotImplementedError is hit — but Task 1 already implemented `build_coverage_tiers` in full, so these should actually pass. If they do, skip to step 4.)

- [ ] **Step 3: Run all coverage_analysis tests to confirm current state**

```
pytest tests/test_coverage_analysis.py -v
```

If tertile and unique_dates tests already PASS (because the Task 1 implementation is complete), proceed directly to the commit step. If not, debug `_assign_tiers` — the most common failure is `pd.qcut` reducing bins due to ties; `duplicates="drop"` with `fillna("low")` handles it but edge cases with all-equal values need the `except ValueError` fallback.

- [ ] **Step 4: Run full test suite to confirm no regressions**

```
python -m pytest --tb=short -q
```

Expected: all prior tests pass plus the new ones.

- [ ] **Step 5: Commit**

```
git add tests/test_coverage_analysis.py
git commit -m "test: verify build_coverage_tiers tertile split and unique-date counting"
```

---

## Task 3: build_coverage_stratified_evaluation

**Files:**
- Modify: `tests/test_coverage_analysis.py`
- Modify: `src/risk_stratification_engine/coverage_analysis.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_coverage_analysis.py`:

```python
from risk_stratification_engine.coverage_analysis import (
    build_coverage_tiers,
    build_coverage_stratified_evaluation,
    build_coverage_flag,
)


_TEST_CHANNEL = {
    "channel_name": "test_channel",
    "horizon_days": 30,
    "threshold_value": 0.50,  # top 50% → threshold = median of risk_30d
    "policy_name": "test",
    "graph_window_size": 4,
    "role": "test",
}


def _stratified_timeline_fixture():
    # Population risk_30d values: 0.10, 0.15, 0.80, 0.90, 0.50, 0.60
    # Sorted: 0.10, 0.15, 0.50, 0.60, 0.80, 0.90 → median (50th pct) ≈ 0.55
    # ev_low (low tier):  risk=[0.10, 0.15], event_within_30d=[0, 1], event_observed=True
    #   → 0.15 < 0.55 when event_within_30d=1 → NOT captured
    # ev_high (high tier): risk=[0.80, 0.90], event_within_30d=[1, 0], event_observed=True
    #   → 0.80 >= 0.55 when event_within_30d=1 → captured
    # no_ev (medium tier): risk=[0.50, 0.60], event_within_30d=[0, 0], event_observed=False
    #   → no event
    return pd.DataFrame(
        [
            {
                "athlete_id": "ev_low", "season_id": "s1", "coverage_tier": "low",
                "measurement_days": 2, "risk_30d": 0.10,
                "event_observed": True, "event_within_30d": 0,
            },
            {
                "athlete_id": "ev_low", "season_id": "s1", "coverage_tier": "low",
                "measurement_days": 2, "risk_30d": 0.15,
                "event_observed": True, "event_within_30d": 1,
            },
            {
                "athlete_id": "ev_high", "season_id": "s1", "coverage_tier": "high",
                "measurement_days": 10, "risk_30d": 0.80,
                "event_observed": True, "event_within_30d": 1,
            },
            {
                "athlete_id": "ev_high", "season_id": "s1", "coverage_tier": "high",
                "measurement_days": 10, "risk_30d": 0.90,
                "event_observed": True, "event_within_30d": 0,
            },
            {
                "athlete_id": "no_ev", "season_id": "s1", "coverage_tier": "medium",
                "measurement_days": 5, "risk_30d": 0.50,
                "event_observed": False, "event_within_30d": 0,
            },
            {
                "athlete_id": "no_ev", "season_id": "s1", "coverage_tier": "medium",
                "measurement_days": 5, "risk_30d": 0.60,
                "event_observed": False, "event_within_30d": 0,
            },
        ]
    )


def test_build_coverage_stratified_evaluation_capture_rates_by_tier():
    result = build_coverage_stratified_evaluation(
        _stratified_timeline_fixture(), _TEST_CHANNEL
    )
    rates = result["tier_capture_rates"]
    assert rates["low"] == 0.0      # 0 captured / 1 observed
    assert rates["high"] == 1.0     # 1 captured / 1 observed
    assert rates["medium"] is None  # 0 observed events


def test_build_coverage_stratified_evaluation_uses_population_wide_threshold():
    # The threshold should be the 50th percentile of ALL risk scores,
    # not per-tier. Verify via population_threshold field.
    result = build_coverage_stratified_evaluation(
        _stratified_timeline_fixture(), _TEST_CHANNEL
    )
    # 50th percentile of [0.10, 0.15, 0.50, 0.60, 0.80, 0.90] ≈ 0.55
    assert 0.50 <= result["population_threshold"] <= 0.60


def test_build_coverage_stratified_evaluation_rows_contain_tier_and_season_entries():
    result = build_coverage_stratified_evaluation(
        _stratified_timeline_fixture(), _TEST_CHANNEL
    )
    rows = result["rows"]
    tier_season_ids = {(r["coverage_tier"], r["season_id"]) for r in rows}
    # Should have "all" entries for each tier
    assert ("low", "all") in tier_season_ids
    assert ("high", "all") in tier_season_ids
    assert ("medium", "all") in tier_season_ids
    # Should have per-season entries for s1
    assert ("low", "s1") in tier_season_ids
    assert ("high", "s1") in tier_season_ids


def test_build_coverage_flag_confounded_when_high_much_greater_than_low():
    channel_results = [
        {"tier_capture_rates": {"low": 0.05, "medium": 0.10, "high": 0.25}},
        {"tier_capture_rates": {"low": 0.03, "medium": 0.08, "high": 0.22}},
    ]
    assert build_coverage_flag(channel_results) == "coverage_confounded"


def test_build_coverage_flag_independent_when_tiers_nearly_equal():
    channel_results = [
        {"tier_capture_rates": {"low": 0.10, "medium": 0.11, "high": 0.12}},
        {"tier_capture_rates": {"low": 0.09, "medium": 0.10, "high": 0.11}},
    ]
    assert build_coverage_flag(channel_results) == "coverage_independent"


def test_build_coverage_flag_mixed_when_difference_is_moderate():
    channel_results = [
        {"tier_capture_rates": {"low": 0.10, "medium": 0.12, "high": 0.18}},
    ]
    assert build_coverage_flag(channel_results) == "mixed"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_coverage_analysis.py -v -k "stratified or coverage_flag"
```

Expected: all six new tests FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement build_coverage_stratified_evaluation and build_coverage_flag**

Replace the `raise NotImplementedError` stubs in `coverage_analysis.py`:

```python
def build_coverage_stratified_evaluation(
    timeline_with_tiers: pd.DataFrame,
    channel: dict,
) -> dict:
    horizon = int(channel["horizon_days"])
    threshold_value = float(channel["threshold_value"])
    risk_col = f"risk_{horizon}d"
    event_col = f"event_within_{horizon}d"

    pop_threshold = (
        float(timeline_with_tiers[risk_col].quantile(1.0 - threshold_value))
        if not timeline_with_tiers.empty
        else 0.0
    )

    rows = []
    for tier in COVERAGE_TIER_LABELS:
        tier_frame = timeline_with_tiers[
            timeline_with_tiers["coverage_tier"] == tier
        ]
        rows.append(
            _stratified_row(
                frame=tier_frame,
                risk_col=risk_col,
                event_col=event_col,
                pop_threshold=pop_threshold,
                channel_name=str(channel["channel_name"]),
                coverage_tier=tier,
                season_id="all",
            )
        )
        for season_id, season_group in tier_frame.groupby("season_id", sort=True):
            rows.append(
                _stratified_row(
                    frame=season_group,
                    risk_col=risk_col,
                    event_col=event_col,
                    pop_threshold=pop_threshold,
                    channel_name=str(channel["channel_name"]),
                    coverage_tier=tier,
                    season_id=str(season_id),
                )
            )

    tier_capture_rates = {
        row["coverage_tier"]: row["capture_rate"]
        for row in rows
        if row["season_id"] == "all"
    }

    return {
        "channel_name": str(channel["channel_name"]),
        "population_threshold": _clean_value(pop_threshold),
        "tier_capture_rates": tier_capture_rates,
        "rows": rows,
    }


def build_coverage_flag(channel_results: list[dict]) -> str:
    diffs = []
    for ch in channel_results:
        rates = ch["tier_capture_rates"]
        high = rates.get("high")
        low = rates.get("low")
        if high is not None and low is not None:
            diffs.append(high - low)
    if not diffs:
        return "mixed"
    mean_diff = sum(diffs) / len(diffs)
    if mean_diff >= 0.15:
        return "coverage_confounded"
    if mean_diff < 0.05:
        return "coverage_independent"
    return "mixed"


def _stratified_row(
    frame: pd.DataFrame,
    risk_col: str,
    event_col: str,
    pop_threshold: float,
    channel_name: str,
    coverage_tier: str,
    season_id: str,
) -> dict:
    athlete_seasons = (
        frame[["athlete_id", "season_id"]].drop_duplicates()
        if not frame.empty
        else pd.DataFrame(columns=["athlete_id", "season_id"])
    )
    athlete_season_count = int(len(athlete_seasons))

    if frame.empty:
        return {
            "channel_name": channel_name,
            "coverage_tier": coverage_tier,
            "season_id": season_id,
            "athlete_season_count": 0,
            "observed_event_count": 0,
            "captured_event_count": 0,
            "capture_rate": None,
            "episodes_per_athlete_season": None,
            "mean_measurement_days": None,
        }

    observed_frame = frame[frame["event_observed"].astype(bool)]
    observed_athlete_seasons = observed_frame[
        ["athlete_id", "season_id"]
    ].drop_duplicates()
    observed_event_count = int(len(observed_athlete_seasons))

    # Captured: observed athlete-season had ≥1 snapshot where
    # event_within_{horizon}d == 1 AND risk >= population threshold.
    captured_count = 0
    if observed_event_count > 0:
        flagged = frame[
            (frame[risk_col] >= pop_threshold)
            & (frame[event_col].fillna(0).astype(int) == 1)
        ]
        flagged_pairs = flagged[["athlete_id", "season_id"]].drop_duplicates()
        captured_count = int(
            observed_athlete_seasons.merge(
                flagged_pairs, on=["athlete_id", "season_id"], how="inner"
            ).shape[0]
        )

    capture_rate = (
        round(float(captured_count) / float(observed_event_count), 6)
        if observed_event_count > 0
        else None
    )

    above_threshold = int((frame[risk_col] >= pop_threshold).sum())
    episodes_per_athlete_season = (
        round(float(above_threshold) / float(athlete_season_count), 6)
        if athlete_season_count > 0
        else None
    )

    mean_measurement_days = (
        round(float(frame["measurement_days"].mean()), 3)
        if "measurement_days" in frame.columns and not frame["measurement_days"].isna().all()
        else None
    )

    return {
        "channel_name": channel_name,
        "coverage_tier": coverage_tier,
        "season_id": season_id,
        "athlete_season_count": athlete_season_count,
        "observed_event_count": observed_event_count,
        "captured_event_count": captured_count,
        "capture_rate": capture_rate,
        "episodes_per_athlete_season": episodes_per_athlete_season,
        "mean_measurement_days": mean_measurement_days,
    }
```

- [ ] **Step 4: Run new tests**

```
pytest tests/test_coverage_analysis.py -v -k "stratified or coverage_flag"
```

Expected: all six new tests PASS.

- [ ] **Step 5: Run full suite**

```
python -m pytest --tb=short -q
```

Expected: all prior tests still pass.

- [ ] **Step 6: Commit**

```
git add src/risk_stratification_engine/coverage_analysis.py tests/test_coverage_analysis.py
git commit -m "feat: implement build_coverage_stratified_evaluation and build_coverage_flag"
```

---

## Task 4: write_coverage_stratified_evaluation_report

**Files:**
- Modify: `tests/test_coverage_analysis.py`
- Modify: `src/risk_stratification_engine/coverage_analysis.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_coverage_analysis.py`:

```python
from pathlib import Path
from risk_stratification_engine.coverage_analysis import (
    build_coverage_tiers,
    build_coverage_stratified_evaluation,
    build_coverage_flag,
    write_coverage_stratified_evaluation_report,
)


def _minimal_report_result():
    return {
        "coverage_flag": "mixed",
        "tier_distribution": {"low": 10, "medium": 10, "high": 10},
        "channel_results": [
            {
                "channel_name": "broad_30d",
                "population_threshold": 0.123,
                "tier_capture_rates": {"low": 0.08, "medium": 0.12, "high": 0.19},
                "rows": [
                    {
                        "channel_name": "broad_30d",
                        "coverage_tier": "low",
                        "season_id": "all",
                        "athlete_season_count": 10,
                        "observed_event_count": 5,
                        "captured_event_count": 0,
                        "capture_rate": 0.08,
                        "episodes_per_athlete_season": 0.5,
                        "mean_measurement_days": 3.0,
                    },
                    {
                        "channel_name": "broad_30d",
                        "coverage_tier": "medium",
                        "season_id": "all",
                        "athlete_season_count": 10,
                        "observed_event_count": 5,
                        "captured_event_count": 1,
                        "capture_rate": 0.12,
                        "episodes_per_athlete_season": 0.8,
                        "mean_measurement_days": 10.0,
                    },
                    {
                        "channel_name": "broad_30d",
                        "coverage_tier": "high",
                        "season_id": "all",
                        "athlete_season_count": 10,
                        "observed_event_count": 5,
                        "captured_event_count": 1,
                        "capture_rate": 0.19,
                        "episodes_per_athlete_season": 1.2,
                        "mean_measurement_days": 30.0,
                    },
                ],
            }
        ],
    }


def test_write_coverage_stratified_evaluation_report_writes_expected_sections(tmp_path):
    path = tmp_path / "report.md"
    write_coverage_stratified_evaluation_report(path, _minimal_report_result())
    text = path.read_text(encoding="utf-8")
    assert "# Coverage-Stratified Evaluation" in text
    assert "Coverage flag: mixed" in text
    assert "Tier Distribution" in text
    assert "broad_30d" in text
    assert "0.123" in text  # population threshold
    assert "Interpretation" in text
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_coverage_analysis.py::test_write_coverage_stratified_evaluation_report_writes_expected_sections -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement write_coverage_stratified_evaluation_report**

Replace the `raise NotImplementedError` stub in `coverage_analysis.py`:

```python
def write_coverage_stratified_evaluation_report(
    path: Path,
    result: dict,
) -> None:
    lines = [
        "# Coverage-Stratified Evaluation",
        "",
        f"Coverage flag: {result['coverage_flag']}",
        "",
        "## Tier Distribution",
        "",
        "| Tier | Athlete-seasons |",
        "|---|---:|",
    ]
    for tier in COVERAGE_TIER_LABELS:
        count = result["tier_distribution"].get(tier, 0)
        lines.append(f"| {tier} | {count} |")

    lines.extend(["", "## Capture Rate by Coverage Tier", ""])

    for ch in result["channel_results"]:
        lines.extend(
            [
                f"### {ch['channel_name']}",
                "",
                f"Population threshold: {_fmt(ch['population_threshold'])}",
                "",
                "| Tier | Capture rate | Burden | Athlete-seasons | Observed events |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in ch["rows"]:
            if row["season_id"] == "all":
                lines.append(
                    f"| {row['coverage_tier']} | "
                    f"{_fmt(row['capture_rate'])} | "
                    f"{_fmt(row['episodes_per_athlete_season'])} | "
                    f"{row['athlete_season_count']} | "
                    f"{row['observed_event_count']} |"
                )
        lines.append("")

    lines.extend(["## Interpretation", "", _interpretation(result["coverage_flag"])])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _interpretation(flag: str) -> str:
    if flag == "coverage_confounded":
        return (
            "High-coverage athletes were captured at substantially higher rates than "
            "low-coverage athletes. Coverage tier appears to be a major driver of "
            "shadow-mode channel performance. The next sprint should test whether "
            "model signal survives after controlling for measurement density."
        )
    if flag == "coverage_independent":
        return (
            "Coverage tier was not a major driver of channel capture rates. "
            "The shadow-mode policy signal appears to hold across coverage levels, "
            "which supports proceeding toward shadow-pilot planning."
        )
    return (
        "The relationship between coverage tier and capture rate is inconsistent "
        "across channels or falls between the confounded and independent thresholds. "
        "Review per-channel tier tables before drawing conclusions."
    )
```

- [ ] **Step 4: Run test**

```
pytest tests/test_coverage_analysis.py -v
```

Expected: all coverage_analysis tests PASS.

- [ ] **Step 5: Run full suite**

```
python -m pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```
git add src/risk_stratification_engine/coverage_analysis.py tests/test_coverage_analysis.py
git commit -m "feat: implement write_coverage_stratified_evaluation_report"
```

---

## Task 5: run_coverage_stratified_evaluation_experiment integration test

**Files:**
- Modify: `tests/test_experiments.py`

This task writes the integration test only. The runner is implemented in Task 6.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_experiments.py`, in the imports block at the top:

```python
from risk_stratification_engine.experiments import (
    ...
    run_coverage_stratified_evaluation_experiment,
)
```

And add this test function:

```python
def test_run_coverage_stratified_evaluation_writes_artifacts(tmp_path):
    measurements_path, injuries_path, detailed_path = _write_policy_fixture_inputs(
        tmp_path
    )

    result = run_coverage_stratified_evaluation_experiment(
        measurements_path=measurements_path,
        injuries_path=injuries_path,
        detailed_injuries_path=detailed_path,
        output_dir=tmp_path,
        experiment_id="coverage_stratified_eval",
        model_variant="l2",
    )

    assert (result / "coverage_tiers.csv").exists()
    assert (result / "coverage_stratified_evaluation.csv").exists()
    assert (result / "coverage_stratified_evaluation.json").exists()
    assert (result / "coverage_stratified_evaluation_report.md").exists()
    assert (result / "config.json").exists()

    tiers = pd.read_csv(result / "coverage_tiers.csv")
    assert set(tiers.columns) >= {
        "athlete_id", "season_id", "measurement_days",
        "measurement_row_count", "source_count", "coverage_tier",
    }
    assert set(tiers["coverage_tier"]).issubset({"low", "medium", "high"})

    eval_csv = pd.read_csv(result / "coverage_stratified_evaluation.csv")
    assert "channel_name" in eval_csv.columns
    assert "coverage_tier" in eval_csv.columns
    assert "capture_rate" in eval_csv.columns

    import json
    payload = json.loads(
        (result / "coverage_stratified_evaluation.json").read_text()
    )
    assert payload["experiment_type"] == "coverage_stratified_evaluation"
    assert "tier_distribution" in payload
    assert "coverage_flag" in payload
    assert payload["coverage_flag"] in {
        "coverage_confounded", "coverage_independent", "mixed"
    }
    assert len(payload["channel_results"]) >= 1

    report = (result / "coverage_stratified_evaluation_report.md").read_text()
    assert "Coverage-Stratified Evaluation" in report
    assert "Coverage flag:" in report
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_experiments.py::test_run_coverage_stratified_evaluation_writes_artifacts -v
```

Expected: `ImportError` — `run_coverage_stratified_evaluation_experiment` does not exist yet.

- [ ] **Step 3: Commit the failing test**

```
git add tests/test_experiments.py
git commit -m "test: add integration test for run_coverage_stratified_evaluation_experiment"
```

---

## Task 6: run_coverage_stratified_evaluation_experiment runner

**Files:**
- Modify: `src/risk_stratification_engine/experiments.py`

- [ ] **Step 1: Add import of coverage_analysis public API at the top of experiments.py**

In `experiments.py`, add after the existing shadow_mode import:

```python
from risk_stratification_engine.coverage_analysis import (
    build_coverage_flag,
    build_coverage_stratified_evaluation,
    build_coverage_tiers,
    write_coverage_stratified_evaluation_report,
)
```

- [ ] **Step 2: Add _coverage_flag private wrapper and run_coverage_stratified_evaluation_experiment**

Add the following functions to `experiments.py`, after `run_season_drift_diagnostic_experiment`:

```python
def run_coverage_stratified_evaluation_experiment(
    measurements_path: str | Path,
    injuries_path: str | Path,
    detailed_injuries_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    model_variant: str = "l2",
) -> Path:
    experiment_dir = _experiment_path(output_dir, experiment_id)
    measurements = load_measurements(measurements_path)
    canonical_injuries = load_injury_events(injuries_path)
    detailed_injuries = pd.read_csv(detailed_injuries_path)

    coverage_tiers = build_coverage_tiers(measurements)

    matrix = build_measurement_matrix(measurements)
    graph_cache: dict[int, pd.DataFrame] = {}
    channel_results = []

    for channel in DEFAULT_SHADOW_MODE_CHANNELS:
        window_size = int(channel["graph_window_size"])
        if window_size not in graph_cache:
            graph_cache[window_size] = build_graph_snapshots(
                matrix, window_size=window_size
            )
        graph_features = graph_cache[window_size]
        if graph_features.empty:
            raise ValueError("no graph snapshots produced")
        policy_injuries = build_policy_injury_events(
            canonical_injuries,
            detailed_injuries,
            policy_name=str(channel["policy_name"]),
        )
        labeled = attach_time_to_event_labels(graph_features, policy_injuries)
        if labeled.empty:
            raise ValueError(
                f"no labeled graph snapshots produced for {channel['policy_name']}"
            )
        model_result = train_discrete_time_risk_model(
            labeled, model_variant=model_variant
        )
        timeline = model_result.timeline

        tier_cols = coverage_tiers[
            ["athlete_id", "season_id", "coverage_tier", "measurement_days"]
        ]
        timeline_with_tiers = timeline.merge(
            tier_cols, on=["athlete_id", "season_id"], how="left"
        )
        timeline_with_tiers["coverage_tier"] = (
            timeline_with_tiers["coverage_tier"].fillna("low")
        )

        channel_result = build_coverage_stratified_evaluation(
            timeline_with_tiers, channel
        )
        channel_results.append(channel_result)

    coverage_flag = build_coverage_flag(channel_results)
    tier_distribution = (
        coverage_tiers["coverage_tier"]
        .value_counts()
        .reindex(["low", "medium", "high"], fill_value=0)
        .astype(int)
        .to_dict()
    ) if not coverage_tiers.empty else {"low": 0, "medium": 0, "high": 0}

    result = {
        "experiment_type": "coverage_stratified_evaluation",
        "tier_distribution": tier_distribution,
        "coverage_flag": coverage_flag,
        "channel_results": channel_results,
    }

    csv_rows = [row for ch in channel_results for row in ch["rows"]]

    write_frame(coverage_tiers, experiment_dir / "coverage_tiers.csv")
    write_frame(
        pd.DataFrame(csv_rows),
        experiment_dir / "coverage_stratified_evaluation.csv",
    )
    _write_json(
        experiment_dir / "config.json",
        {
            "experiment_id": experiment_id,
            "experiment_type": "coverage_stratified_evaluation",
            "measurements_path": str(measurements_path),
            "injuries_path": str(injuries_path),
            "detailed_injuries_path": str(detailed_injuries_path),
            "model_variant": model_variant,
            "channels": list(DEFAULT_SHADOW_MODE_CHANNELS),
        },
    )
    _write_json(
        experiment_dir / "coverage_stratified_evaluation.json", result
    )
    write_coverage_stratified_evaluation_report(
        experiment_dir / "coverage_stratified_evaluation_report.md",
        result,
    )
    return experiment_dir
```

- [ ] **Step 3: Run the integration test**

```
pytest tests/test_experiments.py::test_run_coverage_stratified_evaluation_writes_artifacts -v
```

Expected: PASS.

- [ ] **Step 4: Run full suite**

```
python -m pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```
git add src/risk_stratification_engine/experiments.py
git commit -m "feat: add run_coverage_stratified_evaluation_experiment runner"
```

---

## Task 7: CLI flag and dispatch

**Files:**
- Modify: `src/risk_stratification_engine/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI test**

Add to `tests/test_cli.py`:

```python
def test_cli_runs_coverage_stratified_evaluation_from_live_sources(
    tmp_path,
    monkeypatch,
):
    import risk_stratification_engine.cli as cli
    from risk_stratification_engine.cli import main

    calls = {}

    def fake_prepare_live_source_inputs(data_paths, output_dir):
        from risk_stratification_engine.live_sources import LiveSourcePreparationResult
        return LiveSourcePreparationResult(
            measurements_path=output_dir / "canonical_measurements.csv",
            injuries_path=output_dir / "canonical_injuries.csv",
            detailed_injuries_path=output_dir / "injury_events_detailed.csv",
            audit_path=output_dir / "data_quality_audit.json",
            metadata={"canonical_rows": {"measurements": 1, "injury_events": 1}},
            audit={"coverage": {"athlete_season_count": 1}},
        )

    def fake_run_coverage_stratified_evaluation_experiment(
        measurements_path,
        injuries_path,
        detailed_injuries_path,
        output_dir,
        experiment_id,
        model_variant,
    ):
        calls["coverage"] = {
            "measurements_path": measurements_path,
            "injuries_path": injuries_path,
            "detailed_injuries_path": detailed_injuries_path,
            "output_dir": output_dir,
            "experiment_id": experiment_id,
            "model_variant": model_variant,
        }
        return output_dir / "experiments" / experiment_id

    monkeypatch.setattr(
        cli, "prepare_live_source_inputs", fake_prepare_live_source_inputs
    )
    monkeypatch.setattr(
        cli,
        "run_coverage_stratified_evaluation_experiment",
        fake_run_coverage_stratified_evaluation_experiment,
    )

    exit_code = main(
        [
            "--from-live-sources",
            "--paths-config",
            "config/paths.local.yaml",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "coverage_eval_run",
            "--coverage-stratified-evaluation",
            "--model-variant",
            "l2",
        ]
    )

    assert exit_code == 0
    assert calls["coverage"] == {
        "measurements_path": tmp_path
        / "live_inputs"
        / "coverage_eval_run"
        / "canonical_measurements.csv",
        "injuries_path": tmp_path
        / "live_inputs"
        / "coverage_eval_run"
        / "canonical_injuries.csv",
        "detailed_injuries_path": tmp_path
        / "live_inputs"
        / "coverage_eval_run"
        / "injury_events_detailed.csv",
        "output_dir": tmp_path,
        "experiment_id": "coverage_eval_run",
        "model_variant": "l2",
    }
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_cli.py::test_cli_runs_coverage_stratified_evaluation_from_live_sources -v
```

Expected: FAIL — `run_coverage_stratified_evaluation_experiment` not imported in cli, `--coverage-stratified-evaluation` flag not recognized.

- [ ] **Step 3: Add the import to cli.py**

In `cli.py`, extend the `from risk_stratification_engine.experiments import (...)` block:

```python
from risk_stratification_engine.experiments import (
    run_alert_episode_experiment,
    run_calibration_threshold_experiment,
    run_coverage_stratified_evaluation_experiment,   # add this line
    run_injury_outcome_policy_experiment,
    run_model_robustness_experiment,
    run_outcome_policy_model_comparison_experiment,
    run_policy_decision_sprint_experiment,
    run_research_experiment,
    run_season_drift_diagnostic_experiment,
    run_shadow_mode_stability_experiment,
    run_window_model_robustness_experiment,
    run_window_sensitivity_experiment,
)
```

- [ ] **Step 4: Add the CLI argument to build_parser()**

In `build_parser()`, after the `--season-drift-diagnostic` line:

```python
parser.add_argument("--coverage-stratified-evaluation", action="store_true")
```

- [ ] **Step 5: Add the dispatch block to main()**

In `main()`, after the `if args.season_drift_diagnostic:` block (before the final `else` or the return):

```python
    if args.coverage_stratified_evaluation:
        if detailed_injuries_path is None:
            sibling = injuries_path.parent / "injury_events_detailed.csv"
            if not sibling.exists():
                parser.error(
                    "--coverage-stratified-evaluation requires live-source detailed "
                    "injury events or a sibling injury_events_detailed.csv"
                )
            detailed_injuries_path = sibling
        experiment_dir = run_coverage_stratified_evaluation_experiment(
            measurements_path=measurements_path,
            injuries_path=injuries_path,
            detailed_injuries_path=detailed_injuries_path,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
            model_variant=args.model_variant,
        )
        print(f"Coverage-stratified evaluation written to {experiment_dir}")
        return 0
```

- [ ] **Step 6: Run the CLI test**

```
pytest tests/test_cli.py::test_cli_runs_coverage_stratified_evaluation_from_live_sources -v
```

Expected: PASS.

- [ ] **Step 7: Run full suite**

```
python -m pytest --tb=short -q
```

Expected: all tests pass. Count should be at least 161 (157 existing + 4 new unit tests in test_coverage_analysis.py + the integration test + the CLI test — adjust expectation if counts differ slightly from fixture overlap).

- [ ] **Step 8: Commit**

```
git add src/risk_stratification_engine/cli.py tests/test_cli.py
git commit -m "feat: add --coverage-stratified-evaluation CLI flag and dispatch"
```

---

## Task 8: Verification and live run

- [ ] **Step 1: Confirm full test suite passes clean**

```
python -m pytest --tb=short -q
```

Expected: no failures.

- [ ] **Step 2: Run the live experiment**

```
risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id coverage_stratified_eval_v1 --coverage-stratified-evaluation --model-variant l2
```

Expected: command exits 0, prints path to experiment directory.

- [ ] **Step 3: Verify artifacts exist**

```
ls outputs/experiments/coverage_stratified_eval_v1/
```

Expected: `config.json`, `coverage_tiers.csv`, `coverage_stratified_evaluation.csv`, `coverage_stratified_evaluation.json`, `coverage_stratified_evaluation_report.md`.

- [ ] **Step 4: Check coverage_flag in JSON result**

```
python -c "import json; d=json.load(open('outputs/experiments/coverage_stratified_eval_v1/coverage_stratified_evaluation.json')); print(d['coverage_flag']); print(d['tier_distribution'])"
```

Expected: prints `coverage_confounded`, `coverage_independent`, or `mixed` plus tier counts.

- [ ] **Step 5: Update AGENTS.md**

Add a new `## Latest Completed Step` section at the top of the existing completed steps in `AGENTS.md`, following the same format as prior steps. Record:
- What changed (new module, new runner, new CLI flag)
- Verification (test count, live command)
- Live results (coverage_flag, tier_distribution, per-channel capture rates by tier)
- Interpretation

- [ ] **Step 6: Update README.md** if it lists available experiment modes.

- [ ] **Step 7: Final commit**

```
git add AGENTS.md README.md  # or whichever files changed
git commit -m "feat: add coverage-stratified evaluation sprint artifacts"
git push
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `build_coverage_tiers` — Task 1 + 2
- ✅ Population-wide tertile of `measurement_days` — Task 2
- ✅ `build_coverage_stratified_evaluation(timeline_with_tiers, channel)` — Task 3
- ✅ Population-wide threshold (not tier-local) — Task 3 (explicitly documented in test fixture comments)
- ✅ `write_coverage_stratified_evaluation_report` — Task 4
- ✅ Four artifacts (coverage_tiers.csv, evaluation.csv, evaluation.json, report.md) — Task 6
- ✅ `--coverage-stratified-evaluation` CLI flag — Task 7
- ✅ `coverage_flag` logic (confounded/independent/mixed with 15pp/5pp thresholds) — Task 3
- ✅ Integration test — Task 5 + 6
- ✅ CLI test — Task 7
- ✅ AGENTS.md + live run — Task 8

**Type consistency:**
- `build_coverage_stratified_evaluation(timeline_with_tiers, channel)` — used consistently in Tasks 3 and 6
- `build_coverage_flag(channel_results)` — consistent between coverage_analysis.py (Task 3) and experiments.py import (Task 6)
- `write_coverage_stratified_evaluation_report(path, result)` — consistent between Tasks 4 and 6

**No placeholders:** All steps contain complete code.
