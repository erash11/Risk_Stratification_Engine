from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from risk_stratification_engine.events import DEFAULT_HORIZONS


DECISION_METRICS = {
    "ranking": ("roc_auc", max),
    "calibration": ("brier_skill_score", max),
    "burden_triage": ("top_decile_lift", max),
}
INJURY_HISTORY_FEATURE_SET = "graph_plus_coverage_injury_history"


def build_injury_history_model_comparison_summary(
    comparison_rows: list[dict[str, object]],
) -> dict[str, object]:
    feature_sets = list(dict.fromkeys(str(row["feature_set"]) for row in comparison_rows))
    return {
        "experiment_type": "injury_history_feature_sprint",
        "overall_recommendation": _overall_recommendation(comparison_rows),
        "feature_sets": feature_sets,
        "comparison_rows": [_clean_row(row) for row in comparison_rows],
        "best_by_horizon": _best_by_horizon(comparison_rows),
    }


def write_injury_history_model_comparison_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Injury History Feature Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint adds time-safe prior injury context to scored "
            "athlete-season graph trajectories. It does not use future injury "
            "details or convert daily rows into independent examples."
        ),
        "",
        "## Injury-History Features",
        "",
    ]
    for column in summary.get("injury_history_feature_columns", []):
        lines.append(f"- {column}")
    lines.extend(
        [
            "",
            "## Holdout Comparison",
            "",
            "| Feature set | Horizon | AUROC | Brier skill | Brier | Top-decile lift |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["comparison_rows"]:
        lines.append(
            "| "
            f"{row['feature_set']} | "
            f"{row['horizon_days']}d | "
            f"{_fmt(row['roc_auc'])} | "
            f"{_fmt(row['brier_skill_score'])} | "
            f"{_fmt(row['model_brier_score'])} | "
            f"{_fmt(row['top_decile_lift'])} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _best_by_horizon(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    best: dict[str, object] = {}
    for horizon in DEFAULT_HORIZONS:
        group = frame[frame["horizon_days"].eq(horizon)]
        if group.empty:
            continue
        horizon_best: dict[str, object] = {}
        for mode_name, (metric_name, selector) in DECISION_METRICS.items():
            candidates = group.dropna(subset=[metric_name])
            if candidates.empty:
                continue
            records = candidates.sort_values(["feature_set"]).to_dict("records")
            winner = selector(records, key=lambda candidate: candidate[metric_name])
            horizon_best[mode_name] = {
                "feature_set": str(winner["feature_set"]),
                metric_name: _clean_value(winner[metric_name]),
            }
        best[str(horizon)] = horizon_best
    return best


def _overall_recommendation(rows: list[dict[str, object]]) -> str:
    winners = _best_by_horizon(rows)
    injury_history_wins = sum(
        1
        for horizon_payload in winners.values()
        for mode_payload in horizon_payload.values()
        if mode_payload.get("feature_set") == INJURY_HISTORY_FEATURE_SET
    )
    if injury_history_wins >= 1:
        return "continue_injury_history_research"
    return "keep_coverage_source_reference_under_review"


def _interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"] == "continue_injury_history_research":
        return (
            "Prior injury context improved at least one decision mode. Treat "
            "this as evidence to continue time-safe injury-history validation, "
            "not as pilot clearance."
        )
    return (
        "Prior injury context did not clearly improve the comparison. Keep the "
        "coverage/source model as the reference while awaiting richer session "
        "and exposure context."
    )


def _clean_row(row: dict[str, object]) -> dict[str, object]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if pd.isna(number):
            return None
        return int(number) if number.is_integer() else number
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"
