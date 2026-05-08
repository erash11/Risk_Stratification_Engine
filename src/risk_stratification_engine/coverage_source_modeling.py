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


def build_coverage_source_model_comparison_summary(
    comparison_rows: list[dict[str, object]],
) -> dict[str, object]:
    feature_sets = list(dict.fromkeys(str(row["feature_set"]) for row in comparison_rows))
    return {
        "experiment_type": "coverage_source_aware_model_sprint",
        "overall_recommendation": _overall_recommendation(comparison_rows),
        "feature_sets": feature_sets,
        "comparison_rows": [_clean_row(row) for row in comparison_rows],
        "best_by_horizon": _best_by_horizon(comparison_rows),
    }


def write_coverage_source_model_comparison_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Coverage/Source-Aware Model Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "Coverage and source context are added as explicit covariates, but "
            "the dynamic graph trajectory features remain the core signal. This "
            "keeps the workbench focused on athlete-season trajectories rather "
            "than independent daily-row classification."
        ),
        "",
        "## Coverage/Source Features",
        "",
    ]
    for column in summary.get("coverage_source_feature_columns", []):
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
            row = candidates.sort_values(["feature_set"]).to_dict("records")
            winner = selector(row, key=lambda candidate: candidate[metric_name])
            horizon_best[mode_name] = {
                "feature_set": str(winner["feature_set"]),
                metric_name: _clean_value(winner[metric_name]),
            }
        best[str(horizon)] = horizon_best
    return best


def _overall_recommendation(rows: list[dict[str, object]]) -> str:
    winners = _best_by_horizon(rows)
    coverage_wins = sum(
        1
        for horizon_payload in winners.values()
        for mode_payload in horizon_payload.values()
        if mode_payload.get("feature_set") == "graph_plus_coverage_source"
    )
    if coverage_wins >= 1:
        return "continue_coverage_source_research"
    return "keep_graph_only_baseline_under_review"


def _interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"] == "continue_coverage_source_research":
        return (
            "Coverage/source covariates improved at least some decision modes. "
            "Treat this as research evidence to continue controlled validation, "
            "not as clearance for dashboard or pilot escalation."
        )
    return (
        "Coverage/source covariates did not clearly improve the comparison. "
        "Keep the graph-only baseline as the primary reference while continuing "
        "coverage-adjusted threshold and season-forward validation work."
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
