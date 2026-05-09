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


def build_exposure_load_model_comparison_summary(
    comparison_rows: list[dict[str, object]],
) -> dict[str, object]:
    feature_sets = list(dict.fromkeys(str(row["feature_set"]) for row in comparison_rows))
    return {
        "experiment_type": "exposure_load_feature_sprint",
        "overall_recommendation": _overall_recommendation(comparison_rows),
        "production_readiness": "not_ready_research_validation_required",
        "feature_sets": feature_sets,
        "comparison_rows": [_clean_row(row) for row in comparison_rows],
        "best_by_horizon": _best_by_horizon(comparison_rows),
    }


def write_exposure_load_model_comparison_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Feature Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "Exposure/load context is attached only from participation rows "
            "before each graph snapshot. The athlete-season trajectory remains "
            "the modeling unit; this sprint does not convert the problem into "
            "daily-row injury classification."
        ),
        "",
        "## Exposure Load Features",
        "",
    ]
    for column in summary.get("exposure_load_feature_columns", []):
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
    exposure_wins = sum(
        1
        for horizon_payload in winners.values()
        for mode_payload in horizon_payload.values()
        if mode_payload.get("feature_set") == "graph_plus_coverage_exposure_load"
    )
    if exposure_wins >= 1:
        return "continue_exposure_load_research"
    return "keep_exposure_load_features_under_review"


def _interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"] == "continue_exposure_load_research":
        return (
            "Exposure/load features improved at least one decision mode. "
            "Continue controlled research validation before any pilot or "
            "dashboard escalation."
        )
    return (
        "Exposure/load features did not clearly improve the first comparison. "
        "Keep the features available for diagnostics while reviewing exposure "
        "semantics and duration fields."
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
