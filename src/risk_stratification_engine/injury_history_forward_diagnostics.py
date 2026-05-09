from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


SOURCE_FEATURE_SET = "graph_plus_coverage_source"
INJURY_HISTORY_FEATURE_SET = "graph_plus_coverage_injury_history"


def build_injury_history_forward_diagnostic_summary(
    validation_rows: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    calibration_diagnostics = build_injury_history_calibration_diagnostics(
        validation_rows
    )
    return {
        "experiment_type": "injury_history_forward_diagnostic_sprint",
        "overall_recommendation": _overall_recommendation(
            calibration_diagnostics,
            cases,
        ),
        "calibration_diagnostic_summary": _counter(
            calibration_diagnostics,
            "diagnostic_label",
        ),
        "case_diagnostic_summary": _counter(cases, "diagnostic_label"),
        "case_type_summary": _counter(cases, "case_type"),
        "targeted_test_seasons": sorted(
            {str(case["test_season_id"]) for case in cases if case.get("test_season_id")}
        ),
        "calibration_diagnostics": calibration_diagnostics,
        "cases": [_clean_row(case) for case in cases],
    }


def build_injury_history_calibration_diagnostics(
    validation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(validation_rows)
    if frame.empty or "row_type" not in frame:
        return []
    model_rows = frame[
        frame["row_type"].eq("model_metric")
        & frame["feature_set"].isin([SOURCE_FEATURE_SET, INJURY_HISTORY_FEATURE_SET])
    ].copy()
    if model_rows.empty:
        return []

    diagnostics: list[dict[str, object]] = []
    for (test_season_id, horizon_days), group in model_rows.groupby(
        ["test_season_id", "horizon_days"],
        sort=True,
    ):
        by_feature = {
            str(row["feature_set"]): row
            for row in group.to_dict("records")
        }
        if SOURCE_FEATURE_SET not in by_feature or INJURY_HISTORY_FEATURE_SET not in by_feature:
            continue
        source = by_feature[SOURCE_FEATURE_SET]
        injury_history = by_feature[INJURY_HISTORY_FEATURE_SET]
        row = {
            "test_season_id": str(test_season_id),
            "horizon_days": int(horizon_days),
            "source_roc_auc": _clean_number(source.get("roc_auc")),
            "injury_history_roc_auc": _clean_number(injury_history.get("roc_auc")),
            "delta_roc_auc": _delta(
                injury_history.get("roc_auc"),
                source.get("roc_auc"),
            ),
            "source_brier_skill_score": _clean_number(
                source.get("brier_skill_score")
            ),
            "injury_history_brier_skill_score": _clean_number(
                injury_history.get("brier_skill_score")
            ),
            "delta_brier_skill_score": _delta(
                injury_history.get("brier_skill_score"),
                source.get("brier_skill_score"),
            ),
            "source_model_brier_score": _clean_number(
                source.get("model_brier_score")
            ),
            "injury_history_model_brier_score": _clean_number(
                injury_history.get("model_brier_score")
            ),
            "delta_model_brier_score": _delta(
                injury_history.get("model_brier_score"),
                source.get("model_brier_score"),
            ),
            "source_top_decile_lift": _clean_number(source.get("top_decile_lift")),
            "injury_history_top_decile_lift": _clean_number(
                injury_history.get("top_decile_lift")
            ),
            "delta_top_decile_lift": _delta(
                injury_history.get("top_decile_lift"),
                source.get("top_decile_lift"),
            ),
        }
        row["diagnostic_label"] = _diagnostic_label(row)
        row["priority_tier"] = _priority_tier(str(row["diagnostic_label"]))
        diagnostics.append(row)

    return sorted(
        diagnostics,
        key=lambda row: (
            {"high": 0, "medium": 1, "watch": 2}.get(
                str(row["priority_tier"]),
                3,
            ),
            str(row["test_season_id"]),
            int(row["horizon_days"]),
        ),
    )


def build_injury_history_forward_diagnostic_cases(
    calibration_diagnostics: list[dict[str, object]],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for row in calibration_diagnostics:
        label = str(row["diagnostic_label"])
        if label == "mixed_or_no_injury_history_gain":
            continue
        cases.append(
            {
                "case_type": "season_horizon_calibration_slice",
                "diagnostic_label": label,
                "feature_set": INJURY_HISTORY_FEATURE_SET,
                "test_season_id": row["test_season_id"],
                "horizon_days": row["horizon_days"],
                "priority_tier": row["priority_tier"],
                "target_reason": _case_target_reason(row),
                "delta_roc_auc": row["delta_roc_auc"],
                "delta_brier_skill_score": row["delta_brier_skill_score"],
                "delta_model_brier_score": row["delta_model_brier_score"],
                "delta_top_decile_lift": row["delta_top_decile_lift"],
            }
        )
    return cases


def write_injury_history_forward_diagnostic_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Injury History Forward Diagnostic Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint diagnoses injury-history behavior after complete "
            "athlete-season trajectories have been scored in a season-forward "
            "setup. It does not convert daily rows into independent injury "
            "classification examples."
        ),
        "",
        "## Calibration Diagnostics",
        "",
        "| Test season | Horizon | Diagnostic | AUROC delta | Brier skill delta | Top-decile lift delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary.get("calibration_diagnostics", []):
        lines.append(
            "| "
            f"{row['test_season_id']} | "
            f"{row['horizon_days']}d | "
            f"{row['diagnostic_label']} | "
            f"{_fmt(row.get('delta_roc_auc'))} | "
            f"{_fmt(row.get('delta_brier_skill_score'))} | "
            f"{_fmt(row.get('delta_top_decile_lift'))} |"
        )

    lines.extend(
        [
            "",
            "## Case Diagnostics",
            "",
        ]
    )
    for label, count in summary.get("case_diagnostic_summary", {}).items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _diagnostic_label(row: dict[str, object]) -> str:
    ranking_gain = _positive(row.get("delta_roc_auc"))
    triage_gain = _positive(row.get("delta_top_decile_lift"))
    calibration_loss = _negative(row.get("delta_brier_skill_score")) or _positive(
        row.get("delta_model_brier_score")
    )
    calibration_gain = _positive(row.get("delta_brier_skill_score")) or _negative(
        row.get("delta_model_brier_score")
    )
    if (ranking_gain or triage_gain) and calibration_loss:
        return "ranking_triage_gain_calibration_loss"
    if calibration_gain:
        return "calibration_supported"
    if ranking_gain or triage_gain:
        return "ranking_triage_supported"
    return "mixed_or_no_injury_history_gain"


def _case_target_reason(row: dict[str, object]) -> str:
    if row["diagnostic_label"] == "ranking_triage_gain_calibration_loss":
        return "high_lift_calibration_failure"
    if row["diagnostic_label"] == "calibration_supported":
        return "forward_calibration_comparator"
    return "injury_history_signal_comparator"


def _priority_tier(label: str) -> str:
    if label == "ranking_triage_gain_calibration_loss":
        return "high"
    if label in {"calibration_supported", "ranking_triage_supported"}:
        return "medium"
    return "watch"


def _overall_recommendation(
    calibration_diagnostics: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> str:
    labels = {str(row.get("diagnostic_label")) for row in calibration_diagnostics}
    case_labels = {str(case.get("diagnostic_label")) for case in cases}
    if "ranking_triage_gain_calibration_loss" in labels or case_labels.intersection(
        {"model_miss", "missing_context_or_managed_risk", "explanation_gap"}
    ):
        return "inspect_injury_history_forward_failure_modes"
    if labels.intersection({"calibration_supported", "ranking_triage_supported"}):
        return "continue_injury_history_forward_research"
    return "keep_injury_history_under_review"


def _interpretation(summary: dict[str, object]) -> str:
    diagnostic_summary = summary.get("calibration_diagnostic_summary", {})
    if diagnostic_summary.get("ranking_triage_gain_calibration_loss"):
        return (
            "Prior injury context is helping ranking or triage while hurting "
            "calibration in at least one forward slice. Start with the "
            "2024-2025 high-lift calibration failures and compare them with "
            "2025-2026 calibration comparison rows before any pilot escalation."
        )
    return (
        "Use these rows to separate durable prior-injury signal from coverage, "
        "mechanism, exposure, and managed-risk artifacts before expanding the "
        "model or moving toward dashboard work."
    )


def _counter(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row.get(field)) for row in rows if row.get(field)).items()
        )
    )


def _delta(value: object, reference: object) -> float | None:
    left = _clean_number(value)
    right = _clean_number(reference)
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _positive(value: object) -> bool:
    return value is not None and float(value) > 0.0


def _negative(value: object) -> bool:
    return value is not None and float(value) < 0.0


def _clean_row(row: dict[str, object]) -> dict[str, object]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    number = _clean_number(value)
    if number is not None and not isinstance(value, str):
        return number
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value


def _clean_number(value: object) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if pd.isna(number):
            return None
        return int(number) if number.is_integer() else number
    if not isinstance(value, str) and pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return int(number) if number.is_integer() else number


def _fmt(value: object) -> str:
    number = _clean_number(value)
    if number is None:
        return "n/a"
    return f"{float(number):.3f}"
