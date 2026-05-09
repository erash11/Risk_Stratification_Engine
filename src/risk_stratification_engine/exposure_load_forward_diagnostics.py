from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


SOURCE_FEATURE_SET = "graph_plus_coverage_source"
EXPOSURE_LOAD_FEATURE_SET = "graph_plus_coverage_exposure_load"


def build_exposure_load_forward_diagnostic_summary(
    validation_rows: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    calibration_diagnostics = build_exposure_load_calibration_diagnostics(
        validation_rows
    )
    return {
        "experiment_type": "exposure_load_forward_diagnostic_sprint",
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
        "target_reason_summary": _counter(cases, "target_reason"),
        "targeted_test_seasons": sorted(
            {str(case["test_season_id"]) for case in cases if case.get("test_season_id")}
        ),
        "calibration_diagnostics": calibration_diagnostics,
        "cases": [_clean_row(case) for case in cases],
    }


def build_exposure_load_calibration_diagnostics(
    validation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(validation_rows)
    if frame.empty or "row_type" not in frame:
        return []
    model_rows = frame[
        frame["row_type"].eq("model_metric")
        & frame["feature_set"].isin([SOURCE_FEATURE_SET, EXPOSURE_LOAD_FEATURE_SET])
    ].copy()
    if model_rows.empty:
        return []

    diagnostics: list[dict[str, object]] = []
    for (test_season_id, horizon_days), group in model_rows.groupby(
        ["test_season_id", "horizon_days"],
        sort=True,
    ):
        by_feature = {str(row["feature_set"]): row for row in group.to_dict("records")}
        if (
            SOURCE_FEATURE_SET not in by_feature
            or EXPOSURE_LOAD_FEATURE_SET not in by_feature
        ):
            continue
        source = by_feature[SOURCE_FEATURE_SET]
        exposure_load = by_feature[EXPOSURE_LOAD_FEATURE_SET]
        positive_rate = _first_number(
            exposure_load.get("test_positive_rate"),
            source.get("test_positive_rate"),
        )
        source_mean_predicted = _clean_number(source.get("mean_predicted_risk"))
        exposure_load_mean_predicted = _clean_number(
            exposure_load.get("mean_predicted_risk")
        )
        source_gap = _prediction_gap(source_mean_predicted, positive_rate)
        exposure_load_gap = _prediction_gap(exposure_load_mean_predicted, positive_rate)
        row = {
            "test_season_id": str(test_season_id),
            "horizon_days": int(horizon_days),
            "source_roc_auc": _clean_number(source.get("roc_auc")),
            "exposure_load_roc_auc": _clean_number(exposure_load.get("roc_auc")),
            "delta_roc_auc": _delta(
                exposure_load.get("roc_auc"),
                source.get("roc_auc"),
            ),
            "source_brier_skill_score": _clean_number(
                source.get("brier_skill_score")
            ),
            "exposure_load_brier_skill_score": _clean_number(
                exposure_load.get("brier_skill_score")
            ),
            "delta_brier_skill_score": _delta(
                exposure_load.get("brier_skill_score"),
                source.get("brier_skill_score"),
            ),
            "source_model_brier_score": _clean_number(
                source.get("model_brier_score")
            ),
            "exposure_load_model_brier_score": _clean_number(
                exposure_load.get("model_brier_score")
            ),
            "delta_model_brier_score": _delta(
                exposure_load.get("model_brier_score"),
                source.get("model_brier_score"),
            ),
            "source_top_decile_lift": _clean_number(source.get("top_decile_lift")),
            "exposure_load_top_decile_lift": _clean_number(
                exposure_load.get("top_decile_lift")
            ),
            "delta_top_decile_lift": _delta(
                exposure_load.get("top_decile_lift"),
                source.get("top_decile_lift"),
            ),
            "test_positive_rate": positive_rate,
            "source_mean_predicted_risk": source_mean_predicted,
            "exposure_load_mean_predicted_risk": exposure_load_mean_predicted,
            "source_prediction_to_observed_gap": source_gap,
            "exposure_load_prediction_to_observed_gap": exposure_load_gap,
            "delta_prediction_to_observed_gap": _delta(exposure_load_gap, source_gap),
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


def build_exposure_load_forward_diagnostic_cases(
    calibration_diagnostics: list[dict[str, object]],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for row in calibration_diagnostics:
        label = str(row["diagnostic_label"])
        if label == "mixed_or_no_exposure_load_gain":
            continue
        cases.append(
            {
                "case_type": "season_horizon_calibration_slice",
                "diagnostic_label": label,
                "feature_set": EXPOSURE_LOAD_FEATURE_SET,
                "test_season_id": row["test_season_id"],
                "horizon_days": row["horizon_days"],
                "priority_tier": row["priority_tier"],
                "target_reason": _case_target_reason(row),
                "delta_roc_auc": row["delta_roc_auc"],
                "delta_brier_skill_score": row["delta_brier_skill_score"],
                "delta_model_brier_score": row["delta_model_brier_score"],
                "delta_top_decile_lift": row["delta_top_decile_lift"],
                "test_positive_rate": row["test_positive_rate"],
                "source_mean_predicted_risk": row["source_mean_predicted_risk"],
                "exposure_load_mean_predicted_risk": row[
                    "exposure_load_mean_predicted_risk"
                ],
                "delta_prediction_to_observed_gap": row[
                    "delta_prediction_to_observed_gap"
                ],
            }
        )
    return cases


def write_exposure_load_forward_diagnostic_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Forward Diagnostic Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint diagnoses exposure-load behavior after complete "
            "athlete-season trajectories have been scored in a season-forward "
            "setup. It uses only exposure context available before each graph "
            "snapshot and does not convert daily rows into independent injury "
            "classification examples."
        ),
        "",
        "## Calibration Diagnostics",
        "",
        "| Test season | Horizon | Diagnostic | AUROC delta | Brier skill delta | Top-decile lift delta | Predicted-observed gap delta |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("calibration_diagnostics", []):
        lines.append(
            "| "
            f"{row['test_season_id']} | "
            f"{row['horizon_days']}d | "
            f"{row['diagnostic_label']} | "
            f"{_fmt(row.get('delta_roc_auc'))} | "
            f"{_fmt(row.get('delta_brier_skill_score'))} | "
            f"{_fmt(row.get('delta_top_decile_lift'))} | "
            f"{_fmt(row.get('delta_prediction_to_observed_gap'))} |"
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
    return "mixed_or_no_exposure_load_gain"


def _case_target_reason(row: dict[str, object]) -> str:
    if row["diagnostic_label"] == "ranking_triage_gain_calibration_loss":
        if _positive(row.get("delta_prediction_to_observed_gap")):
            return "over_sharpened_probability_slice"
        return "high_lift_calibration_failure"
    if row["diagnostic_label"] == "calibration_supported":
        return "forward_calibration_comparator"
    return "exposure_load_signal_comparator"


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
    reasons = {str(case.get("target_reason")) for case in cases}
    if "ranking_triage_gain_calibration_loss" in labels or reasons.intersection(
        {"over_sharpened_probability_slice", "high_lift_calibration_failure"}
    ):
        return "inspect_exposure_load_forward_failure_modes"
    if labels.intersection({"calibration_supported", "ranking_triage_supported"}):
        return "continue_exposure_load_forward_research"
    return "keep_exposure_load_under_review"


def _interpretation(summary: dict[str, object]) -> str:
    target_reasons = summary.get("target_reason_summary", {})
    if target_reasons.get("over_sharpened_probability_slice"):
        return (
            "Exposure-load context is improving ranking or triage while creating "
            "probability over-sharpening in at least one forward slice. Start by "
            "reviewing the high-priority season-horizon rows before adding "
            "minute-load terms or treating this as pilot-ready calibration."
        )
    return (
        "Use these rows to separate durable exposure-load signal from coverage, "
        "availability, mechanism, and managed-risk artifacts before expanding "
        "the feature set or moving toward dashboard work."
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


def _prediction_gap(mean_predicted: object, positive_rate: object) -> float | None:
    return _delta(mean_predicted, positive_rate)


def _first_number(*values: object) -> float | int | None:
    for value in values:
        number = _clean_number(value)
        if number is not None:
            return number
    return None


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
