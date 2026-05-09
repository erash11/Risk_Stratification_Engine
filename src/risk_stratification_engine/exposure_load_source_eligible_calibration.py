from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from risk_stratification_engine.exposure_load_forward_diagnostics import (
    build_exposure_load_calibration_diagnostics,
)


def build_exposure_load_source_eligible_calibration_summary(
    validation_rows: list[dict[str, object]],
    source_resolution_policy: dict[str, object],
) -> dict[str, object]:
    diagnostics = build_exposure_load_calibration_diagnostics(validation_rows)
    excluded_test_seasons = _excluded_test_seasons(source_resolution_policy)
    eligible_diagnostics = [
        row
        for row in diagnostics
        if str(row.get("test_season_id")) not in set(excluded_test_seasons)
    ]
    calibration_rows = [
        _calibration_scope_row(
            calibration_scope="all_seasons",
            diagnostics=diagnostics,
            excluded_test_seasons=[],
        ),
        _calibration_scope_row(
            calibration_scope="source_eligible",
            diagnostics=eligible_diagnostics,
            excluded_test_seasons=excluded_test_seasons,
        ),
    ]
    return {
        "experiment_type": "exposure_load_source_eligible_calibration_sprint",
        "overall_recommendation": _overall_recommendation(calibration_rows),
        "production_readiness": source_resolution_policy.get(
            "production_readiness",
            "not_ready_for_probability_or_pilot",
        ),
        "source_resolution_recommendation": source_resolution_policy.get(
            "overall_recommendation",
            "",
        ),
        "excluded_test_seasons": excluded_test_seasons,
        "calibration_rows": calibration_rows,
        "calibration_diagnostics": [_clean_row(row) for row in diagnostics],
    }


def write_exposure_load_source_eligible_calibration_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Source-Eligible Calibration Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint applies the source-resolution eligibility policy to "
            "complete athlete-season trajectories that were already scored in "
            "season-forward validation. It does not treat daily rows as "
            "independent injury-classification examples, and it is not pilot "
            "clearance."
        ),
        "",
        "## Calibration Scope Comparison",
        "",
        "| Scope | Seasons | Excluded seasons | Calibration-loss rows | Calibration-supported rows | Mean Brier skill delta | Mean prediction gap delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("calibration_rows", []):
        lines.append(
            "| "
            f"{row['calibration_scope']} | "
            f"{row['evaluated_test_seasons']} | "
            f"{row['excluded_test_seasons']} | "
            f"{row['ranking_triage_gain_calibration_loss_count']} | "
            f"{row['calibration_supported_count']} | "
            f"{_fmt(row.get('mean_delta_brier_skill_score'))} | "
            f"{_fmt(row.get('mean_delta_prediction_to_observed_gap'))} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _excluded_test_seasons(source_resolution_policy: dict[str, object]) -> list[str]:
    policy_rows = source_resolution_policy.get("policy_rows", [])
    excludes_failed_season = any(
        isinstance(row, dict)
        and row.get("policy_domain") == "season_eligibility"
        and row.get("policy_decision")
        == "exclude_failed_season_from_probability_calibration"
        for row in policy_rows
    )
    if not excludes_failed_season:
        return []
    return sorted(str(season) for season in source_resolution_policy.get("failure_seasons", []))


def _calibration_scope_row(
    calibration_scope: str,
    diagnostics: list[dict[str, object]],
    excluded_test_seasons: list[str],
) -> dict[str, object]:
    labels = Counter(str(row.get("diagnostic_label")) for row in diagnostics)
    seasons = sorted(
        {str(row.get("test_season_id")) for row in diagnostics if row.get("test_season_id")}
    )
    return _clean_row(
        {
            "calibration_scope": calibration_scope,
            "evaluated_test_seasons": ",".join(seasons),
            "excluded_test_seasons": ",".join(excluded_test_seasons),
            "diagnostic_row_count": len(diagnostics),
            "ranking_triage_gain_calibration_loss_count": labels.get(
                "ranking_triage_gain_calibration_loss",
                0,
            ),
            "calibration_supported_count": labels.get("calibration_supported", 0),
            "ranking_triage_supported_count": labels.get(
                "ranking_triage_supported",
                0,
            ),
            "mixed_or_no_exposure_load_gain_count": labels.get(
                "mixed_or_no_exposure_load_gain",
                0,
            ),
            "mean_delta_roc_auc": _mean_metric(diagnostics, "delta_roc_auc"),
            "mean_delta_brier_skill_score": _mean_metric(
                diagnostics,
                "delta_brier_skill_score",
            ),
            "mean_delta_top_decile_lift": _mean_metric(
                diagnostics,
                "delta_top_decile_lift",
            ),
            "mean_delta_prediction_to_observed_gap": _mean_metric(
                diagnostics,
                "delta_prediction_to_observed_gap",
            ),
        }
    )


def _overall_recommendation(calibration_rows: list[dict[str, object]]) -> str:
    by_scope = {row["calibration_scope"]: row for row in calibration_rows}
    eligible = by_scope.get("source_eligible", {})
    eligible_losses = int(
        eligible.get("ranking_triage_gain_calibration_loss_count") or 0
    )
    eligible_supported = int(eligible.get("calibration_supported_count") or 0)
    eligible_rows = int(eligible.get("diagnostic_row_count") or 0)
    if eligible_rows == 0:
        return "insufficient_source_eligible_calibration_evidence"
    if eligible_losses > 0:
        return "calibration_still_blocked_after_source_eligibility"
    if eligible_supported > 0:
        return "probability_research_can_resume_on_source_eligible_seasons"
    return "continue_source_eligible_calibration_research"


def _interpretation(summary: dict[str, object]) -> str:
    recommendation = summary["overall_recommendation"]
    if recommendation == "probability_research_can_resume_on_source_eligible_seasons":
        return (
            "After applying the source-resolution policy, the source-eligible "
            "season-forward slices no longer show exposure-load ranking/triage "
            "gains paired with calibration loss. This reopens probability research "
            "on source-eligible seasons only; it does not clear pilot or dashboard "
            "use."
        )
    if recommendation == "calibration_still_blocked_after_source_eligibility":
        return (
            "Exposure-load calibration failures remain after excluding source-"
            "ineligible seasons, so the next sprint should target calibration "
            "modeling rather than source eligibility."
        )
    return (
        "Use this sprint to decide whether source eligibility alone explains the "
        "exposure-load calibration failure before adding new feature domains."
    )


def _mean_metric(rows: list[dict[str, object]], column: str) -> float | None:
    values = [_clean_number(row.get(column)) for row in rows]
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def clean_source_eligible_calibration_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _clean_row(row: dict[str, object]) -> dict[str, object]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if pd.isna(number):
            return None
        return int(number) if number.is_integer() else round(number, 6)
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value


def _clean_number(value: object) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if pd.isna(number):
            return None
        return int(number) if number.is_integer() else number
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
