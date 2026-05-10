from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON = 1.0
EXPOSURE_LOAD_FEATURE_SET = "graph_plus_coverage_exposure_load"


def build_exposure_load_source_eligible_policy_package(
    validation_rows: list[dict[str, object]],
    source_eligible_calibration: dict[str, object],
    *,
    burden_cap_episodes_per_athlete_season: float = (
        DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON
    ),
) -> dict[str, object]:
    excluded_test_seasons = [
        str(season)
        for season in source_eligible_calibration.get("excluded_test_seasons", [])
    ]
    threshold_rows = _source_eligible_threshold_rows(
        validation_rows,
        excluded_test_seasons,
    )
    policy_rows = _policy_rows(
        threshold_rows,
        burden_cap_episodes_per_athlete_season,
    )
    calibration_status = _source_eligible_calibration_status(
        source_eligible_calibration
    )
    return {
        "experiment_type": "exposure_load_source_eligible_policy_sprint",
        "overall_recommendation": _overall_recommendation(
            policy_rows,
            calibration_status,
        ),
        "production_readiness": source_eligible_calibration.get(
            "production_readiness",
            "not_ready_for_probability_or_pilot",
        ),
        "calibration_recommendation": source_eligible_calibration.get(
            "overall_recommendation",
            "",
        ),
        "calibration_status": calibration_status,
        "excluded_test_seasons": excluded_test_seasons,
        "burden_cap_episodes_per_athlete_season": (
            burden_cap_episodes_per_athlete_season
        ),
        "policy_rows": policy_rows,
        "threshold_rows": threshold_rows,
        "policy_package": _policy_package(policy_rows),
    }


def write_exposure_load_source_eligible_policy_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Source-Eligible Policy Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        (
            "Burden cap: "
            f"{_fmt(summary['burden_cap_episodes_per_athlete_season'])} "
            "episodes per athlete-season"
        ),
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint packages thresholds only after complete athlete-season "
            "trajectories have been scored and source-ineligible seasons have "
            "been excluded. It is not pilot clearance and does not convert daily "
            "rows into independent injury labels."
        ),
        "",
        "## Shadow-Mode Threshold Package",
        "",
        "| Channel | Policy | Status | Mean capture | Mean burden | Mean threshold | Seasons |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary.get("policy_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['recommended_threshold_policy']} | "
            f"{row['recommended_shadow_mode_status']} | "
            f"{_fmt(row.get('mean_capture_rate'))} | "
            f"{_fmt(row.get('mean_episodes_per_athlete_season'))} | "
            f"{_fmt(row.get('mean_selected_threshold_value'))} | "
            f"{row['evaluated_test_seasons']} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _source_eligible_threshold_rows(
    validation_rows: list[dict[str, object]],
    excluded_test_seasons: list[str],
) -> list[dict[str, object]]:
    rows = []
    excluded = set(excluded_test_seasons)
    for row in validation_rows:
        if str(row.get("row_type")) != "alert_policy":
            continue
        if str(row.get("feature_set")) != EXPOSURE_LOAD_FEATURE_SET:
            continue
        if str(row.get("test_season_id")) in excluded:
            continue
        rows.append(_clean_row(row))
    return rows


def _policy_rows(
    threshold_rows: list[dict[str, object]],
    burden_cap_episodes_per_athlete_season: float,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(threshold_rows)
    if frame.empty:
        return []
    policy_rows = []
    for channel_name, channel_group in frame.groupby("channel_name", sort=True):
        candidates = []
        for policy_name, policy_group in channel_group.groupby(
            "threshold_policy",
            sort=True,
        ):
            candidates.append(
                {
                    "threshold_policy": str(policy_name),
                    "mean_capture_rate": _mean_column(
                        policy_group,
                        "unique_event_capture_rate",
                    ),
                    "mean_episodes_per_athlete_season": _mean_column(
                        policy_group,
                        "episodes_per_athlete_season",
                    ),
                    "mean_selected_threshold_value": _mean_column(
                        policy_group,
                        "selected_threshold_value",
                    ),
                    "season_count": int(policy_group["test_season_id"].nunique()),
                    "evaluated_test_seasons": ",".join(
                        sorted(
                            str(season)
                            for season in policy_group["test_season_id"].dropna().unique()
                        )
                    ),
                }
            )
        recommended = _recommend_threshold_policy(
            candidates,
            burden_cap_episodes_per_athlete_season,
        )
        first = channel_group.sort_values(
            ["threshold_policy", "test_season_id"]
        ).iloc[0]
        policy_rows.append(
            _clean_row(
                {
                    "channel_name": str(channel_name),
                    "policy_name": str(first.get("policy_name")),
                    "role": str(first.get("role")),
                    "graph_window_size": first.get("graph_window_size"),
                    "horizon_days": first.get("horizon_days"),
                    "recommended_threshold_policy": recommended.get(
                        "threshold_policy"
                    ),
                    "recommended_shadow_mode_status": _shadow_mode_status(
                        recommended,
                        burden_cap_episodes_per_athlete_season,
                    ),
                    "mean_capture_rate": recommended.get("mean_capture_rate"),
                    "mean_episodes_per_athlete_season": recommended.get(
                        "mean_episodes_per_athlete_season"
                    ),
                    "mean_selected_threshold_value": recommended.get(
                        "mean_selected_threshold_value"
                    ),
                    "evaluated_test_seasons": recommended.get(
                        "evaluated_test_seasons"
                    ),
                    "season_count": recommended.get("season_count"),
                    "threshold_policy_summaries": candidates,
                }
            )
        )
    return policy_rows


def _recommend_threshold_policy(
    candidates: list[dict[str, object]],
    burden_cap_episodes_per_athlete_season: float,
) -> dict[str, object]:
    eligible = [
        row
        for row in candidates
        if row.get("mean_episodes_per_athlete_season") is not None
        and float(row["mean_episodes_per_athlete_season"])
        <= burden_cap_episodes_per_athlete_season
    ]
    pool = eligible or candidates
    return sorted(
        pool,
        key=lambda row: (
            -1.0
            * (
                float(row["mean_capture_rate"])
                if row.get("mean_capture_rate") is not None
                else -1.0
            ),
            float(row["mean_episodes_per_athlete_season"])
            if row.get("mean_episodes_per_athlete_season") is not None
            else float("inf"),
            str(row["threshold_policy"]),
        ),
    )[0]


def _source_eligible_calibration_status(
    source_eligible_calibration: dict[str, object],
) -> str:
    rows = source_eligible_calibration.get("calibration_rows", [])
    source_eligible = next(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and row.get("calibration_scope") == "source_eligible"
        ),
        {},
    )
    losses = int(
        source_eligible.get("ranking_triage_gain_calibration_loss_count") or 0
    )
    supported = int(source_eligible.get("calibration_supported_count") or 0)
    if losses == 0 and supported > 0:
        return "source_eligible_calibration_supported"
    if losses > 0:
        return "source_eligible_calibration_blocked"
    return "source_eligible_calibration_under_review"


def _shadow_mode_status(
    recommended: dict[str, object],
    burden_cap_episodes_per_athlete_season: float,
) -> str:
    burden = recommended.get("mean_episodes_per_athlete_season")
    capture = recommended.get("mean_capture_rate")
    if burden is None or capture is None:
        return "insufficient_shadow_evidence"
    if float(burden) <= burden_cap_episodes_per_athlete_season:
        return "shadow_research_candidate"
    return "burden_above_shadow_target"


def _overall_recommendation(
    policy_rows: list[dict[str, object]],
    calibration_status: str,
) -> str:
    statuses = {str(row.get("recommended_shadow_mode_status")) for row in policy_rows}
    if calibration_status == "source_eligible_calibration_blocked":
        return "calibration_still_blocks_threshold_package"
    if "shadow_research_candidate" in statuses:
        return "advance_source_eligible_shadow_mode_threshold_research"
    return "continue_source_eligible_threshold_research"


def _policy_package(policy_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "source_eligible_research_shadow_mode",
        "deployment_boundary": (
            "research shadow-mode only; not pilot, dashboard, or autonomous intervention"
        ),
        "recommended_channels": [
            {
                "channel_name": row["channel_name"],
                "policy_name": row["policy_name"],
                "horizon_days": row["horizon_days"],
                "threshold_policy": row["recommended_threshold_policy"],
                "mean_selected_threshold_value": row[
                    "mean_selected_threshold_value"
                ],
                "mean_capture_rate": row["mean_capture_rate"],
                "mean_episodes_per_athlete_season": row[
                    "mean_episodes_per_athlete_season"
                ],
            }
            for row in policy_rows
            if row.get("recommended_shadow_mode_status")
            == "shadow_research_candidate"
        ],
        "next_sprint": (
            "source-eligible prospective shadow-mode monitoring with frozen thresholds"
        ),
    }


def _interpretation(summary: dict[str, object]) -> str:
    if (
        summary["overall_recommendation"]
        == "advance_source_eligible_shadow_mode_threshold_research"
    ):
        return (
            "The source-eligible calibration evidence now supports a frozen "
            "research threshold package for shadow-mode monitoring. This advances "
            "model readiness, but product readiness remains blocked until the "
            "package survives prospective shadow-mode review."
        )
    return (
        "Use this package to inspect whether source-eligible threshold choices "
        "can meet burden targets before any pilot, dashboard, or intervention work."
    )


def _mean_column(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


def clean_source_eligible_policy_rows(
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


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "n/a"
    return f"{number:.3f}"
