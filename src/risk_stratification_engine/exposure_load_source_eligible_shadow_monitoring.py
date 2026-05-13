from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


EXPOSURE_LOAD_FEATURE_SET = "graph_plus_coverage_exposure_load"


def build_exposure_load_source_eligible_shadow_monitoring_review(
    validation_rows: list[dict[str, object]],
    source_eligible_policy: dict[str, object],
) -> dict[str, object]:
    excluded_test_seasons = [
        str(season) for season in source_eligible_policy.get("excluded_test_seasons", [])
    ]
    burden_cap = float(
        source_eligible_policy.get(
            "burden_cap_episodes_per_athlete_season",
            1.0,
        )
        or 1.0
    )
    recommended_channels = _recommended_channels(source_eligible_policy)
    monitoring_season_rows = _monitoring_season_rows(
        validation_rows=validation_rows,
        recommended_channels=recommended_channels,
        excluded_test_seasons=excluded_test_seasons,
    )
    monitoring_rows = _monitoring_rows(
        monitoring_season_rows,
        recommended_channels,
        burden_cap,
    )
    return {
        "experiment_type": (
            "exposure_load_source_eligible_shadow_monitoring_sprint"
        ),
        "overall_recommendation": _overall_recommendation(monitoring_rows),
        "production_readiness": source_eligible_policy.get(
            "production_readiness",
            "not_ready_for_probability_or_pilot",
        ),
        "source_policy_recommendation": source_eligible_policy.get(
            "overall_recommendation",
            "",
        ),
        "excluded_test_seasons": excluded_test_seasons,
        "burden_cap_episodes_per_athlete_season": burden_cap,
        "monitoring_rows": monitoring_rows,
        "monitoring_season_rows": monitoring_season_rows,
        "monitoring_boundary": (
            "research shadow-mode only; not pilot, dashboard, or autonomous intervention"
        ),
        "next_sprint": (
            "prospective collection of source-eligible shadow monitoring outcomes"
        ),
    }


def write_exposure_load_source_eligible_shadow_monitoring_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Source-Eligible Shadow Monitoring Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Monitoring boundary: {summary['monitoring_boundary']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint reviews the frozen source-eligible threshold package "
            "against complete athlete-season validation rows. It prepares "
            "prospective shadow review and is not pilot or dashboard clearance."
        ),
        "",
        "## Channel Monitoring Review",
        "",
        "| Channel | Policy | Status | Seasons | Mean capture | Max burden | Mean threshold drift |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("monitoring_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['threshold_policy']} | "
            f"{row['monitoring_status']} | "
            f"{row['source_eligible_season_count']} | "
            f"{_fmt(row.get('mean_capture_rate'))} | "
            f"{_fmt(row.get('max_episodes_per_athlete_season'))} | "
            f"{_fmt(row.get('mean_threshold_absolute_drift'))} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_source_eligible_shadow_monitoring_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _recommended_channels(
    source_eligible_policy: dict[str, object],
) -> list[dict[str, object]]:
    package = source_eligible_policy.get("policy_package", {})
    if isinstance(package, dict):
        channels = package.get("recommended_channels", [])
        if isinstance(channels, list) and channels:
            return [_clean_row(row) for row in channels if isinstance(row, dict)]
    return [
        _clean_row(row)
        for row in source_eligible_policy.get("policy_rows", [])
        if isinstance(row, dict)
        and row.get("recommended_shadow_mode_status") == "shadow_research_candidate"
    ]


def _monitoring_season_rows(
    validation_rows: list[dict[str, object]],
    recommended_channels: list[dict[str, object]],
    excluded_test_seasons: list[str],
) -> list[dict[str, object]]:
    excluded = set(excluded_test_seasons)
    rows = []
    for channel in recommended_channels:
        for row in validation_rows:
            if not _matches_channel(row, channel, excluded):
                continue
            rows.append(
                _clean_row(
                    {
                        **row,
                        "frozen_reference_threshold_value": channel.get(
                            "mean_selected_threshold_value"
                        ),
                        "threshold_absolute_drift": _threshold_drift(row, channel),
                    }
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("channel_name")),
            str(row.get("test_season_id")),
            str(row.get("threshold_policy")),
        ),
    )


def _matches_channel(
    row: dict[str, object],
    channel: dict[str, object],
    excluded_test_seasons: set[str],
) -> bool:
    if str(row.get("row_type")) != "alert_policy":
        return False
    if str(row.get("feature_set")) != EXPOSURE_LOAD_FEATURE_SET:
        return False
    if str(row.get("test_season_id")) in excluded_test_seasons:
        return False
    return (
        str(row.get("channel_name")) == str(channel.get("channel_name"))
        and str(row.get("policy_name")) == str(channel.get("policy_name"))
        and int(row.get("horizon_days") or 0) == int(channel.get("horizon_days") or 0)
        and str(row.get("threshold_policy"))
        == str(channel.get("threshold_policy"))
    )


def _monitoring_rows(
    monitoring_season_rows: list[dict[str, object]],
    recommended_channels: list[dict[str, object]],
    burden_cap: float,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(monitoring_season_rows)
    rows = []
    for channel in recommended_channels:
        channel_name = str(channel.get("channel_name"))
        if frame.empty:
            group = pd.DataFrame()
        else:
            group = frame[frame["channel_name"].astype(str) == channel_name]
        rows.append(
            _clean_row(
                {
                    "channel_name": channel_name,
                    "policy_name": channel.get("policy_name"),
                    "horizon_days": channel.get("horizon_days"),
                    "threshold_policy": channel.get("threshold_policy"),
                    "source_eligible_season_count": _nunique(
                        group,
                        "test_season_id",
                    ),
                    "evaluated_test_seasons": _season_list(group),
                    "mean_capture_rate": _mean_column(
                        group,
                        "unique_event_capture_rate",
                    ),
                    "min_capture_rate": _min_column(
                        group,
                        "unique_event_capture_rate",
                    ),
                    "mean_episodes_per_athlete_season": _mean_column(
                        group,
                        "episodes_per_athlete_season",
                    ),
                    "max_episodes_per_athlete_season": _max_column(
                        group,
                        "episodes_per_athlete_season",
                    ),
                    "burden_within_cap_season_count": _burden_within_cap_count(
                        group,
                        burden_cap,
                    ),
                    "mean_selected_threshold_value": _mean_column(
                        group,
                        "selected_threshold_value",
                    ),
                    "frozen_reference_threshold_value": channel.get(
                        "mean_selected_threshold_value"
                    ),
                    "mean_threshold_absolute_drift": _mean_column(
                        group,
                        "threshold_absolute_drift",
                    ),
                    "monitoring_status": _monitoring_status(group, burden_cap),
                }
            )
        )
    return rows


def _monitoring_status(group: pd.DataFrame, burden_cap: float) -> str:
    season_count = _nunique(group, "test_season_id")
    if season_count == 0:
        return "insufficient_source_eligible_monitoring_rows"
    burdens = pd.to_numeric(
        group.get("episodes_per_athlete_season", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if burdens.empty:
        return "missing_shadow_burden_evidence"
    if season_count >= 2 and float(burdens.max()) <= burden_cap:
        return "ready_for_prospective_shadow_review"
    if float(burdens.max()) <= burden_cap:
        return "limited_source_eligible_shadow_review"
    return "shadow_burden_guardrail_review_needed"


def _overall_recommendation(monitoring_rows: list[dict[str, object]]) -> str:
    statuses = {str(row.get("monitoring_status")) for row in monitoring_rows}
    if "ready_for_prospective_shadow_review" in statuses:
        return "proceed_with_prospective_source_eligible_shadow_monitoring"
    if monitoring_rows:
        return "continue_source_eligible_shadow_monitoring_guardrail_review"
    return "continue_source_eligible_shadow_monitoring_setup"


def _interpretation(summary: dict[str, object]) -> str:
    if (
        summary["overall_recommendation"]
        == "proceed_with_prospective_source_eligible_shadow_monitoring"
    ):
        return (
            "At least one frozen source-eligible channel met the retrospective "
            "shadow burden guardrail across source-eligible seasons. The next "
            "step is prospective shadow review with the same research-only "
            "deployment boundary."
        )
    return (
        "The source-eligible shadow package still needs guardrail review before "
        "prospective monitoring, and it remains outside pilot or dashboard use."
    )


def _threshold_drift(
    row: dict[str, object],
    channel: dict[str, object],
) -> float | None:
    reference = channel.get("mean_selected_threshold_value")
    selected = row.get("selected_threshold_value")
    if reference is None or selected is None:
        return None
    try:
        return abs(float(selected) - float(reference))
    except (TypeError, ValueError):
        return None


def _mean_column(frame: pd.DataFrame, column: str) -> float | None:
    values = _numeric_column(frame, column)
    if values.empty:
        return None
    return round(float(values.mean()), 6)


def _min_column(frame: pd.DataFrame, column: str) -> float | None:
    values = _numeric_column(frame, column)
    if values.empty:
        return None
    return round(float(values.min()), 6)


def _max_column(frame: pd.DataFrame, column: str) -> float | None:
    values = _numeric_column(frame, column)
    if values.empty:
        return None
    return round(float(values.max()), 6)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].dropna().nunique())


def _season_list(frame: pd.DataFrame) -> str:
    if frame.empty or "test_season_id" not in frame:
        return ""
    return ",".join(sorted(str(season) for season in frame["test_season_id"].dropna().unique()))


def _burden_within_cap_count(frame: pd.DataFrame, burden_cap: float) -> int:
    if frame.empty or "episodes_per_athlete_season" not in frame:
        return 0
    values = pd.to_numeric(frame["episodes_per_athlete_season"], errors="coerce")
    return int((values <= burden_cap).sum())


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
