from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


DEFAULT_SHADOW_MODE_CHANNELS = (
    {
        "channel_name": "broad_30d",
        "policy_name": "exclude_concussion",
        "graph_window_size": 4,
        "horizon_days": 30,
        "threshold_value": 0.05,
        "role": "broad 30d early warning",
    },
    {
        "channel_name": "severity_7d",
        "policy_name": "model_safe_time_loss",
        "graph_window_size": 4,
        "horizon_days": 7,
        "threshold_value": 0.10,
        "role": "short-horizon severity triage",
    },
    {
        "channel_name": "severity_14d",
        "policy_name": "model_safe_time_loss",
        "graph_window_size": 4,
        "horizon_days": 14,
        "threshold_value": 0.10,
        "role": "short-horizon severity triage",
    },
    {
        "channel_name": "subtype_lower_extremity_soft_tissue_30d",
        "policy_name": "lower_extremity_soft_tissue",
        "graph_window_size": 2,
        "horizon_days": 30,
        "threshold_value": 0.10,
        "role": "subtype review",
    },
)


def build_shadow_mode_stability_audit(
    stability_rows: pd.DataFrame,
) -> dict[str, object]:
    rows = stability_rows.copy()
    channel_summaries = []
    for channel_name, group in rows.groupby("channel_name", sort=True):
        capture = pd.to_numeric(
            group["unique_event_capture_rate"],
            errors="coerce",
        ).dropna()
        burden = pd.to_numeric(
            group["episodes_per_athlete_season"],
            errors="coerce",
        ).dropna()
        captured = pd.to_numeric(
            group["unique_captured_event_count"],
            errors="coerce",
        ).fillna(0)
        summary = {
            "channel_name": str(channel_name),
            "slice_count": int(group["slice_id"].nunique()),
            "mean_capture_rate": _mean(capture),
            "min_capture_rate": _min(capture),
            "max_capture_rate": _max(capture),
            "capture_rate_range": _range(capture),
            "mean_episodes_per_athlete_season": _mean(burden),
            "max_episodes_per_athlete_season": _max(burden),
            "total_captured_event_count": int(captured.sum()),
        }
        summary["stability_status"] = _stability_status(summary)
        channel_summaries.append(summary)

    statuses = {row["stability_status"] for row in channel_summaries}
    recommendation = (
        "candidate_for_shadow_pilot"
        if statuses and statuses <= {"stable"}
        else "review_before_shadow_pilot"
    )
    return {
        "experiment_type": "shadow_mode_policy_stability",
        "slice_type": "season",
        "slice_count": int(rows["slice_id"].nunique()) if not rows.empty else 0,
        "channel_count": len(channel_summaries),
        "overall_recommendation": recommendation,
        "channel_summaries": channel_summaries,
        "stability_rows": _records(rows),
    }


def write_shadow_mode_stability_report(
    path: Path,
    audit: dict[str, object],
) -> None:
    lines = [
        "# Shadow-Mode Policy Stability",
        "",
        f"Recommendation: {audit['overall_recommendation']}",
        f"Slices: {audit['slice_count']}",
        f"Channels: {audit['channel_count']}",
        "",
        "## Channel Stability",
        "",
        "| Channel | Status | Mean capture | Range | Mean burden | Captured |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in audit["channel_summaries"]:
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['stability_status']} | "
            f"{_fmt(row['mean_capture_rate'])} | "
            f"{_fmt(row['capture_rate_range'])} | "
            f"{_fmt(row['mean_episodes_per_athlete_season'])} | "
            f"{row['total_captured_event_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            _interpretation(audit),
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _stability_status(summary: dict[str, object]) -> str:
    if int(summary["slice_count"]) < 2:
        return "insufficient_slices"
    capture_range = summary["capture_rate_range"]
    if capture_range is None:
        return "insufficient_events"
    return "stable" if float(capture_range) <= 0.10 else "unstable"


def _interpretation(audit: dict[str, object]) -> str:
    if audit["overall_recommendation"] == "candidate_for_shadow_pilot":
        return (
            "All evaluated channels were stable across available season slices. "
            "The policy can move toward monitored shadow-mode review."
        )
    return (
        "At least one channel was unstable or had insufficient slice evidence. "
        "Continue research shadow-mode review before dashboard or intervention work."
    )


def _mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.mean()), 6)


def _min(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.min()), 6)


def _max(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.max()), 6)


def _range(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.max() - values.min()), 6)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _clean_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean_value(val) for key, val in value.items()}
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
