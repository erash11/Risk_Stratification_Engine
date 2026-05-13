from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


READY_STATUS = "ready_for_prospective_shadow_review"
PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"


def build_exposure_load_shadow_channel_lock(
    source_eligible_shadow_monitoring: dict[str, object],
) -> dict[str, object]:
    monitoring_rows = [
        _clean_row(row)
        for row in source_eligible_shadow_monitoring.get("monitoring_rows", [])
        if isinstance(row, dict)
    ]
    locked_channels = [
        _lock_row(row)
        for row in monitoring_rows
        if row.get("monitoring_status") == READY_STATUS
    ]
    held_channels = [
        _held_row(row)
        for row in monitoring_rows
        if row.get("monitoring_status") != READY_STATUS
    ]
    return {
        "experiment_type": "exposure_load_shadow_channel_lock_sprint",
        "overall_recommendation": _channel_lock_recommendation(locked_channels),
        "production_readiness": source_eligible_shadow_monitoring.get(
            "production_readiness",
            PRODUCTION_BLOCKED,
        ),
        "monitoring_recommendation": source_eligible_shadow_monitoring.get(
            "overall_recommendation",
            "",
        ),
        "excluded_test_seasons": source_eligible_shadow_monitoring.get(
            "excluded_test_seasons",
            [],
        ),
        "locked_channels": locked_channels,
        "held_channels": held_channels,
        "launch_boundary": (
            "research shadow monitoring only; not pilot, dashboard, probability-facing deployment, or autonomous intervention"
        ),
    }


def build_exposure_load_shadow_review_protocol(
    shadow_channel_lock: dict[str, object],
) -> dict[str, object]:
    locked_channels = [
        _clean_row(row)
        for row in shadow_channel_lock.get("locked_channels", [])
        if isinstance(row, dict)
    ]
    protocol_rows = [_protocol_row(row) for row in locked_channels]
    return {
        "experiment_type": "exposure_load_shadow_review_protocol_sprint",
        "overall_recommendation": _protocol_recommendation(protocol_rows),
        "production_readiness": shadow_channel_lock.get(
            "production_readiness",
            PRODUCTION_BLOCKED,
        ),
        "protocol_rows": protocol_rows,
        "held_channels": shadow_channel_lock.get("held_channels", []),
        "review_boundary": (
            "prospective research shadow review only; not pilot or dashboard clearance"
        ),
    }


def build_exposure_load_shadow_readiness_register(
    shadow_channel_lock: dict[str, object],
    shadow_review_protocol: dict[str, object],
) -> dict[str, object]:
    locked_channels = [
        _clean_row(row)
        for row in shadow_channel_lock.get("locked_channels", [])
        if isinstance(row, dict)
    ]
    protocol_rows = [
        _clean_row(row)
        for row in shadow_review_protocol.get("protocol_rows", [])
        if isinstance(row, dict)
    ]
    protocol_by_channel = {
        str(row.get("channel_name")): row for row in protocol_rows
    }
    readiness_rows = [
        _readiness_row(row, protocol_by_channel.get(str(row.get("channel_name"))))
        for row in locked_channels
    ]
    held_rows = [
        _clean_row(row)
        for row in shadow_channel_lock.get("held_channels", [])
        if isinstance(row, dict)
    ]
    return {
        "experiment_type": "exposure_load_shadow_readiness_register_sprint",
        "overall_recommendation": _readiness_recommendation(readiness_rows),
        "production_readiness": PRODUCTION_BLOCKED,
        "readiness_rows": readiness_rows,
        "held_channels": held_rows,
        "launch_boundary": shadow_channel_lock.get(
            "launch_boundary",
            "research shadow monitoring only",
        ),
        "next_sprint": (
            "collect prospective shadow monitoring outcomes before calibration or product escalation"
        ),
    }


def write_exposure_load_shadow_channel_lock_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Channel Lock Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Launch boundary: {summary['launch_boundary']}",
        "",
        "## Locked Channels",
        "",
        "| Channel | Policy | Horizon | Threshold policy | Mean capture | Max burden |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in summary.get("locked_channels", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['policy_name']} | "
            f"{row['horizon_days']} | "
            f"{row['threshold_policy']} | "
            f"{_fmt(row.get('mean_capture_rate'))} | "
            f"{_fmt(row.get('max_episodes_per_athlete_season'))} |"
        )
    lines.extend(["", "## Held Channels", ""])
    for row in summary.get("held_channels", []):
        lines.append(f"- `{row['channel_name']}`: {row['hold_reason']}")
    lines.extend(["", "## Interpretation", "", _channel_lock_interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_exposure_load_shadow_review_protocol_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Review Protocol Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Review boundary: {summary['review_boundary']}",
        "",
        "## Protocol Rows",
        "",
        "| Channel | Minimum unit | Required evidence | Stop rule |",
        "|---|---|---|---|",
    ]
    for row in summary.get("protocol_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['minimum_review_unit']} | "
            f"{row['required_evidence']} | "
            f"{row['stop_rule']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The protocol can launch prospective research shadow review, "
                "but it is not pilot or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_exposure_load_shadow_readiness_register_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Readiness Register Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Launch boundary: {summary['launch_boundary']}",
        "",
        "## Readiness Register",
        "",
        "| Channel | Status | Launch action | Product boundary |",
        "|---|---|---|---|",
    ]
    for row in summary.get("readiness_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['readiness_status']} | "
            f"{row['launch_action']} | "
            f"{row['product_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The locked channels are ready for research shadow monitoring "
                "only. Outcome collection must precede calibration updates, "
                "pilot escalation, dashboard work, or autonomous intervention."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_launch_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _lock_row(row: dict[str, object]) -> dict[str, object]:
    return _clean_row(
        {
            "channel_name": row.get("channel_name"),
            "policy_name": row.get("policy_name"),
            "horizon_days": row.get("horizon_days"),
            "threshold_policy": row.get("threshold_policy"),
            "source_eligible_season_count": row.get("source_eligible_season_count"),
            "mean_capture_rate": row.get("mean_capture_rate"),
            "max_episodes_per_athlete_season": row.get(
                "max_episodes_per_athlete_season"
            ),
            "mean_threshold_absolute_drift": row.get(
                "mean_threshold_absolute_drift"
            ),
            "lock_status": "locked_for_research_shadow_review",
        }
    )


def _held_row(row: dict[str, object]) -> dict[str, object]:
    return _clean_row(
        {
            "channel_name": row.get("channel_name"),
            "policy_name": row.get("policy_name"),
            "horizon_days": row.get("horizon_days"),
            "threshold_policy": row.get("threshold_policy"),
            "mean_capture_rate": row.get("mean_capture_rate"),
            "max_episodes_per_athlete_season": row.get(
                "max_episodes_per_athlete_season"
            ),
            "hold_reason": row.get("monitoring_status"),
            "lock_status": "held_from_shadow_launch",
        }
    )


def _protocol_row(row: dict[str, object]) -> dict[str, object]:
    return _clean_row(
        {
            "channel_name": row.get("channel_name"),
            "policy_name": row.get("policy_name"),
            "horizon_days": row.get("horizon_days"),
            "threshold_policy": row.get("threshold_policy"),
            "minimum_review_unit": "complete source-eligible athlete-season",
            "required_evidence": (
                "frozen alert episodes, source eligibility, exposure capture status, outcome adjudication, and alert burden"
            ),
            "stop_rule": (
                "pause channel if prospective burden exceeds 1.0 episodes per athlete-season or source eligibility fails"
            ),
            "review_status": "protocol_ready_for_research_shadow_review",
        }
    )


def _readiness_row(
    channel: dict[str, object],
    protocol: dict[str, object] | None,
) -> dict[str, object]:
    protocol_ready = bool(protocol) and protocol.get("review_status") == (
        "protocol_ready_for_research_shadow_review"
    )
    return _clean_row(
        {
            "channel_name": channel.get("channel_name"),
            "policy_name": channel.get("policy_name"),
            "horizon_days": channel.get("horizon_days"),
            "readiness_status": (
                "research_shadow_launch_ready"
                if protocol_ready
                else "research_shadow_protocol_incomplete"
            ),
            "launch_action": (
                "launch_prospective_source_eligible_shadow_monitoring"
                if protocol_ready
                else "complete_review_protocol_before_launch"
            ),
            "product_boundary": (
                "not pilot, dashboard, probability-facing deployment, or autonomous intervention"
            ),
        }
    )


def _channel_lock_recommendation(locked_channels: list[dict[str, object]]) -> str:
    if locked_channels:
        return "lock_source_eligible_burden_capped_channels_for_shadow_review"
    return "continue_shadow_channel_guardrail_review"


def _protocol_recommendation(protocol_rows: list[dict[str, object]]) -> str:
    if protocol_rows:
        return "launch_research_shadow_review_with_locked_channels"
    return "complete_channel_lock_before_protocol_launch"


def _readiness_recommendation(readiness_rows: list[dict[str, object]]) -> str:
    statuses = {str(row.get("readiness_status")) for row in readiness_rows}
    if "research_shadow_launch_ready" in statuses:
        return "launch_research_shadow_monitoring_without_product_escalation"
    return "continue_research_shadow_launch_preparation"


def _channel_lock_interpretation(summary: dict[str, object]) -> str:
    if summary["locked_channels"]:
        return (
            "The burden-capped source-eligible channels are locked for research "
            "shadow review. Held channels remain outside launch scope until their "
            "guardrails are repaired."
        )
    return "No source-eligible channels are ready to lock for shadow review."


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
