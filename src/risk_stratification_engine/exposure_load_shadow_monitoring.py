from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"


def build_exposure_load_shadow_monitoring_plan(
    adjudication_decision: dict[str, object],
) -> dict[str, object]:
    channel_decisions = [
        _clean_row(row)
        for row in adjudication_decision.get("channel_decision_rows", [])
        if isinstance(row, dict)
    ]
    retained_channel_rows = [
        _retained_channel_row(row)
        for row in channel_decisions
        if row.get("channel_decision") == "continue_shadow_monitoring"
    ]
    paused_channel_rows = [
        _paused_channel_row(row)
        for row in channel_decisions
        if row.get("channel_decision") == "pause_or_revise_before_more_collection"
    ]
    retained_channels = [row["channel_name"] for row in retained_channel_rows]
    paused_channels = [row["channel_name"] for row in paused_channel_rows]
    return {
        "experiment_type": "exposure_load_shadow_monitoring_plan_sprint",
        "overall_recommendation": _overall_recommendation(retained_channels),
        "production_readiness": PRODUCTION_BLOCKED,
        "retained_channels": retained_channels,
        "paused_or_revision_channels": paused_channels,
        "retained_channel_rows": retained_channel_rows,
        "paused_channel_rows": paused_channel_rows,
        "evidence_gate_rows": _evidence_gate_rows(retained_channels),
        "plan_boundary": (
            "shadow monitoring operations only; not probability calibration, pilot, dashboard, or autonomous intervention"
        ),
    }


def write_exposure_load_shadow_monitoring_plan_report(
    path: Path,
    plan: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Monitoring Plan Sprint",
        "",
        f"Recommendation: {plan['overall_recommendation']}",
        f"Production readiness: {plan['production_readiness']}",
        f"Plan boundary: {plan['plan_boundary']}",
        "",
        "## Retained Channels",
        "",
        "| Channel | Status | Minimum new packets | Evidence gate |",
        "|---|---|---:|---|",
    ]
    for row in plan.get("retained_channel_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['monitoring_status']} | "
            f"{row['minimum_new_review_packets']} | "
            f"{row['evidence_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Paused Or Revision Channels",
            "",
            "| Channel | Status | Required action | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row in plan.get("paused_channel_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['monitoring_status']} | "
            f"{row['required_action']} | "
            f"{row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Gates",
            "",
            "| Gate | Status | Requirement |",
            "|---|---|---|",
        ]
    )
    for row in plan.get("evidence_gate_rows", []):
        lines.append(
            "| "
            f"{row['gate_name']} | "
            f"{row['gate_status']} | "
            f"{row['requirement']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This plan launches retained-channel shadow monitoring and "
                "keeps paused channels out of further collection until revised. "
                "It is not probability calibration or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_monitoring_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _retained_channel_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "channel_name": row.get("channel_name"),
        "monitoring_status": "continue_shadow_monitoring",
        "collection_unit": "complete source-eligible athlete-season",
        "minimum_new_review_packets": 4,
        "review_cadence": "review after each complete source-eligible season",
        "evidence_gate": "prospective_shadow_review_before_calibration",
        "source_rule": "stop if source eligibility fails or alert burden exceeds policy cap",
        "decision_rationale": row.get("decision_rationale"),
    }


def _paused_channel_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "channel_name": row.get("channel_name"),
        "monitoring_status": "pause_or_revise",
        "required_action": "revise_threshold_or_channel_definition",
        "reason": row.get("decision_rationale"),
    }


def _evidence_gate_rows(retained_channels: list[str]) -> list[dict[str, object]]:
    retained = ", ".join(retained_channels) if retained_channels else "none"
    return [
        {
            "gate_name": "prospective_shadow_review",
            "gate_status": "required",
            "requirement": (
                "collect at least 4 new complete source-eligible review packets "
                f"for retained channels ({retained})"
            ),
        },
        {
            "gate_name": "probability_calibration",
            "gate_status": "blocked",
            "requirement": (
                "requires prospective shadow review evidence with stable useful, "
                "source-trustworthy, actionable packets"
            ),
        },
        {
            "gate_name": "pilot_dashboard_readiness",
            "gate_status": "blocked",
            "requirement": (
                "requires calibration readiness plus an approved operational "
                "review process"
            ),
        },
    ]


def _overall_recommendation(retained_channels: list[str]) -> str:
    if retained_channels:
        return "launch_retained_channel_shadow_monitoring"
    return "revise_shadow_package_before_monitoring"


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
