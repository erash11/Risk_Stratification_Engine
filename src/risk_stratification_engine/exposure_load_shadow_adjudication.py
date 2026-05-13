from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
REQUIRED_ADJUDICATION_FIELDS = (
    "reviewer_id",
    "review_date",
    "alert_usefulness",
    "outcome_confirmed",
    "source_context_ok",
    "action_taken",
)


def build_exposure_load_shadow_adjudication_package(
    shadow_replay: dict[str, object],
) -> dict[str, object]:
    review_packets = [
        _clean_row(row)
        for row in shadow_replay.get("review_packet_rows", [])
        if isinstance(row, dict)
    ]
    schema_rows = _schema_rows()
    adjudication_template_rows = [
        _template_row(row) for row in review_packets
    ]
    completion_check_rows = [
        _completion_check_row(row) for row in adjudication_template_rows
    ]
    return {
        "experiment_type": "exposure_load_shadow_adjudication_sprint",
        "overall_recommendation": _overall_recommendation(
            adjudication_template_rows
        ),
        "production_readiness": shadow_replay.get(
            "production_readiness",
            PRODUCTION_BLOCKED,
        ),
        "schema_rows": schema_rows,
        "adjudication_template_rows": adjudication_template_rows,
        "completion_check_rows": completion_check_rows,
        "collection_boundary": (
            "research adjudication collection only; not pilot, dashboard, probability-facing deployment, or autonomous intervention"
        ),
        "next_sprint": (
            "collect adjudication values prospectively, then evaluate completeness and usefulness before calibration updates"
        ),
    }


def write_exposure_load_shadow_adjudication_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    template_rows = summary.get("adjudication_template_rows", [])
    completion_rows = summary.get("completion_check_rows", [])
    pending_rows = [
        row
        for row in completion_rows
        if row.get("completion_status") == "pending_required_fields"
    ]
    lines = [
        "# Exposure Load Shadow Adjudication Package Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Collection boundary: {summary['collection_boundary']}",
        "",
        "## Collection Template",
        "",
        f"- Review packet rows: {len(template_rows)}",
        f"- Required fields: {len(REQUIRED_ADJUDICATION_FIELDS)}",
        f"- Pending required-field rows: {len(pending_rows)}",
        "",
        "## Required Fields",
        "",
        "| Field | Type | Required | Allowed values |",
        "|---|---|---|---|",
    ]
    for row in summary.get("schema_rows", []):
        lines.append(
            "| "
            f"{row['field_name']} | "
            f"{row['field_type']} | "
            f"{row['required']} | "
            f"{row['allowed_values']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The review packets now have a prospective collection template "
                "and completion checks. Filling this template is the next "
                "evidence step; this package is not pilot or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_adjudication_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _schema_rows() -> list[dict[str, object]]:
    return [
        _schema_row("review_packet_id", "string", True, ""),
        _schema_row("reviewer_id", "string", True, ""),
        _schema_row("review_date", "date", True, "YYYY-MM-DD"),
        _schema_row(
            "alert_usefulness",
            "category",
            True,
            "useful|noisy|misleading|unclear",
        ),
        _schema_row("outcome_confirmed", "boolean", True, "true|false"),
        _schema_row("source_context_ok", "boolean", True, "true|false"),
        _schema_row(
            "action_taken",
            "category",
            True,
            "none|monitor|communicate|modify_load|clinical_review|other",
        ),
        _schema_row("notes", "string", False, ""),
    ]


def _schema_row(
    field_name: str,
    field_type: str,
    required: bool,
    allowed_values: str,
) -> dict[str, object]:
    return {
        "field_name": field_name,
        "field_type": field_type,
        "required": required,
        "allowed_values": allowed_values,
    }


def _template_row(packet: dict[str, object]) -> dict[str, object]:
    row = {
        "review_packet_id": packet.get("review_packet_id"),
        "channel_name": packet.get("channel_name"),
        "test_season_id": packet.get("test_season_id"),
        "minimum_review_unit": packet.get("minimum_review_unit"),
        "required_evidence": packet.get("required_evidence"),
        "episode_count": packet.get("episode_count"),
        "unique_observed_event_count": packet.get("unique_observed_event_count"),
        "unique_captured_event_count": packet.get("unique_captured_event_count"),
        "missed_event_count": packet.get("missed_event_count"),
        "episodes_per_athlete_season": packet.get("episodes_per_athlete_season"),
        "review_packet_status": packet.get("review_packet_status"),
        "reviewer_id": "",
        "review_date": "",
        "alert_usefulness": "",
        "outcome_confirmed": "",
        "source_context_ok": "",
        "action_taken": "",
        "notes": "",
        "adjudication_status": "pending_review",
    }
    return _clean_row(row)


def _completion_check_row(row: dict[str, object]) -> dict[str, object]:
    missing = [
        field
        for field in REQUIRED_ADJUDICATION_FIELDS
        if str(row.get(field) or "").strip() == ""
    ]
    return {
        "review_packet_id": row.get("review_packet_id"),
        "channel_name": row.get("channel_name"),
        "test_season_id": row.get("test_season_id"),
        "missing_required_fields": ",".join(missing),
        "missing_required_field_count": len(missing),
        "completion_status": (
            "complete"
            if not missing
            else "pending_required_fields"
        ),
    }


def _overall_recommendation(
    adjudication_template_rows: list[dict[str, object]],
) -> str:
    if adjudication_template_rows:
        return "adjudication_template_ready_for_prospective_collection"
    return "complete_replay_packets_before_adjudication_collection"


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
