from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
REQUIRED_COLLECTION_FIELDS = (
    "collection_season_id",
    "packet_start_date",
    "packet_end_date",
    "source_eligible",
    "episode_count",
    "unique_observed_event_count",
    "unique_captured_event_count",
    "reviewer_id",
    "review_date",
    "alert_usefulness",
)


def build_exposure_load_shadow_collection_template(
    monitoring_plan: dict[str, object],
) -> dict[str, object]:
    retained_rows = [
        _clean_row(row)
        for row in monitoring_plan.get("retained_channel_rows", [])
        if isinstance(row, dict)
    ]
    collection_rows = [
        _collection_template_row(row, packet_sequence)
        for row in retained_rows
        for packet_sequence in range(
            1,
            int(row.get("minimum_new_review_packets", 0) or 0) + 1,
        )
    ]
    completion_rows = [
        _completion_check_row(row) for row in collection_rows
    ]
    return {
        "experiment_type": "exposure_load_shadow_collection_template_sprint",
        "overall_recommendation": _overall_recommendation(collection_rows),
        "production_readiness": PRODUCTION_BLOCKED,
        "retained_channels": list(monitoring_plan.get("retained_channels", [])),
        "paused_or_revision_channels": list(
            monitoring_plan.get("paused_or_revision_channels", [])
        ),
        "schema_rows": _schema_rows(),
        "collection_template_rows": collection_rows,
        "completion_check_rows": completion_rows,
        "collection_boundary": (
            "prospective shadow packet collection only; not probability calibration, pilot, dashboard, or autonomous intervention"
        ),
    }


def write_exposure_load_shadow_collection_template_report(
    path: Path,
    template: dict[str, object],
) -> None:
    pending_rows = [
        row
        for row in template.get("completion_check_rows", [])
        if row.get("completion_status") == "pending_required_fields"
    ]
    lines = [
        "# Exposure Load Shadow Collection Template Sprint",
        "",
        f"Recommendation: {template['overall_recommendation']}",
        f"Production readiness: {template['production_readiness']}",
        f"Collection boundary: {template['collection_boundary']}",
        "",
        "## Collection Template",
        "",
        f"- Retained channels: {', '.join(template['retained_channels'])}",
        f"- Paused/revision channels: {', '.join(template['paused_or_revision_channels'])}",
        f"- Collection rows: {len(template['collection_template_rows'])}",
        f"- Pending required-field rows: {len(pending_rows)}",
        "",
        "## Required Fields",
        "",
        "| Field | Type | Required | Allowed values |",
        "|---|---|---|---|",
    ]
    for row in template.get("schema_rows", []):
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
                "This template operationalizes prospective retained-channel "
                "shadow packet collection. It is not probability calibration "
                "or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_collection_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _collection_template_row(
    retained_channel: dict[str, object],
    packet_sequence: int,
) -> dict[str, object]:
    channel_name = str(retained_channel.get("channel_name"))
    return {
        "collection_packet_id": f"{channel_name}__prospective_{packet_sequence:03d}",
        "channel_name": channel_name,
        "packet_sequence": packet_sequence,
        "collection_unit": retained_channel.get("collection_unit"),
        "evidence_gate": retained_channel.get("evidence_gate"),
        "source_rule": retained_channel.get("source_rule"),
        "collection_season_id": "",
        "packet_start_date": "",
        "packet_end_date": "",
        "source_eligible": "",
        "episode_count": "",
        "unique_observed_event_count": "",
        "unique_captured_event_count": "",
        "alert_usefulness": "",
        "outcome_confirmed": "",
        "source_context_ok": "",
        "action_taken": "",
        "reviewer_id": "",
        "review_date": "",
        "notes": "",
        "collection_status": "pending_collection",
    }


def _completion_check_row(row: dict[str, object]) -> dict[str, object]:
    missing = [
        field
        for field in REQUIRED_COLLECTION_FIELDS
        if str(row.get(field) or "").strip() == ""
    ]
    return {
        "collection_packet_id": row.get("collection_packet_id"),
        "channel_name": row.get("channel_name"),
        "packet_sequence": row.get("packet_sequence"),
        "missing_required_fields": ",".join(missing),
        "missing_required_field_count": len(missing),
        "completion_status": (
            "complete" if not missing else "pending_required_fields"
        ),
    }


def _schema_rows() -> list[dict[str, object]]:
    return [
        _schema_row("collection_packet_id", "string", True, ""),
        _schema_row("channel_name", "string", True, ""),
        _schema_row("packet_sequence", "integer", True, ""),
        _schema_row("collection_season_id", "string", True, ""),
        _schema_row("packet_start_date", "date", True, "YYYY-MM-DD"),
        _schema_row("packet_end_date", "date", True, "YYYY-MM-DD"),
        _schema_row("source_eligible", "boolean", True, "true|false"),
        _schema_row("episode_count", "integer", True, ""),
        _schema_row("unique_observed_event_count", "integer", True, ""),
        _schema_row("unique_captured_event_count", "integer", True, ""),
        _schema_row(
            "alert_usefulness",
            "category",
            True,
            "useful|noisy|misleading|unclear",
        ),
        _schema_row("outcome_confirmed", "boolean", False, "true|false"),
        _schema_row("source_context_ok", "boolean", False, "true|false"),
        _schema_row(
            "action_taken",
            "category",
            False,
            "none|monitor|communicate|modify_load|clinical_review|other",
        ),
        _schema_row("reviewer_id", "string", True, ""),
        _schema_row("review_date", "date", True, "YYYY-MM-DD"),
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


def _overall_recommendation(collection_rows: list[dict[str, object]]) -> str:
    if collection_rows:
        return "collect_retained_channel_shadow_packets"
    return "revise_shadow_package_before_collection"


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
