from __future__ import annotations

from datetime import date
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
ALLOWED_ALERT_USEFULNESS = ("useful", "noisy", "misleading", "unclear")
ALLOWED_ACTIONS = (
    "none",
    "monitor",
    "communicate",
    "modify_load",
    "clinical_review",
    "other",
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


def build_exposure_load_shadow_collection_summary(
    collection_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = [_clean_row(row) for row in collection_rows]
    validation_rows = [_collection_validation_row(row) for row in rows]
    complete_packet_ids = {
        row["collection_packet_id"]
        for row in validation_rows
        if row["completion_status"] == "complete_valid"
    }
    complete_rows = [
        row
        for row in rows
        if row.get("collection_packet_id") in complete_packet_ids
    ]
    channel_summary_rows = _collection_channel_summary_rows(rows, complete_rows)
    pending_or_invalid_rows = len(rows) - len(complete_rows)
    complete_source_eligible_rows = sum(
        int(row["complete_source_eligible_rows"])
        for row in channel_summary_rows
    )
    useful_source_ok_actionable_rows = sum(
        int(row["useful_source_ok_actionable_rows"])
        for row in channel_summary_rows
    )
    return {
        "experiment_type": "exposure_load_shadow_collection_summary",
        "overall_recommendation": _collection_summary_recommendation(
            pending_or_invalid_rows,
            channel_summary_rows,
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_readiness": _collection_calibration_readiness(
            pending_or_invalid_rows,
            channel_summary_rows,
        ),
        "total_rows": len(rows),
        "complete_valid_rows": len(complete_rows),
        "pending_or_invalid_rows": pending_or_invalid_rows,
        "complete_source_eligible_rows": complete_source_eligible_rows,
        "useful_source_ok_actionable_rows": useful_source_ok_actionable_rows,
        "validation_rows": validation_rows,
        "channel_summary_rows": channel_summary_rows,
        "interpretation_boundary": (
            "prospective shadow collection summary only; not probability calibration or dashboard clearance"
        ),
    }


def write_exposure_load_shadow_collection_summary_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Collection Summary Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Calibration readiness: {summary['calibration_readiness']}",
        f"Interpretation boundary: {summary['interpretation_boundary']}",
        "",
        "## Completion",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- Complete valid rows: {summary['complete_valid_rows']}",
        f"- Pending or invalid rows: {summary['pending_or_invalid_rows']}",
        f"- Complete source-eligible rows: {summary['complete_source_eligible_rows']}",
        (
            "- Useful, source-trustworthy, actionable rows: "
            f"{summary['useful_source_ok_actionable_rows']}"
        ),
        "",
        "## Channel Summary",
        "",
        (
            "| Channel | Required packets | Complete rows | Source-eligible rows | "
            "Useful/source OK/actionable | Gate |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("channel_summary_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['minimum_required_packets']} | "
            f"{row['complete_valid_rows']} | "
            f"{row['complete_source_eligible_rows']} | "
            f"{row['useful_source_ok_actionable_rows']} | "
            f"{row['calibration_review_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This summary validates prospective retained-channel shadow "
                "collection rows and identifies whether the evidence gate is "
                "ready for a calibration-readiness review. It is not probability "
                "calibration or dashboard clearance."
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


def _collection_validation_row(row: dict[str, object]) -> dict[str, object]:
    missing = [
        field for field in REQUIRED_COLLECTION_FIELDS if _is_blank(row.get(field))
    ]
    invalid = _collection_invalid_fields(row, missing)
    return {
        "collection_packet_id": row.get("collection_packet_id"),
        "channel_name": row.get("channel_name"),
        "packet_sequence": row.get("packet_sequence"),
        "collection_season_id": row.get("collection_season_id"),
        "missing_required_fields": ",".join(missing),
        "missing_required_field_count": len(missing),
        "invalid_fields": ",".join(invalid),
        "invalid_field_count": len(invalid),
        "completion_status": (
            "complete_valid"
            if not missing and not invalid
            else "pending_or_invalid"
        ),
    }


def _collection_invalid_fields(
    row: dict[str, object],
    missing: list[str],
) -> list[str]:
    missing_fields = set(missing)
    invalid: list[str] = []
    packet_start = _parse_iso_date(row.get("packet_start_date"))
    packet_end = _parse_iso_date(row.get("packet_end_date"))
    if "packet_start_date" not in missing_fields and packet_start is None:
        invalid.append("packet_start_date")
    if "packet_end_date" not in missing_fields and packet_end is None:
        invalid.append("packet_end_date")
    if (
        packet_start is not None
        and packet_end is not None
        and packet_end < packet_start
    ):
        invalid.append("packet_end_date")
    if "review_date" not in missing_fields and _parse_iso_date(
        row.get("review_date")
    ) is None:
        invalid.append("review_date")
    if "source_eligible" not in missing_fields and _parse_bool_text(
        row.get("source_eligible")
    ) is None:
        invalid.append("source_eligible")
    for field in (
        "episode_count",
        "unique_observed_event_count",
        "unique_captured_event_count",
    ):
        if (
            field not in missing_fields
            and _parse_nonnegative_int(row.get(field)) is None
        ):
            invalid.append(field)
    observed = _parse_nonnegative_int(row.get("unique_observed_event_count"))
    captured = _parse_nonnegative_int(row.get("unique_captured_event_count"))
    if observed is not None and captured is not None and captured > observed:
        if "unique_captured_event_count" not in invalid:
            invalid.append("unique_captured_event_count")
    if "alert_usefulness" not in missing_fields and _normalized(
        row.get("alert_usefulness")
    ) not in ALLOWED_ALERT_USEFULNESS:
        invalid.append("alert_usefulness")
    if not _is_blank(row.get("outcome_confirmed")) and _parse_bool_text(
        row.get("outcome_confirmed")
    ) is None:
        invalid.append("outcome_confirmed")
    if not _is_blank(row.get("source_context_ok")) and _parse_bool_text(
        row.get("source_context_ok")
    ) is None:
        invalid.append("source_context_ok")
    if not _is_blank(row.get("action_taken")) and _normalized(
        row.get("action_taken")
    ) not in ALLOWED_ACTIONS:
        invalid.append("action_taken")
    return invalid


def _collection_channel_summary_rows(
    all_rows: list[dict[str, object]],
    complete_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    channel_names = sorted(
        {str(row.get("channel_name") or "unknown") for row in all_rows}
    )
    return [
        _collection_channel_summary_row(
            channel_name,
            [
                row
                for row in all_rows
                if str(row.get("channel_name") or "unknown") == channel_name
            ],
            [
                row
                for row in complete_rows
                if str(row.get("channel_name") or "unknown") == channel_name
            ],
        )
        for channel_name in channel_names
    ]


def _collection_channel_summary_row(
    channel_name: str,
    planned_rows: list[dict[str, object]],
    complete_rows: list[dict[str, object]],
) -> dict[str, object]:
    minimum_required_packets = max(
        [
            _parse_nonnegative_int(row.get("packet_sequence")) or 0
            for row in planned_rows
        ]
        or [0]
    )
    source_eligible_rows = [
        row
        for row in complete_rows
        if _parse_bool_text(row.get("source_eligible")) is True
    ]
    useful_rows = [
        row
        for row in source_eligible_rows
        if _normalized(row.get("alert_usefulness")) == "useful"
    ]
    source_ok_rows = [
        row
        for row in source_eligible_rows
        if _parse_bool_text(row.get("source_context_ok")) is True
    ]
    actionable_rows = [
        row
        for row in source_eligible_rows
        if _normalized(row.get("action_taken")) not in {"", "none"}
    ]
    useful_source_ok_actionable = [
        row
        for row in source_eligible_rows
        if _normalized(row.get("alert_usefulness")) == "useful"
        and _parse_bool_text(row.get("source_context_ok")) is True
        and _normalized(row.get("action_taken")) not in {"", "none"}
    ]
    gate = _collection_calibration_review_gate(
        minimum_required_packets,
        len(source_eligible_rows),
        len(useful_source_ok_actionable),
    )
    return {
        "channel_name": channel_name,
        "minimum_required_packets": minimum_required_packets,
        "planned_rows": len(planned_rows),
        "complete_valid_rows": len(complete_rows),
        "complete_source_eligible_rows": len(source_eligible_rows),
        "useful_rows": len(useful_rows),
        "source_context_ok_rows": len(source_ok_rows),
        "actionable_rows": len(actionable_rows),
        "useful_source_ok_actionable_rows": len(useful_source_ok_actionable),
        "calibration_review_gate": gate,
    }


def _collection_calibration_review_gate(
    minimum_required_packets: int,
    source_eligible_rows: int,
    useful_source_ok_actionable_rows: int,
) -> str:
    if source_eligible_rows < minimum_required_packets:
        return "continue_collection"
    if useful_source_ok_actionable_rows <= 0:
        return "revise_channel_before_calibration_review"
    return "ready_for_calibration_readiness_review"


def _collection_summary_recommendation(
    pending_or_invalid_rows: int,
    channel_summary_rows: list[dict[str, object]],
) -> str:
    if pending_or_invalid_rows:
        return "complete_shadow_collection_before_calibration_readiness_review"
    if not channel_summary_rows:
        return "revise_shadow_collection_plan_before_calibration_review"
    gates = {str(row.get("calibration_review_gate")) for row in channel_summary_rows}
    if gates == {"ready_for_calibration_readiness_review"}:
        return "revisit_calibration_readiness_with_prospective_shadow_evidence"
    if "continue_collection" in gates:
        return "continue_retained_channel_shadow_collection"
    return "revise_retained_channels_before_calibration_readiness_review"


def _collection_calibration_readiness(
    pending_or_invalid_rows: int,
    channel_summary_rows: list[dict[str, object]],
) -> str:
    if (
        not pending_or_invalid_rows
        and channel_summary_rows
        and all(
            row.get("calibration_review_gate")
            == "ready_for_calibration_readiness_review"
            for row in channel_summary_rows
        )
    ):
        return "ready_for_calibration_readiness_review_not_calibration_claim"
    return "not_ready_for_calibration_claims"


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


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _normalized(value: object) -> str:
    if _is_blank(value):
        return ""
    return str(value).strip().lower()


def _parse_bool_text(value: object) -> bool | None:
    normalized = _normalized(value)
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _parse_nonnegative_int(value: object) -> int | None:
    if _is_blank(value):
        return None
    try:
        number = int(float(str(value).strip()))
    except ValueError:
        return None
    if str(value).strip() not in {str(number), f"{number}.0"}:
        return None
    if number < 0:
        return None
    return number


def _parse_iso_date(value: object) -> date | None:
    if _is_blank(value):
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None
