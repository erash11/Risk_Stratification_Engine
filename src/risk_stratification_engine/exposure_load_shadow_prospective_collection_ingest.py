from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
LOAD_MODIFICATION_BLOCKED = "blocked"
COMPLETE_STATUS = "complete_practitioner_adjudication"
FORBIDDEN_IDENTIFIER_FIELDS = {
    "athlete_id",
    "athlete_name",
    "external_athlete_id",
    "first_name",
    "full_name",
    "last_name",
    "source_athlete_id",
}


def build_exposure_load_shadow_prospective_collection_ingest(
    operations: dict[str, object],
    completed_rows: list[dict[str, object]],
) -> dict[str, object]:
    payload = _clean_row(operations)
    base_worksheet_rows = [
        _clean_row(row)
        for row in payload.get("collection_worksheet_rows", [])
        if isinstance(row, dict)
    ]
    channel_rows = [
        _clean_row(row)
        for row in payload.get("channel_operation_rows", [])
        if isinstance(row, dict)
    ]
    completed_input_rows = [
        _clean_row(row) for row in completed_rows if isinstance(row, dict)
    ]
    worksheet_by_packet = {
        str(row.get("collection_packet_id") or ""): row
        for row in base_worksheet_rows
        if str(row.get("collection_packet_id") or "")
    }
    updated_by_packet = {
        packet_id: dict(row) for packet_id, row in worksheet_by_packet.items()
    }
    validation_rows: list[dict[str, object]] = []
    seen_complete_packet_ids: set[str] = set()
    ingested_count = 0
    pending_input_rows = 0

    for index, row in enumerate(completed_input_rows, start=1):
        packet_id = str(row.get("collection_packet_id") or "").strip()
        status = str(row.get("collection_status") or "").strip()
        if status != COMPLETE_STATUS:
            pending_input_rows += 1
            validation_rows.append(
                _validation_row(index, packet_id, "pending_input_row", None)
            )
            continue
        if not packet_id or packet_id not in worksheet_by_packet:
            validation_rows.append(
                _validation_row(index, packet_id, "unknown_packet_id", None)
            )
            continue
        if packet_id in seen_complete_packet_ids:
            validation_rows.append(
                _validation_row(index, packet_id, "duplicate_packet_id", None)
            )
            continue
        forbidden_field = _forbidden_identifier_field(row)
        if forbidden_field is not None:
            validation_rows.append(
                _validation_row(
                    index,
                    packet_id,
                    "deidentification_violation",
                    forbidden_field,
                )
            )
            continue
        seen_complete_packet_ids.add(packet_id)
        updated_by_packet[packet_id] = _merged_worksheet_row(
            worksheet_by_packet[packet_id],
            row,
        )
        ingested_count += 1
        validation_rows.append(
            _validation_row(index, packet_id, "ingested_completed_row", None)
        )

    error_count = sum(
        1
        for row in validation_rows
        if row["validation_status"] == "error"
    )
    updated_worksheet_rows = [
        updated_by_packet[str(row.get("collection_packet_id") or "")]
        for row in base_worksheet_rows
    ]
    summary_rows = [
        _summary_row(channel, updated_worksheet_rows, validation_rows)
        for channel in sorted(
            channel_rows,
            key=lambda row: str(row.get("channel_name") or ""),
        )
    ]
    updated_operations = dict(payload)
    updated_operations["collection_worksheet_rows"] = (
        clean_shadow_prospective_collection_ingest_rows(updated_worksheet_rows)
    )
    updated_operations["ingest_source"] = (
        "exposure_load_shadow_prospective_collection_ingest_sprint"
    )
    return {
        "experiment_type": (
            "exposure_load_shadow_prospective_collection_ingest_sprint"
        ),
        "overall_recommendation": _overall_recommendation(
            ingested_count,
            error_count,
        ),
        "milestone_status": "completed_collection_ingest_path_ready",
        "bounded_retest_readiness": _bounded_retest_readiness(ingested_count),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "load_modification_readiness": LOAD_MODIFICATION_BLOCKED,
        "upstream_recommendation": payload.get("overall_recommendation"),
        "known_packet_rows": len(base_worksheet_rows),
        "submitted_input_rows": len(completed_input_rows),
        "pending_input_rows": pending_input_rows,
        "ingested_completed_rows": ingested_count,
        "ingest_error_rows": error_count,
        "ingest_validation_rows": clean_shadow_prospective_collection_ingest_rows(
            validation_rows
        ),
        "ingest_summary_rows": clean_shadow_prospective_collection_ingest_rows(
            summary_rows
        ),
        "updated_collection_worksheet_rows": (
            clean_shadow_prospective_collection_ingest_rows(
                updated_worksheet_rows
            )
        ),
        "updated_operations": updated_operations,
        "interpretation_boundary": (
            "completed worksheet ingest only; not calibration claims, not "
            "probability-facing output, not pilot/dashboard readiness, not "
            "autonomous intervention, and not load modification guidance"
        ),
        "deidentification_boundary": (
            "completed rows must use known collection packet IDs and "
            "de-identified practitioner fields only"
        ),
    }


def write_exposure_load_shadow_prospective_collection_ingest_report(
    path: Path,
    ingest: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Prospective Collection Ingest Sprint",
        "",
        f"Recommendation: {ingest['overall_recommendation']}",
        f"Milestone status: {ingest['milestone_status']}",
        f"Bounded retest readiness: {ingest['bounded_retest_readiness']}",
        f"Production readiness: {ingest['production_readiness']}",
        f"Calibration claim readiness: {ingest['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {ingest['pilot_dashboard_readiness']}",
        f"Load modification readiness: {ingest['load_modification_readiness']}",
        f"Interpretation boundary: {ingest['interpretation_boundary']}",
        f"De-identification boundary: {ingest['deidentification_boundary']}",
        "",
        "## Ingest Results",
        "",
        f"- Known packet rows: {ingest['known_packet_rows']}",
        f"- Submitted input rows: {ingest['submitted_input_rows']}",
        f"- Pending input rows: {ingest['pending_input_rows']}",
        f"- Ingested completed practitioner rows: {ingest['ingested_completed_rows']}",
        f"- Ingest error rows: {ingest['ingest_error_rows']}",
        "",
        "## Channel Summary",
        "",
        "| Channel | Known packets | Ingested completed rows | Error rows | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ingest.get("ingest_summary_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['known_packet_count']} | "
            f"{row['ingested_completed_rows']} | "
            f"{row['ingest_error_rows']} | "
            f"{row['ingest_status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint updates the de-identified prospective collection "
                "worksheet from completed practitioner rows. It only prepares "
                "the next completion-validation run. It is not calibration "
                "claims, probability-facing output, pilot/dashboard clearance, "
                "autonomous intervention, or load modification guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_prospective_collection_ingest_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _validation_row(
    row_number: int,
    packet_id: str,
    error_type: str,
    field_name: str | None,
) -> dict[str, object]:
    is_error = error_type in {
        "unknown_packet_id",
        "duplicate_packet_id",
        "deidentification_violation",
    }
    return {
        "input_row_number": row_number,
        "collection_packet_id": packet_id,
        "error_type": error_type,
        "field_name": field_name or "",
        "validation_status": "error" if is_error else "accepted",
    }


def _merged_worksheet_row(
    base_row: dict[str, object],
    completed_row: dict[str, object],
) -> dict[str, object]:
    merged = dict(base_row)
    for field in base_row:
        if field in completed_row and not _is_blank(completed_row.get(field)):
            merged[field] = completed_row[field]
    merged["collection_status"] = COMPLETE_STATUS
    return merged


def _summary_row(
    channel: dict[str, object],
    updated_worksheet_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
) -> dict[str, object]:
    channel_name = str(channel.get("channel_name") or "")
    packet_ids = {
        str(row.get("collection_packet_id") or "")
        for row in updated_worksheet_rows
        if str(row.get("channel_name") or "") == channel_name
    }
    ingested = sum(
        1
        for row in validation_rows
        if row["collection_packet_id"] in packet_ids
        and row["error_type"] == "ingested_completed_row"
    )
    errors = sum(
        1
        for row in validation_rows
        if row["collection_packet_id"] in packet_ids
        and row["validation_status"] == "error"
    )
    return {
        "channel_name": channel_name,
        "known_packet_count": len(packet_ids),
        "required_packet_count": channel.get("required_packet_count"),
        "ingested_completed_rows": ingested,
        "ingest_error_rows": errors,
        "ingest_status": _channel_ingest_status(ingested, errors),
    }


def _channel_ingest_status(ingested: int, errors: int) -> str:
    if errors:
        return "repair_completed_collection_worksheet_before_validation"
    if ingested:
        return "ready_for_completion_validation_not_claims"
    return "await_completed_practitioner_collection"


def _overall_recommendation(ingested_count: int, error_count: int) -> str:
    if error_count:
        return "repair_completed_collection_worksheet_before_ingest"
    if ingested_count:
        return "rerun_completion_validation_with_ingested_practitioner_rows"
    return "await_completed_practitioner_collection_before_ingest"


def _bounded_retest_readiness(ingested_count: int) -> str:
    if ingested_count:
        return "pending_completion_validation"
    return "pending_completed_practitioner_rows"


def _forbidden_identifier_field(row: dict[str, object]) -> str | None:
    for field in sorted(FORBIDDEN_IDENTIFIER_FIELDS):
        if field in row and not _is_blank(row.get(field)):
            return field
    return None


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


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
