from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
LOAD_MODIFICATION_BLOCKED = "blocked"


def build_exposure_load_shadow_prospective_collection_operations(
    prospective_evidence_gate: dict[str, object],
) -> dict[str, object]:
    gate = _clean_row(prospective_evidence_gate)
    collection_targets = [
        _clean_row(row)
        for row in gate.get("collection_target_rows", [])
        if isinstance(row, dict)
    ]
    packet_targets = [
        _clean_row(row)
        for row in gate.get("packet_target_rows", [])
        if isinstance(row, dict)
    ]
    channel_rows = [
        _channel_operation_row(row, packet_targets)
        for row in sorted(
            collection_targets,
            key=lambda item: str(item.get("channel_name") or ""),
        )
    ]
    packet_manifest_rows = [
        packet
        for channel_row in channel_rows
        for packet in _packet_manifest_rows(channel_row, packet_targets)
    ]
    worksheet_rows = [
        _worksheet_row(packet, _channel_by_name(channel_rows, packet["channel_name"]))
        for packet in packet_manifest_rows
    ]
    checklist_rows = [
        checklist
        for packet in packet_manifest_rows
        for checklist in _packet_checklist_rows(packet)
    ]
    audit_rows = [_audit_trail_row(packet) for packet in packet_manifest_rows]
    packet_documents = [
        {
            "collection_packet_id": packet["collection_packet_id"],
            "packet_filename": packet["packet_filename"],
            "content": _packet_document(packet),
        }
        for packet in packet_manifest_rows
    ]
    return {
        "experiment_type": (
            "exposure_load_shadow_prospective_collection_operations_sprint"
        ),
        "overall_recommendation": _overall_recommendation(packet_manifest_rows),
        "milestone_status": _milestone_status(packet_manifest_rows),
        "retest_readiness": _retest_readiness(packet_manifest_rows),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "load_modification_readiness": LOAD_MODIFICATION_BLOCKED,
        "upstream_recommendation": gate.get("overall_recommendation"),
        "packet_count": len(packet_manifest_rows),
        "channel_operation_rows": clean_shadow_prospective_collection_operation_rows(
            channel_rows
        ),
        "packet_manifest_rows": clean_shadow_prospective_collection_operation_rows(
            packet_manifest_rows
        ),
        "collection_worksheet_rows": clean_shadow_prospective_collection_operation_rows(
            worksheet_rows
        ),
        "packet_checklist_rows": clean_shadow_prospective_collection_operation_rows(
            checklist_rows
        ),
        "audit_trail_rows": clean_shadow_prospective_collection_operation_rows(
            audit_rows
        ),
        "packet_documents": packet_documents,
        "operation_boundary": (
            "prospective reviewer operations only; not probability-facing output, "
            "not calibration claims, not pilot/dashboard readiness, not "
            "autonomous intervention, and not load modification guidance"
        ),
        "deidentification_rule": (
            "use collection packet IDs and de-identified reviewer notes only; "
            "do not enter identifiable athlete information"
        ),
    }


def write_exposure_load_shadow_prospective_collection_reviewer_instructions(
    path: Path,
    operations: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Prospective Collection Reviewer Instructions",
        "",
        f"Recommendation: {operations['overall_recommendation']}",
        f"Retest readiness: {operations['retest_readiness']}",
        f"Production readiness: {operations['production_readiness']}",
        f"Calibration claim readiness: {operations['calibration_claim_readiness']}",
        "",
        "## Boundary",
        "",
        str(operations["operation_boundary"]),
        "",
        "Do not enter identifiable athlete information in packet files, notes, or CSV fields.",
        "Do not make calibration claims, probability-facing statements, pilot/dashboard claims, intervention claims, or load-modification recommendations.",
        "",
        "## Review Steps",
        "",
        "1. Open the packet markdown file listed in the prospective manifest.",
        "2. Confirm source eligibility and the prospective collection window before entering evidence.",
        "3. Record captured and missed event counts using de-identified packet IDs only.",
        "4. Complete practitioner adjudication fields before marking the packet complete.",
        "5. Preserve the no-claim boundary until the required collection target is met and retested.",
        "",
        "## Required Worksheet Fields",
        "",
    ]
    for field in _worksheet_fields():
        lines.append(f"- `{field}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_exposure_load_shadow_prospective_collection_operations_report(
    path: Path,
    operations: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Prospective Collection Operations Sprint",
        "",
        f"Recommendation: {operations['overall_recommendation']}",
        f"Milestone status: {operations['milestone_status']}",
        f"Retest readiness: {operations['retest_readiness']}",
        f"Production readiness: {operations['production_readiness']}",
        f"Calibration claim readiness: {operations['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {operations['pilot_dashboard_readiness']}",
        f"Load modification readiness: {operations['load_modification_readiness']}",
        f"Operation boundary: {operations['operation_boundary']}",
        "",
        "## Operations Package",
        "",
        f"- Reviewer-ready packet count: {operations['packet_count']}",
        f"- Worksheet rows: {len(operations['collection_worksheet_rows'])}",
        f"- Checklist rows: {len(operations['packet_checklist_rows'])}",
        f"- Audit trail seed rows: {len(operations['audit_trail_rows'])}",
        "",
        "## Channel Targets",
        "",
        "| Channel | Required packets | Captured events | Max missed rate | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in operations.get("channel_operation_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['required_packet_count']} | "
            f"{row['required_captured_events']} | "
            f"{row['maximum_allowed_missed_event_rate']} | "
            f"{row['operation_status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint prepares reviewer-ready prospective collection "
                "operations. It is not probability-facing output, calibration "
                "claims, pilot/dashboard clearance, autonomous intervention, or "
                "load modification guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_prospective_collection_operation_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _channel_operation_row(
    target_row: dict[str, object],
    packet_targets: list[dict[str, object]],
) -> dict[str, object]:
    channel_name = str(target_row.get("channel_name") or "")
    matching_packet_targets = _packet_targets_for_channel(packet_targets, channel_name)
    required_packet_count = sum(
        _parse_nonnegative_int(row.get("minimum_packet_count")) or 0
        for row in matching_packet_targets
    )
    if required_packet_count == 0:
        required_packet_count = (
            _parse_nonnegative_int(
                target_row.get("minimum_new_prospective_packets")
            )
            or 0
        )
    return {
        "channel_name": channel_name,
        "required_packet_count": required_packet_count,
        "required_captured_events": (
            _parse_nonnegative_int(target_row.get("minimum_captured_events_needed"))
            or 0
        ),
        "maximum_allowed_missed_event_rate": target_row.get(
            "maximum_allowed_missed_event_rate"
        ),
        "target_decision": target_row.get("target_decision"),
        "operation_status": "ready_for_prospective_collection_not_retest",
        "blocked_use": (
            "probability_output,calibration_claims,pilot_dashboard,"
            "autonomous_intervention,load_modification"
        ),
    }


def _packet_manifest_rows(
    channel_row: dict[str, object],
    packet_targets: list[dict[str, object]],
) -> list[dict[str, object]]:
    channel_name = str(channel_row.get("channel_name") or "")
    rows: list[dict[str, object]] = []
    packet_sequence = 1
    for packet_target in _packet_targets_for_channel(packet_targets, channel_name):
        target_type = str(packet_target.get("target_type") or "prospective_packet")
        count = _parse_nonnegative_int(packet_target.get("minimum_packet_count")) or 0
        for _ in range(count):
            collection_packet_id = (
                f"{channel_name}__prospective_collection_{packet_sequence:03d}"
            )
            rows.append(
                {
                    "collection_packet_id": collection_packet_id,
                    "channel_name": channel_name,
                    "packet_sequence": packet_sequence,
                    "target_type": target_type,
                    "required_action": packet_target.get("required_action"),
                    "packet_filename": f"review_packets/{collection_packet_id}.md",
                    "packet_status": "ready_for_prospective_practitioner_collection",
                    "retest_readiness": "pending_collection",
                }
            )
            packet_sequence += 1
    return rows


def _worksheet_row(
    packet: dict[str, object],
    channel_row: dict[str, object],
) -> dict[str, object]:
    return {
        "collection_packet_id": packet.get("collection_packet_id"),
        "channel_name": packet.get("channel_name"),
        "packet_sequence": packet.get("packet_sequence"),
        "target_type": packet.get("target_type"),
        "required_action": packet.get("required_action"),
        "collection_season_id": "",
        "packet_start_date": "",
        "packet_end_date": "",
        "source_eligible": "",
        "episode_count": "",
        "unique_observed_event_count": "",
        "unique_captured_event_count": "",
        "unique_missed_event_count": "",
        "missed_event_rate": "",
        "alert_usefulness": "",
        "outcome_confirmed": "",
        "source_context_ok": "",
        "action_taken": "",
        "reviewer_id": "",
        "review_date": "",
        "notes": "",
        "target_captured_events_needed": channel_row.get("required_captured_events"),
        "maximum_allowed_missed_event_rate": channel_row.get(
            "maximum_allowed_missed_event_rate"
        ),
        "collection_status": "pending_prospective_collection",
    }


def _packet_checklist_rows(packet: dict[str, object]) -> list[dict[str, object]]:
    items = [
        (
            "confirm_source_eligibility",
            "Confirm packet remains source eligible before evidence review.",
        ),
        (
            "record_prospective_window",
            "Record de-identified season and packet window dates.",
        ),
        (
            "record_captured_and_missed_events",
            "Record observed, captured, and missed event counts.",
        ),
        (
            "complete_practitioner_adjudication",
            "Complete usefulness, outcome, source-context, and action fields.",
        ),
        (
            "preserve_no_claim_boundary",
            "Do not make probability, calibration, pilot, dashboard, intervention, or load-modification claims.",
        ),
    ]
    return [
        {
            "collection_packet_id": packet.get("collection_packet_id"),
            "channel_name": packet.get("channel_name"),
            "packet_sequence": packet.get("packet_sequence"),
            "checklist_item": item,
            "checklist_status": "pending",
            "instruction": instruction,
        }
        for item, instruction in items
    ]


def _audit_trail_row(packet: dict[str, object]) -> dict[str, object]:
    return {
        "collection_packet_id": packet.get("collection_packet_id"),
        "channel_name": packet.get("channel_name"),
        "audit_event": "prospective_packet_created",
        "evidence_status": "not_collected",
        "created_from": "exposure_load_shadow_prospective_evidence_gate",
        "next_required_action": packet.get("required_action"),
    }


def _packet_document(packet: dict[str, object]) -> str:
    lines = [
        f"# Prospective Collection Packet: {packet['collection_packet_id']}",
        "",
        f"Channel: {packet['channel_name']}",
        f"Target type: {packet['target_type']}",
        f"Required action: {packet['required_action']}",
        "",
        "## Boundary",
        "",
        (
            "Use this packet for de-identified prospective practitioner evidence "
            "collection only. Do not use it for probability-facing output, "
            "calibration claims, pilot/dashboard readiness, autonomous "
            "intervention, or load modification guidance."
        ),
        "",
        "## Fields To Complete",
        "",
    ]
    for field in _worksheet_fields():
        lines.append(f"- `{field}`")
    return "\n".join(lines).rstrip() + "\n"


def _packet_targets_for_channel(
    packet_targets: list[dict[str, object]],
    channel_name: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in packet_targets
        if str(row.get("channel_name") or "") == channel_name
    ]


def _channel_by_name(
    channel_rows: list[dict[str, object]],
    channel_name: object,
) -> dict[str, object]:
    for row in channel_rows:
        if str(row.get("channel_name") or "") == str(channel_name or ""):
            return row
    return {}


def _overall_recommendation(packet_rows: list[dict[str, object]]) -> str:
    if not packet_rows:
        return "no_prospective_collection_operations_required"
    return "prepare_prospective_collection_operations_before_retest"


def _milestone_status(packet_rows: list[dict[str, object]]) -> str:
    if not packet_rows:
        return "no_retained_collection_targets"
    return "reviewer_ready_prospective_packet_operations_defined"


def _retest_readiness(packet_rows: list[dict[str, object]]) -> str:
    if not packet_rows:
        return "not_applicable_no_collection_targets"
    return "pending_required_prospective_collection"


def _worksheet_fields() -> list[str]:
    return [
        "collection_packet_id",
        "channel_name",
        "target_type",
        "collection_season_id",
        "packet_start_date",
        "packet_end_date",
        "source_eligible",
        "episode_count",
        "unique_observed_event_count",
        "unique_captured_event_count",
        "unique_missed_event_count",
        "missed_event_rate",
        "alert_usefulness",
        "outcome_confirmed",
        "source_context_ok",
        "action_taken",
        "reviewer_id",
        "review_date",
        "notes",
        "collection_status",
    ]


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


def _parse_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        number = int(float(str(value).strip()))
    except ValueError:
        return None
    if str(value).strip().lower() not in {str(number), f"{number}.0"}:
        return None
    if number < 0:
        return None
    return number
