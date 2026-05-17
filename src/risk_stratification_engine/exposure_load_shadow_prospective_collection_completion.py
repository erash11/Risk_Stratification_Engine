from __future__ import annotations

from datetime import date
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
LOAD_MODIFICATION_BLOCKED = "blocked"
REQUIRED_FIELDS = [
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
]
VALID_USEFULNESS = {"useful", "not_useful", "unclear"}
VALID_ACTIONS = {"monitor", "none", "modified_followup", "other"}


def build_exposure_load_shadow_prospective_collection_completion(
    operations: dict[str, object],
) -> dict[str, object]:
    payload = _clean_row(operations)
    channel_targets = [
        _clean_row(row)
        for row in payload.get("channel_operation_rows", [])
        if isinstance(row, dict)
    ]
    worksheet_rows = [
        _clean_row(row)
        for row in payload.get("collection_worksheet_rows", [])
        if isinstance(row, dict)
    ]
    validation_rows = [_packet_validation_row(row) for row in worksheet_rows]
    channel_rows = [
        _channel_completion_row(target, validation_rows)
        for target in sorted(
            channel_targets,
            key=lambda row: str(row.get("channel_name") or ""),
        )
    ]
    gate_rows = [
        gate
        for channel_row in channel_rows
        for gate in _completion_gate_rows(channel_row)
    ]
    all_channels_ready = bool(channel_rows) and all(
        row["completion_gate"] == "ready_for_bounded_retest_not_claims"
        for row in channel_rows
    )
    return {
        "experiment_type": (
            "exposure_load_shadow_prospective_collection_completion_sprint"
        ),
        "overall_recommendation": _overall_recommendation(all_channels_ready),
        "milestone_status": _milestone_status(all_channels_ready),
        "bounded_retest_readiness": _bounded_retest_readiness(all_channels_ready),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "load_modification_readiness": LOAD_MODIFICATION_BLOCKED,
        "upstream_recommendation": payload.get("overall_recommendation"),
        "total_packet_rows": len(validation_rows),
        "complete_practitioner_packet_rows": sum(
            row["completion_status"] == "complete_valid" for row in validation_rows
        ),
        "packet_validation_rows": clean_shadow_prospective_collection_completion_rows(
            validation_rows
        ),
        "channel_completion_rows": clean_shadow_prospective_collection_completion_rows(
            channel_rows
        ),
        "completion_gate_rows": clean_shadow_prospective_collection_completion_rows(
            gate_rows
        ),
        "interpretation_boundary": (
            "completion validation for bounded retest only; not calibration claims, "
            "not probability-facing output, not pilot/dashboard readiness, not "
            "autonomous intervention, and not load modification guidance"
        ),
    }


def write_exposure_load_shadow_prospective_collection_completion_report(
    path: Path,
    completion: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Prospective Collection Completion Sprint",
        "",
        f"Recommendation: {completion['overall_recommendation']}",
        f"Milestone status: {completion['milestone_status']}",
        f"Bounded retest readiness: {completion['bounded_retest_readiness']}",
        f"Production readiness: {completion['production_readiness']}",
        f"Calibration claim readiness: {completion['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {completion['pilot_dashboard_readiness']}",
        f"Load modification readiness: {completion['load_modification_readiness']}",
        f"Interpretation boundary: {completion['interpretation_boundary']}",
        "",
        "## Packet Validation",
        "",
        f"- Packet rows: {completion['total_packet_rows']}",
        f"- Complete practitioner packet rows: {completion['complete_practitioner_packet_rows']}",
        "",
        "## Channel Completion",
        "",
        "| Channel | Required packets | Complete packets | Captured events | Missed rate | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in completion.get("channel_completion_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['required_packet_count']} | "
            f"{row['complete_practitioner_packet_count']} | "
            f"{row['captured_event_count']} | "
            f"{row['missed_event_rate']} | "
            f"{row['completion_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint validates whether prospective packet evidence is "
                "complete enough for a bounded retest. Passing this gate is not "
                "calibration claims, probability-facing output, pilot/dashboard "
                "clearance, autonomous intervention, or load modification guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_prospective_collection_completion_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _packet_validation_row(row: dict[str, object]) -> dict[str, object]:
    missing = [field for field in REQUIRED_FIELDS if _is_blank(row.get(field))]
    invalid = _invalid_fields(row)
    complete = (
        not missing
        and not invalid
        and str(row.get("collection_status") or "")
        == "complete_practitioner_adjudication"
        and _parse_bool(row.get("source_eligible")) is True
    )
    return {
        "collection_packet_id": row.get("collection_packet_id"),
        "channel_name": row.get("channel_name"),
        "packet_sequence": row.get("packet_sequence"),
        "completion_status": "complete_valid" if complete else "pending_or_invalid",
        "missing_required_fields": ",".join(missing),
        "invalid_fields": ",".join(invalid),
        "collection_status": row.get("collection_status"),
        "source_eligible": _parse_bool(row.get("source_eligible")),
        "unique_observed_event_count": _parse_nonnegative_int(
            row.get("unique_observed_event_count")
        ),
        "unique_captured_event_count": _parse_nonnegative_int(
            row.get("unique_captured_event_count")
        ),
        "unique_missed_event_count": _parse_nonnegative_int(
            row.get("unique_missed_event_count")
        ),
        "missed_event_rate": _parse_rate(row.get("missed_event_rate")),
    }


def _invalid_fields(row: dict[str, object]) -> list[str]:
    invalid: list[str] = []
    if not _is_blank(row.get("packet_start_date")) and _parse_date(
        row.get("packet_start_date")
    ) is None:
        invalid.append("packet_start_date")
    if not _is_blank(row.get("packet_end_date")) and _parse_date(
        row.get("packet_end_date")
    ) is None:
        invalid.append("packet_end_date")
    if not _is_blank(row.get("review_date")) and _parse_date(row.get("review_date")) is None:
        invalid.append("review_date")
    for field in (
        "source_eligible",
        "outcome_confirmed",
        "source_context_ok",
    ):
        if not _is_blank(row.get(field)) and _parse_bool(row.get(field)) is None:
            invalid.append(field)
    for field in (
        "episode_count",
        "unique_observed_event_count",
        "unique_captured_event_count",
        "unique_missed_event_count",
    ):
        if not _is_blank(row.get(field)) and _parse_nonnegative_int(row.get(field)) is None:
            invalid.append(field)
    observed = _parse_nonnegative_int(row.get("unique_observed_event_count"))
    captured = _parse_nonnegative_int(row.get("unique_captured_event_count"))
    missed = _parse_nonnegative_int(row.get("unique_missed_event_count"))
    if observed is not None and captured is not None and captured > observed:
        invalid.append("unique_captured_event_count")
    if observed is not None and missed is not None and missed > observed:
        invalid.append("unique_missed_event_count")
    if not _is_blank(row.get("missed_event_rate")) and _parse_rate(
        row.get("missed_event_rate")
    ) is None:
        invalid.append("missed_event_rate")
    if not _is_blank(row.get("alert_usefulness")) and str(
        row.get("alert_usefulness")
    ) not in VALID_USEFULNESS:
        invalid.append("alert_usefulness")
    if not _is_blank(row.get("action_taken")) and str(row.get("action_taken")) not in VALID_ACTIONS:
        invalid.append("action_taken")
    return invalid


def _channel_completion_row(
    target: dict[str, object],
    validation_rows: list[dict[str, object]],
) -> dict[str, object]:
    channel_name = str(target.get("channel_name") or "")
    rows = [
        row
        for row in validation_rows
        if str(row.get("channel_name") or "") == channel_name
        and row.get("completion_status") == "complete_valid"
    ]
    observed = sum(int(row.get("unique_observed_event_count") or 0) for row in rows)
    captured = sum(int(row.get("unique_captured_event_count") or 0) for row in rows)
    missed = sum(int(row.get("unique_missed_event_count") or 0) for row in rows)
    missed_rate = round(missed / observed, 6) if observed else None
    required_packets = _parse_nonnegative_int(target.get("required_packet_count")) or 0
    required_captured = (
        _parse_nonnegative_int(target.get("required_captured_events")) or 0
    )
    max_missed_rate = _parse_rate(target.get("maximum_allowed_missed_event_rate"))
    packet_gate = len(rows) >= required_packets
    captured_gate = captured >= required_captured
    missed_gate = (
        missed_rate is not None
        and max_missed_rate is not None
        and missed_rate <= max_missed_rate
    )
    completion_gate = _channel_gate(packet_gate, captured_gate, missed_gate)
    return {
        "channel_name": channel_name,
        "required_packet_count": required_packets,
        "complete_practitioner_packet_count": len(rows),
        "required_captured_events": required_captured,
        "captured_event_count": captured,
        "observed_event_count": observed,
        "missed_event_count": missed,
        "missed_event_rate": missed_rate,
        "maximum_allowed_missed_event_rate": max_missed_rate,
        "packet_count_gate": "pass" if packet_gate else "blocked",
        "captured_event_gate": "pass" if captured_gate else "blocked",
        "missed_event_rate_gate": "pass" if missed_gate else "blocked",
        "completion_gate": completion_gate,
    }


def _completion_gate_rows(channel_row: dict[str, object]) -> list[dict[str, object]]:
    ready = channel_row["completion_gate"] == "ready_for_bounded_retest_not_claims"
    return [
        {
            "channel_name": channel_row["channel_name"],
            "gate_name": "prospective_collection_completion",
            "gate_status": "pass" if ready else "blocked",
            "interpretation": channel_row["completion_gate"],
        },
        {
            "channel_name": channel_row["channel_name"],
            "gate_name": "bounded_retest_gate",
            "gate_status": "ready" if ready else "blocked",
            "interpretation": "bounded retest only; no claims",
        },
        {
            "channel_name": channel_row["channel_name"],
            "gate_name": "calibration_claim_boundary",
            "gate_status": "blocked",
            "interpretation": "calibration claims remain blocked after completion validation",
        },
    ]


def _channel_gate(packet_gate: bool, captured_gate: bool, missed_gate: bool) -> str:
    if packet_gate and captured_gate and missed_gate:
        return "ready_for_bounded_retest_not_claims"
    if not packet_gate:
        return "blocked_pending_required_packets"
    if not captured_gate:
        return "blocked_pending_captured_event_target"
    return "blocked_high_missed_event_rate"


def _overall_recommendation(all_channels_ready: bool) -> str:
    if all_channels_ready:
        return "run_bounded_retest_after_completed_prospective_collection"
    return "continue_prospective_collection_before_bounded_retest"


def _milestone_status(all_channels_ready: bool) -> str:
    if all_channels_ready:
        return "prospective_collection_complete_for_retest"
    return "prospective_collection_incomplete"


def _bounded_retest_readiness(all_channels_ready: bool) -> str:
    if all_channels_ready:
        return "ready_for_bounded_retest_not_claims"
    return "blocked_pending_collection"


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _parse_rate(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number < 0 or number > 1:
        return None
    return round(number, 6)


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
