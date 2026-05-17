from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
LOAD_MODIFICATION_BLOCKED = "blocked"
MINIMUM_NEW_PACKETS = 4
MINIMUM_CAPTURED_EVENTS = 8
MAXIMUM_ALLOWED_MISSED_RATE = 0.75


def build_exposure_load_shadow_prospective_evidence_gate(
    stress_test: dict[str, object],
) -> dict[str, object]:
    stress = _clean_row(stress_test)
    channel_rows = [
        _clean_row(row)
        for row in stress.get("channel_stress_rows", [])
        if isinstance(row, dict)
    ]
    scenario_rows = [
        _clean_row(row)
        for row in stress.get("stress_scenario_rows", [])
        if isinstance(row, dict)
    ]
    collection_targets = [
        _collection_target_row(row)
        for row in sorted(channel_rows, key=lambda item: str(item.get("channel_name") or ""))
        if _requires_collection(row)
    ]
    packet_targets = [
        target
        for row in collection_targets
        for target in _packet_target_rows(
            str(row.get("channel_name") or ""),
            [
                scenario
                for scenario in scenario_rows
                if str(scenario.get("channel_name") or "")
                == str(row.get("channel_name") or "")
            ],
        )
    ]
    evidence_gates = [
        gate
        for row in collection_targets
        for gate in _evidence_gate_rows(str(row.get("channel_name") or ""))
    ]
    return {
        "experiment_type": "exposure_load_shadow_prospective_evidence_gate_sprint",
        "overall_recommendation": _overall_recommendation(collection_targets),
        "milestone_status": _milestone_status(collection_targets),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "load_modification_readiness": LOAD_MODIFICATION_BLOCKED,
        "upstream_recommendation": stress.get("overall_recommendation"),
        "gate_definition": {
            "minimum_new_prospective_packets_per_channel": MINIMUM_NEW_PACKETS,
            "minimum_captured_events_per_channel": MINIMUM_CAPTURED_EVENTS,
            "maximum_allowed_missed_event_rate": MAXIMUM_ALLOWED_MISSED_RATE,
            "requires_practitioner_review": True,
            "calibration_claims_allowed": False,
            "probability_or_load_modification_allowed": False,
        },
        "collection_target_rows": clean_shadow_prospective_evidence_gate_rows(
            collection_targets
        ),
        "packet_target_rows": clean_shadow_prospective_evidence_gate_rows(
            packet_targets
        ),
        "evidence_gate_rows": clean_shadow_prospective_evidence_gate_rows(
            evidence_gates
        ),
        "interpretation_boundary": (
            "prospective retained-channel evidence collection gate only; "
            "collection targets do not authorize calibration claims, "
            "probability output, pilot/dashboard readiness, autonomous "
            "intervention, or load modification"
        ),
    }


def write_exposure_load_shadow_prospective_evidence_gate_report(
    path: Path,
    gate: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Prospective Evidence Gate Sprint",
        "",
        f"Recommendation: {gate['overall_recommendation']}",
        f"Milestone status: {gate['milestone_status']}",
        f"Production readiness: {gate['production_readiness']}",
        f"Calibration claim readiness: {gate['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {gate['pilot_dashboard_readiness']}",
        f"Load modification readiness: {gate['load_modification_readiness']}",
        f"Interpretation boundary: {gate['interpretation_boundary']}",
        "",
        "## Collection Targets",
        "",
        "| Channel | New packets | Captured events needed | Max missed rate | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in gate.get("collection_target_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['minimum_new_prospective_packets']} | "
            f"{row['minimum_captured_events_needed']} | "
            f"{row['maximum_allowed_missed_event_rate']} | "
            f"{row['target_decision']} |"
        )
    lines.extend(
        [
            "",
            "## Packet Targets",
            "",
            "| Channel | Target type | Minimum packets | Required action |",
            "|---|---|---:|---|",
        ]
    )
    for row in gate.get("packet_target_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['target_type']} | "
            f"{row['minimum_packet_count']} | "
            f"{row['required_action']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint reaches the prospective evidence collection gate. "
                "It defines the retained-channel evidence needed before any "
                "future retest. It is not calibration claims, probability-facing "
                "output, pilot/dashboard clearance, autonomous intervention, or "
                "load modification guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_prospective_evidence_gate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _collection_target_row(channel_row: dict[str, object]) -> dict[str, object]:
    channel_name = str(channel_row.get("channel_name") or "")
    observed = _parse_nonnegative_int(channel_row.get("observed_event_count")) or 0
    captured = _parse_nonnegative_int(channel_row.get("captured_event_count")) or 0
    missed = _parse_nonnegative_int(channel_row.get("missed_event_count")) or 0
    return {
        "channel_name": channel_name,
        "prior_observed_event_count": observed,
        "prior_captured_event_count": captured,
        "prior_missed_event_count": missed,
        "prior_capture_rate": channel_row.get("capture_rate"),
        "prior_missed_event_rate": channel_row.get("missed_event_rate"),
        "minimum_new_prospective_packets": MINIMUM_NEW_PACKETS,
        "minimum_captured_events_needed": MINIMUM_CAPTURED_EVENTS,
        "maximum_allowed_missed_event_rate": MAXIMUM_ALLOWED_MISSED_RATE,
        "target_decision": "collect_prospective_evidence_then_retest",
        "blocked_use": (
            "probability_output,calibration_claims,pilot_dashboard,"
            "autonomous_intervention,load_modification"
        ),
    }


def _packet_target_rows(
    channel_name: str,
    scenario_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    monitoring_packet_count = _scenario_packet_count(
        scenario_rows,
        "monitoring_context_only",
    )
    missed_only_packet_count = _scenario_packet_count(
        scenario_rows,
        "missed_only_error_bound",
    )
    return [
        {
            "channel_name": channel_name,
            "target_type": "monitoring_context_packet",
            "minimum_packet_count": max(2, monitoring_packet_count),
            "required_action": "capture_practitioner_monitoring_context_and_outcome_followup",
        },
        {
            "channel_name": channel_name,
            "target_type": "missed_only_error_packet",
            "minimum_packet_count": max(1, missed_only_packet_count),
            "required_action": "adjudicate_missed_only_error_context",
        },
        {
            "channel_name": channel_name,
            "target_type": "outcome_context_packet",
            "minimum_packet_count": 1,
            "required_action": "capture_outcome_context_or_mark_unavailable",
        },
    ]


def _evidence_gate_rows(channel_name: str) -> list[dict[str, object]]:
    return [
        {
            "channel_name": channel_name,
            "gate_name": "prospective_collection_required",
            "gate_status": "required",
            "interpretation": "new prospective retained-channel packets are required",
        },
        {
            "channel_name": channel_name,
            "gate_name": "calibration_claim_boundary",
            "gate_status": "blocked",
            "interpretation": "calibration claims remain blocked until retesting",
        },
        {
            "channel_name": channel_name,
            "gate_name": "retest_after_collection",
            "gate_status": "pending",
            "interpretation": "rerun bounded stress testing after prospective collection",
        },
    ]


def _requires_collection(row: dict[str, object]) -> bool:
    return (
        str(row.get("stress_decision") or "")
        == "preserve_limited_monitoring_value_collect_more_evidence"
    )


def _scenario_packet_count(rows: list[dict[str, object]], scenario_name: str) -> int:
    for row in rows:
        if str(row.get("scenario_name") or "") == scenario_name:
            return _parse_nonnegative_int(row.get("packet_count")) or 0
    return 0


def _overall_recommendation(targets: list[dict[str, object]]) -> str:
    if not targets:
        return "close_limited_finding_no_retained_channels_for_collection"
    return "collect_prospective_retained_channel_evidence_before_retesting"


def _milestone_status(targets: list[dict[str, object]]) -> str:
    if not targets:
        return "limited_finding_closeout_ready"
    return "prospective_evidence_collection_gate_defined"


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
