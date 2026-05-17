from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
LOAD_MODIFICATION_BLOCKED = "blocked"
HIGH_MISS_RATE_THRESHOLD = 0.75
LOW_CAPTURE_RATE_THRESHOLD = 0.25


def build_exposure_load_shadow_bounded_calibration_stress_test(
    bounded_protocol: dict[str, object],
) -> dict[str, object]:
    protocol = _clean_row(bounded_protocol)
    channel_protocols = [
        _clean_row(row)
        for row in protocol.get("channel_protocol_rows", [])
        if isinstance(row, dict)
    ]
    evidence_rows = [
        _clean_row(row)
        for row in protocol.get("evidence_use_rows", [])
        if isinstance(row, dict)
    ]
    channel_rows = [
        _channel_stress_row(
            row,
            [
                evidence
                for evidence in evidence_rows
                if str(evidence.get("channel_name") or "")
                == str(row.get("channel_name") or "")
            ],
        )
        for row in sorted(
            channel_protocols,
            key=lambda item: str(item.get("channel_name") or ""),
        )
    ]
    scenario_rows = [
        scenario
        for row in channel_rows
        for scenario in _scenario_rows(
            str(row.get("channel_name") or ""),
            [
                evidence
                for evidence in evidence_rows
                if str(evidence.get("channel_name") or "")
                == str(row.get("channel_name") or "")
            ],
        )
    ]
    gate_rows = [
        gate
        for row in channel_rows
        for gate in _stress_gate_rows(row)
    ]
    return {
        "experiment_type": (
            "exposure_load_shadow_bounded_calibration_stress_test_sprint"
        ),
        "overall_recommendation": _overall_recommendation(channel_rows),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "load_modification_readiness": LOAD_MODIFICATION_BLOCKED,
        "stress_test_status": _stress_test_status(channel_rows),
        "upstream_recommendation": protocol.get("overall_recommendation"),
        "stress_test_definition": {
            "analysis_type": "descriptive_shadow_calibration_stress_test",
            "high_miss_rate_threshold": HIGH_MISS_RATE_THRESHOLD,
            "low_capture_rate_threshold": LOW_CAPTURE_RATE_THRESHOLD,
            "calibration_claims_allowed": False,
            "probability_or_load_modification_allowed": False,
        },
        "channel_stress_rows": clean_shadow_bounded_calibration_stress_test_rows(
            channel_rows
        ),
        "stress_scenario_rows": clean_shadow_bounded_calibration_stress_test_rows(
            scenario_rows
        ),
        "stress_gate_rows": clean_shadow_bounded_calibration_stress_test_rows(
            gate_rows
        ),
        "interpretation_boundary": (
            "descriptive shadow calibration stress test only; high miss "
            "fractions preserve a limited monitoring finding and do not support "
            "calibration claims, probability output, pilot/dashboard readiness, "
            "autonomous intervention, or load modification"
        ),
    }


def write_exposure_load_shadow_bounded_calibration_stress_test_report(
    path: Path,
    stress_test: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Bounded Calibration Stress Test Sprint",
        "",
        f"Recommendation: {stress_test['overall_recommendation']}",
        f"Production readiness: {stress_test['production_readiness']}",
        f"Calibration claim readiness: {stress_test['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {stress_test['pilot_dashboard_readiness']}",
        f"Load modification readiness: {stress_test['load_modification_readiness']}",
        f"Stress test status: {stress_test['stress_test_status']}",
        f"Interpretation boundary: {stress_test['interpretation_boundary']}",
        "",
        "## Channel Stress Results",
        "",
        "| Channel | Capture rate | Missed rate | Classification | Decision |",
        "|---|---:|---:|---|---|",
    ]
    for row in stress_test.get("channel_stress_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['capture_rate']} | "
            f"{row['missed_event_rate']} | "
            f"{row['stress_classification']} | "
            f"{row['stress_decision']} |"
        )
    lines.extend(
        [
            "",
            "## Stress Scenarios",
            "",
            "| Channel | Scenario | Packets | Capture rate | Missed events |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in stress_test.get("stress_scenario_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['scenario_name']} | "
            f"{row['packet_count']} | "
            f"{row['capture_rate']} | "
            f"{row['missed_event_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint runs a descriptive bounded stress test and preserves "
                "a limited calibration finding when miss fractions dominate. "
                "It is not calibration claims, probability-facing output, "
                "pilot/dashboard clearance, autonomous intervention, or load "
                "modification guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_bounded_calibration_stress_test_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _channel_stress_row(
    protocol_row: dict[str, object],
    evidence_rows: list[dict[str, object]],
) -> dict[str, object]:
    channel_name = str(protocol_row.get("channel_name") or "")
    eligible = (
        str(protocol_row.get("protocol_status") or "")
        == "eligible_for_bounded_stress_test_not_claims"
    )
    observed = _sum_int(evidence_rows, "observed_event_count")
    captured = _sum_int(evidence_rows, "captured_event_count")
    missed = _sum_int(evidence_rows, "missed_event_count")
    monitoring_rows = [
        row
        for row in evidence_rows
        if str(row.get("protocol_evidence_role") or "")
        == "monitoring_context_only_not_calibration_claim"
    ]
    monitoring_observed = _sum_int(monitoring_rows, "observed_event_count")
    monitoring_captured = _sum_int(monitoring_rows, "captured_event_count")
    capture_rate = _ratio(captured, observed)
    missed_rate = _ratio(missed, observed)
    if not eligible:
        classification = "not_ready_for_stress_test"
        decision = "complete_protocol_before_stress_test"
    elif missed_rate >= HIGH_MISS_RATE_THRESHOLD or capture_rate < LOW_CAPTURE_RATE_THRESHOLD:
        classification = "high_miss_limited_calibration_signal"
        decision = "preserve_limited_monitoring_value_collect_more_evidence"
    else:
        classification = "bounded_signal_requires_prospective_confirmation"
        decision = "continue_bounded_research_without_claims"
    return {
        "channel_name": channel_name,
        "protocol_status": protocol_row.get("protocol_status"),
        "observed_event_count": observed,
        "captured_event_count": captured,
        "missed_event_count": missed,
        "capture_rate": capture_rate,
        "missed_event_rate": missed_rate,
        "monitoring_context_packet_count": len(monitoring_rows),
        "monitoring_context_capture_rate": _ratio(
            monitoring_captured,
            monitoring_observed,
        ),
        "stress_classification": classification,
        "stress_decision": decision,
        "calibration_claim_status": "blocked",
        "probability_output_status": "blocked",
        "load_modification_status": "blocked",
    }


def _scenario_rows(
    channel_name: str,
    evidence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    monitoring_rows = [
        row
        for row in evidence_rows
        if str(row.get("protocol_evidence_role") or "")
        == "monitoring_context_only_not_calibration_claim"
    ]
    missed_only_rows = [
        row
        for row in evidence_rows
        if str(row.get("protocol_evidence_role") or "")
        == "missed_only_error_case_for_sensitivity_bounds"
    ]
    observed_rows = [
        row for row in evidence_rows if _parse_nonnegative_int(row.get("observed_event_count")) or 0
    ]
    return [
        _scenario_row(channel_name, "observed_replay_all", evidence_rows),
        _scenario_row(channel_name, "exclude_outcome_context_gaps", observed_rows),
        _scenario_row(channel_name, "monitoring_context_only", monitoring_rows),
        _scenario_row(channel_name, "missed_only_error_bound", missed_only_rows),
    ]


def _scenario_row(
    channel_name: str,
    scenario_name: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    observed = _sum_int(rows, "observed_event_count")
    captured = _sum_int(rows, "captured_event_count")
    missed = _sum_int(rows, "missed_event_count")
    return {
        "channel_name": channel_name,
        "scenario_name": scenario_name,
        "packet_count": len(rows),
        "observed_event_count": observed,
        "captured_event_count": captured,
        "missed_event_count": missed,
        "capture_rate": _ratio(captured, observed),
        "missed_event_rate": _ratio(missed, observed),
    }


def _stress_gate_rows(channel_row: dict[str, object]) -> list[dict[str, object]]:
    channel_name = str(channel_row.get("channel_name") or "")
    stress_complete = channel_row.get("stress_decision") != (
        "complete_protocol_before_stress_test"
    )
    return [
        {
            "channel_name": channel_name,
            "gate_name": "stress_test_complete",
            "gate_status": "pass" if stress_complete else "fail",
            "interpretation": "descriptive stress test completed",
        },
        {
            "channel_name": channel_name,
            "gate_name": "calibration_claim_boundary",
            "gate_status": "blocked",
            "interpretation": "stress results do not authorize calibration claims",
        },
        {
            "channel_name": channel_name,
            "gate_name": "prospective_evidence_needed",
            "gate_status": "required",
            "interpretation": "additional prospective evidence is required before claims",
        },
    ]


def _overall_recommendation(channel_rows: list[dict[str, object]]) -> str:
    if not channel_rows or all(
        row.get("stress_decision") == "complete_protocol_before_stress_test"
        for row in channel_rows
    ):
        return "complete_bounded_protocol_before_stress_test"
    if all(
        row.get("stress_decision")
        == "preserve_limited_monitoring_value_collect_more_evidence"
        for row in channel_rows
    ):
        return "preserve_limited_calibration_finding_and_collect_more_prospective_evidence"
    return "continue_bounded_research_and_collect_more_prospective_evidence"


def _stress_test_status(channel_rows: list[dict[str, object]]) -> str:
    if channel_rows and any(
        row.get("stress_decision") != "complete_protocol_before_stress_test"
        for row in channel_rows
    ):
        return "completed_descriptive_stress_test_not_calibration_claim"
    return "not_ready_for_stress_test"


def _sum_int(rows: list[dict[str, object]], key: str) -> int:
    return sum(_parse_nonnegative_int(row.get(key)) or 0 for row in rows)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


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
