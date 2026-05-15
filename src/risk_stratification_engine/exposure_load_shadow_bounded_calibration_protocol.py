from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
LOAD_MODIFICATION_BLOCKED = "blocked"
REQUIRED_CONTROLS = [
    "miss_fraction_control",
    "monitoring_prediction_boundary",
    "packet_review_control",
]


def build_exposure_load_shadow_bounded_calibration_protocol(
    error_control_policy: dict[str, object],
) -> dict[str, object]:
    policy = _clean_row(error_control_policy)
    decisions = [
        _clean_row(row)
        for row in policy.get("decision_rows", [])
        if isinstance(row, dict)
    ]
    dossier_rows = [
        _clean_row(row)
        for row in policy.get("refined_evidence_dossier_rows", [])
        if isinstance(row, dict)
    ]
    control_rows = [
        _clean_row(row)
        for row in policy.get("error_control_rows", [])
        if isinstance(row, dict)
    ]
    channel_rows = [
        _channel_protocol_row(
            decision,
            [
                row
                for row in control_rows
                if str(row.get("channel_name") or "")
                == str(decision.get("channel_name") or "")
            ],
        )
        for decision in sorted(decisions, key=lambda item: str(item.get("channel_name") or ""))
    ]
    evidence_rows = [
        _evidence_use_row(row)
        for row in sorted(
            dossier_rows,
            key=lambda item: (
                str(item.get("channel_name") or ""),
                str(item.get("collection_season_id") or ""),
                str(item.get("collection_packet_id") or ""),
            ),
        )
    ]
    gate_rows = [
        gate
        for row in channel_rows
        for gate in _protocol_gate_rows(row)
    ]
    return {
        "experiment_type": "exposure_load_shadow_bounded_calibration_protocol_sprint",
        "overall_recommendation": _overall_recommendation(channel_rows),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "load_modification_readiness": LOAD_MODIFICATION_BLOCKED,
        "bounded_protocol_status": _bounded_protocol_status(channel_rows),
        "upstream_recommendation": policy.get("overall_recommendation"),
        "protocol_definition": {
            "allowed_analysis": "descriptive_shadow_calibration_stress_test",
            "required_controls": REQUIRED_CONTROLS,
            "monitoring_context_is_not_calibration_evidence": True,
            "probability_or_load_modification_allowed": False,
        },
        "channel_protocol_rows": clean_shadow_bounded_calibration_protocol_rows(
            channel_rows
        ),
        "evidence_use_rows": clean_shadow_bounded_calibration_protocol_rows(
            evidence_rows
        ),
        "protocol_gate_rows": clean_shadow_bounded_calibration_protocol_rows(
            gate_rows
        ),
        "interpretation_boundary": (
            "research-only protocol for a descriptive shadow calibration stress "
            "test; monitoring context is not calibration evidence and cannot be "
            "used for probability output, calibration claims, pilot/dashboard "
            "readiness, autonomous intervention, or load modification"
        ),
    }


def write_exposure_load_shadow_bounded_calibration_protocol_report(
    path: Path,
    protocol: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Bounded Calibration Protocol Sprint",
        "",
        f"Recommendation: {protocol['overall_recommendation']}",
        f"Production readiness: {protocol['production_readiness']}",
        f"Calibration claim readiness: {protocol['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {protocol['pilot_dashboard_readiness']}",
        f"Load modification readiness: {protocol['load_modification_readiness']}",
        f"Bounded protocol status: {protocol['bounded_protocol_status']}",
        f"Interpretation boundary: {protocol['interpretation_boundary']}",
        "",
        "## Channel Protocol",
        "",
        "| Channel | Protocol status | Analysis scope | Next analysis |",
        "|---|---|---|---|",
    ]
    for row in protocol.get("channel_protocol_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['protocol_status']} | "
            f"{row['analysis_scope']} | "
            f"{row['next_analysis']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Use",
            "",
            "| Packet | Channel | Evidence role |",
            "|---|---|---|",
        ]
    )
    for row in protocol.get("evidence_use_rows", []):
        lines.append(
            "| "
            f"{row['collection_packet_id']} | "
            f"{row['channel_name']} | "
            f"{row['protocol_evidence_role']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint authorizes only a descriptive shadow calibration "
                "stress test under explicit error controls. Monitoring context "
                "can be reviewed, but it is not calibration claims, "
                "probability-facing output, pilot/dashboard clearance, "
                "autonomous intervention, or load modification guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_bounded_calibration_protocol_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _channel_protocol_row(
    decision: dict[str, object],
    control_rows: list[dict[str, object]],
) -> dict[str, object]:
    channel_name = str(decision.get("channel_name") or "")
    control_names = {
        str(row.get("control_name") or "")
        for row in control_rows
        if _parse_bool_text(row.get("required")) is True
    }
    retained = _parse_bool_text(decision.get("retained_research_candidate")) is True
    controls_complete = all(control in control_names for control in REQUIRED_CONTROLS)
    if retained and controls_complete:
        status = "eligible_for_bounded_stress_test_not_claims"
        next_analysis = "bounded_calibration_stress_test_without_claims"
    elif retained:
        status = "not_eligible_complete_required_error_controls"
        next_analysis = "complete_error_controls_before_stress_test"
    else:
        status = "not_eligible_collect_more_practitioner_rows"
        next_analysis = "collect_more_practitioner_reviewed_rows"
    return {
        "channel_name": channel_name,
        "protocol_status": status,
        "analysis_scope": "monitoring_context_error_controlled_only",
        "required_control_count": len(control_names & set(REQUIRED_CONTROLS)),
        "required_controls": ",".join(REQUIRED_CONTROLS),
        "next_analysis": next_analysis,
        "calibration_claim_status": "blocked",
        "probability_output_status": "blocked",
        "load_modification_status": "blocked",
    }


def _evidence_use_row(row: dict[str, object]) -> dict[str, object]:
    evidence = dict(row)
    observed = _parse_nonnegative_int(row.get("observed_event_count")) or 0
    captured = _parse_nonnegative_int(row.get("captured_event_count")) or 0
    missed = _parse_nonnegative_int(row.get("missed_event_count")) or 0
    action = _normalized(row.get("action_taken"))
    refinement = _normalized(row.get("dossier_refinement_action"))
    if observed == 0 or "outcome_context" in refinement:
        role = "outcome_context_gap_excluded_from_calibration_signal"
    elif captured == 0 and missed > 0:
        role = "missed_only_error_case_for_sensitivity_bounds"
    elif action == "monitor":
        role = "monitoring_context_only_not_calibration_claim"
    else:
        role = "review_context_before_protocol_use"
    evidence["protocol_evidence_role"] = role
    return evidence


def _protocol_gate_rows(channel_row: dict[str, object]) -> list[dict[str, object]]:
    channel_name = str(channel_row.get("channel_name") or "")
    controls_complete = (
        str(channel_row.get("protocol_status") or "")
        == "eligible_for_bounded_stress_test_not_claims"
    )
    return [
        {
            "channel_name": channel_name,
            "gate_name": "controls_complete",
            "gate_status": "pass" if controls_complete else "fail",
            "interpretation": "required error controls are present",
        },
        {
            "channel_name": channel_name,
            "gate_name": "claim_boundary",
            "gate_status": "blocked",
            "interpretation": "calibration claims remain blocked",
        },
        {
            "channel_name": channel_name,
            "gate_name": "load_modification_boundary",
            "gate_status": "blocked",
            "interpretation": "load modification recommendations remain blocked",
        },
    ]


def _overall_recommendation(channel_rows: list[dict[str, object]]) -> str:
    if not channel_rows:
        return "collect_more_practitioner_reviewed_rows_before_stress_test"
    statuses = {str(row.get("protocol_status") or "") for row in channel_rows}
    if statuses == {"eligible_for_bounded_stress_test_not_claims"}:
        return "run_bounded_calibration_stress_test_without_claims"
    if "eligible_for_bounded_stress_test_not_claims" in statuses:
        return "run_partial_bounded_stress_test_and_collect_more_rows"
    return "collect_more_practitioner_reviewed_rows_before_stress_test"


def _bounded_protocol_status(channel_rows: list[dict[str, object]]) -> str:
    if channel_rows and any(
        row.get("protocol_status") == "eligible_for_bounded_stress_test_not_claims"
        for row in channel_rows
    ):
        return "ready_for_research_only_stress_test_protocol"
    return "not_ready_for_bounded_calibration_protocol"


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


def _parse_bool_text(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()
