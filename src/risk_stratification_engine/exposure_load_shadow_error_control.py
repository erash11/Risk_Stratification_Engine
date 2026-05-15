from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
LOAD_MODIFICATION_BLOCKED = "blocked"
MIN_USEFUL_ACTIONABLE_ROWS = 2
BLOCKED_USES = (
    "probability_output,calibration_claims,pilot_dashboard,"
    "autonomous_intervention,load_modification"
)


def build_exposure_load_shadow_error_control_review(
    calibration_sensitivity: dict[str, object],
) -> dict[str, object]:
    sensitivity = _clean_row(calibration_sensitivity)
    channel_rows = [
        _clean_row(row)
        for row in sensitivity.get("sensitivity_rows", [])
        if isinstance(row, dict)
    ]
    dossier_rows = [
        _clean_row(row)
        for row in sensitivity.get("evidence_dossier_rows", [])
        if isinstance(row, dict)
    ]
    error_rows = [
        _clean_row(row)
        for row in sensitivity.get("error_mode_rows", [])
        if isinstance(row, dict)
    ]
    decisions = [
        _decision_row(
            row,
            [
                error
                for error in error_rows
                if str(error.get("channel_name") or "") == str(row.get("channel_name") or "")
            ],
        )
        for row in sorted(channel_rows, key=lambda item: str(item.get("channel_name") or ""))
    ]
    refined_dossier = [
        _refined_dossier_row(row)
        for row in sorted(
            dossier_rows,
            key=lambda item: (
                str(item.get("channel_name") or ""),
                str(item.get("collection_season_id") or ""),
                str(item.get("collection_packet_id") or ""),
            ),
        )
    ]
    controls = [
        control
        for decision in decisions
        for control in _error_control_rows(
            decision,
            [
                error
                for error in error_rows
                if str(error.get("channel_name") or "")
                == str(decision.get("channel_name") or "")
            ],
        )
    ]
    return {
        "experiment_type": "exposure_load_shadow_error_control_sprint",
        "overall_recommendation": _overall_recommendation(decisions),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "load_modification_readiness": LOAD_MODIFICATION_BLOCKED,
        "bounded_error_control_status": _bounded_status(decisions),
        "upstream_recommendation": sensitivity.get("overall_recommendation"),
        "error_control_definition": {
            "minimum_useful_actionable_rows": MIN_USEFUL_ACTIONABLE_ROWS,
            "high_miss_fraction_requires_control": True,
            "monitoring_useful_is_not_prediction_useful": True,
            "empty_or_missed_only_packets_require_packet_review": True,
            "probability_or_load_modification_allowed": False,
        },
        "decision_rows": clean_shadow_error_control_rows(decisions),
        "refined_evidence_dossier_rows": clean_shadow_error_control_rows(
            refined_dossier
        ),
        "error_control_rows": clean_shadow_error_control_rows(controls),
        "interpretation_boundary": (
            "bounded error-controlled calibration research decision package "
            "only; monitoring usefulness is separated from prediction or "
            "calibration usefulness and does not permit probability output, "
            "calibration claims, pilot/dashboard readiness, autonomous "
            "intervention, or load modification"
        ),
    }


def write_exposure_load_shadow_error_control_report(
    path: Path,
    review: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Error-Control Decision Sprint",
        "",
        f"Recommendation: {review['overall_recommendation']}",
        f"Production readiness: {review['production_readiness']}",
        f"Calibration claim readiness: {review['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {review['pilot_dashboard_readiness']}",
        f"Load modification readiness: {review['load_modification_readiness']}",
        f"Bounded error-control status: {review['bounded_error_control_status']}",
        f"Interpretation boundary: {review['interpretation_boundary']}",
        "",
        "## Channel Decisions",
        "",
        (
            "| Channel | Monitoring status | Prediction/calibration status | "
            "Error-control status | Next gate |"
        ),
        "|---|---|---|---|---|",
    ]
    for row in review.get("decision_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['monitoring_usefulness_status']} | "
            f"{row['prediction_calibration_usefulness_status']} | "
            f"{row['error_control_status']} | "
            f"{row['next_gate_decision']} |"
        )
    lines.extend(
        [
            "",
            "## Required Controls",
            "",
            "| Channel | Control | Required | Action |",
            "|---|---|---|---|",
        ]
    )
    for row in review.get("error_control_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['control_name']} | "
            f"{row['required']} | "
            f"{row['control_action']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This artifact defines error-mode controlled calibration research "
                "as a bounded research-only next gate. Practitioner support can "
                "justify monitoring context only; it is not calibration claims, "
                "probability-facing output, pilot/dashboard clearance, autonomous "
                "intervention, or load modification guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_error_control_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _decision_row(
    sensitivity_row: dict[str, object],
    error_rows: list[dict[str, object]],
) -> dict[str, object]:
    channel_name = str(sensitivity_row.get("channel_name") or "")
    useful_rows = _parse_nonnegative_int(
        sensitivity_row.get("useful_actionable_rows")
    ) or 0
    miss_rate_gate = _normalized(sensitivity_row.get("miss_rate_gate"))
    practitioner_gate = _normalized(
        sensitivity_row.get("practitioner_adjudication_gate")
    )
    error_modes = {_normalized(row.get("error_mode")) for row in error_rows}
    has_high_miss = "high_miss_fraction" in error_modes or miss_rate_gate == "caution"
    has_monitor_boundary = "monitor_only_action_boundary" in error_modes
    if practitioner_gate == "fail":
        next_gate = "complete_practitioner_adjudication"
        retained = False
    elif useful_rows < MIN_USEFUL_ACTIONABLE_ROWS:
        next_gate = "collect_more_practitioner_reviewed_rows"
        retained = False
    elif has_high_miss:
        next_gate = "continue_bounded_calibration_research_with_error_controls"
        retained = True
    else:
        next_gate = "continue_bounded_calibration_research"
        retained = True
    monitoring_status = (
        "supported_for_monitoring_context_only"
        if useful_rows >= MIN_USEFUL_ACTIONABLE_ROWS and has_monitor_boundary
        else "insufficient_monitoring_support"
        if useful_rows < MIN_USEFUL_ACTIONABLE_ROWS
        else "supported_for_monitoring_context"
    )
    prediction_status = (
        "not_established_high_miss_fraction"
        if has_high_miss
        else "bounded_research_candidate_not_claims"
        if retained
        else "not_established"
    )
    return {
        "channel_name": channel_name,
        "retained_research_candidate": retained,
        "useful_actionable_rows": useful_rows,
        "capture_rate": sensitivity_row.get("capture_rate"),
        "missed_event_rate": sensitivity_row.get("missed_event_rate"),
        "monitoring_usefulness_status": monitoring_status,
        "prediction_calibration_usefulness_status": prediction_status,
        "error_control_status": (
            "requires_high_miss_fraction_controls"
            if has_high_miss
            else "standard_bounded_research_controls"
        ),
        "next_gate_decision": next_gate,
        "allowed_use": "shadow_monitoring_review_only",
        "blocked_use": BLOCKED_USES,
    }


def _refined_dossier_row(row: dict[str, object]) -> dict[str, object]:
    observed_events = _parse_nonnegative_int(row.get("observed_event_count")) or 0
    captured_events = _parse_nonnegative_int(row.get("captured_event_count")) or 0
    missed_events = _parse_nonnegative_int(row.get("missed_event_count")) or 0
    usefulness = _normalized(row.get("alert_usefulness"))
    action_taken = _normalized(row.get("action_taken"))
    refined = dict(row)
    refined["dossier_refinement_action"] = _dossier_refinement_action(
        observed_events,
        captured_events,
        missed_events,
        usefulness,
        action_taken,
    )
    refined["monitoring_prediction_boundary"] = (
        "monitoring_useful_not_prediction_evidence"
        if usefulness == "useful" and action_taken == "monitor"
        else "not_monitoring_supported"
    )
    return refined


def _error_control_rows(
    decision: dict[str, object],
    error_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    channel_name = str(decision.get("channel_name") or "")
    error_modes = {_normalized(row.get("error_mode")) for row in error_rows}
    return [
        {
            "channel_name": channel_name,
            "control_name": "miss_fraction_control",
            "required": "high_miss_fraction" in error_modes,
            "control_action": (
                "retain_miss_fraction_threshold_and_report_missed_only_packets"
            ),
        },
        {
            "channel_name": channel_name,
            "control_name": "monitoring_prediction_boundary",
            "required": "monitor_only_action_boundary" in error_modes,
            "control_action": "separate_monitoring_value_from_calibration_evidence",
        },
        {
            "channel_name": channel_name,
            "control_name": "packet_review_control",
            "required": bool(
                {"empty_outcome_packet", "missed_only_packet"} & error_modes
            ),
            "control_action": "review_empty_and_missed_only_packets_before_escalation",
        },
    ]


def _overall_recommendation(decisions: list[dict[str, object]]) -> str:
    if not decisions:
        return "collect_more_practitioner_reviewed_rows_before_calibration_research"
    next_gates = {str(row.get("next_gate_decision") or "") for row in decisions}
    if "complete_practitioner_adjudication" in next_gates:
        return "complete_practitioner_adjudication_before_calibration_research"
    if "collect_more_practitioner_reviewed_rows" in next_gates:
        return "collect_more_practitioner_reviewed_rows_before_calibration_research"
    if "continue_bounded_calibration_research_with_error_controls" in next_gates:
        return "continue_bounded_calibration_research_with_error_controls_not_claims"
    return "continue_bounded_calibration_research_not_claims"


def _bounded_status(decisions: list[dict[str, object]]) -> str:
    if decisions and all(
        str(row.get("next_gate_decision") or "").startswith(
            "continue_bounded_calibration_research"
        )
        for row in decisions
    ):
        return "ready_for_error_controlled_research_decision_not_claims"
    return "not_ready_for_error_controlled_calibration_research"


def _dossier_refinement_action(
    observed_events: int,
    captured_events: int,
    missed_events: int,
    usefulness: str,
    action_taken: str,
) -> str:
    if observed_events == 0:
        return "collect_outcome_context_before_calibration_weight"
    if captured_events == 0 and missed_events > 0:
        return "review_missed_only_packet_before_channel_escalation"
    if usefulness == "useful" and action_taken == "monitor":
        return "preserve_as_monitoring_useful_not_prediction_evidence"
    if usefulness == "useful" and action_taken not in {"", "none"}:
        return "preserve_as_practitioner_supported_evidence"
    return "collect_practitioner_context_before_use"


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


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()
