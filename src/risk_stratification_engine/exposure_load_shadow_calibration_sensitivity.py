from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
PILOT_DASHBOARD_BLOCKED = "blocked"
ADJUDICATED_EVIDENCE = "independent_practitioner_adjudicated_shadow_collection"
READY_RESEARCH_STATUS = "ready_for_bounded_calibration_research_not_claims"
MISS_RATE_CAUTION_THRESHOLD = 0.5


def build_exposure_load_shadow_calibration_sensitivity_review(
    *,
    calibration_readiness: dict[str, object],
    collection_rows: list[dict[str, object]],
    event_crosswalk_rows: list[dict[str, object]],
) -> dict[str, object]:
    readiness = _clean_row(calibration_readiness)
    collections = [_clean_row(row) for row in collection_rows]
    crosswalk = [_clean_row(row) for row in event_crosswalk_rows]
    adjudication_ready = _adjudication_ready(readiness)
    channel_names = sorted(
        {
            str(row.get("channel_name") or "")
            for row in collections + crosswalk
            if str(row.get("channel_name") or "")
        }
    )
    dossier_rows = [
        _evidence_dossier_row(row, crosswalk)
        for row in sorted(
            collections,
            key=lambda item: (
                str(item.get("channel_name") or ""),
                str(item.get("collection_season_id") or ""),
                str(item.get("collection_packet_id") or ""),
            ),
        )
    ]
    sensitivity_rows = [
        _sensitivity_row(
            channel_name,
            [
                row
                for row in collections
                if str(row.get("channel_name") or "") == channel_name
            ],
            [
                row
                for row in crosswalk
                if str(row.get("channel_name") or "") == channel_name
            ],
            adjudication_ready,
        )
        for channel_name in channel_names
    ]
    error_mode_rows = [
        row
        for channel_name in channel_names
        for row in _error_mode_rows(
            channel_name,
            [
                row
                for row in dossier_rows
                if str(row.get("channel_name") or "") == channel_name
            ],
            [
                row
                for row in sensitivity_rows
                if str(row.get("channel_name") or "") == channel_name
            ][0],
        )
    ]
    return {
        "experiment_type": "exposure_load_shadow_calibration_sensitivity_sprint",
        "overall_recommendation": _overall_recommendation(
            adjudication_ready,
            sensitivity_rows,
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "pilot_dashboard_readiness": PILOT_DASHBOARD_BLOCKED,
        "bounded_research_status": (
            "ready_for_bounded_sensitivity_review_not_claims"
            if adjudication_ready
            else "not_ready_practitioner_adjudication_incomplete"
        ),
        "readiness_evidence_basis": readiness.get("evidence_basis"),
        "channel_count": len(channel_names),
        "sensitivity_rows": clean_shadow_calibration_sensitivity_rows(
            sensitivity_rows
        ),
        "evidence_dossier_rows": clean_shadow_calibration_sensitivity_rows(
            dossier_rows
        ),
        "error_mode_rows": clean_shadow_calibration_sensitivity_rows(
            error_mode_rows
        ),
        "interpretation_boundary": (
            "bounded calibration research sensitivity review only; not "
            "calibration claims, probability-facing output, pilot/dashboard "
            "readiness, or autonomous intervention"
        ),
    }


def write_exposure_load_shadow_calibration_sensitivity_report(
    path: Path,
    review: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Calibration Sensitivity Sprint",
        "",
        f"Recommendation: {review['overall_recommendation']}",
        f"Production readiness: {review['production_readiness']}",
        f"Calibration claim readiness: {review['calibration_claim_readiness']}",
        f"Pilot/dashboard readiness: {review['pilot_dashboard_readiness']}",
        f"Bounded research status: {review['bounded_research_status']}",
        f"Interpretation boundary: {review['interpretation_boundary']}",
        "",
        "## Channel Sensitivity",
        "",
        (
            "| Channel | Useful/actionable | Capture rate | Miss rate gate | "
            "Required next action |"
        ),
        "|---|---:|---:|---|---|",
    ]
    for row in review.get("sensitivity_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['useful_actionable_rows']} | "
            f"{row['capture_rate']} | "
            f"{row['miss_rate_gate']} | "
            f"{row['required_next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Error Modes",
            "",
            "| Channel | Error mode | Severity | Packets |",
            "|---|---|---|---:|",
        ]
    )
    for row in review.get("error_mode_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['error_mode']} | "
            f"{row['severity']} | "
            f"{row['packet_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint consolidates practitioner-reviewed retained-channel "
                "evidence into sensitivity, evidence-dossier, and error-mode "
                "artifacts. It can support bounded calibration research, but it "
                "is not calibration claims, probability-facing output, pilot "
                "readiness, dashboard clearance, or intervention guidance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_calibration_sensitivity_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _sensitivity_row(
    channel_name: str,
    collection_rows: list[dict[str, object]],
    crosswalk_rows: list[dict[str, object]],
    adjudication_ready: bool,
) -> dict[str, object]:
    complete_rows = [
        row
        for row in collection_rows
        if _normalized(row.get("collection_status"))
        in {"complete", "complete_practitioner_adjudication", "complete_practitioner_review"}
    ]
    source_eligible_rows = [
        row for row in complete_rows if _parse_bool_text(row.get("source_eligible")) is True
    ]
    useful_actionable_rows = [
        row
        for row in source_eligible_rows
        if _normalized(row.get("alert_usefulness")) == "useful"
        and _parse_bool_text(row.get("source_context_ok")) is True
        and _normalized(row.get("action_taken")) not in {"", "none"}
    ]
    captured_events = sum(
        1 for row in crosswalk_rows if _normalized(row.get("capture_status")) == "captured"
    )
    missed_events = sum(
        1 for row in crosswalk_rows if _normalized(row.get("capture_status")) == "missed"
    )
    observed_events = captured_events + missed_events
    capture_rate = _ratio(captured_events, observed_events)
    missed_rate = _ratio(missed_events, observed_events)
    no_observed_packets = sum(
        1
        for row in collection_rows
        if (_parse_nonnegative_int(row.get("unique_observed_event_count")) or 0) == 0
    )
    practitioner_gate = "pass" if adjudication_ready else "fail"
    usefulness_gate = "pass" if len(useful_actionable_rows) >= 2 else "caution"
    miss_rate_gate = (
        "not_applicable"
        if observed_events == 0
        else "caution"
        if missed_rate >= MISS_RATE_CAUTION_THRESHOLD
        else "pass"
    )
    return {
        "channel_name": channel_name,
        "complete_practitioner_rows": len(complete_rows),
        "source_eligible_rows": len(source_eligible_rows),
        "useful_actionable_rows": len(useful_actionable_rows),
        "captured_event_count": captured_events,
        "missed_event_count": missed_events,
        "observed_event_count": observed_events,
        "capture_rate": capture_rate,
        "missed_event_rate": missed_rate,
        "no_observed_packet_count": no_observed_packets,
        "practitioner_adjudication_gate": practitioner_gate,
        "usefulness_floor_gate": usefulness_gate,
        "miss_rate_gate": miss_rate_gate,
        "calibration_claim_status": "blocked",
        "required_next_action": _required_next_action(
            practitioner_gate,
            usefulness_gate,
            miss_rate_gate,
        ),
    }


def _evidence_dossier_row(
    collection_row: dict[str, object],
    crosswalk_rows: list[dict[str, object]],
) -> dict[str, object]:
    packet_id = str(collection_row.get("collection_packet_id") or "")
    packet_events = [
        row for row in crosswalk_rows if str(row.get("review_packet_id") or "") == packet_id
    ]
    captured_events = sum(
        1 for row in packet_events if _normalized(row.get("capture_status")) == "captured"
    )
    missed_events = sum(
        1 for row in packet_events if _normalized(row.get("capture_status")) == "missed"
    )
    observed_events = captured_events + missed_events
    action_taken = _normalized(collection_row.get("action_taken"))
    usefulness = _normalized(collection_row.get("alert_usefulness"))
    return {
        "collection_packet_id": packet_id,
        "channel_name": collection_row.get("channel_name"),
        "collection_season_id": collection_row.get("collection_season_id"),
        "alert_usefulness": usefulness,
        "outcome_confirmed": _parse_bool_text(collection_row.get("outcome_confirmed")),
        "source_context_ok": _parse_bool_text(collection_row.get("source_context_ok")),
        "action_taken": action_taken,
        "episode_count": _parse_nonnegative_int(collection_row.get("episode_count"))
        or 0,
        "observed_event_count": observed_events,
        "captured_event_count": captured_events,
        "missed_event_count": missed_events,
        "capture_rate": _ratio(captured_events, observed_events),
        "evidence_label": _evidence_label(
            usefulness,
            action_taken,
            observed_events,
            captured_events,
            missed_events,
        ),
        "notes": collection_row.get("notes"),
    }


def _error_mode_rows(
    channel_name: str,
    dossier_rows: list[dict[str, object]],
    sensitivity_row: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    empty_packets = [
        row for row in dossier_rows if int(row.get("observed_event_count") or 0) == 0
    ]
    missed_only_packets = [
        row
        for row in dossier_rows
        if int(row.get("missed_event_count") or 0) > 0
        and int(row.get("captured_event_count") or 0) == 0
    ]
    monitor_only_packets = [
        row
        for row in dossier_rows
        if _normalized(row.get("alert_usefulness")) == "useful"
        and _normalized(row.get("action_taken")) == "monitor"
    ]
    if empty_packets:
        rows.append(
            _error_mode_row(
                channel_name,
                "empty_outcome_packet",
                "medium",
                empty_packets,
                "Retained-channel packet has no observed injury events, limiting usefulness interpretation.",
            )
        )
    if missed_only_packets:
        rows.append(
            _error_mode_row(
                channel_name,
                "missed_only_packet",
                "high",
                missed_only_packets,
                "Packet had observed events but no captured events.",
            )
        )
    if _normalized(sensitivity_row.get("miss_rate_gate")) == "caution":
        rows.append(
            {
                "channel_name": channel_name,
                "error_mode": "high_miss_fraction",
                "severity": "high",
                "packet_count": len(dossier_rows),
                "affected_packet_ids": ",".join(
                    str(row.get("collection_packet_id") or "") for row in dossier_rows
                ),
                "interpretation": (
                    "Captured-event fraction is low enough that bounded calibration "
                    "research must retain explicit miss/error controls."
                ),
            }
        )
    if monitor_only_packets:
        rows.append(
            _error_mode_row(
                channel_name,
                "monitor_only_action_boundary",
                "medium",
                monitor_only_packets,
                "Practitioner-supported rows justify monitoring only, not intervention or load modification.",
            )
        )
    return rows


def _error_mode_row(
    channel_name: str,
    error_mode: str,
    severity: str,
    packet_rows: list[dict[str, object]],
    interpretation: str,
) -> dict[str, object]:
    return {
        "channel_name": channel_name,
        "error_mode": error_mode,
        "severity": severity,
        "packet_count": len(packet_rows),
        "affected_packet_ids": ",".join(
            str(row.get("collection_packet_id") or "") for row in packet_rows
        ),
        "interpretation": interpretation,
    }


def _adjudication_ready(readiness: dict[str, object]) -> bool:
    return (
        _parse_bool_text(readiness.get("independent_adjudication_required")) is False
        and readiness.get("evidence_basis") == ADJUDICATED_EVIDENCE
        and readiness.get("calibration_research_status") == READY_RESEARCH_STATUS
    )


def _overall_recommendation(
    adjudication_ready: bool,
    sensitivity_rows: list[dict[str, object]],
) -> str:
    if not adjudication_ready:
        return "complete_practitioner_adjudication_before_sensitivity_review"
    gates = {str(row.get("miss_rate_gate") or "") for row in sensitivity_rows}
    if "caution" in gates:
        return "continue_bounded_calibration_research_with_error_mode_controls"
    return "continue_bounded_calibration_research_not_claims"


def _required_next_action(
    practitioner_gate: str,
    usefulness_gate: str,
    miss_rate_gate: str,
) -> str:
    if practitioner_gate == "fail":
        return "complete_practitioner_adjudication"
    if usefulness_gate != "pass":
        return "collect_more_practitioner_reviewed_shadow_packets"
    if miss_rate_gate == "caution":
        return "bounded_calibration_research_with_error_mode_controls"
    return "bounded_calibration_research_sensitivity_review"


def _evidence_label(
    usefulness: str,
    action_taken: str,
    observed_events: int,
    captured_events: int,
    missed_events: int,
) -> str:
    if observed_events == 0:
        return "no_observed_event_evidence_gap"
    if captured_events > 0 and usefulness == "useful" and action_taken != "none":
        return "captured_practitioner_useful_monitoring"
    if captured_events == 0 and missed_events > 0 and usefulness != "useful":
        return "missed_only_practitioner_nonuseful"
    if captured_events == 0 and missed_events > 0:
        return "missed_only_practitioner_useful_context"
    return "mixed_review_context"


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
