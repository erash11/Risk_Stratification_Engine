from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
CALIBRATION_CLAIMS_BLOCKED = "not_ready_for_calibration_claims"
ADJUDICATION_CAVEAT = (
    "csv_only_artifact_review_requires_independent_practitioner_adjudication"
)
PRACTITIONER_ADJUDICATED_EVIDENCE = (
    "independent_practitioner_adjudicated_shadow_collection"
)


def build_exposure_load_shadow_calibration_readiness_review(
    collection_summary: dict[str, object],
) -> dict[str, object]:
    summary = _clean_row(collection_summary)
    channel_summary_rows = [
        _clean_row(row)
        for row in summary.get("channel_summary_rows", [])
        if isinstance(row, dict)
    ]
    pending_or_invalid_rows = _parse_nonnegative_int(
        summary.get("pending_or_invalid_rows")
    ) or 0
    complete_valid_rows = _parse_nonnegative_int(
        summary.get("complete_valid_rows")
    ) or 0
    adjudication_satisfied = (
        str(summary.get("independent_practitioner_adjudication_status") or "")
        == "satisfied"
    )
    channel_readiness_rows = [
        _channel_readiness_row(row, pending_or_invalid_rows, adjudication_satisfied)
        for row in channel_summary_rows
    ]
    calibration_research_status = _calibration_research_status(
        pending_or_invalid_rows,
        channel_readiness_rows,
    )
    return {
        "experiment_type": "exposure_load_shadow_calibration_readiness_sprint",
        "overall_recommendation": _overall_recommendation(
            pending_or_invalid_rows,
            channel_readiness_rows,
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_claim_readiness": CALIBRATION_CLAIMS_BLOCKED,
        "calibration_research_status": calibration_research_status,
        "independent_adjudication_required": not adjudication_satisfied,
        "evidence_basis": (
            PRACTITIONER_ADJUDICATED_EVIDENCE
            if adjudication_satisfied
            else ADJUDICATION_CAVEAT
        ),
        "collection_summary_recommendation": summary.get("overall_recommendation"),
        "collection_summary_calibration_readiness": summary.get(
            "calibration_readiness"
        ),
        "collection_summary_complete_valid_rows": complete_valid_rows,
        "collection_summary_pending_or_invalid_rows": pending_or_invalid_rows,
        "collection_summary_practitioner_adjudicated_rows": (
            _parse_nonnegative_int(summary.get("practitioner_adjudicated_rows"))
            or 0
        ),
        "collection_summary_csv_only_review_rows": (
            _parse_nonnegative_int(summary.get("csv_only_review_rows")) or 0
        ),
        "collection_summary_independent_practitioner_adjudication_status": (
            summary.get("independent_practitioner_adjudication_status")
        ),
        "collection_summary_useful_source_ok_actionable_rows": (
            _parse_nonnegative_int(
                summary.get("useful_source_ok_actionable_rows")
            )
            or 0
        ),
        "channel_readiness_rows": channel_readiness_rows,
        "evidence_gap_rows": _evidence_gap_rows(
            pending_or_invalid_rows,
            channel_readiness_rows,
            adjudication_satisfied,
        ),
        "interpretation_boundary": (
            "calibration-readiness review only; not calibrated probability, "
            "pilot, dashboard, or autonomous-intervention clearance"
        ),
    }


def write_exposure_load_shadow_calibration_readiness_report(
    path: Path,
    review: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Calibration Readiness Sprint",
        "",
        f"Recommendation: {review['overall_recommendation']}",
        f"Production readiness: {review['production_readiness']}",
        f"Calibration claim readiness: {review['calibration_claim_readiness']}",
        f"Calibration research status: {review['calibration_research_status']}",
        (
            "independent practitioner adjudication required: "
            f"{review['independent_adjudication_required']}"
        ),
        f"Evidence basis: {review['evidence_basis']}",
        f"Interpretation boundary: {review['interpretation_boundary']}",
        "",
        "## Collection Evidence",
        "",
        (
            "- Complete valid rows: "
            f"{review['collection_summary_complete_valid_rows']}"
        ),
        (
            "- Pending or invalid rows: "
            f"{review['collection_summary_pending_or_invalid_rows']}"
        ),
        (
            "- Useful, source-trustworthy, actionable rows: "
            f"{review['collection_summary_useful_source_ok_actionable_rows']}"
        ),
        (
            "- Practitioner-adjudicated rows: "
            f"{review['collection_summary_practitioner_adjudicated_rows']}"
        ),
        (
            "- CSV-only review rows: "
            f"{review['collection_summary_csv_only_review_rows']}"
        ),
        (
            "- Independent practitioner adjudication status: "
            f"{review['collection_summary_independent_practitioner_adjudication_status']}"
        ),
        "",
        "## Channel Readiness",
        "",
        (
            "| Channel | Collection gate | Complete rows | Useful/source OK/actionable | "
            "Readiness status | Required next action |"
        ),
        "|---|---|---:|---:|---|---|",
    ]
    for row in review.get("channel_readiness_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['collection_gate']} | "
            f"{row['complete_valid_rows']} | "
            f"{row['useful_source_ok_actionable_rows']} | "
            f"{row['readiness_status']} | "
            f"{row['required_next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Gaps",
            "",
            "| Gate | Status | Requirement |",
            "|---|---|---|",
        ]
    )
    for row in review.get("evidence_gap_rows", []):
        lines.append(
            "| "
            f"{row['gate_name']} | "
            f"{row['gate_status']} | "
            f"{row['requirement']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This sprint can identify retained channels as calibration "
                "research candidates, but this is not "
                "calibration claims, probability-facing output, pilot readiness, "
                "or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_calibration_readiness_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _channel_readiness_row(
    row: dict[str, object],
    pending_or_invalid_rows: int,
    adjudication_satisfied: bool,
) -> dict[str, object]:
    collection_gate = str(row.get("calibration_review_gate") or "")
    useful_source_ok_actionable_rows = _parse_nonnegative_int(
        row.get("useful_source_ok_actionable_rows")
    ) or 0
    if pending_or_invalid_rows:
        readiness_status = "not_ready_collection_incomplete"
        required_next_action = "complete_shadow_collection"
    elif collection_gate != "ready_for_calibration_readiness_review":
        readiness_status = "not_ready_revise_or_continue_collection"
        required_next_action = "revise_or_continue_shadow_collection"
    elif useful_source_ok_actionable_rows <= 0:
        readiness_status = "not_ready_no_useful_actionable_evidence"
        required_next_action = "revise_channel_before_calibration_research"
    elif adjudication_satisfied:
        readiness_status = "calibration_research_candidate_practitioner_adjudicated"
        required_next_action = "bounded_calibration_research_sensitivity_review"
    else:
        readiness_status = (
            "candidate_pending_independent_practitioner_adjudication"
        )
        required_next_action = "independent_practitioner_adjudication"
    return {
        "channel_name": row.get("channel_name"),
        "collection_gate": collection_gate,
        "minimum_required_packets": _parse_nonnegative_int(
            row.get("minimum_required_packets")
        )
        or 0,
        "complete_valid_rows": _parse_nonnegative_int(
            row.get("complete_valid_rows")
        )
        or 0,
        "complete_source_eligible_rows": _parse_nonnegative_int(
            row.get("complete_source_eligible_rows")
        )
        or 0,
        "useful_source_ok_actionable_rows": useful_source_ok_actionable_rows,
        "practitioner_adjudicated_rows": _parse_nonnegative_int(
            row.get("practitioner_adjudicated_rows")
        )
        or 0,
        "csv_only_review_rows": _parse_nonnegative_int(
            row.get("csv_only_review_rows")
        )
        or 0,
        "readiness_status": readiness_status,
        "required_next_action": required_next_action,
        "calibration_claim_status": "blocked",
        "rationale": _channel_rationale(row, readiness_status),
    }


def _channel_rationale(
    row: dict[str, object],
    readiness_status: str,
) -> str:
    useful = _parse_nonnegative_int(row.get("useful_source_ok_actionable_rows")) or 0
    complete = _parse_nonnegative_int(row.get("complete_valid_rows")) or 0
    if readiness_status == "candidate_pending_independent_practitioner_adjudication":
        return (
            f"{complete} complete rows and {useful} useful/source-trustworthy/"
            "actionable rows support calibration research framing only; "
            "independent practitioner adjudication remains required."
        )
    if readiness_status == "calibration_research_candidate_practitioner_adjudicated":
        return (
            f"{complete} complete practitioner-adjudicated rows and {useful} "
            "useful/source-trustworthy/actionable rows support bounded "
            "calibration research only; probability-facing claims remain blocked."
        )
    if readiness_status == "not_ready_collection_incomplete":
        return "Collection summary still has pending or invalid rows."
    if readiness_status == "not_ready_no_useful_actionable_evidence":
        return "Channel has no useful/source-trustworthy/actionable rows."
    return "Collection gate is not ready for calibration-readiness review."


def _evidence_gap_rows(
    pending_or_invalid_rows: int,
    channel_readiness_rows: list[dict[str, object]],
    adjudication_satisfied: bool,
) -> list[dict[str, object]]:
    has_candidate = any(
        row.get("readiness_status")
        == "candidate_pending_independent_practitioner_adjudication"
        for row in channel_readiness_rows
    )
    adjudication_requirement = (
        "independent practitioner/source-context adjudication is complete for "
        "the retained-channel rows"
        if adjudication_satisfied
        else "complete independent practitioner/source-context adjudication for "
        "the retained-channel rows before calibration research can advance"
        if has_candidate
        else "complete or revise retained-channel shadow collection before "
        "independent adjudication can support calibration research"
    )
    collection_status = "required" if pending_or_invalid_rows else "satisfied"
    return [
        {
            "gate_name": "shadow_collection_completion",
            "gate_status": collection_status,
            "requirement": (
                "complete retained-channel shadow collection rows with no "
                "pending or invalid rows"
            ),
        },
        {
            "gate_name": "independent_practitioner_adjudication",
            "gate_status": "satisfied" if adjudication_satisfied else "required",
            "requirement": adjudication_requirement,
        },
        {
            "gate_name": "probability_facing_outputs",
            "gate_status": "blocked",
            "requirement": (
                "requires calibration research evidence and approved operating "
                "review; CSV-only evidence is insufficient"
            ),
        },
        {
            "gate_name": "pilot_dashboard_readiness",
            "gate_status": "blocked",
            "requirement": (
                "requires calibrated probability evidence, practitioner "
                "adjudication, and operational approval"
            ),
        },
        {
            "gate_name": "autonomous_intervention",
            "gate_status": "blocked",
            "requirement": (
                "not supported by current research-shadow evidence"
            ),
        },
    ]


def _overall_recommendation(
    pending_or_invalid_rows: int,
    channel_readiness_rows: list[dict[str, object]],
) -> str:
    if pending_or_invalid_rows:
        return "complete_shadow_collection_before_calibration_readiness_review"
    if any(
        row.get("readiness_status")
        == "calibration_research_candidate_practitioner_adjudicated"
        for row in channel_readiness_rows
    ):
        return "advance_to_bounded_calibration_research_not_claims"
    if any(
        row.get("readiness_status")
        == "candidate_pending_independent_practitioner_adjudication"
        for row in channel_readiness_rows
    ):
        return (
            "defer_calibration_claims_pending_independent_practitioner_adjudication"
        )
    return "revise_retained_channels_before_calibration_readiness_review"


def _calibration_research_status(
    pending_or_invalid_rows: int,
    channel_readiness_rows: list[dict[str, object]],
) -> str:
    if pending_or_invalid_rows:
        return "not_ready_collection_incomplete"
    if any(
        row.get("readiness_status")
        == "calibration_research_candidate_practitioner_adjudicated"
        for row in channel_readiness_rows
    ):
        return "ready_for_bounded_calibration_research_not_claims"
    if any(
        row.get("readiness_status")
        == "candidate_pending_independent_practitioner_adjudication"
        for row in channel_readiness_rows
    ):
        return "research_candidate_pending_independent_practitioner_adjudication"
    return "not_ready_revise_or_continue_collection"


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
    if str(value).strip() not in {str(number), f"{number}.0"}:
        return None
    if number < 0:
        return None
    return number
