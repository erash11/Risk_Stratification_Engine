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
RETAINED_COLLECTION_CHANNELS = ("broad_30d", "severity_14d")
REVIEWER_JUDGMENT_FIELDS = (
    "alert_usefulness",
    "outcome_confirmed",
    "source_context_ok",
    "action_taken",
    "reviewer_id",
    "review_date",
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


def build_exposure_load_shadow_collection_evidence_prefill(
    review_packet_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = [_clean_row(row) for row in review_packet_rows]
    retained_rows = [
        row for row in rows if _is_retained_source_eligible_review_packet(row)
    ]
    excluded_rows = [_prefill_excluded_row(row) for row in rows if row not in retained_rows]
    prefilled_rows = [
        _evidence_prefilled_collection_row(row, sequence)
        for channel_name in RETAINED_COLLECTION_CHANNELS
        for sequence, row in enumerate(
            [
                row
                for row in retained_rows
                if str(row.get("channel_name")) == channel_name
            ],
            start=1,
        )
    ]
    validation_rows = [_collection_validation_row(row) for row in prefilled_rows]
    return {
        "experiment_type": (
            "exposure_load_shadow_collection_evidence_prefill_sprint"
        ),
        "overall_recommendation": _evidence_prefill_recommendation(
            prefilled_rows
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_readiness": "not_ready_for_calibration_claims",
        "prefilled_row_count": len(prefilled_rows),
        "excluded_row_count": len(excluded_rows),
        "reviewer_required_field_count": len(REVIEWER_JUDGMENT_FIELDS),
        "retained_channels": list(RETAINED_COLLECTION_CHANNELS),
        "prefilled_collection_rows": prefilled_rows,
        "validation_rows": validation_rows,
        "excluded_rows": excluded_rows,
        "prefill_boundary": (
            "replay-derived evidence prefill only; reviewer judgment fields remain blank"
        ),
    }


def write_exposure_load_shadow_collection_evidence_prefill_report(
    path: Path,
    prefill: dict[str, object],
) -> None:
    pending_rows = [
        row
        for row in prefill.get("validation_rows", [])
        if row.get("completion_status") != "complete_valid"
    ]
    lines = [
        "# Exposure Load Shadow Collection Evidence Prefill Sprint",
        "",
        f"Recommendation: {prefill['overall_recommendation']}",
        f"Production readiness: {prefill['production_readiness']}",
        f"Calibration readiness: {prefill['calibration_readiness']}",
        f"Boundary: {prefill['prefill_boundary']}",
        "",
        "## Prefill Summary",
        "",
        f"- Prefilled retained-channel rows: {prefill['prefilled_row_count']}",
        f"- Excluded source/channel rows: {prefill['excluded_row_count']}",
        f"- Reviewer fields still required: {prefill['reviewer_required_field_count']}",
        f"- Rows still pending validation: {len(pending_rows)}",
        "",
        "## Reviewer Fields Still Required",
        "",
    ]
    for field in REVIEWER_JUDGMENT_FIELDS:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This artifact fills fields already available from shadow replay "
                "evidence: season, source eligibility, episode counts, and "
                "observed/captured event counts. It does not fill reviewer "
                "judgment fields or support calibration/product readiness claims."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_exposure_load_shadow_collection_packet_workflow(
    collection_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = [
        _clean_row(row)
        for row in collection_rows
        if not _is_blank(row.get("collection_packet_id"))
    ]
    packet_manifest_rows = [_packet_manifest_row(row) for row in rows]
    packet_checklist_rows = [
        checklist_row
        for row in packet_manifest_rows
        for checklist_row in _packet_checklist_rows(row)
    ]
    packet_audit_trail_rows = [
        _packet_audit_trail_row(row) for row in packet_manifest_rows
    ]
    packet_documents = [
        {
            "collection_packet_id": row["collection_packet_id"],
            "packet_filename": row["packet_filename"],
            "content": _packet_document(row),
        }
        for row in packet_manifest_rows
    ]
    return {
        "experiment_type": (
            "exposure_load_shadow_collection_packet_workflow_sprint"
        ),
        "overall_recommendation": _packet_workflow_recommendation(
            packet_manifest_rows
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "calibration_readiness": "not_ready_for_calibration_claims",
        "packet_count": len(packet_manifest_rows),
        "packet_manifest_rows": packet_manifest_rows,
        "packet_checklist_rows": packet_checklist_rows,
        "packet_audit_trail_rows": packet_audit_trail_rows,
        "packet_documents": packet_documents,
        "review_boundary": (
            "reviewer packet workflow only; not probability calibration or dashboard clearance"
        ),
        "deidentification_rule": (
            "use collection packet IDs and de-identified reviewer notes only; "
            "do not enter identifiable athlete information"
        ),
    }


def write_exposure_load_shadow_collection_reviewer_instructions(
    path: Path,
    workflow: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Collection Reviewer Instructions",
        "",
        f"Recommendation: {workflow['overall_recommendation']}",
        f"Production readiness: {workflow['production_readiness']}",
        f"Calibration readiness: {workflow['calibration_readiness']}",
        "",
        "## Boundary",
        "",
        str(workflow["review_boundary"]),
        "",
        "Do not enter identifiable athlete information in packet files, notes, or CSV fields.",
        "",
        "## Review Steps",
        "",
        "1. Open the packet markdown file listed in the manifest.",
        "2. Confirm the collection unit and source rule before reviewing evidence.",
        "3. Record only de-identified season/window, eligibility, alert, outcome, and action values in the collection CSV.",
        "4. Use `source_eligible=false` when the packet fails the source rule, then document the de-identified reason in notes.",
        "5. Leave probability, calibration, pilot, dashboard, and intervention claims out of the review.",
        "",
        "## Required CSV Fields",
        "",
    ]
    for field in REQUIRED_COLLECTION_FIELDS:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            "Optional but recommended fields: `outcome_confirmed`, "
            "`source_context_ok`, `action_taken`, and de-identified `notes`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_exposure_load_shadow_collection_packet_workflow_report(
    path: Path,
    workflow: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Collection Packet Workflow Sprint",
        "",
        f"Recommendation: {workflow['overall_recommendation']}",
        f"Production readiness: {workflow['production_readiness']}",
        f"Calibration readiness: {workflow['calibration_readiness']}",
        f"Review boundary: {workflow['review_boundary']}",
        "",
        "## Reviewer Packet Workflow",
        "",
        f"- Reviewer packet count: {workflow['packet_count']}",
        f"- Manifest rows: {len(workflow['packet_manifest_rows'])}",
        f"- Checklist rows: {len(workflow['packet_checklist_rows'])}",
        f"- Audit trail seed rows: {len(workflow['packet_audit_trail_rows'])}",
        "",
        "## Interpretation",
        "",
        (
            "This sprint prepares reviewer packet materials and audit-trail "
            "seeds for retained-channel prospective shadow collection. It does "
            "not complete the evidence rows and does not support probability "
            "calibration or dashboard clearance."
        ),
    ]
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


def _is_retained_source_eligible_review_packet(row: dict[str, object]) -> bool:
    status = _review_packet_status(row)
    return (
        str(row.get("channel_name")) in RETAINED_COLLECTION_CHANNELS
        and _review_packet_source_eligible(row) is True
        and status == "ready_for_research_adjudication"
    )


def _evidence_prefilled_collection_row(
    review_packet: dict[str, object],
    packet_sequence: int,
) -> dict[str, object]:
    season_id = str(review_packet.get("test_season_id") or "")
    season_start, season_end = _season_window_dates(season_id)
    return {
        "collection_packet_id": review_packet.get("review_packet_id"),
        "channel_name": review_packet.get("channel_name"),
        "packet_sequence": packet_sequence,
        "collection_unit": review_packet.get(
            "minimum_review_unit",
            "complete source-eligible athlete-season",
        ),
        "evidence_gate": "historical_replay_evidence_prefill_before_reviewer_judgment",
        "source_rule": "source_eligible=true and replay_status=ready_for_research_adjudication",
        "collection_season_id": season_id,
        "packet_start_date": season_start,
        "packet_end_date": season_end,
        "source_eligible": _review_packet_source_eligible(review_packet),
        "episode_count": _parse_nonnegative_int(
            review_packet.get("episode_count")
        ),
        "unique_observed_event_count": _parse_nonnegative_int(
            review_packet.get("unique_observed_event_count")
        ),
        "unique_captured_event_count": _parse_nonnegative_int(
            review_packet.get("unique_captured_event_count")
        ),
        "alert_usefulness": "",
        "outcome_confirmed": "",
        "source_context_ok": "",
        "action_taken": "",
        "reviewer_id": "",
        "review_date": "",
        "notes": "",
        "collection_status": "pending_reviewer_judgment",
        "evidence_source_packet_id": review_packet.get("review_packet_id"),
        "evidence_source": "exposure_load_shadow_review_packets",
    }


def _prefill_excluded_row(row: dict[str, object]) -> dict[str, object]:
    channel_name = str(row.get("channel_name") or "")
    source_eligible = _review_packet_source_eligible(row)
    status = _review_packet_status(row)
    reasons: list[str] = []
    if channel_name not in RETAINED_COLLECTION_CHANNELS:
        reasons.append("channel_not_retained")
    if source_eligible is not True:
        reasons.append("source_ineligible")
    if status != "ready_for_research_adjudication":
        reasons.append("not_ready_for_research_adjudication")
    return {
        "review_packet_id": row.get("review_packet_id"),
        "channel_name": channel_name,
        "test_season_id": row.get("test_season_id"),
        "source_eligible": source_eligible,
        "replay_status": row.get("replay_status") or row.get("review_packet_status"),
        "exclusion_reason": ",".join(reasons),
    }


def _review_packet_status(row: dict[str, object]) -> str:
    return _normalized(row.get("replay_status") or row.get("review_packet_status"))


def _review_packet_source_eligible(row: dict[str, object]) -> bool | None:
    source_eligible = _parse_bool_text(row.get("source_eligible"))
    if source_eligible is not None:
        return source_eligible
    if _review_packet_status(row) == "ready_for_research_adjudication":
        return True
    return None


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


def _season_window_dates(season_id: str) -> tuple[str, str]:
    if "-" not in season_id:
        return "", ""
    start_year_text, end_year_text = season_id.split("-", maxsplit=1)
    try:
        start_year = int(start_year_text)
        end_year = int(end_year_text)
    except ValueError:
        return "", ""
    return f"{start_year:04d}-07-01", f"{end_year:04d}-06-30"


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


def _packet_manifest_row(row: dict[str, object]) -> dict[str, object]:
    collection_packet_id = str(row.get("collection_packet_id") or "unknown")
    return {
        "collection_packet_id": collection_packet_id,
        "channel_name": row.get("channel_name"),
        "packet_sequence": row.get("packet_sequence"),
        "packet_filename": f"review_packets/{collection_packet_id}.md",
        "collection_unit": row.get("collection_unit"),
        "evidence_gate": row.get("evidence_gate"),
        "source_rule": row.get("source_rule"),
        "collection_status": row.get("collection_status"),
        "packet_status": "ready_for_reviewer_evidence_collection",
        "required_output": (
            "complete the matching collection CSV row with prospective "
            "source-eligible review evidence"
        ),
    }


def _packet_checklist_rows(
    manifest_row: dict[str, object],
) -> list[dict[str, object]]:
    checklist = [
        (
            "confirm_source_eligibility",
            "Confirm source eligibility under the packet source rule.",
            True,
        ),
        (
            "record_packet_window",
            "Record de-identified season, packet start date, and packet end date.",
            True,
        ),
        (
            "record_alert_episode_counts",
            "Record alert episode count and observed/captured event counts.",
            True,
        ),
        (
            "record_alert_usefulness",
            "Classify alert usefulness as useful, noisy, misleading, or unclear.",
            True,
        ),
        (
            "record_source_context_ok",
            "Record whether source context is trustworthy for this packet.",
            False,
        ),
        (
            "record_action_taken",
            "Record de-identified action category, including none when no action occurred.",
            False,
        ),
        (
            "preserve_deidentified_notes",
            "Write concise notes without identifiable athlete information.",
            False,
        ),
    ]
    return [
        {
            "collection_packet_id": manifest_row["collection_packet_id"],
            "channel_name": manifest_row["channel_name"],
            "checklist_order": index,
            "checklist_item": item,
            "completion_rule": rule,
            "required_for_completion": required,
        }
        for index, (item, rule, required) in enumerate(checklist, start=1)
    ]


def _packet_audit_trail_row(
    manifest_row: dict[str, object],
) -> dict[str, object]:
    return {
        "collection_packet_id": manifest_row["collection_packet_id"],
        "channel_name": manifest_row["channel_name"],
        "audit_event": "packet_created_for_review",
        "evidence_status": "not_collected",
        "source_evidence_attached": False,
        "reviewer_id": "",
        "review_date": "",
        "audit_notes": (
            "Seed row; update after reviewer evidence is collected in the "
            "collection template."
        ),
    }


def _packet_document(manifest_row: dict[str, object]) -> str:
    collection_packet_id = manifest_row["collection_packet_id"]
    lines = [
        f"# Collection Packet: {collection_packet_id}",
        "",
        f"Channel: {manifest_row['channel_name']}",
        f"Packet sequence: {manifest_row['packet_sequence']}",
        f"Collection unit: {manifest_row['collection_unit']}",
        f"Evidence gate: {manifest_row['evidence_gate']}",
        f"Source rule: {manifest_row['source_rule']}",
        "",
        "## Review Boundary",
        "",
        (
            "This packet supports prospective shadow collection only. It is not "
            "probability calibration or dashboard clearance."
        ),
        "",
        "## Evidence To Record In Collection CSV",
        "",
        "- `collection_season_id`: de-identified season label",
        "- `packet_start_date`: YYYY-MM-DD",
        "- `packet_end_date`: YYYY-MM-DD",
        "- `source_eligible`: true or false",
        "- `episode_count`: non-negative integer",
        "- `unique_observed_event_count`: non-negative integer",
        "- `unique_captured_event_count`: non-negative integer",
        "- `alert_usefulness`: useful, noisy, misleading, or unclear",
        "- `outcome_confirmed`: true or false when available",
        "- `source_context_ok`: true or false when available",
        "- `action_taken`: none, monitor, communicate, modify_load, clinical_review, or other",
        "- `reviewer_id`: reviewer code, not a full name if not needed",
        "- `review_date`: YYYY-MM-DD",
        "- `notes`: de-identified notes only",
        "",
        "## Checklist",
        "",
    ]
    for row in _packet_checklist_rows(manifest_row):
        lines.append(f"- [ ] {row['completion_rule']}")
    lines.extend(
        [
            "",
            "## De-identification",
            "",
            "Do not enter identifiable athlete information in this packet.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _packet_workflow_recommendation(
    packet_manifest_rows: list[dict[str, object]],
) -> str:
    if packet_manifest_rows:
        return "prepare_retained_channel_reviewer_packets"
    return "revise_collection_template_before_packet_workflow"


def _evidence_prefill_recommendation(
    prefilled_rows: list[dict[str, object]],
) -> str:
    if prefilled_rows:
        return "review_prefilled_retained_channel_shadow_collection_rows"
    return "no_retained_source_eligible_replay_packets_available"


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
