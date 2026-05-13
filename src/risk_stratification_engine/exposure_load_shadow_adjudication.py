from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
REQUIRED_ADJUDICATION_FIELDS = (
    "reviewer_id",
    "review_date",
    "alert_usefulness",
    "outcome_confirmed",
    "source_context_ok",
    "action_taken",
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


def build_exposure_load_shadow_adjudication_package(
    shadow_replay: dict[str, object],
) -> dict[str, object]:
    review_packets = [
        _clean_row(row)
        for row in shadow_replay.get("review_packet_rows", [])
        if isinstance(row, dict)
    ]
    schema_rows = _schema_rows()
    adjudication_template_rows = [
        _template_row(row) for row in review_packets
    ]
    completion_check_rows = [
        _completion_check_row(row) for row in adjudication_template_rows
    ]
    return {
        "experiment_type": "exposure_load_shadow_adjudication_sprint",
        "overall_recommendation": _overall_recommendation(
            adjudication_template_rows
        ),
        "production_readiness": shadow_replay.get(
            "production_readiness",
            PRODUCTION_BLOCKED,
        ),
        "schema_rows": schema_rows,
        "adjudication_template_rows": adjudication_template_rows,
        "completion_check_rows": completion_check_rows,
        "collection_boundary": (
            "research adjudication collection only; not pilot, dashboard, probability-facing deployment, or autonomous intervention"
        ),
        "next_sprint": (
            "collect adjudication values prospectively, then evaluate completeness and usefulness before calibration updates"
        ),
    }


def write_exposure_load_shadow_adjudication_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    template_rows = summary.get("adjudication_template_rows", [])
    completion_rows = summary.get("completion_check_rows", [])
    pending_rows = [
        row
        for row in completion_rows
        if row.get("completion_status") == "pending_required_fields"
    ]
    lines = [
        "# Exposure Load Shadow Adjudication Package Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Collection boundary: {summary['collection_boundary']}",
        "",
        "## Collection Template",
        "",
        f"- Review packet rows: {len(template_rows)}",
        f"- Required fields: {len(REQUIRED_ADJUDICATION_FIELDS)}",
        f"- Pending required-field rows: {len(pending_rows)}",
        "",
        "## Required Fields",
        "",
        "| Field | Type | Required | Allowed values |",
        "|---|---|---|---|",
    ]
    for row in summary.get("schema_rows", []):
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
                "The review packets now have a prospective collection template "
                "and completion checks. Filling this template is the next "
                "evidence step; this package is not pilot or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_exposure_load_shadow_adjudication_summary(
    adjudication_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = [_clean_row(row) for row in adjudication_rows]
    validation_rows = [_validation_row(row) for row in rows]
    complete_packet_ids = {
        row["review_packet_id"]
        for row in validation_rows
        if row["completion_status"] == "complete_valid"
    }
    complete_rows = [
        row for row in rows if row.get("review_packet_id") in complete_packet_ids
    ]
    channel_summary_rows = _channel_summary_rows(complete_rows)
    useful_source_ok_actionable_rows = sum(
        int(row["useful_source_ok_actionable_rows"])
        for row in channel_summary_rows
    )
    pending_or_invalid_rows = len(rows) - len(complete_rows)
    return {
        "experiment_type": "exposure_load_shadow_adjudication_summary",
        "overall_recommendation": _summary_recommendation(
            pending_or_invalid_rows,
            useful_source_ok_actionable_rows,
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "total_rows": len(rows),
        "complete_valid_rows": len(complete_rows),
        "pending_or_invalid_rows": pending_or_invalid_rows,
        "useful_source_ok_actionable_rows": useful_source_ok_actionable_rows,
        "validation_rows": validation_rows,
        "channel_summary_rows": channel_summary_rows,
        "interpretation_boundary": (
            "adjudication summary for shadow monitoring only; not probability calibration or dashboard clearance"
        ),
    }


def build_exposure_load_shadow_adjudication_decision_package(
    adjudication_summary: dict[str, object],
) -> dict[str, object]:
    channel_decision_rows = [
        _channel_decision_row(row)
        for row in adjudication_summary.get("channel_summary_rows", [])
        if isinstance(row, dict)
    ]
    continued_channels = [
        row["channel_name"]
        for row in channel_decision_rows
        if row["channel_decision"] == "continue_shadow_monitoring"
    ]
    paused_channels = [
        row["channel_name"]
        for row in channel_decision_rows
        if row["channel_decision"] == "pause_or_revise_before_more_collection"
    ]
    limited_channels = [
        row["channel_name"]
        for row in channel_decision_rows
        if row["channel_decision"] == "continue_limited_evidence_collection"
    ]
    pending_or_invalid_rows = int(
        adjudication_summary.get("pending_or_invalid_rows", 0) or 0
    )
    return {
        "experiment_type": "exposure_load_shadow_adjudication_decision_sprint",
        "overall_recommendation": _decision_recommendation(
            pending_or_invalid_rows,
            continued_channels,
            paused_channels,
            limited_channels,
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "continued_shadow_channels": continued_channels,
        "paused_or_revision_channels": paused_channels,
        "limited_evidence_channels": limited_channels,
        "channel_decision_rows": channel_decision_rows,
        "decision_boundary": (
            "shadow monitoring decision only; not probability calibration or dashboard clearance"
        ),
        "next_step": (
            "continue shadow collection for retained channels and revise paused channels before probability-facing work"
        ),
    }


def write_exposure_load_shadow_adjudication_decision_report(
    path: Path,
    decision: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Adjudication Decision Sprint",
        "",
        f"Recommendation: {decision['overall_recommendation']}",
        f"Production readiness: {decision['production_readiness']}",
        f"Decision boundary: {decision['decision_boundary']}",
        "",
        "## Channel Decisions",
        "",
        "| Channel | Decision | Complete rows | Useful/actionable rows | Rationale |",
        "|---|---|---:|---:|---|",
    ]
    for row in decision.get("channel_decision_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['channel_decision']} | "
            f"{row['complete_valid_rows']} | "
            f"{row['useful_source_ok_actionable_rows']} | "
            f"{row['decision_rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This package converts completed adjudication evidence into "
                "shadow-monitoring channel decisions. It is not probability "
                "calibration or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_exposure_load_shadow_adjudication_summary_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Adjudication Summary Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Interpretation boundary: {summary['interpretation_boundary']}",
        "",
        "## Completion",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- Complete valid rows: {summary['complete_valid_rows']}",
        f"- Pending or invalid rows: {summary['pending_or_invalid_rows']}",
        (
            "- Useful, source-trustworthy, actionable rows: "
            f"{summary['useful_source_ok_actionable_rows']}"
        ),
        "",
        "## Channel Summary",
        "",
        "| Channel | Rows | Useful | Source OK | Actionable | Useful/source OK/actionable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("channel_summary_rows", []):
        lines.append(
            "| "
            f"{row['channel_name']} | "
            f"{row['complete_valid_rows']} | "
            f"{row['useful_rows']} | "
            f"{row['source_context_ok_rows']} | "
            f"{row['actionable_rows']} | "
            f"{row['useful_source_ok_actionable_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This summary can identify whether completed shadow packets look "
                "useful, source-trustworthy, and actionable. It is not probability "
                "calibration or dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_adjudication_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _schema_rows() -> list[dict[str, object]]:
    return [
        _schema_row("review_packet_id", "string", True, ""),
        _schema_row("reviewer_id", "string", True, ""),
        _schema_row("review_date", "date", True, "YYYY-MM-DD"),
        _schema_row(
            "alert_usefulness",
            "category",
            True,
            "useful|noisy|misleading|unclear",
        ),
        _schema_row("outcome_confirmed", "boolean", True, "true|false"),
        _schema_row("source_context_ok", "boolean", True, "true|false"),
        _schema_row(
            "action_taken",
            "category",
            True,
            "none|monitor|communicate|modify_load|clinical_review|other",
        ),
        _schema_row("notes", "string", False, ""),
    ]


def _validation_row(row: dict[str, object]) -> dict[str, object]:
    missing = [
        field
        for field in REQUIRED_ADJUDICATION_FIELDS
        if _is_blank(row.get(field))
    ]
    invalid = _invalid_fields(row, missing)
    return {
        "review_packet_id": row.get("review_packet_id"),
        "channel_name": row.get("channel_name"),
        "test_season_id": row.get("test_season_id"),
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


def _invalid_fields(
    row: dict[str, object],
    missing: list[str],
) -> list[str]:
    missing_fields = set(missing)
    invalid: list[str] = []
    if "review_date" not in missing_fields and not _is_valid_iso_date(
        row.get("review_date")
    ):
        invalid.append("review_date")
    if "alert_usefulness" not in missing_fields and _normalized(
        row.get("alert_usefulness")
    ) not in ALLOWED_ALERT_USEFULNESS:
        invalid.append("alert_usefulness")
    if "outcome_confirmed" not in missing_fields and _parse_bool_text(
        row.get("outcome_confirmed")
    ) is None:
        invalid.append("outcome_confirmed")
    if "source_context_ok" not in missing_fields and _parse_bool_text(
        row.get("source_context_ok")
    ) is None:
        invalid.append("source_context_ok")
    if "action_taken" not in missing_fields and _normalized(
        row.get("action_taken")
    ) not in ALLOWED_ACTIONS:
        invalid.append("action_taken")
    return invalid


def _channel_summary_rows(
    complete_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    channel_names = sorted(
        {
            str(row.get("channel_name") or "unknown")
            for row in complete_rows
        }
    )
    return [
        _channel_summary_row(
            channel_name,
            [
                row
                for row in complete_rows
                if str(row.get("channel_name") or "unknown") == channel_name
            ],
        )
        for channel_name in channel_names
    ]


def _channel_summary_row(
    channel_name: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    useful_rows = [
        row
        for row in rows
        if _normalized(row.get("alert_usefulness")) == "useful"
    ]
    source_ok_rows = [
        row for row in rows if _parse_bool_text(row.get("source_context_ok")) is True
    ]
    actionable_rows = [
        row
        for row in rows
        if _normalized(row.get("action_taken")) not in {"", "none"}
    ]
    useful_source_ok_actionable = [
        row
        for row in rows
        if _normalized(row.get("alert_usefulness")) == "useful"
        and _parse_bool_text(row.get("source_context_ok")) is True
        and _normalized(row.get("action_taken")) not in {"", "none"}
    ]
    return {
        "channel_name": channel_name,
        "complete_valid_rows": len(rows),
        "useful_rows": len(useful_rows),
        "source_context_ok_rows": len(source_ok_rows),
        "actionable_rows": len(actionable_rows),
        "useful_source_ok_actionable_rows": len(useful_source_ok_actionable),
    }


def _channel_decision_row(row: dict[str, object]) -> dict[str, object]:
    complete_valid_rows = int(row.get("complete_valid_rows", 0) or 0)
    useful_actionable = int(
        row.get("useful_source_ok_actionable_rows", 0) or 0
    )
    channel_decision = _channel_decision(complete_valid_rows, useful_actionable)
    return {
        "channel_name": str(row.get("channel_name") or "unknown"),
        "complete_valid_rows": complete_valid_rows,
        "useful_rows": int(row.get("useful_rows", 0) or 0),
        "source_context_ok_rows": int(row.get("source_context_ok_rows", 0) or 0),
        "actionable_rows": int(row.get("actionable_rows", 0) or 0),
        "useful_source_ok_actionable_rows": useful_actionable,
        "channel_decision": channel_decision,
        "decision_rationale": _channel_decision_rationale(
            complete_valid_rows,
            useful_actionable,
        ),
    }


def _channel_decision(
    complete_valid_rows: int,
    useful_actionable: int,
) -> str:
    if useful_actionable >= 2:
        return "continue_shadow_monitoring"
    if useful_actionable == 1:
        return "continue_limited_evidence_collection"
    if complete_valid_rows > 0:
        return "pause_or_revise_before_more_collection"
    return "complete_adjudication_before_channel_decision"


def _channel_decision_rationale(
    complete_valid_rows: int,
    useful_actionable: int,
) -> str:
    if useful_actionable >= 2:
        return "multiple completed packets were useful, source-trustworthy, and actionable"
    if useful_actionable == 1:
        return "only one completed packet was useful, source-trustworthy, and actionable"
    if complete_valid_rows > 0:
        return "completed packets did not show useful, source-trustworthy, actionable evidence"
    return "no complete valid adjudication rows are available"


def _decision_recommendation(
    pending_or_invalid_rows: int,
    continued_channels: list[str],
    paused_channels: list[str],
    limited_channels: list[str],
) -> str:
    if pending_or_invalid_rows:
        return "complete_adjudication_before_channel_decisions"
    if continued_channels and (paused_channels or limited_channels):
        return "continue_shadow_monitoring_with_channel_revisions"
    if continued_channels:
        return "continue_shadow_monitoring_all_reviewed_channels"
    return "pause_shadow_package_and_revise_before_more_collection"


def _summary_recommendation(
    pending_or_invalid_rows: int,
    useful_source_ok_actionable_rows: int,
) -> str:
    if pending_or_invalid_rows:
        return "complete_adjudication_required_before_operational_summary"
    if useful_source_ok_actionable_rows:
        return "continue_shadow_monitoring_adjudication_collection"
    return "rollback_or_revise_shadow_alert_package_before_more_collection"


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


def _template_row(packet: dict[str, object]) -> dict[str, object]:
    row = {
        "review_packet_id": packet.get("review_packet_id"),
        "channel_name": packet.get("channel_name"),
        "test_season_id": packet.get("test_season_id"),
        "minimum_review_unit": packet.get("minimum_review_unit"),
        "required_evidence": packet.get("required_evidence"),
        "episode_count": packet.get("episode_count"),
        "unique_observed_event_count": packet.get("unique_observed_event_count"),
        "unique_captured_event_count": packet.get("unique_captured_event_count"),
        "missed_event_count": packet.get("missed_event_count"),
        "episodes_per_athlete_season": packet.get("episodes_per_athlete_season"),
        "review_packet_status": packet.get("review_packet_status"),
        "reviewer_id": "",
        "review_date": "",
        "alert_usefulness": "",
        "outcome_confirmed": "",
        "source_context_ok": "",
        "action_taken": "",
        "notes": "",
        "adjudication_status": "pending_review",
    }
    return _clean_row(row)


def _completion_check_row(row: dict[str, object]) -> dict[str, object]:
    missing = [
        field
        for field in REQUIRED_ADJUDICATION_FIELDS
        if str(row.get(field) or "").strip() == ""
    ]
    return {
        "review_packet_id": row.get("review_packet_id"),
        "channel_name": row.get("channel_name"),
        "test_season_id": row.get("test_season_id"),
        "missing_required_fields": ",".join(missing),
        "missing_required_field_count": len(missing),
        "completion_status": (
            "complete"
            if not missing
            else "pending_required_fields"
        ),
    }


def _overall_recommendation(
    adjudication_template_rows: list[dict[str, object]],
) -> str:
    if adjudication_template_rows:
        return "adjudication_template_ready_for_prospective_collection"
    return "complete_replay_packets_before_adjudication_collection"


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if not isinstance(value, str) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _normalized(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_bool_text(value: object) -> bool | None:
    normalized = _normalized(value)
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _is_valid_iso_date(value: object) -> bool:
    if _is_blank(value):
        return False
    parsed = pd.to_datetime(str(value), format="%Y-%m-%d", errors="coerce")
    return not pd.isna(parsed)


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
