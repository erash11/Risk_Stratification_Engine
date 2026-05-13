from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


EXPOSURE_LOAD_FEATURE_SET = "graph_plus_coverage_exposure_load"
PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
BURDEN_STOP_CAP = 1.0


def build_exposure_load_shadow_replay_package(
    validation_rows: list[dict[str, object]],
    shadow_channel_lock: dict[str, object],
    shadow_review_protocol: dict[str, object],
) -> dict[str, object]:
    locked_channels = [
        _clean_row(row)
        for row in shadow_channel_lock.get("locked_channels", [])
        if isinstance(row, dict)
    ]
    excluded_seasons = {
        str(season) for season in shadow_channel_lock.get("excluded_test_seasons", [])
    }
    protocol_by_key = {
        _channel_key(row): _clean_row(row)
        for row in shadow_review_protocol.get("protocol_rows", [])
        if isinstance(row, dict)
    }
    replay_rows = _replay_rows(
        validation_rows,
        locked_channels,
        excluded_seasons,
        protocol_by_key,
    )
    review_packet_rows = [
        _review_packet_row(row)
        for row in replay_rows
        if row["replay_status"] == "ready_for_research_adjudication"
    ]
    stop_rule_rows = [_stop_rule_row(row) for row in replay_rows]
    return {
        "experiment_type": "exposure_load_shadow_replay_sprint",
        "overall_recommendation": _overall_recommendation(review_packet_rows),
        "production_readiness": shadow_channel_lock.get(
            "production_readiness",
            PRODUCTION_BLOCKED,
        ),
        "replay_rows": replay_rows,
        "review_packet_rows": review_packet_rows,
        "stop_rule_rows": stop_rule_rows,
        "held_channels": shadow_channel_lock.get("held_channels", []),
        "launch_boundary": shadow_channel_lock.get(
            "launch_boundary",
            "research shadow monitoring only",
        ),
        "next_sprint": (
            "use review packets to collect prospective adjudication outcomes before calibration or product escalation"
        ),
    }


def write_exposure_load_shadow_replay_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    replay_rows = summary.get("replay_rows", [])
    packet_rows = summary.get("review_packet_rows", [])
    stop_rows = summary.get("stop_rule_rows", [])
    source_stops = sum(
        1 for row in stop_rows if row.get("stop_rule_status") == "source_ineligible_stop"
    )
    burden_stops = sum(
        1 for row in stop_rows if row.get("stop_rule_status") == "burden_stop"
    )
    lines = [
        "# Exposure Load Historical Shadow Replay Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Launch boundary: {summary['launch_boundary']}",
        "",
        "## Replay Summary",
        "",
        f"- Replay rows: {len(replay_rows)}",
        f"- Review packets: {len(packet_rows)}",
        f"- Source-ineligible stop rows: {source_stops}",
        f"- Burden stop rows: {burden_stops}",
        "",
        "## Review Packet Channels",
        "",
        "| Channel | Packets |",
        "|---|---:|",
    ]
    for channel_name, count in _packet_counts(packet_rows).items():
        lines.append(f"| {channel_name} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The current data can generate historical replay logs and review "
                "packets for source-eligible locked channels. This supports "
                "prospective outcome collection setup, but it is not pilot or "
                "dashboard clearance."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_replay_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _replay_rows(
    validation_rows: list[dict[str, object]],
    locked_channels: list[dict[str, object]],
    excluded_seasons: set[str],
    protocol_by_key: dict[tuple[str, str, int, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for channel in locked_channels:
        protocol = protocol_by_key.get(_channel_key(channel), {})
        for row in validation_rows:
            if not _matches_locked_channel(row, channel):
                continue
            source_eligible = str(row.get("test_season_id")) not in excluded_seasons
            rows.append(_replay_row(row, channel, protocol, source_eligible))
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("channel_name")),
            str(row.get("test_season_id")),
        ),
    )


def _replay_row(
    row: dict[str, object],
    channel: dict[str, object],
    protocol: dict[str, object],
    source_eligible: bool,
) -> dict[str, object]:
    burden = _float_or_none(row.get("episodes_per_athlete_season"))
    if not source_eligible:
        replay_status = "source_ineligible_stop"
    elif burden is not None and burden > BURDEN_STOP_CAP:
        replay_status = "burden_stop"
    else:
        replay_status = "ready_for_research_adjudication"
    review_packet_id = (
        f"{row.get('channel_name')}__{row.get('test_season_id')}"
    )
    return _clean_row(
        {
            "review_packet_id": review_packet_id,
            "test_season_id": row.get("test_season_id"),
            "source_eligible": source_eligible,
            "channel_name": row.get("channel_name"),
            "policy_name": row.get("policy_name"),
            "horizon_days": row.get("horizon_days"),
            "threshold_policy": row.get("threshold_policy"),
            "selected_threshold_value": row.get("selected_threshold_value"),
            "episode_count": row.get("episode_count"),
            "true_positive_episode_count": row.get("true_positive_episode_count"),
            "false_positive_episode_count": row.get("false_positive_episode_count"),
            "unique_observed_event_count": row.get("unique_observed_event_count"),
            "unique_captured_event_count": row.get("unique_captured_event_count"),
            "unique_event_capture_rate": row.get("unique_event_capture_rate"),
            "missed_event_count": row.get("missed_event_count"),
            "episodes_per_athlete_season": row.get("episodes_per_athlete_season"),
            "median_start_lead_days": row.get("median_start_lead_days"),
            "minimum_review_unit": protocol.get(
                "minimum_review_unit",
                "complete source-eligible athlete-season",
            ),
            "required_evidence": protocol.get("required_evidence", ""),
            "stop_rule": protocol.get("stop_rule", ""),
            "replay_status": replay_status,
            "launch_boundary": channel.get(
                "launch_boundary",
                "research shadow monitoring only",
            ),
        }
    )


def _review_packet_row(row: dict[str, object]) -> dict[str, object]:
    return _clean_row(
        {
            "review_packet_id": row.get("review_packet_id"),
            "channel_name": row.get("channel_name"),
            "test_season_id": row.get("test_season_id"),
            "minimum_review_unit": row.get("minimum_review_unit"),
            "required_evidence": row.get("required_evidence"),
            "episode_count": row.get("episode_count"),
            "unique_observed_event_count": row.get("unique_observed_event_count"),
            "unique_captured_event_count": row.get("unique_captured_event_count"),
            "missed_event_count": row.get("missed_event_count"),
            "episodes_per_athlete_season": row.get("episodes_per_athlete_season"),
            "review_packet_status": "ready_for_research_adjudication",
            "adjudication_fields_needed": (
                "reviewer_id, review_date, alert_usefulness, outcome_confirmed, source_context_ok, action_taken, notes"
            ),
        }
    )


def _stop_rule_row(row: dict[str, object]) -> dict[str, object]:
    return _clean_row(
        {
            "review_packet_id": row.get("review_packet_id"),
            "channel_name": row.get("channel_name"),
            "test_season_id": row.get("test_season_id"),
            "source_eligible": row.get("source_eligible"),
            "episodes_per_athlete_season": row.get("episodes_per_athlete_season"),
            "stop_rule": row.get("stop_rule"),
            "stop_rule_status": (
                "no_stop_rule_triggered"
                if row.get("replay_status") == "ready_for_research_adjudication"
                else row.get("replay_status")
            ),
        }
    )


def _matches_locked_channel(
    row: dict[str, object],
    channel: dict[str, object],
) -> bool:
    if str(row.get("row_type")) != "alert_policy":
        return False
    if str(row.get("feature_set")) != EXPOSURE_LOAD_FEATURE_SET:
        return False
    return _channel_key(row) == _channel_key(channel)


def _channel_key(row: dict[str, object]) -> tuple[str, str, int, str]:
    return (
        str(row.get("channel_name")),
        str(row.get("policy_name")),
        int(row.get("horizon_days") or 0),
        str(row.get("threshold_policy")),
    )


def _overall_recommendation(review_packet_rows: list[dict[str, object]]) -> str:
    if review_packet_rows:
        return "historical_shadow_replay_ready_for_prospective_collection"
    return "complete_historical_replay_before_prospective_collection"


def _packet_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        channel = str(row.get("channel_name"))
        counts[channel] = counts.get(channel, 0) + 1
    return counts


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


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
