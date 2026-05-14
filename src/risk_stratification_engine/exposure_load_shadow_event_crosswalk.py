from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


PRODUCTION_BLOCKED = "not_ready_for_probability_or_pilot"
REVIEW_BOUNDARY = (
    "captured/missed injury-event crosswalk for independent practitioner "
    "adjudication only; not probability calibration or dashboard clearance"
)


def build_shadow_event_crosswalk_summary(
    packet_contexts: list[dict[str, object]],
    detailed_injuries: pd.DataFrame,
) -> dict[str, object]:
    detail_lookup = _detailed_event_lookup(detailed_injuries)
    event_rows: list[dict[str, object]] = []
    packet_summary_rows: list[dict[str, object]] = []

    for context in packet_contexts:
        packet = dict(context["packet"])
        rows = _packet_event_rows(
            packet=packet,
            episodes=pd.DataFrame(context["episodes"]),
            timeline=pd.DataFrame(context["timeline"]),
            detail_lookup=detail_lookup,
        )
        event_rows.extend(rows)
        packet_summary_rows.append(_packet_summary_row(packet, rows))

    captured_count = sum(
        1 for row in event_rows if row.get("capture_status") == "captured"
    )
    missed_count = sum(1 for row in event_rows if row.get("capture_status") == "missed")
    return {
        "experiment_type": "exposure_load_shadow_event_crosswalk_sprint",
        "overall_recommendation": (
            "use_event_crosswalk_for_independent_practitioner_adjudication"
        ),
        "production_readiness": PRODUCTION_BLOCKED,
        "review_boundary": REVIEW_BOUNDARY,
        "packet_count": len(packet_contexts),
        "total_event_rows": len(event_rows),
        "captured_event_rows": captured_count,
        "missed_event_rows": missed_count,
        "event_crosswalk_rows": clean_shadow_event_crosswalk_rows(event_rows),
        "packet_summary_rows": clean_shadow_event_crosswalk_rows(packet_summary_rows),
    }


def write_exposure_load_shadow_event_crosswalk_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shadow Event Crosswalk Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Production readiness: {summary['production_readiness']}",
        f"Review boundary: {summary['review_boundary']}",
        "",
        "## Crosswalk Summary",
        "",
        f"- Packets: {summary['packet_count']}",
        f"- Event rows: {summary['total_event_rows']}",
        f"- Captured events: {summary['captured_event_rows']}",
        f"- Missed events: {summary['missed_event_rows']}",
        "",
        "## Packet Validation",
        "",
        "| Packet | Captured | Missed | Count check |",
        "|---|---:|---:|---|",
    ]
    for row in summary["packet_summary_rows"]:
        lines.append(
            "| "
            f"{row['review_packet_id']} | "
            f"{row['crosswalk_captured_event_count']} | "
            f"{row['crosswalk_missed_event_count']} | "
            f"{row['aggregate_count_check']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This artifact lists the captured/missed injury-event crosswalk "
                "needed to support independent practitioner adjudication. It is "
                "not calibration evidence, probability-facing output, pilot "
                "clearance, or dashboard readiness."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def clean_shadow_event_crosswalk_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


def _packet_event_rows(
    *,
    packet: dict[str, object],
    episodes: pd.DataFrame,
    timeline: pd.DataFrame,
    detail_lookup: dict[tuple[str, str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    true_positive = _true_positive_episodes(episodes)
    event_rows = _observed_events(timeline)
    for event in event_rows:
        linked = _linked_episodes(true_positive, event)
        nearest = _nearest_episode(linked)
        details = detail_lookup.get(_event_key(event), {})
        rows.append(
            _clean_row(
                {
                    "review_packet_id": packet.get("review_packet_id"),
                    "channel_name": packet.get("channel_name"),
                    "test_season_id": packet.get("test_season_id"),
                    "policy_name": packet.get("policy_name"),
                    "horizon_days": packet.get("horizon_days"),
                    "threshold_policy": packet.get("threshold_policy"),
                    "selected_threshold_value": packet.get(
                        "selected_threshold_value"
                    ),
                    "capture_status": "captured" if len(linked) else "missed",
                    "athlete_id": event["athlete_id"],
                    "season_id": event["season_id"],
                    "event_date": event["event_date"],
                    "injury_type": event["injury_type"],
                    "injury_event_id": details.get("injury_event_id"),
                    "classification": details.get("classification"),
                    "pathology": details.get("pathology"),
                    "body_area": details.get("body_area"),
                    "tissue_type": details.get("tissue_type"),
                    "side": details.get("side"),
                    "recurrent": details.get("recurrent"),
                    "caused_unavailability": details.get("caused_unavailability"),
                    "activity": details.get("activity"),
                    "activity_group": details.get("activity_group"),
                    "participation_level": details.get("participation_level"),
                    "duration_days": details.get("duration_days"),
                    "time_loss_days": details.get("time_loss_days"),
                    "modified_available_days": details.get(
                        "modified_available_days"
                    ),
                    "source_file": details.get("source_file"),
                    "source_row_number": details.get("source_row_number"),
                    "linked_alert_episode_count": len(linked),
                    "nearest_alert_start_date": _episode_value(nearest, "start_date"),
                    "nearest_alert_peak_date": _episode_value(nearest, "peak_date"),
                    "nearest_alert_end_date": _episode_value(nearest, "end_date"),
                    "nearest_days_from_start_to_event": _episode_value(
                        nearest,
                        "days_from_start_to_event",
                    ),
                    "nearest_days_from_peak_to_event": _episode_value(
                        nearest,
                        "days_from_peak_to_event",
                    ),
                    "nearest_days_from_end_to_event": _episode_value(
                        nearest,
                        "days_from_end_to_event",
                    ),
                    "nearest_peak_risk": _episode_value(nearest, "peak_risk"),
                    "nearest_mean_risk": _episode_value(nearest, "mean_risk"),
                    "evidence_boundary": REVIEW_BOUNDARY,
                }
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["review_packet_id"]),
            str(row["capture_status"]),
            str(row["event_date"]),
            str(row["athlete_id"]),
            str(row["injury_type"]),
        ),
    )


def _packet_summary_row(
    packet: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    captured_count = sum(1 for row in rows if row["capture_status"] == "captured")
    missed_count = sum(1 for row in rows if row["capture_status"] == "missed")
    expected_captured = _int_or_none(packet.get("unique_captured_event_count"))
    expected_missed = _int_or_none(packet.get("missed_event_count"))
    expected_observed = _int_or_none(packet.get("unique_observed_event_count"))
    checks = [
        expected_captured is None or expected_captured == captured_count,
        expected_missed is None or expected_missed == missed_count,
        expected_observed is None or expected_observed == len(rows),
    ]
    return {
        "review_packet_id": packet.get("review_packet_id"),
        "channel_name": packet.get("channel_name"),
        "test_season_id": packet.get("test_season_id"),
        "expected_observed_event_count": expected_observed,
        "expected_captured_event_count": expected_captured,
        "expected_missed_event_count": expected_missed,
        "crosswalk_observed_event_count": len(rows),
        "crosswalk_captured_event_count": captured_count,
        "crosswalk_missed_event_count": missed_count,
        "aggregate_count_check": "matches_replay_counts"
        if all(checks)
        else "review_count_mismatch",
    }


def _observed_events(timeline: pd.DataFrame) -> list[dict[str, object]]:
    if timeline.empty or "event_observed" not in timeline:
        return []
    date_column = "event_date" if "event_date" in timeline.columns else "injury_date"
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    observed = timeline[timeline["event_observed"].map(_as_bool)]
    for _, row in observed.iterrows():
        event = {
            "athlete_id": str(row["athlete_id"]),
            "season_id": str(row["season_id"]),
            "event_date": _date_string(row.get(date_column)),
            "injury_type": str(_clean_value(row.get("injury_type")) or ""),
        }
        rows[_event_key(event)] = event
    return sorted(
        rows.values(),
        key=lambda row: (
            row["event_date"],
            row["athlete_id"],
            row["season_id"],
            row["injury_type"],
        ),
    )


def _true_positive_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty or "event_within_horizon_after_start" not in episodes:
        return pd.DataFrame()
    return episodes[episodes["event_within_horizon_after_start"].map(_as_bool)].copy()


def _linked_episodes(episodes: pd.DataFrame, event: dict[str, object]) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    mask = (
        episodes["athlete_id"].astype(str).eq(event["athlete_id"])
        & episodes["season_id"].astype(str).eq(event["season_id"])
        & episodes["injury_type"].map(lambda value: str(_clean_value(value) or "")).eq(
            event["injury_type"]
        )
    )
    return episodes.loc[mask].copy()


def _nearest_episode(episodes: pd.DataFrame) -> pd.Series | None:
    if episodes.empty:
        return None
    sortable = episodes.copy()
    sortable["_sort_days"] = pd.to_numeric(
        sortable.get("days_from_start_to_event"),
        errors="coerce",
    )
    sortable["_sort_days"] = sortable["_sort_days"].fillna(float("inf"))
    sortable = sortable.sort_values(["_sort_days", "start_date", "peak_date"])
    return sortable.iloc[0]


def _detailed_event_lookup(
    detailed_injuries: pd.DataFrame,
) -> dict[tuple[str, str, str, str], dict[str, object]]:
    if detailed_injuries.empty:
        return {}
    frame = detailed_injuries.copy()
    if "injury_date" not in frame:
        return {}
    frame["injury_date"] = frame["injury_date"].map(_date_string)
    lookup = {}
    for key, group in frame.groupby(
        ["athlete_id", "season_id", "injury_date", "injury_type"],
        sort=True,
        dropna=False,
    ):
        first = group.iloc[0].to_dict()
        first["injury_event_id"] = ";".join(
            str(value)
            for value in group.get("injury_event_id", pd.Series(dtype=object))
            if _clean_value(value)
        )
        lookup[tuple(str(value) for value in key)] = first
    return lookup


def _event_key(event: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(event["athlete_id"]),
        str(event["season_id"]),
        str(event["event_date"]),
        str(event["injury_type"]),
    )


def _episode_value(row: pd.Series | None, column: str) -> object:
    if row is None or column not in row.index:
        return None
    return _clean_value(row.get(column))


def _date_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return str(value)
    return str(date.date())


def _int_or_none(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


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
        return int(number) if number.is_integer() else number
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value
