from __future__ import annotations

import ast
from collections import Counter

import pandas as pd


def build_qualitative_case_review(
    episodes: pd.DataFrame,
    alert_timeline: pd.DataFrame,
    quality: dict[str, object],
) -> dict[str, object]:
    cases = []
    representative_cases = quality.get("representative_cases", {})
    for horizon_key, threshold_payload in representative_cases.items():
        for threshold, case_payload in threshold_payload.items():
            for case_type, case_ref in case_payload.items():
                if case_ref is None:
                    continue
                case = _build_case(
                    case_type=case_type,
                    case_ref=case_ref,
                    episodes=episodes,
                    alert_timeline=alert_timeline,
                    horizon=int(horizon_key),
                    threshold=str(threshold),
                )
                if case is not None:
                    cases.append(case)

    diagnostic_summary = dict(sorted(Counter(case["diagnostic_label"] for case in cases).items()))
    return {
        "case_count": len(cases),
        "cases": cases,
        "diagnostic_summary": diagnostic_summary,
    }


def _build_case(
    case_type: str,
    case_ref: dict[str, object],
    episodes: pd.DataFrame,
    alert_timeline: pd.DataFrame,
    horizon: int,
    threshold: str,
) -> dict[str, object] | None:
    if case_type == "missed_injury":
        event_rows = _matching_event_rows(alert_timeline, case_ref)
        if event_rows.empty:
            return None
        context = _timeline_context(event_rows)
        diagnostics = _data_diagnostics(event_rows)
        label = _missed_injury_label(diagnostics)
        return {
            "case_type": case_type,
            "review_label": "Missed injury",
            "diagnostic_label": label,
            "athlete_id": str(case_ref["athlete_id"]),
            "season_id": str(case_ref["season_id"]),
            "horizon_days": horizon,
            "threshold": threshold,
            "event_date": _date_key(case_ref.get("event_date")),
            "injury_type": _clean_value(case_ref.get("injury_type")),
            "data_diagnostics": diagnostics,
            "timeline_context": context,
        }

    episode = _matching_episode(episodes, case_ref, horizon, threshold)
    if episode is None:
        return None
    season_rows = _season_rows(alert_timeline, episode["athlete_id"], episode["season_id"])
    context = _episode_context(
        season_rows,
        int(episode["start_time_index"]),
        int(episode["end_time_index"]),
    )
    diagnostics = _data_diagnostics(season_rows)
    label = _episode_label(case_type, diagnostics, episode)
    return {
        "case_type": case_type,
        "review_label": _review_label(case_type),
        "diagnostic_label": label,
        "athlete_id": str(episode["athlete_id"]),
        "season_id": str(episode["season_id"]),
        "horizon_days": horizon,
        "threshold": threshold,
        "start_date": str(episode["start_date"]),
        "peak_date": str(episode["peak_date"]),
        "end_date": str(episode["end_date"]),
        "days_from_start_to_event": _clean_number(
            episode.get("days_from_start_to_event")
        ),
        "days_from_peak_to_event": _clean_number(episode.get("days_from_peak_to_event")),
        "days_from_end_to_event": _clean_number(episode.get("days_from_end_to_event")),
        "peak_risk": _clean_number(episode.get("peak_risk")),
        "mean_risk": _clean_number(episode.get("mean_risk")),
        "injury_type": _clean_value(episode.get("injury_type")),
        "top_model_features": _feature_records(episode.get("top_model_features")),
        "elevated_z_features": _feature_list(episode.get("elevated_z_features")),
        "data_diagnostics": diagnostics,
        "timeline_context": context,
    }


def _matching_episode(
    episodes: pd.DataFrame,
    case_ref: dict[str, object],
    horizon: int,
    threshold: str,
) -> pd.Series | None:
    if episodes.empty:
        return None
    threshold_kind, threshold_value = threshold.split(":", maxsplit=1)
    matches = episodes[
        (episodes["athlete_id"].astype(str) == str(case_ref["athlete_id"]))
        & (episodes["season_id"].astype(str) == str(case_ref["season_id"]))
        & (episodes["horizon_days"].astype(int) == horizon)
        & (episodes["threshold_kind"].astype(str) == threshold_kind)
        & (episodes["threshold_value"].astype(float) == float(threshold_value))
    ]
    if "start_date" in case_ref:
        matches = matches[matches["start_date"].astype(str) == str(case_ref["start_date"])]
    if matches.empty:
        return None
    return matches.sort_values(["peak_risk", "start_time_index"], ascending=[False, True]).iloc[0]


def _matching_event_rows(
    alert_timeline: pd.DataFrame,
    case_ref: dict[str, object],
) -> pd.DataFrame:
    rows = _season_rows(alert_timeline, case_ref["athlete_id"], case_ref["season_id"])
    if rows.empty:
        return rows
    event_date = str(case_ref.get("event_date", ""))
    injury_type = str(case_ref.get("injury_type", ""))
    if event_date and "event_date" in rows.columns:
        event_date_key = _date_key(event_date)
        rows = rows[rows["event_date"].map(_date_key) == event_date_key]
    if injury_type and "injury_type" in rows.columns:
        rows = rows[rows["injury_type"].astype(str) == injury_type]
    return rows.sort_values("time_index")


def _season_rows(
    alert_timeline: pd.DataFrame,
    athlete_id: object,
    season_id: object,
) -> pd.DataFrame:
    if alert_timeline.empty:
        return alert_timeline
    return alert_timeline[
        (alert_timeline["athlete_id"].astype(str) == str(athlete_id))
        & (alert_timeline["season_id"].astype(str) == str(season_id))
    ].sort_values("time_index")


def _episode_context(
    season_rows: pd.DataFrame,
    start_time_index: int,
    end_time_index: int,
) -> list[dict[str, object]]:
    context = season_rows[
        (season_rows["time_index"].astype(int) >= start_time_index)
        & (season_rows["time_index"].astype(int) <= end_time_index)
    ]
    return _timeline_context(context)


def _timeline_context(rows: pd.DataFrame) -> list[dict[str, object]]:
    columns = [
        "time_index",
        "snapshot_date",
        "risk_7d",
        "risk_14d",
        "risk_30d",
        "top_feature_30d",
        "top_contribution_30d",
        "elevated_z_features",
        "event_observed",
        "event_date",
        "days_to_event",
        "injury_type",
        "event_window_quality",
        "nearest_measurement_gap_days",
        "primary_model_event",
    ]
    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                column: _json_value(row[column])
                for column in columns
                if column in row.index
            }
        )
    return records


def _data_diagnostics(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {
            "event_window_quality": None,
            "nearest_measurement_gap_days": None,
            "primary_model_event": None,
        }
    first = rows.iloc[0]
    return {
        "event_window_quality": _clean_value(first.get("event_window_quality")),
        "nearest_measurement_gap_days": _clean_number(
            first.get("nearest_measurement_gap_days")
        ),
        "primary_model_event": _as_bool(first.get("primary_model_event"))
        if "primary_model_event" in first.index
        else None,
    }


def _episode_label(
    case_type: str,
    diagnostics: dict[str, object],
    episode: pd.Series,
) -> str:
    if case_type == "true_positive_episode":
        if diagnostics.get("event_window_quality") == "modelable":
            return "model_signal_supported"
        return "possible_label_or_measurement_gap"
    if case_type == "false_positive_episode":
        return "missing_context_or_managed_risk"
    if case_type == "high_intra_individual_deviation_episode":
        if _as_bool(episode.get("event_within_horizon_after_start")):
            return "model_signal_supported"
        return "explanation_gap"
    return "needs_review"


def _missed_injury_label(diagnostics: dict[str, object]) -> str:
    quality = diagnostics.get("event_window_quality")
    gap = diagnostics.get("nearest_measurement_gap_days")
    if quality != "modelable" or (gap is not None and float(gap) > 30):
        return "possible_label_or_measurement_gap"
    return "model_miss"


def _review_label(case_type: str) -> str:
    return {
        "true_positive_episode": "Useful warning",
        "false_positive_episode": "Noisy alert",
        "high_intra_individual_deviation_episode": "High own-baseline departure",
    }.get(case_type, "Needs review")


def _date_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return str(parsed.date())


def _feature_records(value: object) -> list[dict[str, object]]:
    parsed = _parse_collection(value)
    if not isinstance(parsed, list):
        return []
    records = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("feature"):
            continue
        records.append(
            {
                "feature": str(item["feature"]),
                "mean_abs_contribution": _clean_number(
                    item.get("mean_abs_contribution")
                ),
            }
        )
    return records


def _feature_list(value: object) -> list[str]:
    parsed = _parse_collection(value)
    if isinstance(parsed, list | tuple | set):
        return [str(item) for item in parsed if str(item)]
    if isinstance(parsed, str) and parsed:
        return [parsed]
    return []


def _parse_collection(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, list | tuple | set | dict):
        return value
    if not isinstance(value, str) and pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
    return value


def _json_value(value: object) -> object:
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(val) for key, val in value.items()}
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return _clean_number(value)
    return value


def _clean_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _clean_number(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}
