from __future__ import annotations

import pandas as pd


EXPOSURE_LOAD_FEATURE_COLUMNS = (
    "exposure_training_sessions_7d",
    "exposure_training_sessions_14d",
    "exposure_training_sessions_28d",
    "exposure_games_prior_count",
    "exposure_days_since_last_game",
    "exposure_full_participations_28d",
    "exposure_modified_participations_28d",
    "exposure_no_participations_28d",
    "exposure_days_since_last_modified_or_no_participation",
    "exposure_practice_sessions_28d",
    "exposure_lift_sessions_28d",
    "exposure_conditioning_sessions_28d",
    "exposure_rtp_sessions_28d",
    "exposure_game_events_28d",
)


def attach_exposure_load_features(
    graph_features: pd.DataFrame,
    exposure_participations: pd.DataFrame,
) -> pd.DataFrame:
    out = graph_features.copy()
    for column in EXPOSURE_LOAD_FEATURE_COLUMNS:
        out[column] = 0
    if out.empty or exposure_participations.empty:
        return out

    out["season_id"] = out["season_id"].astype(str)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    participations = _normalized_participations(exposure_participations)
    if participations.empty:
        return out

    rows = []
    for _, snapshot in out.iterrows():
        athlete_rows = participations[
            participations["athlete_id"].eq(snapshot["athlete_id"])
            & participations["season_id"].eq(str(snapshot["season_id"]))
            & participations["date"].lt(snapshot["snapshot_date"])
        ]
        rows.append(_prior_exposure_features(snapshot, athlete_rows))

    feature_frame = pd.DataFrame(rows, index=out.index)
    for column in EXPOSURE_LOAD_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(0)
        out[column] = out[column].round().astype(int)
    return out


def _normalized_participations(participations: pd.DataFrame) -> pd.DataFrame:
    required = {
        "athlete_id",
        "date",
        "season_id",
        "event_id",
        "event_type",
        "exposure_category",
        "participation_category",
    }
    if missing := sorted(required - set(participations.columns)):
        raise ValueError(
            "exposure_participations is missing required columns: "
            + ", ".join(missing)
        )
    frame = participations.copy()
    frame["athlete_id"] = frame["athlete_id"].fillna("").astype(str).str.strip()
    frame = frame[frame["athlete_id"].ne("")]
    if "athlete_match_status" in frame:
        frame = frame[frame["athlete_match_status"].astype(str).eq("matched")]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["season_id"] = frame["season_id"].astype(str)
    for column in ("event_id", "event_type", "exposure_category"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["participation_category"] = (
        frame["participation_category"].fillna("").astype(str).str.strip().str.lower()
    )
    return frame


def _prior_exposure_features(
    snapshot: pd.Series,
    prior: pd.DataFrame,
) -> dict[str, int]:
    if prior.empty:
        return {column: 0 for column in EXPOSURE_LOAD_FEATURE_COLUMNS}
    snapshot_date = snapshot["snapshot_date"]
    windows = {
        days: prior[prior["date"].ge(snapshot_date - pd.Timedelta(days=days))]
        for days in (7, 14, 28)
    }
    prior_28 = windows[28]
    training = {days: window[window["event_type"].eq("training")] for days, window in windows.items()}
    games = prior[prior["event_type"].eq("game")]
    games_28 = prior_28[prior_28["event_type"].eq("game")]
    limited = prior[
        prior["participation_category"].isin({"modified", "no_participation"})
    ]
    category_flags = _category_flags(prior_28)
    return {
        "exposure_training_sessions_7d": _unique_event_count(training[7]),
        "exposure_training_sessions_14d": _unique_event_count(training[14]),
        "exposure_training_sessions_28d": _unique_event_count(training[28]),
        "exposure_games_prior_count": _unique_event_count(games),
        "exposure_days_since_last_game": _days_since_last(snapshot_date, games),
        "exposure_full_participations_28d": _participation_count(prior_28, "full"),
        "exposure_modified_participations_28d": _participation_count(
            prior_28,
            "modified",
        ),
        "exposure_no_participations_28d": _participation_count(
            prior_28,
            "no_participation",
        ),
        "exposure_days_since_last_modified_or_no_participation": _days_since_last(
            snapshot_date,
            limited,
        ),
        "exposure_practice_sessions_28d": _unique_event_count(
            prior_28[category_flags["practice"]]
        ),
        "exposure_lift_sessions_28d": _unique_event_count(
            prior_28[category_flags["lift"]]
        ),
        "exposure_conditioning_sessions_28d": _unique_event_count(
            prior_28[category_flags["conditioning"]]
        ),
        "exposure_rtp_sessions_28d": _unique_event_count(
            prior_28[category_flags["rtp"]]
        ),
        "exposure_game_events_28d": _unique_event_count(games_28),
    }


def _category_flags(frame: pd.DataFrame) -> dict[str, pd.Series]:
    category = frame["exposure_category"].astype(str).str.lower()
    event_type = frame["event_type"].astype(str).str.lower()
    return {
        "practice": (
            category.str.contains("practice")
            | category.isin({"walkthrough", "scrimmage", "position_drills"})
        ),
        "lift": category.str.contains("weight_room"),
        "conditioning": (
            category.str.contains("conditioning") | category.str.contains("speed_power")
        ),
        "rtp": category.eq("rtp"),
        "game": event_type.eq("game") | category.eq("game"),
    }


def _participation_count(frame: pd.DataFrame, category: str) -> int:
    return int(frame["participation_category"].eq(category).sum())


def _unique_event_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame["event_id"].nunique())


def _days_since_last(snapshot_date: pd.Timestamp, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int((snapshot_date - frame["date"].max()).days)
