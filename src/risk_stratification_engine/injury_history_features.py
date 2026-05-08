from __future__ import annotations

import pandas as pd


INJURY_HISTORY_FEATURE_COLUMNS = (
    "injury_history_prior_injury_count",
    "injury_history_prior_same_season_injury_count",
    "injury_history_days_since_last_injury",
    "injury_history_prior_time_loss_days_sum",
    "injury_history_prior_time_loss_days_max",
    "injury_history_prior_lower_extremity_injury_count",
    "injury_history_prior_soft_tissue_injury_count",
    "injury_history_prior_lower_extremity_soft_tissue_count",
    "injury_history_prior_game_injury_count",
    "injury_history_prior_practice_injury_count",
    "injury_history_prior_s_and_c_injury_count",
    "injury_history_prior_caused_unavailability_count",
)

LOWER_EXTREMITY_BODY_AREAS = {
    "ankle",
    "foot",
    "groin/hip",
    "hip",
    "knee",
    "lower leg",
    "thigh",
}


def attach_injury_history_features(
    graph_features: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
) -> pd.DataFrame:
    out = graph_features.copy()
    for column in INJURY_HISTORY_FEATURE_COLUMNS:
        out[column] = 0
    if out.empty or detailed_injuries.empty:
        return out

    out["season_id"] = out["season_id"].astype(str)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    injuries = _normalized_injury_frame(detailed_injuries)
    if injuries.empty:
        return out

    rows = []
    for _, snapshot in out.iterrows():
        athlete_injuries = injuries[injuries["athlete_id"].eq(snapshot["athlete_id"])]
        prior = athlete_injuries[
            athlete_injuries["injury_date"].lt(snapshot["snapshot_date"])
        ]
        rows.append(_prior_injury_features(snapshot, prior))
    feature_frame = pd.DataFrame(rows, index=out.index)
    for column in INJURY_HISTORY_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(0)
        out[column] = out[column].round().astype(int)
    return out


def _normalized_injury_frame(detailed_injuries: pd.DataFrame) -> pd.DataFrame:
    frame = detailed_injuries.copy()
    if "athlete_id" not in frame or "injury_date" not in frame:
        return pd.DataFrame()
    frame["season_id"] = _column(frame, "season_id", "").astype(str)
    frame["injury_date"] = pd.to_datetime(frame["injury_date"], errors="coerce")
    frame = frame.dropna(subset=["injury_date"])
    frame["time_loss_days"] = pd.to_numeric(_column(frame, "time_loss_days", 0), errors="coerce").fillna(0)
    for column in ("body_area", "tissue_type", "classification", "activity_group"):
        frame[column] = _column(frame, column, "").fillna("").astype(str)
    frame["caused_unavailability_bool"] = frame.get(
        "caused_unavailability",
        pd.Series(False, index=frame.index),
    ).map(_truthy)
    frame["is_lower_extremity"] = frame["body_area"].map(_is_lower_extremity)
    frame["is_soft_tissue"] = frame.apply(_is_soft_tissue, axis=1)
    frame["activity_key"] = frame["activity_group"].map(_activity_key)
    return frame


def _prior_injury_features(
    snapshot: pd.Series,
    prior: pd.DataFrame,
) -> dict[str, int]:
    if prior.empty:
        return {column: 0 for column in INJURY_HISTORY_FEATURE_COLUMNS}
    same_season = prior["season_id"].astype(str).eq(str(snapshot["season_id"]))
    last_injury_date = prior["injury_date"].max()
    lower = prior["is_lower_extremity"]
    soft = prior["is_soft_tissue"]
    activity = prior["activity_key"]
    return {
        "injury_history_prior_injury_count": int(len(prior)),
        "injury_history_prior_same_season_injury_count": int(same_season.sum()),
        "injury_history_days_since_last_injury": int(
            (snapshot["snapshot_date"] - last_injury_date).days
        ),
        "injury_history_prior_time_loss_days_sum": int(prior["time_loss_days"].sum()),
        "injury_history_prior_time_loss_days_max": int(prior["time_loss_days"].max()),
        "injury_history_prior_lower_extremity_injury_count": int(lower.sum()),
        "injury_history_prior_soft_tissue_injury_count": int(soft.sum()),
        "injury_history_prior_lower_extremity_soft_tissue_count": int(
            (lower & soft).sum()
        ),
        "injury_history_prior_game_injury_count": int(activity.eq("game").sum()),
        "injury_history_prior_practice_injury_count": int(
            activity.eq("practice").sum()
        ),
        "injury_history_prior_s_and_c_injury_count": int(activity.eq("s_and_c").sum()),
        "injury_history_prior_caused_unavailability_count": int(
            prior["caused_unavailability_bool"].sum()
        ),
    }


def _is_lower_extremity(value: object) -> bool:
    return str(value).strip().lower() in LOWER_EXTREMITY_BODY_AREAS


def _is_soft_tissue(row: pd.Series) -> bool:
    tissue = str(row.get("tissue_type", "")).strip().lower()
    classification = str(row.get("classification", "")).strip().lower()
    return (
        "muscle" in tissue
        or "tendon" in tissue
        or "muscle" in classification
        or "tendin" in classification
        or "soft" in classification
    )


def _activity_key(value: object) -> str:
    text = str(value).strip().lower().replace("&", "and")
    if text == "game":
        return "game"
    if text == "practice":
        return "practice"
    if text in {"s and c", "s c", "s+c", "strength and conditioning"}:
        return "s_and_c"
    return "other"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _column(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index)
