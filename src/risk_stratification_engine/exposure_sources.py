from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from risk_stratification_engine.io import write_frame
from risk_stratification_engine.live_sources import stable_athlete_id


FOOTBALL_SQUAD_ID = "94"

REQUIRED_EXPOSURE_FILES = (
    "athletes.csv",
    "squads.csv",
    "training_sessions.csv",
    "training_session_participations.csv",
    "games.csv",
    "game_participations.csv",
)

API_PERFORMANCE_SOURCE_TERMS = (
    "perch",
    "perks",
    "forcedecks",
    "force decks",
    "vald",
    "vault",
    "nordbord",
    "groinbar",
    "smart speed",
    "smartspeed",
    "catapult",
)

TRAINING_EXPOSURE_CATEGORIES = {
    "practice": "practice",
    "practice - shells": "practice_shells",
    "practice - full pads": "practice_full_pads",
    "practice - helmets": "practice_helmets",
    "practice - no pads": "practice_no_pads",
    "practice - spiders": "practice_spiders",
    "walkthrough": "walkthrough",
    "scrimmage": "scrimmage",
    "position drills": "position_drills",
    "conditioning": "conditioning",
    "s&c - conditioning": "conditioning",
    "speed-power": "speed_power",
    "speed-power + weight room": "speed_power_weight_room",
    "speed-power + conditioning": "speed_power_conditioning",
    "speed-power + cond + weight room": "speed_power_conditioning_weight_room",
    "speed-power + conditioning + weight room": (
        "speed_power_conditioning_weight_room"
    ),
    "conditioning + weight room": "conditioning_weight_room",
    "weight room": "weight_room",
    "rtp": "rtp",
}

CANDIDATE_FEATURE_DEFINITIONS = (
    "prior_7_day_training_session_count",
    "prior_14_day_training_session_count",
    "prior_28_day_training_session_count",
    "prior_7_day_training_duration_minutes",
    "prior_14_day_training_duration_minutes",
    "prior_28_day_training_duration_minutes",
    "prior_7_day_full_modified_no_participation_counts",
    "prior_14_day_full_modified_no_participation_counts",
    "prior_28_day_full_modified_no_participation_counts",
    "prior_game_count",
    "prior_game_participation_minutes",
    "days_since_last_game",
    "days_since_last_modified_or_no_participation_session",
    "coarse_practice_type_exposure_counts",
)


@dataclass(frozen=True)
class ExposurePreparationResult:
    events_path: Path
    participations_path: Path
    audit_path: Path
    audit: dict[str, Any]


def prepare_exposure_inputs(
    exposure_dir: str | Path,
    output_dir: str | Path,
) -> ExposurePreparationResult:
    source_dir = Path(exposure_dir)
    output = Path(output_dir)
    frames = _read_exposure_files(source_dir)

    athletes = _prepare_athletes(frames["athletes.csv"])
    training_events, training_audit = _build_training_events(
        frames["training_sessions.csv"]
    )
    game_events, game_audit = _build_game_events(frames["games.csv"])

    training_participations = _build_participations(
        frames["training_session_participations.csv"],
        athletes,
        training_events,
        event_type="training",
        external_event_column="ExternalSessionId",
    )
    game_participations = _build_participations(
        frames["game_participations.csv"],
        athletes,
        game_events,
        event_type="game",
        external_event_column="ExternalGameId",
    )

    events = (
        pd.concat([training_events, game_events], ignore_index=True)
        .sort_values(["date", "event_type", "external_event_id"])
        .reset_index(drop=True)
    )
    participations = (
        pd.concat([training_participations, game_participations], ignore_index=True)
        .sort_values(["date", "event_type", "external_event_id", "external_athlete_id"])
        .reset_index(drop=True)
    )

    audit = _build_audit(
        frames=frames,
        training_audit=training_audit,
        game_audit=game_audit,
        training_participations=training_participations,
        game_participations=game_participations,
        training_raw_participations=frames["training_session_participations.csv"],
        game_raw_participations=frames["game_participations.csv"],
        training_events=training_events,
        game_events=game_events,
    )

    events_path = output / "exposure_events.csv"
    participations_path = output / "exposure_participations.csv"
    audit_path = output / "exposure_cleaning_audit.json"
    write_frame(events, events_path)
    write_frame(participations, participations_path)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return ExposurePreparationResult(
        events_path=events_path,
        participations_path=participations_path,
        audit_path=audit_path,
        audit=audit,
    )


def _read_exposure_files(source_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [
        filename for filename in REQUIRED_EXPOSURE_FILES if not (source_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "exposure source folder is missing required files: "
            + ", ".join(missing)
        )
    return {
        filename: pd.read_csv(source_dir / filename, dtype=str, keep_default_na=False)
        for filename in REQUIRED_EXPOSURE_FILES
    }


def _prepare_athletes(athletes: pd.DataFrame) -> pd.DataFrame:
    out = athletes.copy()
    out["external_athlete_id"] = _text(out, "ExternalAthleteId")
    out["athlete_name"] = (
        _text(out, "FirstName").str.strip()
        + " "
        + _text(out, "LastName").str.strip()
    ).str.strip()
    out["athlete_id"] = out["athlete_name"].map(stable_athlete_id)
    out["football_squad_member"] = _text(out, "ExternalSquadIds").map(
        _contains_football_squad
    )
    return out.loc[
        :,
        [
            "external_athlete_id",
            "athlete_id",
            "football_squad_member",
            "Position",
            "ExternalSquadIds",
        ],
    ].rename(
        columns={
            "Position": "source_position",
            "ExternalSquadIds": "source_external_squad_ids",
        }
    )


def _build_training_events(
    training_sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = training_sessions.copy()
    rows["external_event_id"] = _text(rows, "ExternalSessionId")
    rows["external_squad_id"] = _text(rows, "ExternalSquadId")
    rows["source_name"] = _text(rows, "Name")
    rows["session_type"] = _text(rows, "SessionType")
    rows["retention_reason"] = rows.apply(_training_retention_reason, axis=1)
    retained = rows.loc[rows["retention_reason"].eq("retained")].copy()
    retained["event_id"] = "training:" + retained["external_event_id"]
    retained["event_type"] = "training"
    retained["date"] = _parse_datetime(retained["StartDateTime"]).dt.date.astype(str)
    retained["start_datetime"] = _parse_datetime(retained["StartDateTime"]).astype(str)
    retained["end_datetime"] = _parse_datetime(retained["EndDateTime"]).astype(str)
    retained["season_id"] = _season_id_for_datetimes(
        _parse_datetime(retained["StartDateTime"])
    )
    retained["exposure_category"] = retained.apply(_training_category, axis=1)
    retained["duration_minutes"] = pd.to_numeric(
        retained.get("Duration"), errors="coerce"
    )
    retained["rtp_flag"] = retained["exposure_category"].eq("rtp")
    football_rows = rows.loc[rows["external_squad_id"].eq(FOOTBALL_SQUAD_ID)]
    audit = {
        "source_events": int(len(rows)),
        "football_events": int(len(football_rows)),
        "non_football_events": int(
            rows["retention_reason"].eq("non_football_squad").sum()
        ),
        "excluded_by_reason": _value_counts(
            football_rows.loc[
                ~football_rows["retention_reason"].eq("retained"),
                "retention_reason",
            ]
        ),
        "retained_by_season": _value_counts(retained["season_id"]),
        "retained_by_session_type": _value_counts(retained["session_type"]),
        "unclassified_session_types": _value_counts(
            rows.loc[rows["retention_reason"].eq("unclassified"), "session_type"]
        ),
    }
    return retained.loc[:, _EVENT_COLUMNS], audit


def _build_game_events(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = games.copy()
    rows["external_event_id"] = _text(rows, "ExternalGameId")
    rows["external_squad_id"] = _text(rows, "ExternalSquadId")
    rows["retention_reason"] = rows["external_squad_id"].map(
        lambda value: "retained" if value == FOOTBALL_SQUAD_ID else "non_football_squad"
    )
    retained = rows.loc[rows["retention_reason"].eq("retained")].copy()
    retained["event_id"] = "game:" + retained["external_event_id"]
    retained["event_type"] = "game"
    retained["source_name"] = _text(retained, "Name")
    retained["session_type"] = ""
    retained["date"] = _parse_datetime(retained["StartDateTime"]).dt.date.astype(str)
    retained["start_datetime"] = _parse_datetime(retained["StartDateTime"]).astype(str)
    retained["end_datetime"] = _parse_datetime(retained["EndDateTime"]).astype(str)
    retained["season_id"] = _season_id_for_datetimes(
        _parse_datetime(retained["StartDateTime"])
    )
    retained["exposure_category"] = "game"
    retained["duration_minutes"] = pd.to_numeric(
        retained.get("Duration"), errors="coerce"
    )
    retained["rtp_flag"] = False
    football_rows = rows.loc[rows["external_squad_id"].eq(FOOTBALL_SQUAD_ID)]
    audit = {
        "source_events": int(len(rows)),
        "football_events": int(len(football_rows)),
        "non_football_events": int(
            rows["retention_reason"].eq("non_football_squad").sum()
        ),
        "excluded_by_reason": _value_counts(
            football_rows.loc[
                ~football_rows["retention_reason"].eq("retained"),
                "retention_reason",
            ]
        ),
        "retained_by_season": _value_counts(retained["season_id"]),
    }
    return retained.loc[:, _EVENT_COLUMNS], audit


_EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "external_event_id",
    "external_squad_id",
    "date",
    "start_datetime",
    "end_datetime",
    "season_id",
    "source_name",
    "session_type",
    "exposure_category",
    "duration_minutes",
    "rtp_flag",
]


def _build_participations(
    raw_participations: pd.DataFrame,
    athletes: pd.DataFrame,
    retained_events: pd.DataFrame,
    *,
    event_type: str,
    external_event_column: str,
) -> pd.DataFrame:
    rows = raw_participations.copy()
    rows["external_event_id"] = _text(rows, external_event_column)
    event_map = retained_events.loc[
        :,
        [
            "event_id",
            "event_type",
            "external_event_id",
            "date",
            "season_id",
            "exposure_category",
        ],
    ]
    joined = rows.merge(event_map, on="external_event_id", how="inner")
    joined["external_athlete_id"] = _text(joined, "ExternalAthleteId")
    eligible_athletes = athletes.loc[athletes["football_squad_member"]].copy()
    joined = joined.merge(
        eligible_athletes,
        on="external_athlete_id",
        how="left",
    )
    joined["athlete_id"] = joined["athlete_id"].fillna("")
    joined["athlete_match_status"] = joined["athlete_id"].map(
        lambda value: "matched" if str(value).strip() else "unmatched_athlete_id"
    )
    joined["participation_level"] = _text(joined, "ParticipationLevel")
    joined["participation_category"] = joined["participation_level"].map(
        _participation_category
    )
    joined["duration_minutes"] = pd.to_numeric(joined.get("Duration"), errors="coerce")
    joined["rpe"] = pd.to_numeric(joined.get("Rpe"), errors="coerce")
    joined["workload_unit_amount"] = pd.to_numeric(
        joined.get("WorkloadUnitAmount"), errors="coerce"
    )
    joined["workload_unit_type"] = _text(joined, "WorkloadUnitType")
    joined["participation_level_reason"] = _text(joined, "ParticipationLevelReason")
    joined["participation_level_reason_other"] = _text(
        joined, "ParticipationLevelReasonOther"
    )
    joined["related_external_issue_type"] = _text(joined, "RelatedExternalIssueType")
    joined["related_external_issue_id"] = _text(joined, "RelatedExternalIssueId")
    joined["source_file"] = (
        "training_session_participations.csv"
        if event_type == "training"
        else "game_participations.csv"
    )
    return joined.loc[:, _PARTICIPATION_COLUMNS]


_PARTICIPATION_COLUMNS = [
    "event_id",
    "event_type",
    "external_event_id",
    "external_athlete_id",
    "athlete_id",
    "athlete_match_status",
    "date",
    "season_id",
    "exposure_category",
    "participation_level",
    "participation_category",
    "participation_level_reason",
    "participation_level_reason_other",
    "related_external_issue_type",
    "related_external_issue_id",
    "duration_minutes",
    "rpe",
    "workload_unit_amount",
    "workload_unit_type",
    "source_file",
]


def _build_audit(
    *,
    frames: dict[str, pd.DataFrame],
    training_audit: dict[str, Any],
    game_audit: dict[str, Any],
    training_participations: pd.DataFrame,
    game_participations: pd.DataFrame,
    training_raw_participations: pd.DataFrame,
    game_raw_participations: pd.DataFrame,
    training_events: pd.DataFrame,
    game_events: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "input_files": {
            filename: {"rows": int(len(frame))}
            for filename, frame in frames.items()
        },
        "football_scope": _football_scope_summary(
            frames=frames,
            training_participations=training_participations,
            game_participations=game_participations,
        ),
        "event_counts": {
            "training": {
                "source_events": training_audit["source_events"],
                "football_events": training_audit["football_events"],
                "non_football_events": training_audit["non_football_events"],
                "retained_events": int(len(training_events)),
                "excluded_by_reason": training_audit["excluded_by_reason"],
                "retained_by_season": training_audit["retained_by_season"],
                "retained_by_session_type": training_audit[
                    "retained_by_session_type"
                ],
                "unclassified_session_types": training_audit[
                    "unclassified_session_types"
                ],
            },
            "game": {
                "source_events": game_audit["source_events"],
                "football_events": game_audit["football_events"],
                "non_football_events": game_audit["non_football_events"],
                "retained_events": int(len(game_events)),
                "excluded_by_reason": game_audit["excluded_by_reason"],
                "retained_by_season": game_audit["retained_by_season"],
            },
        },
        "participation_counts": {
            "training_rows": int(len(training_participations)),
            "game_rows": int(len(game_participations)),
            "training_by_participation_category": _value_counts(
                training_participations["participation_category"]
            ),
            "game_by_participation_category": _value_counts(
                game_participations["participation_category"]
            ),
        },
        "athlete_matching": {
            "training": _athlete_match_summary(training_participations),
            "game": _athlete_match_summary(game_participations),
        },
        "duplicate_keys": {
            "training_participation_duplicate_keys": _duplicate_key_count(
                training_raw_participations,
                external_event_column="ExternalSessionId",
            ),
            "game_participation_duplicate_keys": _duplicate_key_count(
                game_raw_participations,
                external_event_column="ExternalGameId",
            ),
        },
        "missing_duration": {
            "training_events_missing_duration": _missing_count(
                training_events["duration_minutes"]
            ),
            "game_events_missing_duration": _missing_count(
                game_events["duration_minutes"]
            ),
            "training_participations_missing_duration": _missing_count(
                training_participations["duration_minutes"]
            ),
            "game_participations_missing_duration": _missing_count(
                game_participations["duration_minutes"]
            ),
        },
        "candidate_feature_definitions": list(CANDIDATE_FEATURE_DEFINITIONS),
    }


def _football_scope_summary(
    *,
    frames: dict[str, pd.DataFrame],
    training_participations: pd.DataFrame,
    game_participations: pd.DataFrame,
) -> dict[str, int]:
    football_training_ids = set(
        _text(
            frames["training_sessions.csv"].loc[
                _text(frames["training_sessions.csv"], "ExternalSquadId").eq(
                    FOOTBALL_SQUAD_ID
                )
            ],
            "ExternalSessionId",
        )
    )
    football_game_ids = set(
        _text(
            frames["games.csv"].loc[
                _text(frames["games.csv"], "ExternalSquadId").eq(FOOTBALL_SQUAD_ID)
            ],
            "ExternalGameId",
        )
    )
    raw_training = frames["training_session_participations.csv"]
    raw_games = frames["game_participations.csv"]
    football_training_rows = _text(raw_training, "ExternalSessionId").isin(
        football_training_ids
    )
    football_game_rows = _text(raw_games, "ExternalGameId").isin(football_game_ids)
    return {
        "training_events": len(football_training_ids),
        "game_events": len(football_game_ids),
        "training_participation_rows": int(football_training_rows.sum()),
        "game_participation_rows": int(football_game_rows.sum()),
        "retained_training_participation_rows": int(len(training_participations)),
        "retained_game_participation_rows": int(len(game_participations)),
    }


def _training_retention_reason(row: pd.Series) -> str:
    if _normalized(row.get("ExternalSquadId")) != FOOTBALL_SQUAD_ID:
        return "non_football_squad"
    text = f"{_normalized(row.get('SessionType'))} {_normalized(row.get('Name'))}"
    if any(term in text for term in API_PERFORMANCE_SOURCE_TERMS):
        return "api_performance_source"
    if _training_category(row) != "unclassified":
        return "retained"
    return "unclassified"


def _training_category(row: pd.Series) -> str:
    session_type = _normalized(row.get("SessionType"))
    name = _normalized(row.get("Name"))
    if session_type in TRAINING_EXPOSURE_CATEGORIES:
        return TRAINING_EXPOSURE_CATEGORIES[session_type]
    if name in TRAINING_EXPOSURE_CATEGORIES:
        return TRAINING_EXPOSURE_CATEGORIES[name]
    if "practice" in session_type or "practice" in name:
        return "practice"
    return "unclassified"


def _participation_category(value: object) -> str:
    normalized = _normalized(value)
    if not normalized:
        return "unknown"
    if "no participation" in normalized:
        return "no_participation"
    if "modified" in normalized:
        return "modified"
    if "partial" in normalized:
        return "partial"
    if "full" in normalized:
        return "full"
    return "other"


def _athlete_match_summary(participations: pd.DataFrame) -> dict[str, int]:
    matched = int(participations["athlete_match_status"].eq("matched").sum())
    unmatched = int(
        participations["athlete_match_status"].eq("unmatched_athlete_id").sum()
    )
    return {
        "matched_participation_rows": matched,
        "unmatched_participation_rows": unmatched,
        "unique_external_athlete_ids": int(
            participations["external_athlete_id"].nunique()
        ),
        "unique_matched_athletes": int(
            participations.loc[
                participations["athlete_match_status"].eq("matched"), "athlete_id"
            ].nunique()
        ),
    }


def _duplicate_key_count(
    participations: pd.DataFrame,
    *,
    external_event_column: str,
) -> int:
    if participations.empty:
        return 0
    groups = (
        participations.assign(
            external_event_id=_text(participations, external_event_column),
            external_athlete_id=_text(participations, "ExternalAthleteId"),
        )
        .groupby(["external_event_id", "external_athlete_id"], as_index=False)
        .size()
    )
    return int((groups["size"] > 1).sum())


def _contains_football_squad(value: object) -> bool:
    squad_ids = [part.strip() for part in re.split(r"[|,;]", str(value)) if part.strip()]
    return FOOTBALL_SQUAD_ID in squad_ids


def _season_id_for_datetimes(datetimes: pd.Series) -> pd.Series:
    years = datetimes.dt.year
    start_years = years.where(datetimes.dt.month >= 7, years - 1)
    end_years = start_years + 1
    season_ids = start_years.astype("Int64").astype(str) + "-" + end_years.astype(
        "Int64"
    ).astype(str)
    return season_ids.mask(datetimes.isna(), "")


def _parse_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _text(frame_or_series: pd.DataFrame | pd.Series, column: str) -> pd.Series:
    if isinstance(frame_or_series, pd.Series):
        return frame_or_series.astype(str).str.strip()
    if column not in frame_or_series.columns:
        return pd.Series([""] * len(frame_or_series), index=frame_or_series.index)
    return frame_or_series[column].astype(str).str.strip()


def _normalized(value: object) -> str:
    return " ".join(str(value or "").lower().replace("\u00a0", " ").split())


def _value_counts(values: pd.Series) -> dict[str, int]:
    if values.empty:
        return {}
    counts = values.fillna("").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items() if str(key)}


def _missing_count(values: pd.Series) -> int:
    return int(values.isna().sum())
