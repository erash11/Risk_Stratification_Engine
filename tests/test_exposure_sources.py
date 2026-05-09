import json
from pathlib import Path

import pandas as pd
import pytest

from risk_stratification_engine.live_sources import stable_athlete_id


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_minimal_exposure_source(folder: Path) -> None:
    folder.mkdir(parents=True)
    _write_csv(
        folder / "athletes.csv",
        [
            {
                "ExternalAthleteId": "1",
                "FirstName": "Jane",
                "LastName": "Athlete",
                "Position": "Football - Offense: Receiver",
                "ExternalSquadIds": "94|105",
            },
            {
                "ExternalAthleteId": "2",
                "FirstName": "Zandbox",
                "LastName": "Athlete",
                "Position": "Football - Defense: Safety",
                "ExternalSquadIds": "106",
            },
        ],
    )
    _write_csv(
        folder / "squads.csv",
        [
            {"ExternalSquadId": "94", "Name": "Football", "Sport": "Football"},
            {"ExternalSquadId": "106", "Name": "Zandbox", "Sport": "Football"},
        ],
    )
    _write_csv(
        folder / "training_sessions.csv",
        [
            {
                "ExternalSessionId": "100",
                "ExternalSquadId": "94",
                "StartDateTime": "2025-08-01T14:00:00Z",
                "EndDateTime": "2025-08-01T16:00:00Z",
                "Name": "Practice - Shells",
                "SessionType": "Practice - Shells",
                "Duration": "120",
                "SessionCategory": "Training",
                "GameDayPlus": "",
                "GameDayMinus": "",
            },
            {
                "ExternalSessionId": "101",
                "ExternalSquadId": "94",
                "StartDateTime": "2025-08-02T14:00:00Z",
                "EndDateTime": "2025-08-02T15:00:00Z",
                "Name": "Perch - Weight Room",
                "SessionType": "Perch - Weight Room",
                "Duration": "60",
                "SessionCategory": "Training",
                "GameDayPlus": "",
                "GameDayMinus": "",
            },
            {
                "ExternalSessionId": "102",
                "ExternalSquadId": "94",
                "StartDateTime": "2025-08-03T14:00:00Z",
                "EndDateTime": "2025-08-03T15:00:00Z",
                "Name": "Catapult - FB School",
                "SessionType": "Catapult - FB School",
                "Duration": "60",
                "SessionCategory": "Training",
                "GameDayPlus": "",
                "GameDayMinus": "",
            },
            {
                "ExternalSessionId": "103",
                "ExternalSquadId": "106",
                "StartDateTime": "2025-08-04T14:00:00Z",
                "EndDateTime": "2025-08-04T15:00:00Z",
                "Name": "Practice - Shells",
                "SessionType": "Practice - Shells",
                "Duration": "60",
                "SessionCategory": "Training",
                "GameDayPlus": "",
                "GameDayMinus": "",
            },
        ],
    )
    _write_csv(
        folder / "training_session_participations.csv",
        [
            {
                "ExternalSessionId": "100",
                "ExternalGameId": "",
                "ExternalAthleteId": "1",
                "ParticipationLevel": "Full",
                "ParticipationLevelReason": "",
                "RelatedExternalIssueType": "",
                "RelatedExternalIssueId": "",
                "Duration": "110",
                "Rpe": "7",
                "WorkloadUnitAmount": "",
                "WorkloadUnitType": "",
                "ParticipationLevelReasonOther": "",
            },
            {
                "ExternalSessionId": "100",
                "ExternalGameId": "",
                "ExternalAthleteId": "999",
                "ParticipationLevel": "Modified",
                "ParticipationLevelReason": "Return to play",
                "RelatedExternalIssueType": "Issue",
                "RelatedExternalIssueId": "abc",
                "Duration": "",
                "Rpe": "",
                "WorkloadUnitAmount": "",
                "WorkloadUnitType": "",
                "ParticipationLevelReasonOther": "",
            },
            {
                "ExternalSessionId": "100",
                "ExternalGameId": "",
                "ExternalAthleteId": "2",
                "ParticipationLevel": "Full",
                "ParticipationLevelReason": "",
                "RelatedExternalIssueType": "",
                "RelatedExternalIssueId": "",
                "Duration": "100",
                "Rpe": "",
                "WorkloadUnitAmount": "",
                "WorkloadUnitType": "",
                "ParticipationLevelReasonOther": "",
            },
            {
                "ExternalSessionId": "101",
                "ExternalGameId": "",
                "ExternalAthleteId": "1",
                "ParticipationLevel": "Full",
                "ParticipationLevelReason": "",
                "RelatedExternalIssueType": "",
                "RelatedExternalIssueId": "",
                "Duration": "60",
                "Rpe": "",
                "WorkloadUnitAmount": "",
                "WorkloadUnitType": "",
                "ParticipationLevelReasonOther": "",
            },
        ],
    )
    _write_csv(
        folder / "games.csv",
        [
            {
                "ExternalGameId": "200",
                "ExternalSquadId": "94",
                "StartDateTime": "2025-09-06T17:00:00Z",
                "EndDateTime": "2025-09-06T20:00:00Z",
                "Name": "",
                "Duration": "180",
                "Competition": "Big 12",
                "OpponentTeam": "Opponent",
                "Venue": "Home",
                "SeasonSegment": "Regular",
            },
            {
                "ExternalGameId": "201",
                "ExternalSquadId": "106",
                "StartDateTime": "2025-09-06T17:00:00Z",
                "EndDateTime": "2025-09-06T20:00:00Z",
                "Name": "",
                "Duration": "180",
                "Competition": "Other",
                "OpponentTeam": "Opponent",
                "Venue": "Home",
                "SeasonSegment": "Regular",
            },
        ],
    )
    _write_csv(
        folder / "game_participations.csv",
        [
            {
                "ExternalGameId": "200",
                "ExternalSessionId": "",
                "ExternalAthleteId": "1",
                "ParticipationLevel": "Full Participation",
                "ParticipationLevelReason": "",
                "RelatedExternalIssueType": "",
                "RelatedExternalIssueId": "",
                "Duration": "180",
                "Rpe": "",
                "WorkloadUnitAmount": "",
                "WorkloadUnitType": "",
                "ParticipationLevelReasonOther": "",
            },
            {
                "ExternalGameId": "200",
                "ExternalSessionId": "",
                "ExternalAthleteId": "999",
                "ParticipationLevel": "No Participation - Medical",
                "ParticipationLevelReason": "Medical",
                "RelatedExternalIssueType": "Issue",
                "RelatedExternalIssueId": "def",
                "Duration": "",
                "Rpe": "",
                "WorkloadUnitAmount": "",
                "WorkloadUnitType": "",
                "ParticipationLevelReasonOther": "",
            },
        ],
    )


def _import_exposure_module():
    try:
        import risk_stratification_engine.exposure_sources as exposure_sources
    except ModuleNotFoundError as exc:
        pytest.fail(f"expected exposure_sources module to exist: {exc}")
    return exposure_sources


def test_prepare_exposure_inputs_filters_football_and_writes_clean_artifacts(tmp_path):
    exposure_sources = _import_exposure_module()
    source_dir = tmp_path / "Baylor_Exposure_Data"
    _write_minimal_exposure_source(source_dir)

    result = exposure_sources.prepare_exposure_inputs(
        source_dir,
        tmp_path / "prepared_exposure",
    )

    events = pd.read_csv(result.events_path)
    participations = pd.read_csv(result.participations_path)
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))

    assert events["event_id"].tolist() == ["training:100", "game:200"]
    assert events["season_id"].tolist() == ["2025-2026", "2025-2026"]
    assert events["event_type"].tolist() == ["training", "game"]
    assert events.loc[0, "exposure_category"] == "practice_shells"
    assert events.loc[1, "exposure_category"] == "game"
    assert participations["event_id"].tolist() == [
        "training:100",
        "training:100",
        "training:100",
        "game:200",
        "game:200",
    ]
    assert participations.loc[0, "athlete_id"] == stable_athlete_id("Jane Athlete")
    assert participations.loc[1, "athlete_match_status"] == "unmatched_athlete_id"
    zandbox_row = participations.loc[participations["external_athlete_id"].eq(2)].iloc[
        0
    ]
    assert zandbox_row["athlete_match_status"] == "unmatched_athlete_id"
    assert participations.loc[3, "participation_category"] == "full"
    assert participations.loc[4, "participation_category"] == "no_participation"
    assert audit["event_counts"]["training"]["retained_events"] == 1
    assert audit["event_counts"]["training"]["football_events"] == 3
    assert audit["event_counts"]["training"]["non_football_events"] == 1
    assert audit["football_scope"]["training_participation_rows"] == 4
    assert audit["football_scope"]["game_participation_rows"] == 2
    assert audit["football_scope"]["retained_training_participation_rows"] == 3
    assert audit["football_scope"]["retained_game_participation_rows"] == 2
    assert audit["event_counts"]["training"]["excluded_by_reason"] == {
        "api_performance_source": 2
    }
    assert audit["event_counts"]["game"]["retained_events"] == 1
    assert audit["athlete_matching"]["training"]["unmatched_participation_rows"] == 2
    assert audit["athlete_matching"]["game"]["unmatched_participation_rows"] == 1
    assert audit["duplicate_keys"]["training_participation_duplicate_keys"] == 0
    assert audit["duplicate_keys"]["game_participation_duplicate_keys"] == 0
    assert "prior_7_day_training_session_count" in audit["candidate_feature_definitions"]


def test_prepare_exposure_inputs_reports_duplicate_athlete_event_keys(tmp_path):
    exposure_sources = _import_exposure_module()
    source_dir = tmp_path / "Baylor_Exposure_Data"
    _write_minimal_exposure_source(source_dir)
    path = source_dir / "training_session_participations.csv"
    frame = pd.read_csv(path, dtype=str)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(path, index=False)

    result = exposure_sources.prepare_exposure_inputs(
        source_dir,
        tmp_path / "prepared_exposure",
    )

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["duplicate_keys"]["training_participation_duplicate_keys"] == 1
