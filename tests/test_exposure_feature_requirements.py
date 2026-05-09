import json
from pathlib import Path

import pandas as pd

from risk_stratification_engine.exposure_feature_requirements import (
    build_exposure_feature_requirements,
    build_exposure_feature_requirements_summary,
    write_exposure_feature_requirements_report,
)


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "training:1",
                "event_type": "training",
                "date": "2025-08-01",
                "season_id": "2025-2026",
                "exposure_category": "practice_shells",
                "duration_minutes": 120,
            },
            {
                "event_id": "training:2",
                "event_type": "training",
                "date": "2025-08-02",
                "season_id": "2025-2026",
                "exposure_category": "weight_room",
                "duration_minutes": 60,
            },
            {
                "event_id": "game:1",
                "event_type": "game",
                "date": "2025-09-01",
                "season_id": "2025-2026",
                "exposure_category": "game",
                "duration_minutes": 180,
            },
        ]
    )


def _sample_participations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "training:1",
                "event_type": "training",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2025-08-01",
                "season_id": "2025-2026",
                "exposure_category": "practice_shells",
                "participation_category": "full",
                "duration_minutes": 110,
            },
            {
                "event_id": "training:1",
                "event_type": "training",
                "athlete_id": "a2",
                "athlete_match_status": "matched",
                "date": "2025-08-01",
                "season_id": "2025-2026",
                "exposure_category": "practice_shells",
                "participation_category": "modified",
                "duration_minutes": "",
            },
            {
                "event_id": "training:2",
                "event_type": "training",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2025-08-02",
                "season_id": "2025-2026",
                "exposure_category": "weight_room",
                "participation_category": "no_participation",
                "duration_minutes": "",
            },
            {
                "event_id": "game:1",
                "event_type": "game",
                "athlete_id": "a1",
                "athlete_match_status": "matched",
                "date": "2025-09-01",
                "season_id": "2025-2026",
                "exposure_category": "game",
                "participation_category": "full",
                "duration_minutes": "",
            },
        ]
    )


def _sample_audit() -> dict[str, object]:
    return {
        "event_counts": {
            "training": {
                "retained_events": 2,
                "excluded_by_reason": {"api_performance_source": 1},
                "unclassified_session_types": {},
                "retained_by_session_type": {
                    "Practice - Shells": 1,
                    "Weight Room": 1,
                },
            },
            "game": {"retained_events": 1},
        },
        "participation_counts": {
            "training_rows": 3,
            "game_rows": 1,
            "training_by_participation_category": {
                "full": 1,
                "modified": 1,
                "no_participation": 1,
            },
            "game_by_participation_category": {"full": 1},
        },
        "athlete_matching": {
            "training": {
                "matched_participation_rows": 3,
                "unmatched_participation_rows": 0,
            },
            "game": {
                "matched_participation_rows": 1,
                "unmatched_participation_rows": 0,
            },
        },
        "duplicate_keys": {
            "training_participation_duplicate_keys": 0,
            "game_participation_duplicate_keys": 0,
        },
        "missing_duration": {
            "training_participations_missing_duration": 2,
            "game_participations_missing_duration": 1,
        },
        "candidate_feature_definitions": [
            "prior_7_day_training_session_count",
            "prior_14_day_training_duration_minutes",
            "prior_game_count",
        ],
    }


def test_build_exposure_feature_requirements_prioritizes_count_and_status_features():
    requirements = build_exposure_feature_requirements(
        _sample_events(),
        _sample_participations(),
        _sample_audit(),
    )

    domains = {row["requirement_domain"]: row for row in requirements}
    assert domains["session_count_load"]["readiness_status"] == "ready"
    assert domains["participation_status"]["readiness_status"] == "ready"
    assert domains["duration_load"]["readiness_status"] == "caution"
    assert domains["game_exposure"]["readiness_status"] == "caution"
    assert "prior_7_day_training_session_count" in domains["session_count_load"][
        "recommended_first_pass_features"
    ]
    assert "training_duration_missing_share=0.6667" in domains["duration_load"][
        "evidence_summary"
    ]


def test_build_exposure_feature_requirements_summary_tracks_readiness_and_next_step():
    requirements = build_exposure_feature_requirements(
        _sample_events(),
        _sample_participations(),
        _sample_audit(),
    )

    summary = build_exposure_feature_requirements_summary(
        _sample_events(),
        _sample_participations(),
        _sample_audit(),
        requirements,
    )

    assert summary["experiment_type"] == "exposure_feature_requirements_sprint"
    assert summary["overall_recommendation"] == (
        "proceed_with_count_and_status_features_first"
    )
    assert summary["category_summary"]["training"]["practice_shells"] == 1
    assert summary["duration_summary"]["game"]["missing_share"] == 1.0
    assert summary["readiness_summary"] == {"caution": 2, "ready": 3}


def test_write_exposure_feature_requirements_report(tmp_path):
    requirements = build_exposure_feature_requirements(
        _sample_events(),
        _sample_participations(),
        _sample_audit(),
    )
    summary = build_exposure_feature_requirements_summary(
        _sample_events(),
        _sample_participations(),
        _sample_audit(),
        requirements,
    )

    report_path = tmp_path / "exposure_feature_requirements_report.md"
    write_exposure_feature_requirements_report(report_path, summary)

    report = report_path.read_text(encoding="utf-8")
    assert "Exposure Feature Requirements Sprint" in report
    assert "count and participation-status features first" in report


def _write_exposure_artifacts(folder: Path) -> None:
    folder.mkdir(parents=True)
    _sample_events().to_csv(folder / "exposure_events.csv", index=False)
    _sample_participations().to_csv(
        folder / "exposure_participations.csv",
        index=False,
    )
    (folder / "exposure_cleaning_audit.json").write_text(
        json.dumps(_sample_audit()),
        encoding="utf-8",
    )


def test_run_exposure_feature_requirements_sprint_writes_artifacts(tmp_path):
    from risk_stratification_engine.experiments import (
        run_exposure_feature_requirements_sprint_experiment,
    )

    exposure_dir = tmp_path / "exposure_inputs" / "prepared"
    _write_exposure_artifacts(exposure_dir)

    result = run_exposure_feature_requirements_sprint_experiment(
        exposure_events_path=exposure_dir / "exposure_events.csv",
        exposure_participations_path=exposure_dir / "exposure_participations.csv",
        exposure_audit_path=exposure_dir / "exposure_cleaning_audit.json",
        output_dir=tmp_path,
        experiment_id="exposure_feature_requirements_v1",
    )

    assert (result / "exposure_category_summary.csv").exists()
    assert (result / "exposure_duration_summary.csv").exists()
    assert (result / "exposure_feature_requirements.csv").exists()
    summary = json.loads(
        (result / "exposure_feature_requirements.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["overall_recommendation"] == (
        "proceed_with_count_and_status_features_first"
    )
