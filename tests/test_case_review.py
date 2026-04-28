from __future__ import annotations

import pandas as pd

from risk_stratification_engine.case_review import build_qualitative_case_review


def _alert_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-01-01",
                "risk_7d": 0.10,
                "risk_14d": 0.20,
                "risk_30d": 0.90,
                "top_feature_30d": "mean_abs_correlation",
                "top_contribution_30d": 0.70,
                "elevated_z_features": ["z_mean_abs_correlation"],
                "event_observed": True,
                "event_date": "2026-01-10",
                "days_to_event": 9,
                "injury_type": "hamstring",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 2,
                "primary_model_event": True,
            },
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "time_index": 1,
                "snapshot_date": "2026-01-02",
                "risk_7d": 0.08,
                "risk_14d": 0.18,
                "risk_30d": 0.82,
                "top_feature_30d": "mean_abs_correlation",
                "top_contribution_30d": 0.60,
                "elevated_z_features": [],
                "event_observed": True,
                "event_date": "2026-01-10",
                "days_to_event": 8,
                "injury_type": "hamstring",
                "event_window_quality": "modelable",
                "nearest_measurement_gap_days": 2,
                "primary_model_event": True,
            },
            {
                "athlete_id": "a2",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-02-01",
                "risk_7d": 0.03,
                "risk_14d": 0.05,
                "risk_30d": 0.10,
                "top_feature_30d": "edge_density",
                "top_contribution_30d": -0.20,
                "elevated_z_features": [],
                "event_observed": True,
                "event_date": "2026-02-20",
                "days_to_event": 19,
                "injury_type": "ankle",
                "event_window_quality": "out_of_window",
                "nearest_measurement_gap_days": 45,
                "primary_model_event": False,
            },
            {
                "athlete_id": "a3",
                "season_id": "2026",
                "time_index": 0,
                "snapshot_date": "2026-03-01",
                "risk_7d": 0.04,
                "risk_14d": 0.08,
                "risk_30d": 0.80,
                "top_feature_30d": "mean_abs_correlation",
                "top_contribution_30d": 0.50,
                "elevated_z_features": ["z_graph_instability"],
                "event_observed": False,
                "event_date": "2026-03-30",
                "days_to_event": 29,
                "injury_type": "censored",
                "event_window_quality": "censored",
                "nearest_measurement_gap_days": None,
                "primary_model_event": False,
            },
        ]
    )


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "athlete_id": "a1",
                "season_id": "2026",
                "horizon_days": 30,
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
                "start_time_index": 0,
                "end_time_index": 1,
                "peak_time_index": 0,
                "start_date": "2026-01-01",
                "peak_date": "2026-01-01",
                "end_date": "2026-01-02",
                "peak_risk": 0.90,
                "mean_risk": 0.86,
                "injury_type": "hamstring",
                "days_from_start_to_event": 9,
                "days_from_peak_to_event": 9,
                "days_from_end_to_event": 8,
                "event_within_horizon_after_start": True,
                "top_model_features": [
                    {"feature": "mean_abs_correlation", "mean_abs_contribution": 0.7}
                ],
                "elevated_z_features": ["z_mean_abs_correlation"],
            },
            {
                "athlete_id": "a3",
                "season_id": "2026",
                "horizon_days": 30,
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
                "start_time_index": 0,
                "end_time_index": 0,
                "peak_time_index": 0,
                "start_date": "2026-03-01",
                "peak_date": "2026-03-01",
                "end_date": "2026-03-01",
                "peak_risk": 0.80,
                "mean_risk": 0.80,
                "injury_type": "censored",
                "days_from_start_to_event": None,
                "days_from_peak_to_event": None,
                "days_from_end_to_event": None,
                "event_within_horizon_after_start": False,
                "top_model_features": [
                    {"feature": "mean_abs_correlation", "mean_abs_contribution": 0.5}
                ],
                "elevated_z_features": ["z_graph_instability"],
            },
        ]
    )


def _quality() -> dict[str, object]:
    return {
        "quality_rows": [
            {
                "horizon_days": 30,
                "threshold": "percentile:0.05",
                "threshold_kind": "percentile",
                "threshold_value": 0.05,
            }
        ],
        "representative_cases": {
            "30": {
                "percentile:0.05": {
                    "true_positive_episode": {
                        "athlete_id": "a1",
                        "season_id": "2026",
                        "horizon_days": 30,
                        "threshold": "percentile:0.05",
                        "start_date": "2026-01-01",
                        "peak_date": "2026-01-01",
                        "end_date": "2026-01-02",
                    },
                    "false_positive_episode": {
                        "athlete_id": "a3",
                        "season_id": "2026",
                        "horizon_days": 30,
                        "threshold": "percentile:0.05",
                        "start_date": "2026-03-01",
                        "peak_date": "2026-03-01",
                        "end_date": "2026-03-01",
                    },
                    "missed_injury": {
                        "athlete_id": "a2",
                        "season_id": "2026",
                        "event_date": "2026-02-20 00:00:00",
                        "injury_type": "ankle",
                    },
                    "high_intra_individual_deviation_episode": {
                        "athlete_id": "a3",
                        "season_id": "2026",
                        "horizon_days": 30,
                        "threshold": "percentile:0.05",
                        "start_date": "2026-03-01",
                        "peak_date": "2026-03-01",
                        "end_date": "2026-03-01",
                    },
                }
            }
        },
    }


def test_build_qualitative_case_review_selects_and_labels_representative_cases():
    review = build_qualitative_case_review(_episodes(), _alert_timeline(), _quality())

    assert review["case_count"] == 4
    cases = {case["case_type"]: case for case in review["cases"]}

    assert cases["true_positive_episode"]["diagnostic_label"] == "model_signal_supported"
    assert cases["true_positive_episode"]["review_label"] == "Useful warning"
    assert cases["false_positive_episode"]["diagnostic_label"] == (
        "missing_context_or_managed_risk"
    )
    assert cases["missed_injury"]["diagnostic_label"] == (
        "possible_label_or_measurement_gap"
    )
    assert cases["high_intra_individual_deviation_episode"]["diagnostic_label"] == (
        "explanation_gap"
    )


def test_build_qualitative_case_review_includes_timeline_context_and_data_diagnostics():
    review = build_qualitative_case_review(_episodes(), _alert_timeline(), _quality())

    true_positive = next(
        case for case in review["cases"] if case["case_type"] == "true_positive_episode"
    )
    missed = next(case for case in review["cases"] if case["case_type"] == "missed_injury")

    assert true_positive["data_diagnostics"] == {
        "event_window_quality": "modelable",
        "nearest_measurement_gap_days": 2,
        "primary_model_event": True,
    }
    assert true_positive["timeline_context"] == [
        {
            "time_index": 0,
            "snapshot_date": "2026-01-01",
            "risk_7d": 0.1,
            "risk_14d": 0.2,
            "risk_30d": 0.9,
            "top_feature_30d": "mean_abs_correlation",
            "top_contribution_30d": 0.7,
            "elevated_z_features": ["z_mean_abs_correlation"],
            "event_observed": True,
            "event_date": "2026-01-10",
            "days_to_event": 9,
            "injury_type": "hamstring",
            "event_window_quality": "modelable",
            "nearest_measurement_gap_days": 2,
            "primary_model_event": True,
        },
        {
            "time_index": 1,
            "snapshot_date": "2026-01-02",
            "risk_7d": 0.08,
            "risk_14d": 0.18,
            "risk_30d": 0.82,
            "top_feature_30d": "mean_abs_correlation",
            "top_contribution_30d": 0.6,
            "elevated_z_features": [],
            "event_observed": True,
            "event_date": "2026-01-10",
            "days_to_event": 8,
            "injury_type": "hamstring",
            "event_window_quality": "modelable",
            "nearest_measurement_gap_days": 2,
            "primary_model_event": True,
        },
    ]
    assert missed["event_date"] == "2026-02-20"
    assert missed["data_diagnostics"]["event_window_quality"] == "out_of_window"


def test_build_qualitative_case_review_summarizes_diagnostic_labels():
    review = build_qualitative_case_review(_episodes(), _alert_timeline(), _quality())

    assert review["diagnostic_summary"] == {
        "explanation_gap": 1,
        "missing_context_or_managed_risk": 1,
        "model_signal_supported": 1,
        "possible_label_or_measurement_gap": 1,
    }
