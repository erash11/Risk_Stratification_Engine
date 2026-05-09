from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from risk_stratification_engine.exposure_load_failure_modes import (
    EXPOSURE_LOAD_FEATURE_DOMAINS,
)
from risk_stratification_engine.exposure_load_features import (
    EXPOSURE_LOAD_FEATURE_COLUMNS,
)


DOMAIN_FEATURES = {
    "game_exposure": (
        "exposure_games_prior_count",
        "exposure_days_since_last_game",
        "exposure_game_events_28d",
    ),
    "participation_status": (
        "exposure_full_participations_28d",
        "exposure_modified_participations_28d",
        "exposure_no_participations_28d",
        "exposure_days_since_last_modified_or_no_participation",
    ),
    "category_specific_load": (
        "exposure_practice_sessions_28d",
        "exposure_lift_sessions_28d",
        "exposure_conditioning_sessions_28d",
        "exposure_rtp_sessions_28d",
    ),
    "training_session_load": (
        "exposure_training_sessions_7d",
        "exposure_training_sessions_14d",
        "exposure_training_sessions_28d",
    ),
}


def build_exposure_load_shift_context_summary(
    exposure_events: list[dict[str, object]],
    exposure_participations: list[dict[str, object]],
    exposure_load_features: list[dict[str, object]],
    exposure_load_diagnostics: list[dict[str, object]],
    exposure_load_failure_modes: dict[str, object],
) -> dict[str, object]:
    events = _normalized_events(pd.DataFrame(exposure_events))
    participations = _normalized_participations(pd.DataFrame(exposure_participations))
    features = _normalized_features(pd.DataFrame(exposure_load_features))
    diagnostics = pd.DataFrame(exposure_load_diagnostics)

    failure_seasons = _failure_seasons(exposure_load_failure_modes, diagnostics)
    comparator_seasons = _comparator_seasons(exposure_load_failure_modes, diagnostics)
    target_seasons = sorted(set(failure_seasons + comparator_seasons))
    shift_context_rows = _shift_context_rows(
        target_seasons,
        failure_seasons,
        comparator_seasons,
        events,
        participations,
        features,
    )
    failure_context = _aggregate_domain_context(shift_context_rows, failure_seasons)
    comparator_context = _aggregate_domain_context(shift_context_rows, comparator_seasons)
    driver_context_rows = _driver_context_rows(
        exposure_load_failure_modes.get("top_driver_features", []),
        failure_context,
        comparator_context,
    )
    shift_context_cases = _shift_context_cases(
        diagnostics,
        failure_seasons,
        driver_context_rows,
    )

    return {
        "experiment_type": "exposure_load_shift_context_sprint",
        "overall_recommendation": _overall_recommendation(
            failure_seasons,
            driver_context_rows,
        ),
        "failure_seasons": failure_seasons,
        "comparator_seasons": comparator_seasons,
        "diagnostic_label_summary": _counter(
            exposure_load_diagnostics,
            "diagnostic_label",
        ),
        "target_reason_summary": _counter(exposure_load_diagnostics, "target_reason"),
        "failure_context": failure_context,
        "comparator_context": comparator_context,
        "shift_context_rows": shift_context_rows,
        "driver_context_rows": driver_context_rows,
        "shift_context_cases": shift_context_cases,
    }


def write_exposure_load_shift_context_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Shift Context Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint reviews shifted exposure-load domains after complete "
            "athlete-season trajectories have already been scored. It joins "
            "the failed forward slice back to schedule, roster, availability, "
            "and managed-risk context without changing the model or creating "
            "independent daily injury-classification rows."
        ),
        "",
        "## Target Seasons",
        "",
        "- Failure seasons: " + ", ".join(summary.get("failure_seasons", [])),
        "- Comparator seasons: " + ", ".join(summary.get("comparator_seasons", [])),
        "",
        "## Driver Context",
        "",
        "| Feature | Domain | Shift | Context signal | Failure mean | Comparator mean |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in summary.get("driver_context_rows", []):
        lines.append(
            "| "
            f"{row['feature_name']} | "
            f"{row['context_domain']} | "
            f"{row['shift_direction']} | "
            f"{row['context_signal']} | "
            f"{_fmt(row.get('failure_context_value'))} | "
            f"{_fmt(row.get('comparator_context_value'))} |"
        )
    lines.extend(
        [
            "",
            "## Season Context",
            "",
            "| Season | Group | Domain | Key signal | Events | Athletes | Modified rate |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary.get("shift_context_rows", []):
        lines.append(
            "| "
            f"{row['season_id']} | "
            f"{row['context_group']} | "
            f"{row['context_domain']} | "
            f"{row['primary_context_signal']} | "
            f"{_fmt(row.get('event_count'))} | "
            f"{_fmt(row.get('matched_athlete_count'))} | "
            f"{_fmt(row.get('modified_participation_rate'))} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _shift_context_rows(
    seasons: list[str],
    failure_seasons: list[str],
    comparator_seasons: list[str],
    events: pd.DataFrame,
    participations: pd.DataFrame,
    features: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for season in seasons:
        event_metrics = _event_metrics(events[events["season_id"].eq(season)])
        participation_metrics = _participation_metrics(
            participations[participations["season_id"].eq(season)]
        )
        feature_metrics = _feature_metrics(features[features["season_id"].eq(season)])
        group = "failure" if season in failure_seasons else "comparator"
        for domain, feature_names in DOMAIN_FEATURES.items():
            row = {
                "season_id": season,
                "context_group": group,
                "context_domain": domain,
                **event_metrics,
                **participation_metrics,
            }
            for feature_name in feature_names:
                row[_feature_metric_name(feature_name)] = feature_metrics.get(
                    feature_name
                )
            row["primary_context_signal"] = _primary_context_signal(domain, row)
            rows.append(_clean_row(row))
    return rows


def _event_metrics(events: pd.DataFrame) -> dict[str, object]:
    if events.empty:
        return {
            "event_count": 0,
            "game_event_count": 0,
            "training_event_count": 0,
            "practice_event_count": 0,
            "lift_event_count": 0,
            "conditioning_event_count": 0,
            "rtp_event_count": 0,
            "event_duration_recorded_rate": None,
            "event_median_duration_minutes": None,
        }
    event_type = events["event_type"].astype(str).str.lower()
    category = events["exposure_category"].astype(str).str.lower()
    duration = pd.to_numeric(events.get("duration_minutes"), errors="coerce")
    return {
        "event_count": int(events["event_id"].nunique())
        if "event_id" in events
        else int(len(events)),
        "game_event_count": int((event_type.eq("game") | category.eq("game")).sum()),
        "training_event_count": int(event_type.eq("training").sum()),
        "practice_event_count": int(
            (category.str.contains("practice") | category.eq("scrimmage")).sum()
        ),
        "lift_event_count": int(category.str.contains("weight_room").sum()),
        "conditioning_event_count": int(
            (
                category.str.contains("conditioning")
                | category.str.contains("speed_power")
            ).sum()
        ),
        "rtp_event_count": int(_truthy_series(events.get("rtp_flag")).sum()),
        "event_duration_recorded_rate": _recorded_rate(duration),
        "event_median_duration_minutes": _median(duration),
    }


def _participation_metrics(participations: pd.DataFrame) -> dict[str, object]:
    if participations.empty:
        return {
            "matched_athlete_count": 0,
            "matched_participation_count": 0,
            "full_participation_count": 0,
            "modified_participation_count": 0,
            "no_participation_count": 0,
            "modified_participation_rate": None,
            "no_participation_rate": None,
            "linked_issue_participation_count": 0,
            "participation_duration_recorded_rate": None,
            "rpe_recorded_rate": None,
            "workload_recorded_rate": None,
        }
    frame = participations.copy()
    if "athlete_match_status" in frame:
        frame = frame[frame["athlete_match_status"].astype(str).eq("matched")]
    category = frame["participation_category"].astype(str).str.lower()
    total = len(frame)
    linked_issue = (
        frame.get("related_external_issue_id", pd.Series([], dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    return {
        "matched_athlete_count": int(frame["athlete_id"].nunique())
        if "athlete_id" in frame
        else 0,
        "matched_participation_count": int(total),
        "full_participation_count": int(category.eq("full").sum()),
        "modified_participation_count": int(category.eq("modified").sum()),
        "no_participation_count": int(category.eq("no_participation").sum()),
        "modified_participation_rate": _safe_rate(int(category.eq("modified").sum()), total),
        "no_participation_rate": _safe_rate(
            int(category.eq("no_participation").sum()),
            total,
        ),
        "linked_issue_participation_count": int(linked_issue.sum()),
        "participation_duration_recorded_rate": _recorded_rate(
            _numeric_column(frame, "duration_minutes")
        ),
        "rpe_recorded_rate": _recorded_rate(_numeric_column(frame, "rpe")),
        "workload_recorded_rate": _recorded_rate(
            _numeric_column(frame, "workload_unit_amount")
        ),
    }


def _feature_metrics(features: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for feature_name in EXPOSURE_LOAD_FEATURE_COLUMNS:
        if feature_name not in features:
            continue
        metrics[feature_name] = _mean(pd.to_numeric(features[feature_name], errors="coerce"))
    return metrics


def _aggregate_domain_context(
    rows: list[dict[str, object]],
    seasons: list[str],
) -> dict[str, dict[str, object]]:
    if not rows or not seasons:
        return {}
    frame = pd.DataFrame(rows)
    frame = frame[frame["season_id"].astype(str).isin(seasons)]
    if frame.empty:
        return {}
    summary: dict[str, dict[str, object]] = {}
    for domain, group in frame.groupby("context_domain", sort=True):
        metrics: dict[str, object] = {}
        for column in group.columns:
            if column in {"season_id", "context_group", "context_domain", "primary_context_signal"}:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                metrics[column] = _mean(values)
        summary[str(domain)] = _clean_row(metrics)
    return summary


def _driver_context_rows(
    top_driver_features: object,
    failure_context: dict[str, dict[str, object]],
    comparator_context: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(top_driver_features, list):
        return []
    rows: list[dict[str, object]] = []
    for driver in top_driver_features:
        if not isinstance(driver, dict):
            continue
        feature_name = str(driver.get("feature_name") or "")
        if not feature_name:
            continue
        domain = str(
            driver.get("feature_domain")
            or EXPOSURE_LOAD_FEATURE_DOMAINS.get(feature_name, "other_exposure_load")
        )
        metric_name = _feature_metric_name(feature_name)
        failure_value = failure_context.get(domain, {}).get(metric_name)
        comparator_value = comparator_context.get(domain, {}).get(metric_name)
        rows.append(
            _clean_row(
                {
                    "feature_name": feature_name,
                    "context_domain": domain,
                    "shift_direction": driver.get("shift_direction"),
                    "driver_score": driver.get("driver_score"),
                    "context_metric": metric_name,
                    "failure_context_value": failure_value,
                    "comparator_context_value": comparator_value,
                    "context_signal": _context_signal(
                        feature_name,
                        str(driver.get("shift_direction") or ""),
                    ),
                    "review_question": _review_question(feature_name, domain),
                }
            )
        )
    return rows


def _shift_context_cases(
    diagnostics: pd.DataFrame,
    failure_seasons: list[str],
    driver_context_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if diagnostics.empty or "diagnostic_label" not in diagnostics:
        return []
    cases = diagnostics[
        diagnostics["diagnostic_label"].astype(str).eq(
            "ranking_triage_gain_calibration_loss"
        )
    ].copy()
    if "test_season_id" in cases:
        cases = cases[cases["test_season_id"].astype(str).isin(failure_seasons)]
    primary_signal = _primary_case_signal(driver_context_rows)
    records: list[dict[str, object]] = []
    for row in cases.to_dict(orient="records"):
        row["primary_context_signal"] = primary_signal
        row["context_review_status"] = "requires_schedule_roster_availability_review"
        records.append(_clean_row(row))
    return records


def _primary_case_signal(driver_context_rows: list[dict[str, object]]) -> str:
    for row in driver_context_rows:
        signal = str(row.get("context_signal") or "")
        if signal.startswith("elevated_game"):
            return "elevated_game_exposure"
    if driver_context_rows:
        return str(driver_context_rows[0].get("context_signal") or "review_context")
    return "review_context"


def _normalized_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"season_id", "event_type", "exposure_category"}
    for column in required - set(events.columns):
        events[column] = ""
    if "event_id" not in events:
        events["event_id"] = range(len(events))
    if "duration_minutes" not in events:
        events["duration_minutes"] = None
    if "rtp_flag" not in events:
        events["rtp_flag"] = False
    events = events.copy()
    events["season_id"] = events["season_id"].fillna("").astype(str)
    return events


def _normalized_participations(participations: pd.DataFrame) -> pd.DataFrame:
    required = {"season_id", "athlete_id", "participation_category"}
    for column in required - set(participations.columns):
        participations[column] = ""
    participations = participations.copy()
    participations["season_id"] = participations["season_id"].fillna("").astype(str)
    return participations


def _normalized_features(features: pd.DataFrame) -> pd.DataFrame:
    if "season_id" not in features:
        features["season_id"] = ""
    features = features.copy()
    features["season_id"] = features["season_id"].fillna("").astype(str)
    return features


def _failure_seasons(
    failure_modes: dict[str, object],
    diagnostics: pd.DataFrame,
) -> list[str]:
    seasons = failure_modes.get("failure_seasons")
    if isinstance(seasons, list) and seasons:
        return sorted(str(season) for season in seasons)
    return _seasons_with_label(diagnostics, "ranking_triage_gain_calibration_loss")


def _comparator_seasons(
    failure_modes: dict[str, object],
    diagnostics: pd.DataFrame,
) -> list[str]:
    seasons = failure_modes.get("comparator_seasons")
    if isinstance(seasons, list) and seasons:
        return sorted(str(season) for season in seasons)
    return _seasons_with_label(diagnostics, "calibration_supported")


def _seasons_with_label(diagnostics: pd.DataFrame, label: str) -> list[str]:
    if diagnostics.empty or "diagnostic_label" not in diagnostics:
        return []
    if "test_season_id" not in diagnostics:
        return []
    seasons = diagnostics.loc[
        diagnostics["diagnostic_label"].astype(str).eq(label),
        "test_season_id",
    ]
    return sorted(str(value) for value in seasons.dropna().unique())


def _feature_metric_name(feature_name: str) -> str:
    prefix = "exposure_"
    suffix = feature_name[len(prefix) :] if feature_name.startswith(prefix) else feature_name
    if suffix.endswith("_count"):
        suffix = suffix[: -len("_count")]
    return f"{suffix}_mean"


def _primary_context_signal(domain: str, row: dict[str, object]) -> str:
    if domain == "game_exposure" and float(row.get("game_event_count") or 0) > 0:
        return "game_schedule_context"
    if domain == "participation_status" and float(row.get("modified_participation_count") or 0) > 0:
        return "managed_availability_context"
    if domain == "category_specific_load" and float(row.get("lift_event_count") or 0) > 0:
        return "lift_schedule_context"
    if domain == "training_session_load" and float(row.get("training_event_count") or 0) > 0:
        return "training_density_context"
    return "limited_context"


def _context_signal(feature_name: str, shift_direction: str) -> str:
    if feature_name == "exposure_days_since_last_game":
        if shift_direction == "elevated_in_failure":
            return "longer_gap_since_game_in_failed_season"
        return "shorter_gap_since_game_in_failed_season"
    if feature_name in {"exposure_games_prior_count", "exposure_game_events_28d"}:
        if shift_direction == "elevated_in_failure":
            return "elevated_game_exposure_in_failed_season"
        return "reduced_game_exposure_in_failed_season"
    if feature_name == "exposure_practice_sessions_28d":
        if shift_direction == "elevated_in_failure":
            return "elevated_practice_exposure_in_failed_season"
        return "reduced_practice_exposure_in_failed_season"
    if feature_name == "exposure_lift_sessions_28d":
        if shift_direction == "reduced_in_failure":
            return "reduced_lift_exposure_in_failed_season"
        return "elevated_lift_exposure_in_failed_season"
    if feature_name in {
        "exposure_modified_participations_28d",
        "exposure_no_participations_28d",
    }:
        if shift_direction == "reduced_in_failure":
            return "reduced_availability_flagging_in_failed_season"
        return "elevated_availability_flagging_in_failed_season"
    if feature_name == "exposure_days_since_last_modified_or_no_participation":
        return "longer_gap_since_availability_flag_in_failed_season"
    return f"{shift_direction or 'shifted'}_{feature_name}"


def _review_question(feature_name: str, domain: str) -> str:
    if domain == "game_exposure":
        return "Was the failed season schedule denser, differently timed, or more game-heavy?"
    if domain == "participation_status":
        return "Did availability flagging or managed-risk documentation change in the failed season?"
    if feature_name == "exposure_lift_sessions_28d":
        return "Did weight-room scheduling, capture, or roster participation change in the failed season?"
    if domain == "category_specific_load":
        return "Did category-specific training capture shift relative to comparator seasons?"
    return "Review this shifted exposure domain before expanding model features."


def _overall_recommendation(
    failure_seasons: list[str],
    driver_context_rows: list[dict[str, object]],
) -> str:
    if failure_seasons and driver_context_rows:
        return "review_schedule_roster_availability_context"
    if failure_seasons:
        return "review_failure_season_context_without_driver_mapping"
    return "keep_exposure_load_context_under_review"


def _interpretation(summary: dict[str, object]) -> str:
    if summary["overall_recommendation"] == "review_schedule_roster_availability_context":
        return (
            "The failed season should be reviewed against schedule density, "
            "roster participation, availability flagging, and managed-risk "
            "documentation before exposure-load is used for probability-facing "
            "outputs, pilot escalation, dashboard work, or minute-load expansion."
        )
    return (
        "No specific shifted context driver was mapped. Keep exposure-load in "
        "research validation and review the source artifacts before model expansion."
    )


def _counter(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row.get(field)) for row in rows if row.get(field)).items()
        )
    )


def _truthy_series(values: object) -> pd.Series:
    if values is None:
        return pd.Series([], dtype=bool)
    series = pd.Series(values)
    if series.empty:
        return pd.Series([], dtype=bool)
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def _recorded_rate(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return _safe_rate(int(values.notna().sum()), len(values))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _median(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "n/a"
    return f"{number:.3f}"


def clean_shift_context_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_clean_row(row) for row in rows]


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
