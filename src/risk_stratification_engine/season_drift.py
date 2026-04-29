from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


DEFAULT_DRIFT_CHANNELS = (
    "broad_30d",
    "severity_7d",
    "severity_14d",
    "subtype_lower_extremity_soft_tissue_30d",
)
MODEL_SAFE_TIME_LOSS_MAX_DAYS = 365


def build_season_drift_diagnostics(
    measurements: pd.DataFrame,
    canonical_injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    shadow_mode_rows: pd.DataFrame,
) -> dict[str, object]:
    measurement_rows = _measurement_rows(measurements)
    canonical_rows = _canonical_injury_rows(canonical_injuries)
    detailed_rows = _detailed_injury_rows(detailed_injuries)
    shadow_rows = _shadow_mode_rows(shadow_mode_rows)
    season_ids = sorted(
        set(measurement_rows)
        | set(canonical_rows)
        | set(detailed_rows)
        | set(shadow_rows)
    )
    coverage_median = _median_positive(
        row["measurement_row_count"] for row in measurement_rows.values()
    )
    highest_by_channel = _highest_capture_season_by_channel(shadow_mode_rows)

    season_rows = []
    for season_id in season_ids:
        row = {
            "season_id": season_id,
            **_empty_measurement_row(),
            **_empty_canonical_row(),
            **_empty_detailed_row(),
        }
        row.update(measurement_rows.get(season_id, {}))
        row.update(canonical_rows.get(season_id, {}))
        row.update(detailed_rows.get(season_id, {}))
        row.update(shadow_rows.get(season_id, {}))
        row["primary_drift_flag"] = _primary_drift_flag(
            row=row,
            coverage_median=coverage_median,
            highest_capture_seasons=set(highest_by_channel.values()),
        )
        season_rows.append(row)

    low_capture_seasons = [
        row["season_id"]
        for row in season_rows
        if row["primary_drift_flag"] == "low_capture_with_events"
    ]
    coverage_warning_seasons = [
        row["season_id"]
        for row in season_rows
        if row["primary_drift_flag"] == "low_measurement_coverage"
    ]
    latest_season = max(season_ids) if season_ids else None
    return {
        "experiment_type": "season_drift_diagnostic",
        "season_count": len(season_rows),
        "latest_season": latest_season,
        "summary": {
            "latest_season": latest_season,
            "highest_capture_season_by_channel": highest_by_channel,
            "low_capture_seasons": low_capture_seasons,
            "coverage_warning_seasons": coverage_warning_seasons,
            "overall_interpretation": _interpretation(
                low_capture_seasons,
                coverage_warning_seasons,
            ),
        },
        "season_rows": [_clean_row(row) for row in season_rows],
    }


def write_season_drift_diagnostic_report(
    path: Path,
    diagnostics: dict[str, object],
) -> None:
    summary = diagnostics["summary"]
    lines = [
        "# Season Drift Diagnostics",
        "",
        f"Seasons: {diagnostics['season_count']}",
        f"Latest season: {summary['latest_season']}",
        "",
        "## Interpretation",
        str(summary["overall_interpretation"]),
        "",
        "## Season Table",
        "",
        "| Season | Measurements | Athletes | Observed events | Detailed events | Broad 30d capture | Severity 7d capture | Flag |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in diagnostics["season_rows"]:
        lines.append(
            "| "
            f"{row['season_id']} | "
            f"{row['measurement_row_count']} | "
            f"{row['athlete_count']} | "
            f"{row['observed_event_count']} | "
            f"{row['detailed_event_count']} | "
            f"{_fmt(row.get('broad_30d_capture_rate'))} | "
            f"{_fmt(row.get('severity_7d_capture_rate'))} | "
            f"{row['primary_drift_flag']} |"
        )
    lines.extend(
        [
            "",
            "## Highest Capture By Channel",
            "",
        ]
    )
    for channel, season_id in summary["highest_capture_season_by_channel"].items():
        lines.append(f"- {channel}: {season_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _measurement_rows(measurements: pd.DataFrame) -> dict[str, dict[str, object]]:
    frame = measurements.copy()
    if frame.empty:
        return {}
    frame["season_id"] = frame["season_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    rows = {}
    for season_id, group in frame.groupby("season_id", sort=True):
        athlete_dates = group.loc[:, ["athlete_id", "date"]].drop_duplicates()
        rows[str(season_id)] = {
            "athlete_count": int(group["athlete_id"].nunique()),
            "measurement_row_count": int(len(group)),
            "measurement_date_count": int(group["date"].nunique()),
            "athlete_measurement_day_count": int(len(athlete_dates)),
            "source_count": int(group["source"].nunique()),
            "source_counts": _top_counts(group["source"], limit=None),
            "metric_count": int(group["metric_name"].nunique()),
            "metric_counts": _top_counts(group["metric_name"], limit=8),
            "median_days_between_measurements": _median_days_between_measurements(
                group
            ),
        }
    return rows


def _canonical_injury_rows(
    canonical_injuries: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    frame = canonical_injuries.copy()
    if frame.empty:
        return {}
    frame["season_id"] = frame["season_id"].astype(str)
    frame["event_observed"] = frame["event_observed"].astype(bool)
    if "primary_model_event" in frame:
        frame["primary_model_event"] = frame["primary_model_event"].astype(bool)
    else:
        frame["primary_model_event"] = frame["event_observed"]
    rows = {}
    for season_id, group in frame.groupby("season_id", sort=True):
        rows[str(season_id)] = {
            "canonical_athlete_season_count": int(
                group[["athlete_id", "season_id"]].drop_duplicates().shape[0]
            ),
            "observed_event_count": int(group["event_observed"].sum()),
            "primary_model_event_count": int(group["primary_model_event"].sum()),
        }
    return rows


def _detailed_injury_rows(
    detailed_injuries: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    frame = detailed_injuries.copy()
    if frame.empty:
        return {}
    frame["season_id"] = frame["season_id"].astype(str)
    rows = {}
    for season_id, group in frame.groupby("season_id", sort=True):
        buckets = [_time_loss_bucket(_number_or_none(value)) for value in group.get(
            "time_loss_days",
            pd.Series(dtype=object),
        )]
        rows[str(season_id)] = {
            "detailed_event_count": int(len(group)),
            "model_safe_time_loss_event_count": int(
                group.apply(_is_model_safe_time_loss, axis=1).sum()
            ),
            "lower_extremity_soft_tissue_event_count": int(
                group.apply(_is_lower_extremity_soft_tissue, axis=1).sum()
            ),
            "severe_time_loss_event_count": int(
                group.apply(_is_severe_time_loss, axis=1).sum()
            ),
            "concussion_event_count": int(group.apply(_is_concussion, axis=1).sum()),
            "top_body_areas": _top_counts(group.get("body_area", pd.Series()), 5),
            "top_classifications": _top_counts(
                group.get("classification", pd.Series()),
                5,
            ),
            "time_loss_bucket_counts": dict(Counter(buckets)),
        }
    return rows


def _shadow_mode_rows(
    shadow_mode_rows: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    frame = shadow_mode_rows.copy()
    if frame.empty:
        return {}
    frame["slice_id"] = frame["slice_id"].astype(str)
    rows: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        season_row = rows.setdefault(str(row["slice_id"]), {})
        channel = str(row["channel_name"])
        season_row[f"{channel}_capture_rate"] = _round_or_none(
            row.get("unique_event_capture_rate")
        )
        season_row[f"{channel}_captured_event_count"] = _int_or_zero(
            row.get("unique_captured_event_count")
        )
        season_row[f"{channel}_observed_event_count"] = _int_or_zero(
            row.get("unique_observed_event_count")
        )
        season_row[f"{channel}_episode_count"] = _int_or_zero(row.get("episode_count"))
        season_row[f"{channel}_episodes_per_athlete_season"] = _round_or_none(
            row.get("episodes_per_athlete_season")
        )
    return rows


def _highest_capture_season_by_channel(
    shadow_mode_rows: pd.DataFrame,
) -> dict[str, str]:
    frame = shadow_mode_rows.copy()
    if frame.empty:
        return {}
    frame["slice_id"] = frame["slice_id"].astype(str)
    frame["unique_event_capture_rate"] = pd.to_numeric(
        frame["unique_event_capture_rate"],
        errors="coerce",
    )
    output = {}
    for channel, group in frame.dropna(
        subset=["unique_event_capture_rate"]
    ).groupby("channel_name", sort=True):
        best = group.sort_values(
            ["unique_event_capture_rate", "slice_id"],
            ascending=[False, False],
        ).iloc[0]
        output[str(channel)] = str(best["slice_id"])
    return output


def _primary_drift_flag(
    row: dict[str, object],
    coverage_median: float | None,
    highest_capture_seasons: set[str],
) -> str:
    season_id = str(row["season_id"])
    observed_events = int(row.get("observed_event_count") or 0)
    capture_rates = [
        row.get(f"{channel}_capture_rate") for channel in DEFAULT_DRIFT_CHANNELS
    ]
    numeric_capture = [
        float(value) for value in capture_rates if value is not None and pd.notna(value)
    ]
    if observed_events and numeric_capture and max(numeric_capture) == 0.0:
        return "low_capture_with_events"
    measurement_rows = int(row.get("measurement_row_count") or 0)
    if coverage_median and measurement_rows < coverage_median * 0.5:
        return "low_measurement_coverage"
    if season_id in highest_capture_seasons:
        return "reference_high_capture"
    return "no_primary_flag"


def _empty_measurement_row() -> dict[str, object]:
    return {
        "athlete_count": 0,
        "measurement_row_count": 0,
        "measurement_date_count": 0,
        "athlete_measurement_day_count": 0,
        "source_count": 0,
        "source_counts": {},
        "metric_count": 0,
        "metric_counts": {},
        "median_days_between_measurements": None,
    }


def _empty_canonical_row() -> dict[str, object]:
    return {
        "canonical_athlete_season_count": 0,
        "observed_event_count": 0,
        "primary_model_event_count": 0,
    }


def _empty_detailed_row() -> dict[str, object]:
    return {
        "detailed_event_count": 0,
        "model_safe_time_loss_event_count": 0,
        "lower_extremity_soft_tissue_event_count": 0,
        "severe_time_loss_event_count": 0,
        "concussion_event_count": 0,
        "top_body_areas": {},
        "top_classifications": {},
        "time_loss_bucket_counts": {},
    }


def _top_counts(values: pd.Series, limit: int | None = 5) -> dict[str, int]:
    clean = values.dropna().astype(str)
    counts = Counter(clean)
    items = counts.most_common(limit) if limit is not None else sorted(counts.items())
    return {str(key): int(value) for key, value in items}


def _median_days_between_measurements(group: pd.DataFrame) -> float | None:
    deltas = []
    for _, athlete_group in group.groupby("athlete_id", sort=False):
        dates = athlete_group["date"].dropna().drop_duplicates().sort_values()
        if len(dates) < 2:
            continue
        deltas.extend(dates.diff().dropna().dt.days.tolist())
    if not deltas:
        return None
    return round(float(pd.Series(deltas).median()), 3)


def _median_positive(values) -> float | None:
    numeric = [float(value) for value in values if value]
    if not numeric:
        return None
    return float(pd.Series(numeric).median())


def _is_model_safe_time_loss(row: pd.Series) -> bool:
    value = _number_or_none(row.get("time_loss_days"))
    return value is not None and 0 < value <= MODEL_SAFE_TIME_LOSS_MAX_DAYS


def _is_severe_time_loss(row: pd.Series) -> bool:
    value = _number_or_none(row.get("time_loss_days"))
    return value is not None and 29 <= value <= MODEL_SAFE_TIME_LOSS_MAX_DAYS


def _is_lower_extremity_soft_tissue(row: pd.Series) -> bool:
    body = _text(row.get("body_area"))
    classification = _text(row.get("classification"))
    pathology = _text(row.get("pathology"))
    lower_terms = (
        "ankle",
        "foot",
        "groin",
        "hamstring",
        "hip",
        "knee",
        "leg",
        "lower",
        "thigh",
    )
    soft_terms = ("muscle", "soft", "strain", "tendon")
    return any(term in body for term in lower_terms) and (
        any(term in classification for term in soft_terms)
        or any(term in pathology for term in soft_terms)
    )


def _is_concussion(row: pd.Series) -> bool:
    return "concussion" in " ".join(
        [
            _text(row.get("injury_type")),
            _text(row.get("classification")),
            _text(row.get("pathology")),
        ]
    )


def _time_loss_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0:
        return "negative"
    if value == 0:
        return "0_days"
    if value <= 7:
        return "1_7_days"
    if value <= 28:
        return "8_28_days"
    if value <= MODEL_SAFE_TIME_LOSS_MAX_DAYS:
        return "29_365_days"
    return "extreme_366_plus_days"


def _number_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _int_or_zero(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _round_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _interpretation(
    low_capture_seasons: list[str],
    coverage_warning_seasons: list[str],
) -> str:
    if low_capture_seasons or coverage_warning_seasons:
        return (
            "Season drift should be reviewed through coverage and injury mix before "
            "treating shadow-mode differences as pure model performance."
        )
    return (
        "No single coverage or low-capture warning dominates the season comparison; "
        "continue reviewing policy stability with coverage and injury mix context."
    )


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _clean_row(row: dict[str, object]) -> dict[str, object]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if not isinstance(value, str) and pd.isna(value):
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value
