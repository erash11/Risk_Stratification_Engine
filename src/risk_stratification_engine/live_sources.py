from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from risk_stratification_engine.config import DataSourcePaths
from risk_stratification_engine.data_quality import build_data_quality_audit
from risk_stratification_engine.io import write_frame


GPS_METRICS = (
    "total_player_load",
    "total_distance_m",
    "max_velocity_ms",
    "high_speed_distance_m",
    "accel_density",
    "explosive_efforts",
    "sprint_distance_m",
    "total_duration_s",
    "total_accel_load",
    "average_velocity_ms",
    "meterage_per_minute",
    "player_load_per_minute",
    "ima_high_accel",
    "ima_high_decel",
    "total_contacts",
    "total_throws",
)

METRIC_LIMITS = {
    "forceplate": 24,
    "perch": 8,
}

EVENT_WINDOW_QUALITY_ORDER = (
    "censored",
    "low_confidence",
    "modelable",
    "no_measurements",
    "out_of_window",
)

MODELABLE_EVENT_WINDOW_DAYS = 14
LOW_CONFIDENCE_EVENT_WINDOW_DAYS = 30


@dataclass(frozen=True)
class LiveSourcePreparationResult:
    measurements_path: Path
    injuries_path: Path
    detailed_injuries_path: Path
    metadata_path: Path
    audit_path: Path
    metadata: dict[str, Any]
    audit: dict[str, Any]


def stable_athlete_id(name: object) -> str:
    normalized = _normalize_name(name)
    if not normalized:
        return ""
    return "ath_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def aggregate_same_day_measurements(
    measurements: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_columns = [
        "athlete_id",
        "date",
        "season_id",
        "source",
        "metric_name",
    ]
    duplicate_groups = (
        measurements.groupby(group_columns, as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    duplicate_group_count = int((duplicate_groups["row_count"] > 1).sum())
    aggregated = (
        measurements.groupby(group_columns, as_index=False)
        .agg(metric_value=("metric_value", "mean"))
        .sort_values(["athlete_id", "season_id", "date", "source", "metric_name"])
        .reset_index(drop=True)
    )
    summary = {
        "policy": (
            "mean metric_value per athlete_id, season_id, date, source, metric_name"
        ),
        "input_rows": int(len(measurements)),
        "output_rows": int(len(aggregated)),
        "duplicate_same_day_metric_groups": duplicate_group_count,
        "aggregated_rows_removed": int(len(measurements) - len(aggregated)),
    }
    return aggregated, summary


def canonicalize_wide_measurements(
    frame: pd.DataFrame,
    *,
    source: str,
    date_column: str,
    name_column: str,
    metric_columns: list[str],
) -> pd.DataFrame:
    base = frame.loc[:, [date_column, name_column, *metric_columns]].copy()
    base["date"] = pd.to_datetime(base[date_column], errors="coerce")
    base["athlete_id"] = base[name_column].map(stable_athlete_id)
    base["season_id"] = _season_id_for_dates(base["date"])
    long = base.melt(
        id_vars=["athlete_id", "date", "season_id"],
        value_vars=metric_columns,
        var_name="raw_metric_name",
        value_name="metric_value",
    )
    long["source"] = source
    long["metric_name"] = source + "__" + long["raw_metric_name"].map(_metric_slug)
    return _clean_measurements(long)


def canonicalize_long_measurements(
    frame: pd.DataFrame,
    *,
    source: str,
    date_column: str,
    name_column: str,
    metric_name_column: str,
    metric_value_column: str,
) -> pd.DataFrame:
    out = frame.loc[
        :, [date_column, name_column, metric_name_column, metric_value_column]
    ].copy()
    out["date"] = pd.to_datetime(out[date_column], errors="coerce")
    out["athlete_id"] = out[name_column].map(stable_athlete_id)
    out["season_id"] = _season_id_for_dates(out["date"])
    out["source"] = source
    out["metric_name"] = source + "__" + out[metric_name_column].map(_metric_slug)
    out["metric_value"] = out[metric_value_column]
    return _clean_measurements(out)


def build_injury_event_rows(
    measurements: pd.DataFrame,
    injury_export: pd.DataFrame,
) -> pd.DataFrame:
    measurement_dates = measurements.assign(
        date=pd.to_datetime(measurements["date"], errors="coerce")
    )
    season_windows = (
        measurement_dates.groupby(["athlete_id", "season_id"], as_index=False)
        .agg(
            first_measurement_date=("date", "min"),
            censor_date=("date", "max"),
        )
        .sort_values(["athlete_id", "season_id"])
    )

    injuries = injury_export.copy()
    injuries["injury_date"] = pd.to_datetime(injuries["Issue Date"], errors="coerce")
    injuries["athlete_id"] = injuries["Athlete"].map(stable_athlete_id)
    injuries["season_id"] = _season_id_for_dates(injuries["injury_date"])
    injuries["injury_type"] = _injury_type(injuries)
    earliest_events = (
        injuries.dropna(subset=["injury_date"])
        .loc[injuries["athlete_id"].ne("")]
        .sort_values(["athlete_id", "season_id", "injury_date"])
        .drop_duplicates(["athlete_id", "season_id"], keep="first")
        .loc[:, ["athlete_id", "season_id", "injury_date", "injury_type"]]
    )

    events = season_windows.merge(
        earliest_events,
        on=["athlete_id", "season_id"],
        how="left",
    )
    events["event_observed"] = events["injury_date"].notna()
    events.loc[~events["event_observed"], "injury_type"] = "censored"
    events = _attach_event_window_quality(events, measurement_dates)
    return events.loc[
        :,
        [
            "athlete_id",
            "season_id",
            "injury_date",
            "injury_type",
            "event_observed",
            "censor_date",
            "nearest_measurement_date",
            "nearest_measurement_gap_days",
            "event_window_quality",
            "primary_model_event",
        ],
    ]


DETAILED_INJURY_EVENT_COLUMNS = (
    "injury_event_id",
    "athlete_id",
    "season_id",
    "injury_date",
    "issue_entry_date",
    "issue_resolved_date",
    "injury_type",
    "pathology",
    "classification",
    "body_area",
    "tissue_type",
    "side",
    "recurrent",
    "caused_unavailability",
    "activity",
    "activity_group",
    "activity_group_type",
    "participation_level",
    "session_completed",
    "competition",
    "game_date",
    "training_session_type",
    "surface_type",
    "duration_days",
    "time_loss_days",
    "modified_available_days",
    "not_modified_available_days",
    "raw_code",
    "icd_code",
    "source_file",
    "source_row_number",
)


def build_detailed_injury_event_rows(injury_export: pd.DataFrame) -> pd.DataFrame:
    injuries = injury_export.copy()
    output = pd.DataFrame(index=injuries.index)
    output["athlete_id"] = _text_column(injuries, "Athlete").map(stable_athlete_id)
    output["injury_date"] = pd.to_datetime(
        _text_column(injuries, "Issue Date"),
        errors="coerce",
    )
    output["season_id"] = _season_id_for_dates(output["injury_date"])
    output["issue_entry_date"] = pd.to_datetime(
        _text_column(injuries, "Issue Entry Date"),
        errors="coerce",
    )
    output["issue_resolved_date"] = pd.to_datetime(
        _text_column(injuries, "Issue Resolved Date"),
        errors="coerce",
    )
    output["injury_type"] = _injury_type(injuries)
    output["pathology"] = _text_column(injuries, "Pathology")
    output["classification"] = _text_column(injuries, "Classification")
    output["body_area"] = _text_column(injuries, "Body Area")
    output["tissue_type"] = _text_column(injuries, "Tissue Type")
    output["side"] = _text_column(injuries, "Side")
    output["recurrent"] = _text_column(injuries, "Recurrent")
    output["caused_unavailability"] = _text_column(injuries, "Caused Unavailability")
    output["activity"] = _text_column(injuries, "Activity")
    output["activity_group"] = _text_column(injuries, "Activity Group")
    output["activity_group_type"] = _text_column(injuries, "Activity Group Type")
    output["participation_level"] = _text_column(injuries, "Participation Level")
    output["session_completed"] = _text_column(injuries, "Session Completed")
    output["competition"] = _text_column(injuries, "Competition")
    output["game_date"] = pd.to_datetime(
        _text_column(injuries, "Game Date/Time"),
        errors="coerce",
    )
    output["training_session_type"] = _text_column(injuries, "Training Session Type")
    output["surface_type"] = _text_column(injuries, "Surface Type")
    output["duration_days"] = _numeric_column(injuries, "Duration")
    output["time_loss_days"] = _numeric_column(
        injuries,
        "Status duration: Unavailable - time-loss",
    )
    output["modified_available_days"] = _numeric_column(
        injuries,
        "Status duration: Available - modified",
    )
    output["not_modified_available_days"] = _numeric_column(
        injuries,
        "Status duration: Available - not modified",
    )
    output["raw_code"] = _text_column(injuries, "Code")
    output["icd_code"] = _text_column(injuries, "ICD")
    output["source_file"] = _text_column(injuries, "_source_file")
    output["source_row_number"] = _integer_column(injuries, "_source_row_number")
    output = output.loc[
        output["athlete_id"].ne("") & output["injury_date"].notna()
    ].reset_index(drop=True)
    output["injury_event_id"] = _stable_injury_event_ids(output)
    return output.loc[:, list(DETAILED_INJURY_EVENT_COLUMNS)]


def prepare_live_source_inputs(
    paths: DataSourcePaths,
    output_dir: str | Path,
) -> LiveSourcePreparationResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_frames, source_metadata, source_identities = _load_source_measurements(paths)
    measurements = (
        pd.concat(source_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["athlete_id", "season_id", "date", "source", "metric_name"])
    )
    measurements, aggregation_summary = aggregate_same_day_measurements(measurements)
    injury_export = _load_injury_exports(paths.injury_csv)
    source_metadata["sources"]["injury"] = _injury_source_summary(injury_export)
    source_identities["injury"] = _identity_frame(injury_export, "Athlete")
    detailed_injuries = build_detailed_injury_event_rows(injury_export)
    injuries = build_injury_event_rows(measurements, injury_export)

    measurements_path = output_path / "canonical_measurements.csv"
    injuries_path = output_path / "canonical_injuries.csv"
    detailed_injuries_path = output_path / "injury_events_detailed.csv"
    metadata_path = output_path / "prep_metadata.json"
    audit_path = output_path / "data_quality_audit.json"

    write_frame(_csv_ready(measurements), measurements_path)
    write_frame(_csv_ready(injuries), injuries_path)
    write_frame(_csv_ready(detailed_injuries), detailed_injuries_path)

    metadata = _preparation_metadata(
        paths,
        source_metadata,
        measurements,
        injuries,
        detailed_injuries,
        aggregation_summary,
    )
    audit = build_data_quality_audit(measurements, injuries, source_identities)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return LiveSourcePreparationResult(
        measurements_path=measurements_path,
        injuries_path=injuries_path,
        detailed_injuries_path=detailed_injuries_path,
        metadata_path=metadata_path,
        audit_path=audit_path,
        metadata=metadata,
        audit=audit,
    )


def _load_source_measurements(
    paths: DataSourcePaths,
) -> tuple[list[pd.DataFrame], dict[str, Any], dict[str, pd.DataFrame]]:
    duckdb = _require_duckdb()
    metadata: dict[str, Any] = {"sources": {}}
    identities: dict[str, pd.DataFrame] = {}

    gps = _read_duckdb(
        duckdb,
        paths.gps_db,
        """
        select a.name, s.session_date, ats.*
        from athlete_sessions ats
        join athletes a on a.catapult_id = ats.athlete_id
        join sessions s on s.session_id = ats.session_id
        """,
    )
    gps_metrics = [column for column in GPS_METRICS if column in gps.columns]
    gps_frame = canonicalize_wide_measurements(
        gps,
        source="gps",
        date_column="session_date",
        name_column="name",
        metric_columns=gps_metrics,
    )
    metadata["sources"]["gps"] = _source_summary(gps, gps_metrics)
    identities["gps"] = _identity_frame(gps, "name")

    forceplate = _read_duckdb(
        duckdb,
        paths.forceplate_db,
        "select athlete_name, test_date, metric_name, metric_value from raw_tests",
    )
    forceplate["metric_value_numeric"] = pd.to_numeric(
        forceplate["metric_value"], errors="coerce"
    )
    forceplate_metrics = _top_metric_names(
        forceplate,
        metric_name_column="metric_name",
        metric_value_column="metric_value_numeric",
        limit=METRIC_LIMITS["forceplate"],
    )
    forceplate_frame = canonicalize_long_measurements(
        forceplate.loc[forceplate["metric_name"].isin(forceplate_metrics)],
        source="forceplate",
        date_column="test_date",
        name_column="athlete_name",
        metric_name_column="metric_name",
        metric_value_column="metric_value_numeric",
    )
    metadata["sources"]["forceplate"] = _source_summary(
        forceplate,
        forceplate_metrics,
    )
    identities["forceplate"] = _identity_frame(forceplate, "athlete_name")

    bodyweight = pd.read_csv(paths.bodyweight_csv)
    bodyweight_frame = canonicalize_wide_measurements(
        bodyweight,
        source="bodyweight",
        date_column="DATE",
        name_column="NAME",
        metric_columns=["WEIGHT"],
    )
    metadata["sources"]["bodyweight"] = _source_summary(bodyweight, ["WEIGHT"])
    identities["bodyweight"] = _identity_frame(bodyweight, "NAME")

    perch = _read_duckdb(
        duckdb,
        paths.perch_db,
        "select name_normalized, test_date, exercise, one_rm_lbs from perch_1rm",
    )
    perch_metrics = _top_metric_names(
        perch,
        metric_name_column="exercise",
        metric_value_column="one_rm_lbs",
        limit=METRIC_LIMITS["perch"],
    )
    perch_frame = canonicalize_long_measurements(
        perch.loc[perch["exercise"].isin(perch_metrics)],
        source="perch",
        date_column="test_date",
        name_column="name_normalized",
        metric_name_column="exercise",
        metric_value_column="one_rm_lbs",
    )
    metadata["sources"]["perch"] = _source_summary(perch, perch_metrics)
    identities["perch"] = _identity_frame(perch, "name_normalized")
    return [gps_frame, forceplate_frame, bodyweight_frame, perch_frame], metadata, identities


def _load_injury_exports(path: Path) -> pd.DataFrame:
    paths = _injury_export_paths(path)
    frames = []
    for export_path in paths:
        frame = pd.read_csv(export_path)
        frame["_source_file"] = export_path.name
        frame["_source_row_number"] = range(1, len(frame) + 1)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    raw_columns = [
        column
        for column in combined.columns
        if column not in {"_source_file", "_source_row_number"}
    ]
    return combined.drop_duplicates(subset=raw_columns).reset_index(drop=True)


def _injury_export_paths(path: Path) -> list[Path]:
    if re.fullmatch(r"injuries-summary-export-.+\.csv", path.name):
        paths = sorted(path.parent.glob("injuries-summary-export-*.csv"))
        if paths:
            return paths
    return [path]


def _normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if "," in text:
        parts = [part.strip() for part in text.split(",", maxsplit=1)]
        if all(parts):
            text = f"{parts[1]} {parts[0]}"
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _season_id_for_dates(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    start_year = parsed.dt.year.where(parsed.dt.month >= 7, parsed.dt.year - 1)
    return start_year.astype("Int64").astype(str) + "-" + (
        start_year + 1
    ).astype("Int64").astype(str)


def _metric_slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _clean_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["metric_value"] = pd.to_numeric(cleaned["metric_value"], errors="coerce")
    cleaned = cleaned.loc[
        cleaned["athlete_id"].ne("")
        & cleaned["date"].notna()
        & cleaned["metric_value"].notna()
    ]
    return cleaned.loc[
        :,
        ["athlete_id", "date", "season_id", "source", "metric_name", "metric_value"],
    ].reset_index(drop=True)


def _attach_event_window_quality(
    events: pd.DataFrame,
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    output = events.copy()
    output["nearest_measurement_date"] = pd.NaT
    output["nearest_measurement_gap_days"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["event_window_quality"] = "censored"
    output["primary_model_event"] = False

    measurement_dates = (
        measurements.loc[:, ["athlete_id", "season_id", "date"]]
        .drop_duplicates()
        .sort_values(["athlete_id", "season_id", "date"])
    )
    grouped_dates = {
        key: group["date"].reset_index(drop=True)
        for key, group in measurement_dates.groupby(["athlete_id", "season_id"])
    }

    for index, row in output.loc[output["event_observed"]].iterrows():
        dates = grouped_dates.get((row["athlete_id"], row["season_id"]))
        if dates is None or dates.empty:
            output.at[index, "event_window_quality"] = "no_measurements"
            continue
        gaps = (dates - row["injury_date"]).abs().dt.days
        nearest_position = gaps.idxmin()
        nearest_gap = int(gaps.loc[nearest_position])
        output.at[index, "nearest_measurement_date"] = dates.loc[nearest_position]
        output.at[index, "nearest_measurement_gap_days"] = nearest_gap
        quality = _event_window_quality(nearest_gap)
        output.at[index, "event_window_quality"] = quality
        output.at[index, "primary_model_event"] = quality == "modelable"
    return output


def _event_window_quality(nearest_gap_days: int) -> str:
    if nearest_gap_days <= MODELABLE_EVENT_WINDOW_DAYS:
        return "modelable"
    if nearest_gap_days <= LOW_CONFIDENCE_EVENT_WINDOW_DAYS:
        return "low_confidence"
    return "out_of_window"


def _event_window_quality_counts(injuries: pd.DataFrame) -> dict[str, int]:
    counts = injuries["event_window_quality"].value_counts().to_dict()
    return {
        label: int(counts.get(label, 0))
        for label in EVENT_WINDOW_QUALITY_ORDER
    }


def _identity_frame(frame: pd.DataFrame, name_column: str) -> pd.DataFrame:
    identities = pd.DataFrame({"athlete_id": frame[name_column].map(stable_athlete_id)})
    return identities.loc[identities["athlete_id"].ne("")].drop_duplicates()


def _injury_type(injuries: pd.DataFrame) -> pd.Series:
    injury_type = pd.Series("", index=injuries.index, dtype=object)
    for column in ("Classification", "Pathology", "Type"):
        if column in injuries:
            values = injuries[column].fillna("").astype(str).str.strip()
            injury_type = injury_type.where(
                injury_type.astype(str).str.strip().ne(""),
                values,
            )
    return injury_type.where(injury_type.astype(str).str.strip().ne(""), "unspecified")


def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[column].fillna("").astype(str).str.strip()


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("Float64")


def _integer_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")
    return pd.to_numeric(frame[column], errors="coerce").astype("Int64")


def _stable_injury_event_ids(events: pd.DataFrame) -> pd.Series:
    seeds = events.loc[
        :,
        [
            "athlete_id",
            "injury_date",
            "issue_entry_date",
            "raw_code",
            "pathology",
            "classification",
            "source_file",
            "source_row_number",
        ],
    ].copy()
    for column in ("injury_date", "issue_entry_date"):
        seeds[column] = pd.to_datetime(seeds[column], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    seed_text = seeds.fillna("").astype(str).agg("|".join, axis=1)
    return "inj_" + seed_text.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    )


def _top_metric_names(
    frame: pd.DataFrame,
    *,
    metric_name_column: str,
    metric_value_column: str,
    limit: int,
) -> list[str]:
    counts = (
        frame.dropna(subset=[metric_value_column])
        .groupby(metric_name_column)
        .size()
        .sort_values(ascending=False)
    )
    return [str(value) for value in counts.head(limit).index.tolist()]


def _require_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required to prepare live source inputs. "
            "Install the project dependencies with `python -m pip install -e .`."
        ) from exc
    return duckdb


def _read_duckdb(duckdb_module, path: Path, query: str) -> pd.DataFrame:
    connection = duckdb_module.connect(str(path), read_only=True)
    try:
        return connection.execute(query).fetchdf()
    finally:
        connection.close()


def _source_summary(frame: pd.DataFrame, selected_metrics: list[str]) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "selected_metrics": selected_metrics,
    }


def _injury_source_summary(frame: pd.DataFrame) -> dict[str, Any]:
    files = (
        sorted(frame["_source_file"].dropna().astype(str).unique().tolist())
        if "_source_file" in frame
        else []
    )
    return {
        "row_count": int(len(frame)),
        "columns": [
            str(column)
            for column in frame.columns
            if column not in {"_source_file", "_source_row_number"}
        ],
        "files": files,
    }


def _preparation_metadata(
    paths: DataSourcePaths,
    source_metadata: dict[str, Any],
    measurements: pd.DataFrame,
    injuries: pd.DataFrame,
    detailed_injuries: pd.DataFrame,
    aggregation_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "paths": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for name, path in paths.as_dict().items()
        },
        "identity_policy": (
            "stable sha256 hash of normalized athlete name, first 12 hex characters"
        ),
        "season_policy": "season starts July 1; season_id is YYYY-YYYY+1",
        "event_policy": (
            "earliest injury issue date per athlete-season; non-event rows censored "
            "at last measurement date"
        ),
        "event_window_policy": {
            "modelable_days": MODELABLE_EVENT_WINDOW_DAYS,
            "low_confidence_days": LOW_CONFIDENCE_EVENT_WINDOW_DAYS,
            "labels": list(EVENT_WINDOW_QUALITY_ORDER),
        },
        "event_window_quality_counts": _event_window_quality_counts(injuries),
        "primary_model_event_count": int(injuries["primary_model_event"].sum()),
        "aggregation": aggregation_summary,
        **source_metadata,
        "canonical_rows": {
            "measurements": int(len(measurements)),
            "injury_events": int(len(injuries)),
            "detailed_injury_events": int(len(detailed_injuries)),
            "observed_events": int(injuries["event_observed"].sum()),
        },
        "canonical_distinct": {
            "athletes": int(measurements["athlete_id"].nunique()),
            "athlete_seasons": int(
                measurements[["athlete_id", "season_id"]].drop_duplicates().shape[0]
            ),
            "measurement_metrics": int(measurements["metric_name"].nunique()),
        },
    }


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output
