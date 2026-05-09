from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from risk_stratification_engine.exposure_load_features import (
    EXPOSURE_LOAD_FEATURE_COLUMNS,
)


EXPOSURE_LOAD_FEATURE_DOMAINS = {
    "exposure_training_sessions_7d": "training_session_load",
    "exposure_training_sessions_14d": "training_session_load",
    "exposure_training_sessions_28d": "training_session_load",
    "exposure_games_prior_count": "game_exposure",
    "exposure_days_since_last_game": "game_exposure",
    "exposure_game_events_28d": "game_exposure",
    "exposure_full_participations_28d": "participation_status",
    "exposure_modified_participations_28d": "participation_status",
    "exposure_no_participations_28d": "participation_status",
    "exposure_days_since_last_modified_or_no_participation": "participation_status",
    "exposure_practice_sessions_28d": "category_specific_load",
    "exposure_lift_sessions_28d": "category_specific_load",
    "exposure_conditioning_sessions_28d": "category_specific_load",
    "exposure_rtp_sessions_28d": "category_specific_load",
}


def build_exposure_load_failure_mode_summary(
    exposure_feature_rows: list[dict[str, object]],
    diagnostic_rows: list[dict[str, object]],
) -> dict[str, object]:
    features = pd.DataFrame(exposure_feature_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    failure_seasons = _seasons_with_label(
        diagnostics,
        "ranking_triage_gain_calibration_loss",
    )
    comparator_seasons = _seasons_with_label(diagnostics, "calibration_supported")
    feature_shift_summary = _feature_shift_rows(
        features,
        failure_seasons,
        comparator_seasons,
    )
    domain_shift_summary = _domain_shift_rows(feature_shift_summary)
    top_driver_features = sorted(
        feature_shift_summary,
        key=lambda row: (
            -float(row.get("driver_score") or 0.0),
            str(row["feature_domain"]),
            str(row["feature_name"]),
        ),
    )[:8]
    return {
        "experiment_type": "exposure_load_failure_mode_sprint",
        "overall_recommendation": _overall_recommendation(
            failure_seasons,
            feature_shift_summary,
        ),
        "failure_seasons": failure_seasons,
        "comparator_seasons": comparator_seasons,
        "diagnostic_label_summary": _counter(diagnostic_rows, "diagnostic_label"),
        "feature_shift_summary": feature_shift_summary,
        "domain_shift_summary": domain_shift_summary,
        "top_driver_features": top_driver_features,
    }


def write_exposure_load_failure_mode_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Exposure Load Failure Mode Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint diagnoses exposure feature distributions after complete "
            "athlete-season trajectories have already been scored. It uses only "
            "time-safe exposure features attached before graph snapshots and does "
            "not create independent daily injury-classification rows."
        ),
        "",
        "## Failure Seasons",
        "",
        "- Failure seasons: " + ", ".join(summary.get("failure_seasons", [])),
        "- Comparator seasons: " + ", ".join(summary.get("comparator_seasons", [])),
        "",
        "## Top Feature Shifts",
        "",
        "| Feature | Domain | Direction | Failure mean | Comparator mean | Driver score |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary.get("top_driver_features", []):
        lines.append(
            "| "
            f"{row['feature_name']} | "
            f"{row['feature_domain']} | "
            f"{row['shift_direction']} | "
            f"{_fmt(row.get('failure_mean'))} | "
            f"{_fmt(row.get('comparator_mean'))} | "
            f"{_fmt(row.get('driver_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Domain Shifts",
            "",
            "| Domain | Shifted features | Mean absolute shift | Max driver score |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary.get("domain_shift_summary", []):
        lines.append(
            "| "
            f"{row['feature_domain']} | "
            f"{row['shifted_feature_count']} | "
            f"{_fmt(row.get('mean_abs_shift'))} | "
            f"{_fmt(row.get('max_driver_score'))} |"
        )
    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _feature_shift_rows(
    features: pd.DataFrame,
    failure_seasons: list[str],
    comparator_seasons: list[str],
) -> list[dict[str, object]]:
    if features.empty or "season_id" not in features:
        return []
    frame = features.copy()
    frame["season_id"] = frame["season_id"].astype(str)
    failure = frame[frame["season_id"].isin(failure_seasons)]
    comparator = frame[frame["season_id"].isin(comparator_seasons)]
    rows: list[dict[str, object]] = []
    for feature_name in EXPOSURE_LOAD_FEATURE_COLUMNS:
        if feature_name not in frame:
            continue
        failure_values = pd.to_numeric(failure[feature_name], errors="coerce")
        comparator_values = pd.to_numeric(comparator[feature_name], errors="coerce")
        failure_mean = _mean(failure_values)
        comparator_mean = _mean(comparator_values)
        mean_delta = _delta(failure_mean, comparator_mean)
        relative_shift = _relative_shift(failure_mean, comparator_mean)
        row = {
            "feature_name": feature_name,
            "feature_domain": EXPOSURE_LOAD_FEATURE_DOMAINS.get(
                feature_name,
                "other_exposure_load",
            ),
            "failure_seasons": ",".join(failure_seasons),
            "comparator_seasons": ",".join(comparator_seasons),
            "failure_snapshot_count": int(failure_values.notna().sum()),
            "comparator_snapshot_count": int(comparator_values.notna().sum()),
            "failure_mean": failure_mean,
            "comparator_mean": comparator_mean,
            "mean_delta": mean_delta,
            "relative_shift": relative_shift,
            "failure_nonzero_rate": _nonzero_rate(failure_values),
            "comparator_nonzero_rate": _nonzero_rate(comparator_values),
            "failure_p90": _quantile(failure_values, 0.9),
            "comparator_p90": _quantile(comparator_values, 0.9),
            "failure_max": _max(failure_values),
            "comparator_max": _max(comparator_values),
            "shift_direction": _shift_direction(mean_delta),
            "driver_score": _driver_score(mean_delta, relative_shift),
        }
        rows.append(_clean_row(row))
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("driver_score") or 0.0),
            str(row["feature_name"]),
        ),
    )


def _domain_shift_rows(feature_shift_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    if not feature_shift_summary:
        return []
    frame = pd.DataFrame(feature_shift_summary)
    rows: list[dict[str, object]] = []
    for domain, group in frame.groupby("feature_domain", sort=True):
        shifted = group[group["shift_direction"].ne("no_shift")]
        rows.append(
            {
                "feature_domain": str(domain),
                "feature_count": int(len(group)),
                "shifted_feature_count": int(len(shifted)),
                "mean_abs_shift": _mean(
                    pd.to_numeric(group["mean_delta"], errors="coerce").abs()
                ),
                "max_driver_score": _max(
                    pd.to_numeric(group["driver_score"], errors="coerce")
                ),
                "top_feature": str(
                    group.sort_values("driver_score", ascending=False).iloc[0][
                        "feature_name"
                    ]
                ),
            }
        )
    return sorted(
        [_clean_row(row) for row in rows],
        key=lambda row: (
            -float(row.get("max_driver_score") or 0.0),
            str(row["feature_domain"]),
        ),
    )


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


def _overall_recommendation(
    failure_seasons: list[str],
    feature_shift_summary: list[dict[str, object]],
) -> str:
    if failure_seasons and any(
        float(row.get("driver_score") or 0.0) > 0 for row in feature_shift_summary
    ):
        return "inspect_exposure_feature_shift_drivers"
    if failure_seasons:
        return "inspect_exposure_failure_without_feature_shift"
    return "keep_exposure_load_failure_modes_under_review"


def _interpretation(summary: dict[str, object]) -> str:
    top_domains = [
        str(row["feature_domain"])
        for row in summary.get("domain_shift_summary", [])[:3]
    ]
    if top_domains:
        return (
            "The failed season should be reviewed first through the shifted "
            f"exposure domains: {', '.join(top_domains)}. These rows are a "
            "diagnostic for model behavior, not a new pilot-ready feature set."
        )
    return (
        "No clear exposure feature distribution shift was detected. Keep the "
        "failure-mode review in research validation before changing model scope."
    )


def _counter(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row.get(field)) for row in rows if row.get(field)).items()
        )
    )


def _mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


def _nonzero_rate(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.ne(0).mean()), 6)


def _quantile(values: pd.Series, q: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.quantile(q)), 6)


def _max(values: pd.Series) -> float | int | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    value = float(numeric.max())
    return int(value) if value.is_integer() else round(value, 6)


def _delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _relative_shift(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    if float(right) == 0.0:
        return None
    return round(float(left) / float(right), 6)


def _driver_score(mean_delta: object, relative_shift: object) -> float:
    if relative_shift is not None:
        return round(abs(float(relative_shift) - 1.0), 6)
    if mean_delta is not None:
        return round(abs(float(mean_delta)), 6)
    return 0.0


def _shift_direction(mean_delta: object) -> str:
    if mean_delta is None or float(mean_delta) == 0.0:
        return "no_shift"
    if float(mean_delta) > 0.0:
        return "elevated_in_failure"
    return "reduced_in_failure"


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


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"
