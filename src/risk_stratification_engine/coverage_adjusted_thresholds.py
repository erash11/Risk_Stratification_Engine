from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from risk_stratification_engine.alert_episodes import build_alert_episodes
from risk_stratification_engine.episode_quality import build_alert_episode_quality


THRESHOLD_POLICIES = (
    "season_local_percentile",
    "coverage_tier_local_percentile",
    "burden_capped_percentile",
)
DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON = 1.0


def build_coverage_adjusted_threshold_policy_rows(
    timeline: pd.DataFrame,
    channel: dict[str, object],
    *,
    burden_cap_episodes_per_athlete_season: float = (
        DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON
    ),
    candidate_percentiles: tuple[float, ...] | None = None,
) -> list[dict[str, object]]:
    if timeline.empty:
        return []
    frame = timeline.copy()
    frame["season_id"] = frame["season_id"].astype(str)
    if "coverage_tier" not in frame.columns:
        frame["coverage_tier"] = "unknown"

    horizon = int(channel["horizon_days"])
    base_percentile = float(channel["threshold_value"])
    candidate_percentiles = _candidate_percentiles(base_percentile, candidate_percentiles)

    rows: list[dict[str, object]] = []
    for season_id, season_timeline in frame.groupby("season_id", sort=True):
        rows.append(
            _policy_row(
                channel=channel,
                timeline=season_timeline,
                threshold_policy="season_local_percentile",
                threshold_scope="season",
                slice_id=str(season_id),
                coverage_tier="all",
                selected_percentile=base_percentile,
                burden_cap=None,
                horizon=horizon,
            )
        )
        for tier, tier_timeline in season_timeline.groupby("coverage_tier", sort=True):
            rows.append(
                _policy_row(
                    channel=channel,
                    timeline=tier_timeline,
                    threshold_policy="coverage_tier_local_percentile",
                    threshold_scope="season_coverage_tier",
                    slice_id=str(season_id),
                    coverage_tier=str(tier),
                    selected_percentile=base_percentile,
                    burden_cap=None,
                    horizon=horizon,
                )
            )

        selected = _burden_capped_percentile(
            timeline=season_timeline,
            horizon=horizon,
            candidate_percentiles=candidate_percentiles,
            burden_cap=burden_cap_episodes_per_athlete_season,
        )
        rows.append(
            _policy_row(
                channel=channel,
                timeline=season_timeline,
                threshold_policy="burden_capped_percentile",
                threshold_scope="season",
                slice_id=str(season_id),
                coverage_tier="all",
                selected_percentile=selected,
                burden_cap=burden_cap_episodes_per_athlete_season,
                horizon=horizon,
            )
        )
    return rows


def build_coverage_adjusted_threshold_summary(
    rows: list[dict[str, object]],
    *,
    burden_cap_episodes_per_athlete_season: float = (
        DEFAULT_BURDEN_CAP_EPISODES_PER_ATHLETE_SEASON
    ),
) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    recommendations: dict[str, object] = {}
    if not frame.empty:
        for channel_name, channel_group in frame.groupby("channel_name", sort=True):
            policy_rows = []
            for policy_name, policy_group in channel_group.groupby(
                "threshold_policy",
                sort=True,
            ):
                policy_rows.append(
                    {
                        "threshold_policy": str(policy_name),
                        "mean_capture_rate": _mean(
                            policy_group["unique_event_capture_rate"]
                        ),
                        "mean_episodes_per_athlete_season": _mean(
                            policy_group["episodes_per_athlete_season"]
                        ),
                        "mean_selected_threshold_value": _mean_column(
                            policy_group, "selected_threshold_value"
                        ),
                    }
                )
            recommended = _recommended_policy(
                policy_rows,
                burden_cap_episodes_per_athlete_season,
            )
            recommendations[str(channel_name)] = {
                "channel_name": str(channel_name),
                "recommended_policy": recommended["threshold_policy"]
                if recommended
                else None,
                "mean_capture_rate": recommended["mean_capture_rate"]
                if recommended
                else None,
                "mean_episodes_per_athlete_season": recommended[
                    "mean_episodes_per_athlete_season"
                ]
                if recommended
                else None,
                "policy_summaries": policy_rows,
            }

    return {
        "experiment_type": "coverage_adjusted_threshold_sprint",
        "overall_recommendation": "continue_threshold_research",
        "threshold_policies": list(THRESHOLD_POLICIES),
        "burden_cap_episodes_per_athlete_season": (
            burden_cap_episodes_per_athlete_season
        ),
        "channel_recommendations": recommendations,
        "policy_rows": [_clean_record(row) for row in rows],
    }


def write_coverage_adjusted_threshold_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Coverage-Adjusted Threshold Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        (
            "Burden cap: "
            f"{_fmt(summary['burden_cap_episodes_per_athlete_season'])} "
            "episodes per athlete-season"
        ),
        "",
        "## Peterson Guardrail",
        "",
        (
            "Thresholds are adjusted only after complete athlete-season "
            "trajectories are scored. The sprint does not relabel or resample "
            "daily rows as independent examples."
        ),
        "",
        "## Threshold Policies",
        "",
    ]
    for policy_name in summary["threshold_policies"]:
        lines.append(f"- {policy_name}")

    lines.extend(
        [
            "",
            "## Channel Recommendations",
            "",
            "| Channel | Recommended policy | Mean capture | Mean burden |",
            "|---|---|---:|---:|",
        ]
    )
    for channel_name, row in summary["channel_recommendations"].items():
        lines.append(
            "| "
            f"{channel_name} | "
            f"{row['recommended_policy']} | "
            f"{_fmt(row['mean_capture_rate'])} | "
            f"{_fmt(row['mean_episodes_per_athlete_season'])} |"
        )

    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _policy_row(
    *,
    channel: dict[str, object],
    timeline: pd.DataFrame,
    threshold_policy: str,
    threshold_scope: str,
    slice_id: str,
    coverage_tier: str,
    selected_percentile: float,
    burden_cap: float | None,
    horizon: int,
) -> dict[str, object]:
    episodes = build_alert_episodes(
        timeline,
        horizons=(horizon,),
        percentile_thresholds=(selected_percentile,),
    )
    quality_rows = build_alert_episode_quality(episodes, timeline)["quality_rows"]
    quality = (
        quality_rows[0]
        if quality_rows
        else _empty_quality_row(timeline, horizon, selected_percentile)
    )
    return {
        "threshold_policy": threshold_policy,
        "channel_name": str(channel["channel_name"]),
        "role": str(channel["role"]),
        "slice_type": "season",
        "slice_id": slice_id,
        "coverage_tier": coverage_tier,
        "policy_name": str(channel["policy_name"]),
        "graph_window_size": int(channel["graph_window_size"]),
        "horizon_days": horizon,
        "threshold_scope": threshold_scope,
        "selected_threshold_value": float(selected_percentile),
        "burden_cap_episodes_per_athlete_season": burden_cap,
        "episode_count": quality["episode_count"],
        "true_positive_episode_count": quality["true_positive_episode_count"],
        "false_positive_episode_count": quality["false_positive_episode_count"],
        "unique_observed_event_count": quality["unique_observed_event_count"],
        "unique_captured_event_count": quality["unique_captured_event_count"],
        "unique_event_capture_rate": quality["unique_event_capture_rate"],
        "missed_event_count": quality["missed_event_count"],
        "episodes_per_athlete_season": quality["episodes_per_athlete_season"],
        "median_start_lead_days": quality["median_start_lead_days"],
    }


def _burden_capped_percentile(
    *,
    timeline: pd.DataFrame,
    horizon: int,
    candidate_percentiles: tuple[float, ...],
    burden_cap: float,
) -> float:
    best_percentile = candidate_percentiles[-1]
    for percentile in candidate_percentiles:
        episodes = build_alert_episodes(
            timeline,
            horizons=(horizon,),
            percentile_thresholds=(percentile,),
        )
        quality_rows = build_alert_episode_quality(episodes, timeline)["quality_rows"]
        if quality_rows:
            burden = quality_rows[0]["episodes_per_athlete_season"]
        else:
            burden = 0.0
        best_percentile = percentile
        if burden is not None and float(burden) <= burden_cap:
            break
    return float(best_percentile)


def _candidate_percentiles(
    base_percentile: float,
    explicit: tuple[float, ...] | None,
) -> tuple[float, ...]:
    values = explicit or (
        base_percentile,
        base_percentile / 2.0,
        base_percentile / 4.0,
        0.01,
    )
    return tuple(
        sorted({float(value) for value in values if value > 0.0}, reverse=True)
    )


def _empty_quality_row(
    timeline: pd.DataFrame,
    horizon: int,
    threshold_value: float,
) -> dict[str, object]:
    observed = timeline[
        timeline.get("event_observed", pd.Series(dtype=bool)).astype(bool)
    ]
    if {"athlete_id", "season_id", "event_date", "injury_type"}.issubset(
        observed.columns
    ):
        event_count = int(
            observed[["athlete_id", "season_id", "event_date", "injury_type"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        event_count = 0
    athlete_seasons = (
        int(timeline[["athlete_id", "season_id"]].drop_duplicates().shape[0])
        if {"athlete_id", "season_id"}.issubset(timeline.columns)
        else 0
    )
    return {
        "horizon_days": horizon,
        "threshold_kind": "percentile",
        "threshold_value": threshold_value,
        "episode_count": 0,
        "true_positive_episode_count": 0,
        "false_positive_episode_count": 0,
        "unique_observed_event_count": event_count,
        "unique_captured_event_count": 0,
        "unique_event_capture_rate": 0.0 if event_count else None,
        "missed_event_count": event_count,
        "episodes_per_athlete_season": 0.0 if athlete_seasons else None,
        "median_start_lead_days": None,
    }


def _recommended_policy(
    policy_rows: list[dict[str, object]],
    burden_cap: float,
) -> dict[str, object] | None:
    if not policy_rows:
        return None
    eligible = [
        row
        for row in policy_rows
        if row["mean_episodes_per_athlete_season"] is not None
        and float(row["mean_episodes_per_athlete_season"]) <= burden_cap
    ]
    candidates = eligible or policy_rows
    return sorted(
        candidates,
        key=lambda row: (
            -1.0
            * (
                float(row["mean_capture_rate"])
                if row["mean_capture_rate"] is not None
                else -1.0
            ),
            float(row["mean_episodes_per_athlete_season"])
            if row["mean_episodes_per_athlete_season"] is not None
            else float("inf"),
            str(row["threshold_policy"]),
        ),
    )[0]


def _mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


def _mean_column(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    return _mean(frame[column])


def _interpretation(summary: dict[str, object]) -> str:
    return (
        "Use these results to decide whether tier-local thresholds or burden caps "
        "reduce alert load without erasing event capture. This is threshold-policy "
        "research, not dashboard or pilot clearance."
    )


def _clean_record(row: dict[str, object]) -> dict[str, object]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if pd.isna(number):
            return None
        return int(number) if number.is_integer() else number
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"
