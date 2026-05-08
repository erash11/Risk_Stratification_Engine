from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from risk_stratification_engine.events import DEFAULT_HORIZONS


DECISION_METRICS = {
    "ranking": ("roc_auc", max),
    "calibration": ("brier_skill_score", max),
    "burden_triage": ("top_decile_lift", max),
}


def build_season_forward_validation_summary(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    test_seasons = (
        sorted(str(value) for value in frame["test_season_id"].dropna().unique())
        if "test_season_id" in frame
        else []
    )
    return {
        "experiment_type": "season_forward_validation_sprint",
        "overall_recommendation": _overall_recommendation(frame),
        "split_policy": "season_forward_train_prior_evaluate_next",
        "evaluated_test_seasons": test_seasons,
        "row_count": len(rows),
        "best_by_horizon": _best_by_horizon(frame),
        "alert_policy_summary": _alert_policy_summary(frame),
        "validation_rows": [_clean_row(row) for row in rows],
    }


def write_season_forward_validation_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Season-Forward Validation Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Split policy: {summary['split_policy']}",
        "",
        "## Peterson Guardrail",
        "",
        (
            "This sprint is designed to train on earlier seasons and evaluate later seasons "
            "after complete athlete-season trajectories are scored. It preserves "
            "the longitudinal unit of analysis instead of treating daily rows as "
            "independent injury-classification examples."
        ),
        "",
        "## Model Comparison",
        "",
        "| Test season | Feature set | Horizon | AUROC | Brier skill | Top-decile lift |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["validation_rows"]:
        if row.get("row_type") != "model_metric":
            continue
        lines.append(
            "| "
            f"{row['test_season_id']} | "
            f"{row['feature_set']} | "
            f"{row['horizon_days']}d | "
            f"{_fmt(row.get('roc_auc'))} | "
            f"{_fmt(row.get('brier_skill_score'))} | "
            f"{_fmt(row.get('top_decile_lift'))} |"
        )

    lines.extend(
        [
            "",
            "## Alert Policy Forward Check",
            "",
            "| Channel | Recommended policy | Mean capture | Mean burden |",
            "|---|---|---:|---:|",
        ]
    )
    for channel_name, row in summary["alert_policy_summary"].items():
        lines.append(
            "| "
            f"{channel_name} | "
            f"{row['recommended_threshold_policy']} | "
            f"{_fmt(row.get('mean_capture_rate'))} | "
            f"{_fmt(row.get('mean_episodes_per_athlete_season'))} |"
        )

    lines.extend(["", "## Interpretation", "", _interpretation(summary)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _best_by_horizon(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or "row_type" not in frame:
        return {}
    model_rows = frame[frame["row_type"].eq("model_metric")]
    best: dict[str, object] = {}
    for horizon in DEFAULT_HORIZONS:
        group = model_rows[model_rows["horizon_days"].eq(horizon)]
        if group.empty:
            continue
        horizon_best: dict[str, object] = {}
        for mode_name, (metric_name, selector) in DECISION_METRICS.items():
            if metric_name not in group:
                continue
            candidates = group.dropna(subset=[metric_name])
            if candidates.empty:
                continue
            records = candidates.sort_values(
                ["test_season_id", "feature_set"]
            ).to_dict("records")
            winner = selector(records, key=lambda candidate: candidate[metric_name])
            horizon_best[mode_name] = {
                "feature_set": str(winner["feature_set"]),
                "test_season_id": str(winner["test_season_id"]),
                metric_name: _clean_value(winner[metric_name]),
            }
        best[str(horizon)] = horizon_best
    return best


def _alert_policy_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or "row_type" not in frame:
        return {}
    alert_rows = frame[frame["row_type"].eq("alert_policy")]
    if alert_rows.empty:
        return {}
    summary: dict[str, object] = {}
    for channel_name, channel_group in alert_rows.groupby("channel_name", sort=True):
        policy_rows = []
        for policy_name, policy_group in channel_group.groupby(
            "threshold_policy",
            sort=True,
        ):
            policy_rows.append(
                {
                    "threshold_policy": str(policy_name),
                    "mean_capture_rate": _mean_column(
                        policy_group, "unique_event_capture_rate"
                    ),
                    "mean_episodes_per_athlete_season": _mean_column(
                        policy_group, "episodes_per_athlete_season"
                    ),
                }
            )
        recommended = _recommend_alert_policy(policy_rows)
        summary[str(channel_name)] = {
            "channel_name": str(channel_name),
            "recommended_threshold_policy": recommended["threshold_policy"]
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
    return summary


def _recommend_alert_policy(
    policy_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    if not policy_rows:
        return None
    eligible = [
        row
        for row in policy_rows
        if row["mean_episodes_per_athlete_season"] is not None
        and float(row["mean_episodes_per_athlete_season"]) <= 1.0
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


def _overall_recommendation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "insufficient_season_forward_evidence"
    return "continue_season_forward_research"


def _interpretation(summary: dict[str, object]) -> str:
    return (
        "Use this sprint to decide whether current graph and coverage/source "
        "signals survive forward-season validation. Results remain research-only "
        "until a channel shows repeatable forward capture with tolerable burden."
    )


def _mean_column(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


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
        return int(number) if number.is_integer() else number
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"
