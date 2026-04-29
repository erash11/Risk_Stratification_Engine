from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path

import pandas as pd


DEFAULT_POLICY_DECISION_POLICIES = (
    "any_injury",
    "exclude_concussion",
    "model_safe_time_loss",
    "lower_extremity_soft_tissue",
    "severe_time_loss",
)
DEFAULT_POLICY_WINDOW_SIZES = (2, 4, 7)


def build_two_channel_alert_policy(
    comparison_rows: pd.DataFrame,
    policy_summaries: dict[str, object] | None = None,
) -> dict[str, object]:
    rows = comparison_rows.copy()
    broad = _select_best(
        rows,
        policy_names=("exclude_concussion", "any_injury"),
        horizon_days=30,
        thresholds=("percentile:0.05",),
        metric_name="brier_skill_score",
    )
    severity_views = [
        _compact_view(
            _select_best(
                rows,
                policy_names=("model_safe_time_loss",),
                horizon_days=horizon,
                metric_name="unique_event_capture_rate",
            )
        )
        for horizon in (7, 14)
    ]
    severity_views = [view for view in severity_views if view is not None]
    subtype = _select_best(
        rows,
        policy_names=("lower_extremity_soft_tissue",),
        horizon_days=30,
        metric_name="unique_event_capture_rate",
    )
    return {
        "experiment_type": "two_channel_alert_policy",
        "policy_summaries": policy_summaries or {},
        "channels": {
            "broad_early_warning": _channel_row(
                broad,
                role="30d broad early-warning channel",
                selection_metric="brier_skill_score",
            ),
            "severity_short_horizon": {
                "role": "7d/14d severity-oriented channel",
                "policy_name": "model_safe_time_loss",
                "selection_metric": "unique_event_capture_rate",
                "views": severity_views,
            },
            "subtype_review": _channel_row(
                subtype,
                role="30d lower-extremity soft-tissue review channel",
                selection_metric="unique_event_capture_rate",
            ),
        },
        "interpretation": (
            "Use a broad channel for 30d early warning and a model-safe "
            "time-loss channel for short-horizon severity triage."
        ),
    }


def build_policy_window_sensitivity(
    comparison_rows: pd.DataFrame,
) -> dict[str, object]:
    rows = comparison_rows.copy()
    recommendations = {
        "broad_early_warning": _channel_row(
            _select_best(
                rows,
                policy_names=("exclude_concussion", "any_injury"),
                horizon_days=30,
                thresholds=("percentile:0.05",),
                metric_name="brier_skill_score",
            ),
            role="best 30d broad channel window",
            selection_metric="brier_skill_score",
        ),
        "severity_7d": _channel_row(
            _select_best(
                rows,
                policy_names=("model_safe_time_loss",),
                horizon_days=7,
                metric_name="unique_event_capture_rate",
            ),
            role="best 7d severity channel window",
            selection_metric="unique_event_capture_rate",
        ),
        "severity_14d": _channel_row(
            _select_best(
                rows,
                policy_names=("model_safe_time_loss",),
                horizon_days=14,
                metric_name="unique_event_capture_rate",
            ),
            role="best 14d severity channel window",
            selection_metric="unique_event_capture_rate",
        ),
        "subtype_30d": _channel_row(
            _select_best(
                rows,
                policy_names=("lower_extremity_soft_tissue",),
                horizon_days=30,
                metric_name="unique_event_capture_rate",
            ),
            role="best 30d subtype-review channel window",
            selection_metric="unique_event_capture_rate",
        ),
    }
    return {
        "experiment_type": "policy_window_sensitivity",
        "window_sizes": sorted(
            int(size) for size in rows.get("graph_window_size", pd.Series()).dropna().unique()
        ),
        "comparison_row_count": int(len(rows)),
        "recommendations": recommendations,
        "comparison_rows": _records(rows),
    }


def build_operational_policy_package(
    two_channel_policy: dict[str, object],
    window_sensitivity: dict[str, object],
) -> dict[str, object]:
    channels = two_channel_policy.get("channels", {})
    return {
        "experiment_type": "operational_policy_package",
        "sprint_count": 3,
        "status": "research_shadow_mode",
        "recommended_policy": {
            "broad_early_warning": {
                **dict(channels.get("broad_early_warning") or {}),
                "window_recommendation": window_sensitivity.get(
                    "recommendations", {}
                ).get("broad_early_warning"),
            },
            "severity_short_horizon": {
                **dict(channels.get("severity_short_horizon") or {}),
                "window_recommendations": {
                    "7d": window_sensitivity.get("recommendations", {}).get(
                        "severity_7d"
                    ),
                    "14d": window_sensitivity.get("recommendations", {}).get(
                        "severity_14d"
                    ),
                },
            },
            "subtype_review": {
                **dict(channels.get("subtype_review") or {}),
                "window_recommendation": window_sensitivity.get(
                    "recommendations", {}
                ).get("subtype_30d"),
            },
        },
        "not_recommended_primary_targets": [
            "severe_time_loss",
            "concussion_only",
        ],
        "deployment_boundary": (
            "Research workbench only: use for shadow-mode review and case "
            "audit, not autonomous operational intervention."
        ),
        "next_sprint": "shadow-mode policy stability audit",
    }


def write_two_channel_alert_policy_report(
    path: Path,
    policy: dict[str, object],
) -> None:
    channels = policy["channels"]
    broad = channels.get("broad_early_warning") or {}
    severity = channels.get("severity_short_horizon") or {}
    subtype = channels.get("subtype_review") or {}
    lines = [
        "# Two-Channel Alert Policy",
        "",
        "## Broad Early Warning",
        _channel_line(broad),
        "",
        "## Severity Short Horizon",
    ]
    for view in severity.get("views", []):
        lines.append(
            f"- {view['horizon_days']}d {severity['policy_name']} at "
            f"{view['threshold']}"
        )
    lines.extend(
        [
            "",
            "## Subtype Review",
            _channel_line(subtype),
            "",
            "## Interpretation",
            str(policy["interpretation"]),
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_policy_window_sensitivity_report(
    path: Path,
    sensitivity: dict[str, object],
) -> None:
    lines = [
        "# Policy Window Sensitivity",
        "",
        f"Rows: {sensitivity['comparison_row_count']}",
        f"Windows: {', '.join(str(size) for size in sensitivity['window_sizes'])}",
        "",
        "## Recommendations",
    ]
    for name, row in sensitivity["recommendations"].items():
        lines.append(f"- {name}: {_channel_line(row)}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_operational_policy_package_report(
    path: Path,
    package: dict[str, object],
) -> None:
    lines = [
        "# Operational Policy Package",
        "",
        f"Status: {package['status']}",
        "",
        "## Recommended Policy",
    ]
    for name, row in package["recommended_policy"].items():
        policy_name = row.get("policy_name", "n/a")
        lines.append(f"- {name}: {policy_name}")
    lines.extend(
        [
            "",
            "## Not Recommended As Primary Targets",
            ", ".join(package["not_recommended_primary_targets"]),
            "",
            "## Boundary",
            package["deployment_boundary"],
            "",
            "## Next Sprint",
            package["next_sprint"],
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _select_best(
    rows: pd.DataFrame,
    *,
    policy_names: tuple[str, ...],
    horizon_days: int,
    metric_name: str,
    thresholds: tuple[str, ...] | None = None,
) -> dict[str, object] | None:
    if rows.empty or metric_name not in rows:
        return None
    subset = rows[
        rows["policy_name"].isin(policy_names)
        & rows["horizon_days"].astype(int).eq(int(horizon_days))
    ].copy()
    if thresholds is not None:
        subset = subset[subset["threshold"].isin(thresholds)]
    subset = subset.dropna(subset=[metric_name])
    if subset.empty:
        return None
    sort_columns = [metric_name, "policy_name", "threshold"]
    ascending = [False, True, True]
    if "episodes_per_athlete_season" in subset:
        sort_columns.append("episodes_per_athlete_season")
        ascending.append(True)
    return _row_dict(subset.sort_values(sort_columns, ascending=ascending).iloc[0])


def _channel_row(
    row: dict[str, object] | None,
    *,
    role: str,
    selection_metric: str,
) -> dict[str, object]:
    if row is None:
        return {
            "role": role,
            "selection_metric": selection_metric,
            "policy_name": None,
        }
    return {
        "role": role,
        "selection_metric": selection_metric,
        **row,
    }


def _compact_view(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "horizon_days": int(row["horizon_days"]),
        "threshold": str(row["threshold"]),
    }


def _channel_line(row: dict[str, object]) -> str:
    if not row or row.get("policy_name") is None:
        return "n/a"
    window = row.get("graph_window_size")
    window_text = f", window {int(window)}" if window is not None else ""
    return (
        f"{row.get('policy_name')} {row.get('horizon_days')}d "
        f"{row.get('threshold')}{window_text}"
    )


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [_row_dict(row) for _, row in frame.iterrows()]


def _row_dict(row: pd.Series) -> dict[str, object]:
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
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
