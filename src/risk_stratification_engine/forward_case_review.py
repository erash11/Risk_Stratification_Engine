from __future__ import annotations

from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


def build_forward_case_review_summary(
    cases: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "experiment_type": "forward_case_review_sprint",
        "overall_recommendation": _overall_recommendation(cases),
        "case_count": len(cases),
        "targeted_channel_count": len(
            {str(case.get("channel_name")) for case in cases if case.get("channel_name")}
        ),
        "targeted_test_seasons": sorted(
            {str(case.get("test_season_id")) for case in cases if case.get("test_season_id")}
        ),
        "diagnostic_summary": _counter(cases, "diagnostic_label"),
        "case_type_summary": _counter(cases, "case_type"),
        "channel_summary": _channel_summary(cases),
        "cases": [_clean_case(case) for case in cases],
    }


def write_forward_case_review_report(
    path: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Forward Case Review Sprint",
        "",
        f"Recommendation: {summary['overall_recommendation']}",
        f"Cases reviewed: {summary['case_count']}",
        "",
        "## Targeting",
        "",
        (
            "This sprint reviews forward-surviving windows and channels from "
            "season-forward validation, rather than searching all historical "
            "alerts again."
        ),
        "",
        "## Diagnostic Summary",
        "",
    ]
    for label, count in summary["diagnostic_summary"].items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "## Channel Summary",
            "",
            "| Channel | Cases | Dominant diagnostic |",
            "|---|---:|---|",
        ]
    )
    for channel_name, row in summary["channel_summary"].items():
        lines.append(
            "| "
            f"{channel_name} | "
            f"{row['case_count']} | "
            f"{row['dominant_diagnostic_label']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Use these cases to decide whether apparent misses and noisy "
                "alerts are due to missing exposure, intervention, baseline, or "
                "mechanism context, or whether the graph signal itself is too "
                "weak for the target channel."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _overall_recommendation(cases: list[dict[str, object]]) -> str:
    if not cases:
        return "insufficient_forward_cases"
    return "continue_targeted_case_review"


def _counter(cases: list[dict[str, object]], field: str) -> dict[str, int]:
    counts = Counter(str(case.get(field)) for case in cases if case.get(field))
    return dict(sorted(counts.items()))


def _channel_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(cases)
    if frame.empty or "channel_name" not in frame:
        return {}
    summary: dict[str, object] = {}
    for channel_name, group in frame.groupby("channel_name", sort=True):
        diagnostic_counts = Counter(str(value) for value in group["diagnostic_label"])
        dominant = sorted(
            diagnostic_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        summary[str(channel_name)] = {
            "channel_name": str(channel_name),
            "case_count": int(len(group)),
            "test_seasons": sorted(str(value) for value in group["test_season_id"].unique()),
            "diagnostic_summary": dict(sorted(diagnostic_counts.items())),
            "dominant_diagnostic_label": dominant,
        }
    return summary


def _clean_case(case: dict[str, object]) -> dict[str, object]:
    return {str(key): _clean_value(value) for key, value in case.items()}


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
