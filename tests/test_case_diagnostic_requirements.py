from risk_stratification_engine.case_diagnostic_requirements import (
    build_case_diagnostic_requirements,
    build_case_diagnostic_requirements_summary,
    write_case_diagnostic_requirements_report,
)


def test_build_case_diagnostic_requirements_prioritizes_missing_context_domains():
    cases = [
        {
            "channel_name": "broad_30d",
            "case_type": "true_positive_episode",
            "diagnostic_label": "model_signal_supported",
            "target_reason": "forward_calibration_survivor",
        },
        {
            "channel_name": "severity_14d",
            "case_type": "missed_injury",
            "diagnostic_label": "model_miss",
            "target_reason": "forward_ranking_survivor",
        },
        {
            "channel_name": "severity_14d",
            "case_type": "false_positive_episode",
            "diagnostic_label": "missing_context_or_managed_risk",
            "target_reason": "forward_calibration_survivor",
        },
        {
            "channel_name": "subtype_lower_extremity_soft_tissue_30d",
            "case_type": "missed_injury",
            "diagnostic_label": "model_miss",
            "target_reason": "forward_ranking_survivor",
        },
        {
            "channel_name": "severity_14d",
            "case_type": "high_intra_individual_deviation_episode",
            "diagnostic_label": "explanation_gap",
            "target_reason": "forward_calibration_survivor",
        },
    ]

    requirements = build_case_diagnostic_requirements(cases)

    domains = {row["requirement_domain"]: row for row in requirements}
    assert {
        "exposure_load",
        "intervention_availability",
        "baseline_frailty",
        "injury_mechanism",
        "explanation_fidelity",
    }.issubset(domains)
    assert domains["exposure_load"]["priority_tier"] == "critical"
    assert "practice_intensity" in domains["exposure_load"]["missing_data_fields"]
    assert domains["intervention_availability"]["evidence_case_count"] == 1
    assert "availability_status" in domains["intervention_availability"][
        "missing_data_fields"
    ]
    assert "mechanism_context_features" in domains["injury_mechanism"][
        "modeling_action"
    ]
    assert domains["explanation_fidelity"]["triggering_diagnostic_labels"] == [
        "explanation_gap"
    ]


def test_case_diagnostic_requirements_summary_and_report_name_production_blockers(
    tmp_path,
):
    cases = [
        {
            "channel_name": "severity_14d",
            "case_type": "missed_injury",
            "diagnostic_label": "model_miss",
            "target_reason": "forward_ranking_survivor",
        },
        {
            "channel_name": "severity_14d",
            "case_type": "false_positive_episode",
            "diagnostic_label": "missing_context_or_managed_risk",
            "target_reason": "forward_calibration_survivor",
        },
    ]
    requirements = build_case_diagnostic_requirements(cases)
    summary = build_case_diagnostic_requirements_summary(requirements, cases)

    assert summary["experiment_type"] == "case_diagnostic_requirements_sprint"
    assert summary["overall_recommendation"] == (
        "prioritize_data_acquisition_before_production"
    )
    assert summary["production_readiness"] == "not_ready_missing_context"
    assert summary["case_count"] == 2
    assert summary["critical_requirement_count"] >= 1

    output = tmp_path / "case_diagnostic_requirements_report.md"
    write_case_diagnostic_requirements_report(output, summary)

    report = output.read_text()
    assert "Case Diagnostic Requirements Sprint" in report
    assert "production blockers" in report
    assert "exposure, intervention, baseline/frailty, and injury mechanism" in report
