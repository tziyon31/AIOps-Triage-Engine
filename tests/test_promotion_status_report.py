from scripts.build_promotion_status_report import (
    build_mlflow_filter_examples,
    build_promotion_status_report,
    build_status_policy_checks,
)


def make_summary(
    *,
    run_id: str,
    status: str,
    evidence_status: str = "passed",
    current: str = "false",
    candidate: str = "false",
    reason: str = "not_evaluated",
) -> dict:
    return {
        "run_id": run_id,
        "candidate_status": status,
        "candidate": candidate,
        "current_candidate": current,
        "promotion_reason": reason,
        "promotion_rejection_reason": "",
        "promotion_evaluated_at": "",
        "run_owner": "tziyon31",
        "variant_name": "variant-a",
        "comparison_group_id": "group-a",
        "metrics": {
            "f1_macro": 0.9,
            "offline_decision_latency_p95_ms": 20.0,
        },
        "promotion_evidence": {
            "status": evidence_status,
            "missing_params": [],
            "missing_metrics": [],
            "missing_tags": [],
            "missing_artifacts": [],
        },
    }


def make_contract() -> dict:
    return {
        "contract_name": "promotion_evidence_contract",
        "contract_version": "v1",
        "contract_path": "config/promotion_evidence_contract.yaml",
        "status": "learning_contract_not_production_registry",
    }


def test_status_policy_checks_pass_for_single_current_and_valid_selected():
    summaries = [
        make_summary(
            run_id="run-1",
            status="selected",
            current="true",
            candidate="true",
            reason="candidate_policy_and_baseline_passed",
        ),
        make_summary(
            run_id="run-2",
            status="rejected",
            reason="promotion_evidence_contract_failed",
        ),
    ]

    checks = build_status_policy_checks(summaries=summaries)

    assert checks["status"] == "passed"
    assert checks["checks"]["single_current_candidate"] is True
    assert checks["checks"]["selected_runs_have_passed_evidence"] is True
    assert checks["checks"]["rejected_runs_have_reason"] is True


def test_status_policy_checks_fail_when_multiple_current_candidates():
    summaries = [
        make_summary(
            run_id="run-1",
            status="selected",
            current="true",
            candidate="true",
        ),
        make_summary(
            run_id="run-2",
            status="selected",
            current="true",
            candidate="true",
        ),
    ]

    checks = build_status_policy_checks(summaries=summaries)

    assert checks["status"] == "failed"
    assert checks["checks"]["single_current_candidate"] is False


def test_status_policy_checks_fail_when_selected_evidence_failed():
    summaries = [
        make_summary(
            run_id="run-1",
            status="selected",
            current="true",
            candidate="true",
            evidence_status="failed",
        ),
    ]

    checks = build_status_policy_checks(summaries=summaries)

    assert checks["status"] == "failed"
    assert checks["checks"]["selected_runs_have_passed_evidence"] is False
    assert checks["selected_with_failed_evidence"] == ["run-1"]


def test_build_promotion_status_report_counts_statuses():
    summaries = [
        make_summary(run_id="run-1", status="selected"),
        make_summary(run_id="run-2", status="rejected"),
        make_summary(run_id="run-3", status="rejected"),
        make_summary(run_id="run-4", status="not_evaluated"),
    ]

    report = build_promotion_status_report(
        experiment_name="log-triage-decision-engine",
        run_summaries=summaries,
        contract=make_contract(),
    )

    assert report["status_counts"]["selected"] == 1
    assert report["status_counts"]["rejected"] == 2
    assert report["status_counts"]["not_evaluated"] == 1


def test_mlflow_filter_examples_include_candidate_status_filters():
    filters = build_mlflow_filter_examples()

    assert filters["selected"] == "tags.candidate_status = 'selected'"
    assert filters["rejected"] == "tags.candidate_status = 'rejected'"
    assert filters["current_candidate"] == "tags.current_candidate = 'true'"
