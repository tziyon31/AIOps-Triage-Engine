from scripts.promote import build_candidate_selection_report


def make_run(
    *,
    run_id: str,
    variant_name: str,
    f1_macro: float,
    latency_p95: float = 20.0,
    low_confidence_rate: float = 0.10,
    approval_required_rate: float = 0.20,
    weakest_class_recall: float = 0.80,
) -> dict:
    return {
        "run_id": run_id,
        "run_name": variant_name,
        "status": "FINISHED",
        "start_time": "2026-01-01 00:00:00",
        "tags": {
            "variant_name": variant_name,
            "model_family": variant_name,
        },
        "params": {},
        "metrics": {
            "f1_macro": f1_macro,
            "combined_weakest_class_recall": weakest_class_recall,
            "low_confidence_rate": low_confidence_rate,
            "approval_required_rate": approval_required_rate,
            "offline_decision_latency_p95_ms": latency_p95,
        },
    }


def make_policy() -> dict:
    return {
        "policy_name": "decision_engine_candidate_selection",
        "policy_version": "v1",
        "policy_path": "config/candidate_selection.yaml",
        "status": "learning_policy_not_production_gate",
        "baseline": {
            "required": True,
            "minimum_f1_improvement_over_baseline": 0.02,
        },
        "thresholds": {
            "minimum_f1_macro": 0.80,
            "minimum_weakest_class_recall": 0.70,
            "maximum_low_confidence_rate": 0.20,
            "maximum_approval_required_rate": 0.50,
            "maximum_offline_decision_latency_p95_ms": 200.0,
        },
        "tie_breakers": [
            "highest_f1_macro",
            "lowest_offline_decision_latency_p95_ms",
            "lowest_low_confidence_rate",
            "lowest_approval_required_rate",
            "earliest_start_time",
            "run_id",
        ],
        "decisions": {
            "pass": "candidate_ready",
            "fail": "candidate_rejected",
            "review": "human_review_required",
        },
    }


def valid_contract() -> dict:
    return {
        "status": "valid",
        "comparison_group_id": "group-a",
        "comparison_type": "model_family",
        "changed_variable": "model_family",
        "checks": {
            "single_comparison_group_id": True,
            "single_comparison_type": True,
            "single_changed_variable": True,
            "controlled_variables_passed": True,
            "at_least_two_runs": True,
        },
    }


def valid_controlled_variable_validation() -> dict:
    return {
        "status": "passed",
        "checks": [],
    }


def test_build_candidate_selection_report_selects_best_ready_candidate():
    baseline = make_run(
        run_id="baseline",
        variant_name="baseline",
        f1_macro=0.80,
    )

    candidate_a = make_run(
        run_id="candidate-a",
        variant_name="candidate-a",
        f1_macro=0.83,
        latency_p95=20.0,
    )

    candidate_b = make_run(
        run_id="candidate-b",
        variant_name="candidate-b",
        f1_macro=0.90,
        latency_p95=30.0,
    )

    report = build_candidate_selection_report(
        experiment_name="log-triage-decision-engine",
        comparison_group_id="group-a",
        comparison_contract=valid_contract(),
        controlled_variable_validation=valid_controlled_variable_validation(),
        baseline_run=baseline,
        candidate_runs=[candidate_a, candidate_b],
        policy=make_policy(),
        apply=False,
    )

    assert report["selection_status"] == "selected"
    assert report["selected_candidate"]["run_id"] == "candidate-b"


def test_candidate_selection_rejects_candidate_without_baseline_improvement():
    baseline = make_run(
        run_id="baseline",
        variant_name="baseline",
        f1_macro=0.80,
    )

    candidate = make_run(
        run_id="candidate-a",
        variant_name="candidate-a",
        f1_macro=0.81,
    )

    report = build_candidate_selection_report(
        experiment_name="log-triage-decision-engine",
        comparison_group_id="group-a",
        comparison_contract=valid_contract(),
        controlled_variable_validation=valid_controlled_variable_validation(),
        baseline_run=baseline,
        candidate_runs=[candidate],
        policy=make_policy(),
        apply=False,
    )

    assert report["selection_status"] == "no_candidate_selected"
    assert report["selected_candidate"] is None
    assert report["candidate_evaluations"][0]["eligible"] is False


def test_candidate_selection_skips_baseline_run_if_it_is_in_candidate_pool():
    baseline = make_run(
        run_id="baseline",
        variant_name="baseline",
        f1_macro=0.80,
    )

    candidate = make_run(
        run_id="candidate-a",
        variant_name="candidate-a",
        f1_macro=0.85,
    )

    report = build_candidate_selection_report(
        experiment_name="log-triage-decision-engine",
        comparison_group_id="group-a",
        comparison_contract=valid_contract(),
        controlled_variable_validation=valid_controlled_variable_validation(),
        baseline_run=baseline,
        candidate_runs=[baseline, candidate],
        policy=make_policy(),
        apply=False,
    )

    assert report["selection_status"] == "selected"
    assert report["selected_candidate"]["run_id"] == "candidate-a"

    baseline_eval = report["candidate_evaluations"][0]
    assert baseline_eval["eligible"] is False
    assert baseline_eval["reason"] == "candidate_is_baseline_run"


def test_candidate_selection_does_not_select_when_comparison_invalid():
    baseline = make_run(
        run_id="baseline",
        variant_name="baseline",
        f1_macro=0.80,
    )

    candidate = make_run(
        run_id="candidate-a",
        variant_name="candidate-a",
        f1_macro=0.90,
    )

    contract = valid_contract()
    contract["status"] = "invalid"

    report = build_candidate_selection_report(
        experiment_name="log-triage-decision-engine",
        comparison_group_id="group-a",
        comparison_contract=contract,
        controlled_variable_validation=valid_controlled_variable_validation(),
        baseline_run=baseline,
        candidate_runs=[candidate],
        policy=make_policy(),
        apply=False,
    )

    assert report["selection_status"] == "no_candidate_selected"
    assert report["selected_candidate"] is None
