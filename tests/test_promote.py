from scripts.promote import (
    build_baseline_guard,
    build_candidate_lifecycle_plan,
    build_candidate_selection_report,
    build_candidate_status_transition,
    build_candidate_status_transitions,
    build_current_candidate_state,
)


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


def test_candidate_lifecycle_plan_supersedes_previous_current_candidate():
    plan = build_candidate_lifecycle_plan(
        selected_run_id="new-run",
        current_candidate_run_ids=["old-run"],
    )

    assert plan["new_current_candidate_run_id"] == "new-run"
    assert plan["previous_current_candidate_run_ids"] == ["old-run"]
    assert plan["superseded_run_ids"] == ["old-run"]
    assert plan["already_current"] is False


def test_candidate_lifecycle_plan_is_idempotent_when_selected_is_already_current():
    plan = build_candidate_lifecycle_plan(
        selected_run_id="same-run",
        current_candidate_run_ids=["same-run"],
    )

    assert plan["new_current_candidate_run_id"] == "same-run"
    assert plan["superseded_run_ids"] == []
    assert plan["already_current"] is True


def test_current_candidate_state_contains_selected_candidate_metadata():
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "selected_candidate": {
            "run_id": "new-run",
            "variant_name": "manual_tfidf_logistic_regression",
        },
        "baseline_run": {
            "run_id": "baseline-run",
        },
        "comparison_group_id": "group-a",
        "experiment_name": "log-triage-decision-engine",
        "candidate_selection_policy": {
            "policy_name": "decision_engine_candidate_selection",
            "policy_version": "v1",
        },
        "selection_status": "selected",
        "mode": "apply",
    }

    lifecycle_plan = {
        "previous_current_candidate_run_ids": ["old-run"],
        "superseded_run_ids": ["old-run"],
    }

    state = build_current_candidate_state(
        report=report,
        lifecycle_plan=lifecycle_plan,
    )

    assert state["current_candidate_run_id"] == "new-run"
    assert state["previous_current_candidate_run_ids"] == ["old-run"]
    assert state["superseded_run_ids"] == ["old-run"]
    assert state["baseline_run_id"] == "baseline-run"


def test_baseline_guard_passes_when_no_current_candidate_exists():
    guard = build_baseline_guard(
        baseline_run_id="baseline",
        current_candidate_run_ids=[],
    )

    assert guard["status"] == "passed"
    assert guard["mode"] == "bootstrap_no_current_candidate"


def test_baseline_guard_passes_when_baseline_matches_current_candidate():
    guard = build_baseline_guard(
        baseline_run_id="current-run",
        current_candidate_run_ids=["current-run"],
    )

    assert guard["status"] == "passed"
    assert guard["mode"] == "baseline_matches_current_candidate"


def test_baseline_guard_fails_when_baseline_does_not_match_current_candidate():
    guard = build_baseline_guard(
        baseline_run_id="weak-baseline",
        current_candidate_run_ids=["current-run"],
    )

    assert guard["status"] == "failed"
    assert guard["reason"] == "baseline_run_id_does_not_match_current_candidate"


def test_baseline_guard_fails_when_multiple_current_candidates_exist():
    guard = build_baseline_guard(
        baseline_run_id="current-a",
        current_candidate_run_ids=["current-a", "current-b"],
    )

    assert guard["status"] == "failed"
    assert guard["reason"] == "multiple_current_candidates_found"


def test_baseline_guard_allows_explicit_override_with_reason():
    guard = build_baseline_guard(
        baseline_run_id="manual-baseline",
        current_candidate_run_ids=["current-run"],
        allow_non_current_baseline=True,
        baseline_override_reason="bootstrap correction test",
    )

    assert guard["status"] == "passed_with_override"
    assert guard["mode"] == "non_current_baseline_override"


def test_baseline_guard_blocks_override_without_reason():
    guard = build_baseline_guard(
        baseline_run_id="manual-baseline",
        current_candidate_run_ids=["current-run"],
        allow_non_current_baseline=True,
        baseline_override_reason=None,
    )

    assert guard["status"] == "failed"


def test_candidate_selection_report_blocks_selection_when_baseline_guard_fails():
    baseline = make_run(
        run_id="weak-baseline",
        variant_name="weak-baseline",
        f1_macro=0.80,
    )

    candidate = make_run(
        run_id="candidate-a",
        variant_name="candidate-a",
        f1_macro=0.90,
    )

    baseline_guard = build_baseline_guard(
        baseline_run_id="weak-baseline",
        current_candidate_run_ids=["current-strong-run"],
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
        current_candidate_run_ids=["current-strong-run"],
        baseline_guard=baseline_guard,
    )

    assert report["selection_status"] == "blocked_by_baseline_guard"
    assert report["selected_candidate"] is None
    assert report["baseline_guard"]["status"] == "failed"


def test_candidate_rejected_when_promotion_evidence_fails(monkeypatch):
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

    def fake_evidence_result(*, run_id, contract):
        return {
            "status": "failed",
            "contract_name": "promotion_evidence_contract",
            "contract_version": "v1",
            "missing_params": [],
            "missing_metrics": ["f1_macro"],
            "missing_tags": [],
            "missing_artifacts": [],
        }

    monkeypatch.setattr(
        "scripts.promote.build_promotion_evidence_result",
        fake_evidence_result,
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
        promotion_evidence_contract={
            "contract_name": "promotion_evidence_contract",
            "contract_version": "v1",
            "contract_path": "config/promotion_evidence_contract.yaml",
            "status": "learning_contract_not_production_registry",
        },
    )

    assert report["selection_status"] == "no_candidate_selected"
    assert report["selected_candidate"] is None
    assert report["candidate_evaluations"][0]["eligible"] is False
    assert (
        report["candidate_evaluations"][0]["reason"]
        == "promotion_evidence_contract_failed"
    )


def test_candidate_selected_when_promotion_evidence_passes(monkeypatch):
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

    def fake_evidence_result(*, run_id, contract):
        return {
            "status": "passed",
            "contract_name": "promotion_evidence_contract",
            "contract_version": "v1",
            "missing_params": [],
            "missing_metrics": [],
            "missing_tags": [],
            "missing_artifacts": [],
        }

    monkeypatch.setattr(
        "scripts.promote.build_promotion_evidence_result",
        fake_evidence_result,
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
        promotion_evidence_contract={
            "contract_name": "promotion_evidence_contract",
            "contract_version": "v1",
            "contract_path": "config/promotion_evidence_contract.yaml",
            "status": "learning_contract_not_production_registry",
        },
    )

    assert report["selection_status"] == "selected"
    assert report["selected_candidate"]["run_id"] == "candidate-a"


def test_candidate_status_transition_marks_selected_candidate():
    evaluation = {
        "run_summary": {
            "run_id": "candidate-a",
            "variant_name": "candidate-a",
        },
        "eligible": True,
        "reason": "candidate_policy_and_baseline_passed",
    }

    transition = build_candidate_status_transition(
        evaluation=evaluation,
        selected_run_id="candidate-a",
        baseline_run_id="baseline",
    )

    assert transition["action"] == "update"
    assert transition["new_candidate_status"] == "selected"
    assert transition["candidate"] == "true"
    assert transition["current_candidate"] == "true"


def test_candidate_status_transition_marks_eligible_non_selected_candidate():
    evaluation = {
        "run_summary": {
            "run_id": "candidate-b",
            "variant_name": "candidate-b",
        },
        "eligible": True,
        "reason": "candidate_policy_and_baseline_passed",
    }

    transition = build_candidate_status_transition(
        evaluation=evaluation,
        selected_run_id="candidate-a",
        baseline_run_id="baseline",
    )

    assert transition["action"] == "update"
    assert transition["new_candidate_status"] == "eligible"
    assert transition["candidate"] == "false"
    assert transition["current_candidate"] == "false"


def test_candidate_status_transition_marks_rejected_candidate():
    evaluation = {
        "run_summary": {
            "run_id": "candidate-c",
            "variant_name": "candidate-c",
        },
        "eligible": False,
        "reason": "promotion_evidence_contract_failed",
    }

    transition = build_candidate_status_transition(
        evaluation=evaluation,
        selected_run_id="candidate-a",
        baseline_run_id="baseline",
    )

    assert transition["action"] == "update"
    assert transition["new_candidate_status"] == "rejected"
    assert transition["promotion_reason"] == "promotion_evidence_contract_failed"
    assert transition["candidate"] == "false"


def test_candidate_status_transition_skips_baseline_run():
    evaluation = {
        "run_summary": {
            "run_id": "baseline",
            "variant_name": "baseline",
        },
        "eligible": False,
        "reason": "candidate_is_baseline_run",
    }

    transition = build_candidate_status_transition(
        evaluation=evaluation,
        selected_run_id="candidate-a",
        baseline_run_id="baseline",
    )

    assert transition["action"] == "skip"
    assert transition["new_candidate_status"] is None
    assert transition["promotion_reason"] == "baseline_run_status_not_changed"


def test_candidate_selection_report_includes_status_transitions(monkeypatch):
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

    def fake_evidence_result(*, run_id, contract):
        return {
            "status": "passed",
            "contract_name": "promotion_evidence_contract",
            "contract_version": "v1",
            "missing_params": [],
            "missing_metrics": [],
            "missing_tags": [],
            "missing_artifacts": [],
        }

    monkeypatch.setattr(
        "scripts.promote.build_promotion_evidence_result",
        fake_evidence_result,
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
        promotion_evidence_contract={
            "contract_name": "promotion_evidence_contract",
            "contract_version": "v1",
            "contract_path": "config/promotion_evidence_contract.yaml",
            "status": "learning_contract_not_production_registry",
        },
    )

    assert report["candidate_status_transitions"][0]["run_id"] == "candidate-a"
    assert (
        report["candidate_status_transitions"][0]["new_candidate_status"]
        == "selected"
    )
