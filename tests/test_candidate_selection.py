from pathlib import Path

import pytest

from src.log_triage.candidate_selection import (
    compare_to_baseline,
    evaluate_candidate_run,
    load_candidate_selection_policy,
)


def make_run(
    *,
    run_id: str,
    f1_macro: float | None = 0.90,
    weakest_class_recall: float | None = 0.85,
    low_confidence_rate: float | None = 0.10,
    approval_required_rate: float | None = 0.20,
    latency_p95_ms: float | None = 15.0,
) -> dict:
    metrics = {}

    if f1_macro is not None:
        metrics["f1_macro"] = f1_macro
    if weakest_class_recall is not None:
        metrics["combined_weakest_class_recall"] = weakest_class_recall
    if low_confidence_rate is not None:
        metrics["low_confidence_rate"] = low_confidence_rate
    if approval_required_rate is not None:
        metrics["approval_required_rate"] = approval_required_rate
    if latency_p95_ms is not None:
        metrics["offline_decision_latency_p95_ms"] = latency_p95_ms

    return {
        "run_id": run_id,
        "metrics": metrics,
    }


def test_candidate_selection_policy_loads():
    policy = load_candidate_selection_policy()

    assert policy["policy_name"] == "decision_engine_candidate_selection"
    assert policy["policy_version"] == "v1"
    assert policy["baseline"]["required"] is True
    assert policy["baseline"]["minimum_f1_improvement_over_baseline"] == 0.02
    assert Path("config/candidate_selection.yaml").exists()


def test_evaluate_candidate_ready_when_all_checks_pass():
    policy = load_candidate_selection_policy()
    run = make_run(run_id="ready-run")

    result = evaluate_candidate_run(run, policy)

    assert result["decision"] == "candidate_ready"
    assert result["failed_checks"] == []
    assert result["missing_metrics"] == []


def test_evaluate_candidate_rejected_when_f1_below_threshold():
    policy = load_candidate_selection_policy()
    run = make_run(run_id="low-f1", f1_macro=0.70)

    result = evaluate_candidate_run(run, policy)

    assert result["decision"] == "candidate_rejected"
    assert "f1_macro" in result["failed_checks"]


def test_evaluate_candidate_requires_review_when_metric_missing():
    policy = load_candidate_selection_policy()
    run = make_run(run_id="missing-metric", low_confidence_rate=None)

    result = evaluate_candidate_run(run, policy)

    assert result["decision"] == "human_review_required"
    assert "low_confidence_rate" in result["missing_metrics"]


def test_compare_to_baseline_rejects_insufficient_improvement():
    policy = load_candidate_selection_policy()
    baseline = make_run(run_id="baseline", f1_macro=0.90)
    candidate = make_run(run_id="candidate", f1_macro=0.91)

    result = compare_to_baseline(
        candidate_run=candidate,
        baseline_run=baseline,
        policy=policy,
    )

    assert result["decision"] == "candidate_rejected"
    assert result["reason"] == "insufficient_improvement_over_baseline"
    assert result["actual_improvement"] == pytest.approx(0.01)


def test_compare_to_baseline_passes_when_improvement_is_enough():
    policy = load_candidate_selection_policy()
    baseline = make_run(run_id="baseline", f1_macro=0.90)
    candidate = make_run(run_id="candidate", f1_macro=0.93)

    result = compare_to_baseline(
        candidate_run=candidate,
        baseline_run=baseline,
        policy=policy,
    )

    assert result["decision"] == "candidate_ready"
    assert result["reason"] == "sufficient_improvement_over_baseline"
    assert result["actual_improvement"] == pytest.approx(0.03)
