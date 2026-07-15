from scripts.build_experiment_history_report import (
    build_experiment_history_report,
    group_runs_by_comparison_group,
)


def make_run(
    *,
    run_id: str,
    group_id: str,
    variant_name: str,
    f1_macro: float,
    low_confidence_rate: float = 0.1,
    weakest_class_recall: float = 0.9,
    latency_p95_ms: float = 10.0,
) -> dict:
    return {
        "run_id": run_id,
        "run_name": variant_name,
        "status": "FINISHED",
        "start_time": "2026-01-01 00:00:00",
        "tags": {
            "variant_name": variant_name,
            "variant_type": "model_family",
            "comparison_type": "model_family",
            "changed_variable": "model_family",
            "controlled_variables": (
                "raw_data,train_split,test_split,"
                "feature_pipeline,policy,evaluation_code"
            ),
            "comparison_group_id": group_id,
            "experiment_name": "model_family_baseline",
            "experiment_config_path": "config/experiments/model_family.yaml",
            "experiment_config_sha256": "experiment-config-sha",
            "split_sha256": "split-sha",
            "train_split_sha256": "train-split-sha",
            "test_split_sha256": "test-split-sha",
            "feature_pipeline_name": "manual_features_plus_tfidf",
            "feature_pipeline_sha256": "feature-pipeline-sha",
            "vectorizer_name": "TfidfVectorizer",
            "model_family": variant_name,
            "policy_sha256": "policy-sha",
            "training_data_sha256": "training-data-sha",
            "evaluation_code_sha256": "evaluation-code-sha",
        },
        "params": {},
        "metrics": {
            "f1_macro": f1_macro,
            "combined_accuracy": f1_macro,
            "combined_weakest_class_recall": weakest_class_recall,
            "low_confidence_rate": low_confidence_rate,
            "approval_required_rate": 0.2,
            "offline_decision_latency_p95_ms": latency_p95_ms,
        },
    }


def test_group_runs_by_comparison_group():
    runs = [
        make_run(
            run_id="run-1",
            group_id="group-a",
            variant_name="lr",
            f1_macro=0.9,
        ),
        make_run(
            run_id="run-2",
            group_id="group-b",
            variant_name="sgd",
            f1_macro=0.8,
        ),
    ]

    grouped = group_runs_by_comparison_group(runs)

    assert sorted(grouped.keys()) == ["group-a", "group-b"]


def test_experiment_history_report_marks_valid_group_ready():
    runs = [
        make_run(
            run_id="run-1",
            group_id="group-a",
            variant_name="manual_tfidf_logistic_regression",
            f1_macro=0.9,
        ),
        make_run(
            run_id="run-2",
            group_id="group-a",
            variant_name="manual_tfidf_sgd_log_loss",
            f1_macro=0.8,
        ),
    ]

    report = build_experiment_history_report(
        experiment_name="log-triage-decision-engine",
        runs=runs,
    )

    assert report["total_comparison_groups"] == 1
    assert report["valid_comparison_groups"] == 1
    assert report["ready_for_candidate_selection_groups"] == 1

    group = report["comparison_groups"][0]

    assert group["status"] == "valid"
    assert group["ready_for_candidate_selection"] is True
    assert group["changed_variable"] == "model_family"
    assert group["blocked_reason"] is None


def test_experiment_history_report_marks_single_run_group_invalid():
    runs = [
        make_run(
            run_id="run-1",
            group_id="group-a",
            variant_name="manual_tfidf_logistic_regression",
            f1_macro=0.9,
        ),
    ]

    report = build_experiment_history_report(
        experiment_name="log-triage-decision-engine",
        runs=runs,
    )

    assert report["total_comparison_groups"] == 1
    assert report["valid_comparison_groups"] == 0
    assert report["ready_for_candidate_selection_groups"] == 0

    group = report["comparison_groups"][0]

    assert group["status"] == "invalid"
    assert group["ready_for_candidate_selection"] is False
    assert "at_least_two_runs" in group["blocked_reason"]


def test_experiment_history_recommends_candidate_selection_for_clean_valid_group():
    runs = [
        make_run(
            run_id="run-1",
            group_id="group-a",
            variant_name="manual_tfidf_logistic_regression",
            f1_macro=0.95,
            low_confidence_rate=0.05,
            weakest_class_recall=0.9,
            latency_p95_ms=10.0,
        ),
        make_run(
            run_id="run-2",
            group_id="group-a",
            variant_name="manual_tfidf_sgd_log_loss",
            f1_macro=0.80,
            low_confidence_rate=0.10,
            weakest_class_recall=0.85,
            latency_p95_ms=10.0,
        ),
    ]

    report = build_experiment_history_report(
        experiment_name="log-triage-decision-engine",
        runs=runs,
    )

    recommendation = report["comparison_groups"][0][
        "next_experiment_recommendation"
    ]

    assert recommendation["recommendation_type"] == "candidate_selection"
    assert (
        recommendation["recommended_next_experiment"]
        == "candidate_selection_policy"
    )

    assert (
        report["overall_next_step"]["recommended_next_experiment"]
        == "candidate_selection_policy"
    )
    assert "candidate_selection_command" in report["overall_next_step"]
    assert (
        "scripts/promote.py"
        in report["overall_next_step"]["candidate_selection_command"]
    )


def test_experiment_history_recommends_confidence_experiment_when_low_confidence_high():
    runs = [
        make_run(
            run_id="run-1",
            group_id="group-a",
            variant_name="manual_tfidf_logistic_regression",
            f1_macro=0.95,
            low_confidence_rate=0.35,
            weakest_class_recall=0.9,
        ),
        make_run(
            run_id="run-2",
            group_id="group-a",
            variant_name="manual_tfidf_sgd_log_loss",
            f1_macro=0.80,
            low_confidence_rate=0.10,
            weakest_class_recall=0.85,
        ),
    ]

    report = build_experiment_history_report(
        experiment_name="log-triage-decision-engine",
        runs=runs,
    )

    recommendation = report["comparison_groups"][0][
        "next_experiment_recommendation"
    ]

    assert recommendation["recommendation_type"] == (
        "confidence_or_feature_experiment"
    )
    assert recommendation["recommended_next_experiment"] == (
        "confidence_threshold_or_feature_pipeline"
    )


def test_experiment_history_recommends_rerun_for_invalid_group():
    runs = [
        make_run(
            run_id="run-1",
            group_id="group-a",
            variant_name="manual_tfidf_logistic_regression",
            f1_macro=0.9,
        ),
    ]

    report = build_experiment_history_report(
        experiment_name="log-triage-decision-engine",
        runs=runs,
    )

    recommendation = report["comparison_groups"][0][
        "next_experiment_recommendation"
    ]

    assert recommendation["recommendation_type"] == "rerun_invalid_comparison"
    assert recommendation["recommended_next_experiment"] == (
        "rerun_same_experiment_after_fixing_contract"
    )


def test_experiment_history_report_includes_candidate_selection_command():
    runs = [
        make_run(
            run_id="run-1",
            group_id="group-a",
            variant_name="manual_tfidf_logistic_regression",
            f1_macro=0.9,
        ),
        make_run(
            run_id="run-2",
            group_id="group-a",
            variant_name="manual_tfidf_sgd_log_loss",
            f1_macro=0.8,
        ),
    ]

    report = build_experiment_history_report(
        experiment_name="log-triage-decision-engine",
        runs=runs,
        candidate_policy_path="config/candidate_selection.yaml",
    )

    group = report["comparison_groups"][0]

    assert "candidate_selection_command" in group
    assert "scripts/promote.py" in group["candidate_selection_command"]
    assert "--comparison-group-id group-a" in group["candidate_selection_command"]
    assert "--baseline-run-id <BASELINE_RUN_ID>" in group[
        "candidate_selection_command"
    ]
    assert "--candidate-policy-path config/candidate_selection.yaml" in group[
        "candidate_selection_command"
    ]
    assert "--apply" not in group["candidate_selection_command"]

    assert "candidate_selection_command" in report["overall_next_step"]
    assert (
        "scripts/promote.py"
        in report["overall_next_step"]["candidate_selection_command"]
    )
