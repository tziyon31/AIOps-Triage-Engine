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
            "low_confidence_rate": 0.1,
            "approval_required_rate": 0.2,
            "offline_decision_latency_p95_ms": 10.0,
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
