from scripts.compare_runs import (
    build_candidate_selection_input,
    build_comparison_contract,
    validate_controlled_variables,
)


def build_run(
    *,
    run_id: str,
    variant_name: str,
    comparison_group_id: str = "group-1",
    comparison_type: str = "model_family",
    changed_variable: str = "model_family",
    training_data_sha256: str = "data-sha",
    train_split_sha256: str = "train-sha",
    test_split_sha256: str = "test-sha",
    feature_pipeline_sha256: str = "feature-sha",
    policy_sha256: str = "policy-sha",
    evaluation_code_sha256: str = "eval-sha",
):
    return {
        "run_id": run_id,
        "run_name": variant_name,
        "tags": {
            "variant_name": variant_name,
            "variant_type": "model_family",
            "comparison_type": comparison_type,
            "changed_variable": changed_variable,
            "controlled_variables": (
                "raw_data,train_split,test_split,"
                "feature_pipeline,policy,evaluation_code"
            ),
            "comparison_group_id": comparison_group_id,
            "training_data_sha256": training_data_sha256,
            "train_split_sha256": train_split_sha256,
            "test_split_sha256": test_split_sha256,
            "feature_pipeline_sha256": feature_pipeline_sha256,
            "policy_sha256": policy_sha256,
            "evaluation_code_sha256": evaluation_code_sha256,
            "model_family": "LogisticRegression",
            "feature_pipeline_name": "manual_features_plus_tfidf",
        },
        "metrics": {
            "f1_macro": 1.0,
            "combined_accuracy": 1.0,
            "offline_decision_latency_p95_ms": 1.5,
        },
    }


def test_controlled_variable_validation_passes_when_hashes_match():
    runs = [
        build_run(run_id="run-1", variant_name="lr"),
        build_run(run_id="run-2", variant_name="sgd"),
    ]

    result = validate_controlled_variables(runs)

    assert result["status"] == "passed"


def test_controlled_variable_validation_fails_when_hashes_differ():
    runs = [
        build_run(run_id="run-1", variant_name="lr"),
        build_run(
            run_id="run-2",
            variant_name="sgd",
            feature_pipeline_sha256="different-feature-sha",
        ),
    ]

    result = validate_controlled_variables(runs)

    assert result["status"] == "failed"


def test_comparison_contract_valid_when_group_is_consistent():
    runs = [
        build_run(run_id="run-1", variant_name="lr"),
        build_run(run_id="run-2", variant_name="sgd"),
    ]

    validation = validate_controlled_variables(runs)
    contract = build_comparison_contract(
        runs=runs,
        controlled_variable_validation=validation,
    )

    assert contract["status"] == "valid"
    assert contract["comparison_type"] == "model_family"
    assert contract["changed_variable"] == "model_family"
    assert contract["checks"]["at_least_two_runs"] is True


def test_candidate_selection_input_is_ready_only_for_valid_contract():
    runs = [
        build_run(run_id="run-1", variant_name="lr"),
        build_run(run_id="run-2", variant_name="sgd"),
    ]

    validation = validate_controlled_variables(runs)
    contract = build_comparison_contract(
        runs=runs,
        controlled_variable_validation=validation,
    )

    candidate_input = build_candidate_selection_input(
        runs=runs,
        comparison_contract=contract,
    )

    assert candidate_input["ready_for_candidate_selection"] is True
    assert len(candidate_input["candidate_pool"]) == 2
