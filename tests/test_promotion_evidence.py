from src.log_triage.promotion_evidence import (
    build_default_promotion_report_text,
    load_promotion_evidence_contract,
    validate_promotion_evidence,
)


def make_complete_run() -> dict:
    return {
        "run_id": "run-1",
        "params": {
            "model_family": "LogisticRegression",
            "model_max_iter": "1000",
            "feature_pipeline_name": "manual_features_plus_tfidf",
            "feature_pipeline_sha256": "abc",
            "variant_name": "manual_tfidf_logistic_regression",
            "experiment_config_sha256": "def",
        },
        "metrics": {
            "f1_macro": 0.90,
            "combined_accuracy": 0.90,
            "combined_weakest_class_recall": 0.80,
            "low_confidence_rate": 0.10,
            "approval_required_rate": 0.20,
            "offline_decision_latency_p95_ms": 20.0,
        },
        "tags": {
            "candidate_status": "not_evaluated",
            "promotion_reason": "not_evaluated",
            "run_owner": "tziyon31",
            "comparison_group_id": "group-a",
            "variant_name": "manual_tfidf_logistic_regression",
            "comparison_type": "model_family",
            "changed_variable": "model_family",
            "training_data_sha256": "data",
            "train_split_sha256": "train",
            "test_split_sha256": "test",
            "policy_sha256": "policy",
            "evaluation_code_sha256": "eval",
            "experiment_config_sha256": "config",
        },
    }


def test_load_promotion_evidence_contract():
    contract = load_promotion_evidence_contract()

    assert contract["contract_name"] == "promotion_evidence_contract"
    assert contract["contract_version"] == "v1"
    assert "required_metrics" in contract


def test_validate_promotion_evidence_passes_when_complete():
    contract = load_promotion_evidence_contract()

    result = validate_promotion_evidence(
        run=make_complete_run(),
        artifact_paths=["manifest.json", "promotion_report.md"],
        contract=contract,
    )

    assert result["status"] == "passed"
    assert result["missing_params"] == []
    assert result["missing_metrics"] == []
    assert result["missing_tags"] == []
    assert result["missing_artifacts"] == []


def test_validate_promotion_evidence_fails_when_metric_missing():
    contract = load_promotion_evidence_contract()
    run = make_complete_run()
    del run["metrics"]["f1_macro"]

    result = validate_promotion_evidence(
        run=run,
        artifact_paths=["manifest.json", "promotion_report.md"],
        contract=contract,
    )

    assert result["status"] == "failed"
    assert result["missing_metrics"] == ["f1_macro"]


def test_validate_promotion_evidence_fails_when_artifact_missing():
    contract = load_promotion_evidence_contract()

    result = validate_promotion_evidence(
        run=make_complete_run(),
        artifact_paths=["manifest.json"],
        contract=contract,
    )

    assert result["status"] == "failed"
    assert result["missing_artifacts"] == ["promotion_report.md"]


def test_build_default_promotion_report_text_contains_status_and_owner():
    contract = load_promotion_evidence_contract()

    text = build_default_promotion_report_text(
        run_id="run-1",
        variant_name="manual_tfidf_logistic_regression",
        candidate_status="not_evaluated",
        promotion_reason="not_evaluated",
        run_owner="tziyon31",
        contract=contract,
    )

    assert "candidate_status" not in text
    assert "Candidate status" in text
    assert "not_evaluated" in text
    assert "tziyon31" in text
    assert "promotion_evidence_contract" in text
