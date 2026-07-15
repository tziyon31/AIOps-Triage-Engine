from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.build_tuning_plan_report import (
    build_tuning_plan_report,
    load_tuning_plan,
    summarize_parameter_experiment,
)


def make_run(
    *,
    run_id: str,
    variant_name: str,
    f1_macro: float,
    model_max_iter: str = "1000",
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
            "comparison_group_id": "group-a",
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
        "params": {
            "model_max_iter": model_max_iter,
        },
        "metrics": {
            "f1_macro": f1_macro,
            "combined_accuracy": f1_macro,
            "combined_weakest_class_recall": 0.85,
            "low_confidence_rate": 0.1,
            "approval_required_rate": 0.2,
            "offline_decision_latency_p95_ms": 10.0,
        },
    }


def make_plan(*, experiments: list[dict[str, Any]] | None = None) -> dict:
    return {
        "tuning_plan_name": "logistic_regression_tuning",
        "policy_version": "v1",
        "status": "planning_only_no_promotion",
        "description": "test plan",
        "winner_metric": "f1_macro",
        "require_valid_comparison_groups": True,
        "require_final_validation": True,
        "experiments": experiments
        or [
            {
                "label": "max_iter_tuning",
                "comparison_group_id": "group-a",
                "changed_parameter": "model_max_iter",
                "notes": "test",
            }
        ],
        "final_validation": {
            "required": True,
            "recommended_experiment_name": "final_candidate_tournament",
            "recommended_changed_variable": "candidate_config",
            "baseline_required": True,
        },
    }


def test_loads_tuning_plan_yaml():
    plan = load_tuning_plan("config/tuning_plans/logistic_regression_tuning.yaml")

    assert plan["tuning_plan_name"] == "logistic_regression_tuning"
    assert plan["winner_metric"] == "f1_macro"
    assert len(plan["experiments"]) == 3
    assert Path("config/tuning_plans/logistic_regression_tuning.yaml").exists()


def test_invalid_comparison_group_blocks_plan():
    summary = summarize_parameter_experiment(
        label="max_iter_tuning",
        comparison_group_id="group-a",
        changed_parameter="model_max_iter",
        notes="test",
        runs=[
            make_run(
                run_id="run-1",
                variant_name="variant-a",
                f1_macro=0.9,
            ),
        ],
        winner_metric="f1_macro",
        require_valid_comparison_groups=True,
    )

    assert summary["status"] == "blocked"
    assert summary["comparison_status"] != "valid"

    report = build_tuning_plan_report(
        plan=make_plan(),
        experiment_summaries=[summary],
    )

    assert report["status"] == "blocked"
    assert report["experiment_winners"] == []
    assert len(report["blocked_experiments"]) == 1


def test_valid_group_returns_one_winner():
    summary = summarize_parameter_experiment(
        label="max_iter_tuning",
        comparison_group_id="group-a",
        changed_parameter="model_max_iter",
        notes="test",
        runs=[
            make_run(
                run_id="run-1",
                variant_name="variant-low",
                f1_macro=0.85,
                model_max_iter="500",
            ),
            make_run(
                run_id="run-2",
                variant_name="variant-high",
                f1_macro=0.93,
                model_max_iter="2000",
            ),
        ],
        winner_metric="f1_macro",
        require_valid_comparison_groups=True,
    )

    assert summary["status"] == "valid"
    assert summary["winner_run_id"] == "run-2"
    assert summary["winner_metric_value"] == pytest.approx(0.93)


def test_report_status_is_requires_final_validation():
    summary = summarize_parameter_experiment(
        label="max_iter_tuning",
        comparison_group_id="group-a",
        changed_parameter="model_max_iter",
        notes="test",
        runs=[
            make_run(
                run_id="run-1",
                variant_name="variant-a",
                f1_macro=0.90,
            ),
            make_run(
                run_id="run-2",
                variant_name="variant-b",
                f1_macro=0.88,
            ),
        ],
        winner_metric="f1_macro",
        require_valid_comparison_groups=True,
    )

    report = build_tuning_plan_report(
        plan=make_plan(),
        experiment_summaries=[summary],
    )

    assert report["status"] == "requires_final_validation"


def test_combined_candidate_draft_promotable_is_false():
    summary = summarize_parameter_experiment(
        label="max_iter_tuning",
        comparison_group_id="group-a",
        changed_parameter="model_max_iter",
        notes="test",
        runs=[
            make_run(
                run_id="run-1",
                variant_name="variant-a",
                f1_macro=0.90,
            ),
            make_run(
                run_id="run-2",
                variant_name="variant-b",
                f1_macro=0.88,
            ),
        ],
        winner_metric="f1_macro",
        require_valid_comparison_groups=True,
    )

    report = build_tuning_plan_report(
        plan=make_plan(),
        experiment_summaries=[summary],
    )

    assert report["combined_candidate_draft"]["promotable"] is False


def test_recommended_next_step_is_run_final_candidate_tournament():
    summary = summarize_parameter_experiment(
        label="max_iter_tuning",
        comparison_group_id="group-a",
        changed_parameter="model_max_iter",
        notes="test",
        runs=[
            make_run(
                run_id="run-1",
                variant_name="variant-a",
                f1_macro=0.90,
            ),
            make_run(
                run_id="run-2",
                variant_name="variant-b",
                f1_macro=0.88,
            ),
        ],
        winner_metric="f1_macro",
        require_valid_comparison_groups=True,
    )

    report = build_tuning_plan_report(
        plan=make_plan(),
        experiment_summaries=[summary],
    )

    assert (
        report["recommended_next_step"]["type"]
        == "run_final_candidate_tournament"
    )
    assert (
        report["recommended_next_step"]["recommended_experiment_name"]
        == "final_candidate_tournament"
    )


def test_template_tuning_plan_yaml_loads():
    plan = load_tuning_plan("config/tuning_plans/logistic_regression_tuning.yaml")

    assert yaml.safe_load(
        Path("config/tuning_plans/logistic_regression_tuning.yaml").read_text(
            encoding="utf-8"
        )
    )["tuning_plan_name"] == plan["tuning_plan_name"]
