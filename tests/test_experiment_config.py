from pathlib import Path

from src.log_triage.config import load_training_config
from src.log_triage.experiments import (
    load_experiment_config,
    validate_experiment_config,
)


def test_load_model_family_experiment_config():
    config = load_experiment_config("config/experiments/model_family.yaml")

    assert config.experiment_name == "model_family_baseline"
    assert config.comparison_type == "model_family"
    assert config.changed_variable == "model_family"
    assert config.experiment_config_sha256
    assert len(config.variants) == 2

    variant_names = [variant.variant_name for variant in config.variants]

    assert variant_names == [
        "manual_tfidf_logistic_regression",
        "manual_tfidf_sgd_log_loss",
    ]


def test_experiment_config_controls_feature_pipeline_for_model_family():
    config = load_experiment_config("config/experiments/model_family.yaml")

    assert "feature_pipeline" in config.controlled_variables

    for variant in config.variants:
        assert variant.comparison_type == "model_family"
        assert variant.changed_variable == "model_family"
        assert variant.controlled_variables == config.controlled_variables


def test_model_family_experiment_config_matches_training_config():
    config = load_experiment_config("config/experiments/model_family.yaml")
    training_config = load_training_config()

    errors = validate_experiment_config(
        experiment_config=config,
        training_config=training_config,
    )

    assert errors == []


def test_experiment_config_validation_fails_on_random_seed_mismatch():
    config = load_experiment_config("config/experiments/model_family.yaml")
    training_config = load_training_config()
    training_config = dict(training_config)
    training_config["random_seed"] = 999

    errors = validate_experiment_config(
        experiment_config=config,
        training_config=training_config,
    )

    assert errors
    assert "random_state" in errors[0]


def test_config_driven_experiments_doc_exists():
    assert Path(
        "docs/stage5/module_5_6_config_driven_experiments.md"
    ).exists()
