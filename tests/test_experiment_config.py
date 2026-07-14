from src.log_triage.experiments import load_experiment_config


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
