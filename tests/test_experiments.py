from src.log_triage.experiments import (
    build_feature_pipeline_identity,
    build_model_family_variants,
    stable_json_sha256,
)


def test_stable_json_sha256_is_order_independent():
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}

    assert stable_json_sha256(first) == stable_json_sha256(second)


def test_model_family_variants_define_experiment_contract():
    variants = build_model_family_variants(
        model_max_iter=1000,
        random_state=42,
    )

    assert [variant.variant_name for variant in variants] == [
        "manual_tfidf_logistic_regression",
        "manual_tfidf_sgd_log_loss",
    ]

    for variant in variants:
        assert variant.variant_type == "model_family"
        assert variant.comparison_type == "model_family"
        assert variant.changed_variable == "model_family"
        assert "feature_pipeline" in variant.controlled_variables
        assert "evaluation_code" in variant.controlled_variables
        assert variant.model_family in {"LogisticRegression", "SGDClassifier"}


def test_feature_pipeline_identity_changes_when_vectorizer_params_change():
    first = build_feature_pipeline_identity(
        feature_pipeline_name="manual_features_plus_tfidf",
        vectorizer_name="TfidfVectorizer",
        vectorizer_params={"max_features": 1000},
        manual_feature_names=["error_count", "warning_count"],
    )

    second = build_feature_pipeline_identity(
        feature_pipeline_name="manual_features_plus_tfidf",
        vectorizer_name="TfidfVectorizer",
        vectorizer_params={"max_features": 2000},
        manual_feature_names=["error_count", "warning_count"],
    )

    assert first.feature_pipeline_sha256 != second.feature_pipeline_sha256
