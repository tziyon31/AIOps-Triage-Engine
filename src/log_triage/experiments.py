from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODEL_FAMILY_CONTROLLED_VARIABLES = [
    "raw_data",
    "train_split",
    "test_split",
    "feature_pipeline",
    "policy",
    "evaluation_code",
]


@dataclass(frozen=True)
class VariantDefinition:
    variant_name: str
    variant_type: str
    comparison_type: str
    changed_variable: str
    controlled_variables: list[str]
    model_family: str
    model_params: dict[str, Any]


@dataclass(frozen=True)
class FeaturePipelineIdentity:
    feature_pipeline_name: str
    vectorizer_name: str
    vectorizer_params: dict[str, Any]
    feature_pipeline_sha256: str


@dataclass(frozen=True)
class EvaluationCodeIdentity:
    evaluation_code_sha256: str
    source_files: list[dict[str, str]]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    comparison_type: str
    changed_variable: str
    controlled_variables: list[str]
    feature_pipeline: dict[str, Any]
    variants: list[VariantDefinition]
    experiment_config_path: str
    experiment_config_sha256: str


def stable_json_sha256(payload: dict[str, Any] | list[Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_EVALUATION_CODE_FILES = [
    "src/log_triage/evaluation.py",
    "src/log_triage/train.py",
    "src/log_triage/schemas.py",
    "src/log_triage/policy.py",
]


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_evaluation_code_identity(
    source_files: list[str] | None = None,
) -> EvaluationCodeIdentity:
    files = source_files or DEFAULT_EVALUATION_CODE_FILES
    file_identities = []

    for relative_path in files:
        absolute_path = PROJECT_ROOT / relative_path

        if not absolute_path.exists():
            raise FileNotFoundError(f"Evaluation code file not found: {relative_path}")

        file_identities.append(
            {
                "path": relative_path,
                "sha256": file_sha256(absolute_path),
            }
        )

    return EvaluationCodeIdentity(
        evaluation_code_sha256=stable_json_sha256(file_identities),
        source_files=file_identities,
    )


def evaluation_code_identity_to_dict(
    identity: EvaluationCodeIdentity,
) -> dict[str, Any]:
    return asdict(identity)


def build_feature_pipeline_identity(
    *,
    feature_pipeline_name: str,
    vectorizer_name: str,
    vectorizer_params: dict[str, Any],
    manual_feature_names: list[str],
) -> FeaturePipelineIdentity:
    payload = {
        "feature_pipeline_name": feature_pipeline_name,
        "vectorizer_name": vectorizer_name,
        "vectorizer_params": vectorizer_params,
        "manual_feature_names": manual_feature_names,
    }

    return FeaturePipelineIdentity(
        feature_pipeline_name=feature_pipeline_name,
        vectorizer_name=vectorizer_name,
        vectorizer_params=vectorizer_params,
        feature_pipeline_sha256=stable_json_sha256(payload),
    )


def build_model_family_variants(
    *,
    model_max_iter: int,
    random_state: int,
) -> list[VariantDefinition]:
    return [
        VariantDefinition(
            variant_name="manual_tfidf_logistic_regression",
            variant_type="model_family",
            comparison_type="model_family",
            changed_variable="model_family",
            controlled_variables=DEFAULT_MODEL_FAMILY_CONTROLLED_VARIABLES,
            model_family="LogisticRegression",
            model_params={
                "model_family": "LogisticRegression",
                "model_max_iter": model_max_iter,
                "random_state": random_state,
            },
        ),
        VariantDefinition(
            variant_name="manual_tfidf_sgd_log_loss",
            variant_type="model_family",
            comparison_type="model_family",
            changed_variable="model_family",
            controlled_variables=DEFAULT_MODEL_FAMILY_CONTROLLED_VARIABLES,
            model_family="SGDClassifier",
            model_params={
                "model_family": "SGDClassifier",
                "loss": "log_loss",
                "model_max_iter": model_max_iter,
                "random_state": random_state,
                "tol": 1e-3,
            },
        ),
    ]


def variant_to_dict(variant: VariantDefinition) -> dict[str, Any]:
    return asdict(variant)


def feature_pipeline_identity_to_dict(
    identity: FeaturePipelineIdentity,
) -> dict[str, Any]:
    return asdict(identity)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw_config, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {config_path}")

    required_fields = [
        "experiment_name",
        "comparison_type",
        "changed_variable",
        "controlled_variables",
        "feature_pipeline",
        "variants",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in raw_config
    ]

    if missing_fields:
        raise ValueError(
            f"Experiment config missing required fields: {missing_fields}"
        )

    variants = [
        VariantDefinition(
            variant_name=item["variant_name"],
            variant_type=item["variant_type"],
            comparison_type=raw_config["comparison_type"],
            changed_variable=raw_config["changed_variable"],
            controlled_variables=list(raw_config["controlled_variables"]),
            model_family=item["model_family"],
            model_params=dict(item.get("model_params", {})),
        )
        for item in raw_config["variants"]
    ]

    return ExperimentConfig(
        experiment_name=raw_config["experiment_name"],
        comparison_type=raw_config["comparison_type"],
        changed_variable=raw_config["changed_variable"],
        controlled_variables=list(raw_config["controlled_variables"]),
        feature_pipeline=dict(raw_config["feature_pipeline"]),
        variants=variants,
        experiment_config_path=str(config_path),
        experiment_config_sha256=file_sha256(config_path),
    )


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "comparison_type": config.comparison_type,
        "changed_variable": config.changed_variable,
        "controlled_variables": config.controlled_variables,
        "feature_pipeline": config.feature_pipeline,
        "variants": [variant_to_dict(variant) for variant in config.variants],
        "experiment_config_path": config.experiment_config_path,
        "experiment_config_sha256": config.experiment_config_sha256,
    }


def normalize_ngram_range(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)

    return value


def validate_experiment_config(
    *,
    experiment_config: ExperimentConfig,
    training_config: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    changed_variable = experiment_config.changed_variable
    feature_pipeline = experiment_config.feature_pipeline
    vectorizer_params = dict(feature_pipeline.get("vectorizer_params", {}))
    tfidf_cfg = training_config.get("tfidf", {})

    if (
        feature_pipeline.get("vectorizer_name") == "TfidfVectorizer"
        and changed_variable not in {
            "feature_pipeline",
            "vectorizer",
            "vectorizer_name",
            "vectorizer_params",
            "tfidf_max_features",
            "tfidf_ngram_range",
        }
    ):
        expected_max_features = tfidf_cfg.get("max_features")
        actual_max_features = vectorizer_params.get("max_features")

        if actual_max_features != expected_max_features:
            errors.append(
                "Experiment config feature_pipeline.vectorizer_params.max_features "
                f"({actual_max_features}) does not match training.yaml "
                f"tfidf.max_features ({expected_max_features})."
            )

        expected_ngram_range = normalize_ngram_range(
            tfidf_cfg.get("ngram_range")
        )
        actual_ngram_range = normalize_ngram_range(
            vectorizer_params.get("ngram_range")
        )

        if actual_ngram_range != expected_ngram_range:
            errors.append(
                "Experiment config feature_pipeline.vectorizer_params.ngram_range "
                f"({actual_ngram_range}) does not match training.yaml "
                f"tfidf.ngram_range ({expected_ngram_range})."
            )

    expected_random_seed = training_config.get("random_seed")

    for variant in experiment_config.variants:
        variant_random_state = variant.model_params.get("random_state")

        if (
            changed_variable not in {"random_state", "seed"}
            and variant_random_state != expected_random_seed
        ):
            errors.append(
                f"Variant {variant.variant_name} model_params.random_state "
                f"({variant_random_state}) does not match training.yaml "
                f"random_seed ({expected_random_seed})."
            )

    expected_model_max_iter = training_config.get("model", {}).get("max_iter")

    for variant in experiment_config.variants:
        variant_max_iter = variant.model_params.get("model_max_iter")

        if (
            changed_variable not in {"model_max_iter", "max_iter"}
            and variant_max_iter != expected_model_max_iter
        ):
            errors.append(
                f"Variant {variant.variant_name} model_params.model_max_iter "
                f"({variant_max_iter}) does not match training.yaml "
                f"model.max_iter ({expected_model_max_iter})."
            )

    return errors
