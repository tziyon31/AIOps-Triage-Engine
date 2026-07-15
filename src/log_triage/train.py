from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import warnings
from contextlib import nullcontext
from time import perf_counter
from typing import cast
from uuid import uuid4

import joblib  # type: ignore[import-not-found]

from scipy.sparse import csr_matrix, hstack  # type: ignore[import-not-found]
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-not-found]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.linear_model import (  # type: ignore[import-not-found]
    LogisticRegression,
    SGDClassifier,
)
from sklearn.metrics import (  # type: ignore[import-not-found]
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]

from src.log_triage.artifact_version import (
    POLICY_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    build_artifact_file_hashes,
    build_content_hashes,
    build_versioned_artifact_dir_name,
    find_latest_artifact_for_version,
    load_yaml_snapshot,
    parse_versioned_artifact_dir,
    resolve_patch,
)
from src.log_triage.config import (
    ACTION_RISK,
    ALLOWED_ACTIONS,
    ARTIFACT_TYPE,
    FORBIDDEN_ACTIONS,
    MIN_CONFIDENCE,
    REQUIRES_APPROVAL,
    SCHEMA_VERSION,
    SIMILARITY_THRESHOLD,
    load_training_config,
)
from src.log_triage.data import load_raw_logs
from src.log_triage.evaluation import (
    build_classification_evaluation,
    build_confusion_matrix_markdown,
    build_confusion_matrix_text,
    build_decision_quality_evaluation,
    build_offline_latency_evaluation,
    flatten_decision_quality_metrics,
    flatten_offline_latency_metrics,
    flatten_per_class_metrics,
)
from src.log_triage.experiments import (
    ExperimentConfig,
    VariantDefinition,
    build_evaluation_code_identity,
    build_feature_pipeline_identity,
    evaluation_code_identity_to_dict,
    experiment_config_to_dict,
    feature_pipeline_identity_to_dict,
    load_experiment_config,
    validate_experiment_config,
    variant_to_dict,
)
from src.log_triage.features import MANUAL_FEATURE_NAMES
from src.log_triage.promotion_evidence import (
    build_default_promotion_report_text,
    load_promotion_evidence_contract,
)
from src.log_triage.pipeline import (
    FEATURE_NAMES,
    KNOWN_ACTIONS,
    extract_features,
    extract_label,
    parse_log_line,
)
from src.log_triage.policy import validate as validate_policy
from src.log_triage.schemas import build_decision

TRAINING_CONFIG = load_training_config()
TEST_SIZE = TRAINING_CONFIG["test_size"]
RANDOM_STATE = TRAINING_CONFIG["random_seed"]
MODEL_MAX_ITER = TRAINING_CONFIG["model"]["max_iter"]
DEFAULT_EXPERIMENT_CONFIG_PATH = "config/experiments/model_family.yaml"

ACTION_DISPLAY_NAME = {
    "open_ticket": "open_ticket",
    "ignore": "ignore",
    "scale_up": "suggest_scale_up",
}


def decode_label(encoded_label):
    return KNOWN_ACTIONS[int(encoded_label)]


def build_decision_contract() -> dict:
    return {
        "output_schema_version": SCHEMA_VERSION,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "forbidden_actions": sorted(FORBIDDEN_ACTIONS),
        "confidence_thresholds": {
            "min_for_suggestion": MIN_CONFIDENCE,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
        "action_risk": ACTION_RISK,
        "requires_approval": REQUIRES_APPROVAL,
    }


def build_training_config() -> dict:
    return {
        "source": "config/training.yaml",
        "config": dict(TRAINING_CONFIG),
        "derived": {
            "text_representation": "manual_features_plus_tfidf",
            "model_type": TRAINING_CONFIG["model"]["type"],
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "model_max_iter": MODEL_MAX_ITER,
        },
    }


def load_experiment_config_for_run() -> ExperimentConfig:
    config_path = os.getenv(
        "LOG_TRIAGE_EXPERIMENT_CONFIG",
        DEFAULT_EXPERIMENT_CONFIG_PATH,
    )

    return load_experiment_config(config_path)


def select_variants_for_run(
    experiment_config: ExperimentConfig,
) -> list[VariantDefinition]:
    variants = experiment_config.variants

    if os.getenv("LOG_TRIAGE_COMPARE_VARIANTS") == "1":
        return variants

    requested_variant = os.getenv(
        "LOG_TRIAGE_VARIANT",
        variants[0].variant_name,
    )

    selected = [
        variant
        for variant in variants
        if variant.variant_name == requested_variant
    ]

    if not selected:
        valid_names = ", ".join(variant.variant_name for variant in variants)
        raise ValueError(
            f"Unknown LOG_TRIAGE_VARIANT={requested_variant}. "
            f"Valid variants from {experiment_config.experiment_config_path}: "
            f"{valid_names}"
        )

    return selected


def build_vectorizer_kwargs_from_experiment_config(
    experiment_config: ExperimentConfig,
) -> dict:
    feature_pipeline_cfg = experiment_config.feature_pipeline
    vectorizer_params = dict(feature_pipeline_cfg.get("vectorizer_params", {}))

    if isinstance(vectorizer_params.get("ngram_range"), list):
        vectorizer_params["ngram_range"] = tuple(
            vectorizer_params["ngram_range"]
        )

    return {
        key: value
        for key, value in vectorizer_params.items()
        if value is not None
    }


def build_comparison_group_id(
    experiment_config: ExperimentConfig,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_config_sha = experiment_config.experiment_config_sha256[:8]
    short_random = uuid4().hex[:8]

    return (
        f"{experiment_config.experiment_name}-"
        f"{short_config_sha}-"
        f"{timestamp}-"
        f"{short_random}"
    )


def build_model_for_variant(variant: VariantDefinition):
    params = variant.model_params

    if variant.model_family == "LogisticRegression":
        return LogisticRegression(
            max_iter=int(params["model_max_iter"]),
            random_state=int(params["random_state"]),
        )

    if variant.model_family == "SGDClassifier":
        return SGDClassifier(
            loss=str(params["loss"]),
            max_iter=int(params["model_max_iter"]),
            random_state=int(params["random_state"]),
            tol=float(params["tol"]),
        )

    raise ValueError(f"Unsupported model_family={variant.model_family}")


def build_decision_artifact(
    model,
    vectorizer,
    manual_feature_names: list[str],
    known_actions: list[str],
    primary_model_accuracy: float,
    test_size: int,
    metrics: dict,
    training_warnings: dict | None = None,
    evaluation: dict | None = None,
    variant: dict | None = None,
    feature_pipeline: dict | None = None,
    evaluation_code: dict | None = None,
    experiment_config: dict | None = None,
) -> dict:
    variant_info = variant or {
        "variant_name": "manual_tfidf_logistic_regression",
        "variant_type": "model_family",
        "comparison_type": "model_family",
        "changed_variable": "model_family",
        "controlled_variables": [
            "raw_data",
            "train_split",
            "test_split",
            "feature_pipeline",
            "policy",
            "evaluation_code",
        ],
        "model_family": TRAINING_CONFIG["model"]["type"],
        "model_params": {},
    }

    feature_pipeline_info = feature_pipeline or {
        "feature_pipeline_name": "manual_features_plus_tfidf",
        "vectorizer_name": "TfidfVectorizer",
        "vectorizer_params": {},
        "feature_pipeline_sha256": "unknown",
    }

    evaluation_code_info = evaluation_code or {
        "evaluation_code_sha256": "unknown",
        "source_files": [],
    }

    experiment_config_info = experiment_config or {
        "experiment_name": "unknown",
        "experiment_config_path": "unknown",
        "experiment_config_sha256": "unknown",
    }

    return {
        "run_id": str(uuid4()),
        "model": model,
        "vectorizer": vectorizer,
        "manual_feature_names": manual_feature_names,
        "known_actions": known_actions,
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": variant_info.get(
            "model_family",
            TRAINING_CONFIG["model"]["type"],
        ),
        "text_representation": feature_pipeline_info.get(
            "feature_pipeline_name",
            "manual_features_plus_tfidf",
        ),
        "decision_contract": build_decision_contract(),
        "metrics": {
            # `accuracy` is the primary quality metric for the saved artifact.
            "accuracy": round(float(primary_model_accuracy), 4),
            "test_size": int(test_size),
            **metrics,
        },
        "training_config": build_training_config(),
        "training_warnings": training_warnings or {},
        "evaluation": evaluation or {},
        "variant": variant_info,
        "feature_pipeline": feature_pipeline_info,
        "evaluation_code": evaluation_code_info,
        "experiment_config": experiment_config_info,
    }


def create_artifact_dir(artifact: dict) -> Path:
    artifact_cfg = TRAINING_CONFIG["artifact"]
    name = artifact_cfg["name"]
    major = artifact_cfg["major"]
    minor = artifact_cfg["minor"]
    output_dir = artifact_cfg["output_dir"]

    current_hashes = build_content_hashes(artifact)
    last_patch, last_hashes = find_latest_artifact_for_version(
        output_dir=output_dir,
        name=name,
        major=major,
        minor=minor,
    )
    patch = resolve_patch(current_hashes, last_patch, last_hashes)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dir_name = build_versioned_artifact_dir_name(
        name=name,
        major=major,
        minor=minor,
        patch=patch,
        timestamp=timestamp,
    )

    artifact_dir = Path(output_dir) / dir_name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def is_mlflow_enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI")) and (
        os.getenv("LOG_TRIAGE_DISABLE_MLFLOW") != "1"
    )


def mlflow_run_context(run_name: str = "train-decision-engine"):
    if not is_mlflow_enabled():
        return nullcontext(None)

    import mlflow  # type: ignore[import-not-found]

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "log-triage-decision-engine",
    )

    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def build_mlflow_params(
    *,
    variant: dict | None = None,
    feature_pipeline: dict | None = None,
) -> dict:
    tfidf_cfg = TRAINING_CONFIG.get("tfidf", {})

    params = {
        "model_type": TRAINING_CONFIG["model"]["type"],
        "model_max_iter": MODEL_MAX_ITER,
        "random_seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "tfidf_max_features": tfidf_cfg.get("max_features"),
        "tfidf_ngram_range": str(tfidf_cfg.get("ngram_range")),
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
    }

    if variant:
        for key, value in variant.get("model_params", {}).items():
            if value is not None:
                params[f"variant_{key}"] = value

    if feature_pipeline:
        params["feature_pipeline_name"] = feature_pipeline.get(
            "feature_pipeline_name"
        )
        params["vectorizer_name"] = feature_pipeline.get("vectorizer_name")

        for key, value in feature_pipeline.get("vectorizer_params", {}).items():
            if value is not None:
                params[f"vectorizer_{key}"] = str(value)

    return params


def compute_artifact_sha256_from_manifest(manifest: dict) -> str:
    """
    Build a stable artifact identity hash from the manifest content.

    This is not a replacement for the per-file hashes.
    It is a compact fingerprint for the artifact package identity.
    """
    artifact_identity = {
        "schema_version": manifest.get("schema_version"),
        "artifact_type": manifest.get("artifact_type"),
        "files": manifest.get("files", {}),
        "hashes": manifest.get("hashes", {}),
        "decision_contract": manifest.get("decision_contract", {}),
    }

    encoded = json.dumps(
        artifact_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def log_mlflow_outputs(artifact: dict, artifact_dir: Path) -> None:
    if not is_mlflow_enabled():
        print("\nMLflow logging skipped: MLFLOW_TRACKING_URI is not set.")
        return

    import mlflow  # type: ignore[import-not-found]

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    hashes = manifest["hashes"]

    variant = artifact.get("variant", {})
    feature_pipeline = artifact.get("feature_pipeline", {})
    evaluation_code = artifact.get("evaluation_code", {})
    experiment_config = artifact.get("experiment_config", {})

    for key, value in build_mlflow_params(
        variant=variant,
        feature_pipeline=feature_pipeline,
    ).items():
        if value is not None:
            mlflow.log_param(key, value)

    evidence_params = {
        "model_family": variant.get("model_family"),
        "variant_name": variant.get("variant_name"),
        "feature_pipeline_sha256": feature_pipeline.get("feature_pipeline_sha256"),
        "experiment_config_sha256": experiment_config.get(
            "experiment_config_sha256"
        ),
    }
    for key, value in evidence_params.items():
        if value is not None:
            mlflow.log_param(key, value)

    for key, value in artifact["metrics"].items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, float(value))

    run_owner = os.getenv("LOG_TRIAGE_RUN_OWNER", "tziyon31")

    mlflow.set_tags(
        {
            "stage": "5",
            "component": "decision-engine",
            "git_sha": manifest["git_sha"],
            "training_config_sha256": hashes["training_config_sha256"],
            "policy_sha256": hashes["policy_sha256"],
            "config_sha256": hashes["config_sha256"],
            "training_data_sha256": hashes["training_data_sha256"],
            "artifact_id": manifest["artifact_id"],
            "artifact_run_id": manifest["run_id"],
            "artifact_version": manifest.get(
                "model_version",
                manifest.get("version", "unknown"),
            ),
            "artifact_sha256": compute_artifact_sha256_from_manifest(manifest),
            "model_sha256": hashes["model_sha256"],
            "vectorizer_sha256": hashes["vectorizer_sha256"],
            "known_actions_sha256": hashes["known_actions_sha256"],
            "candidate_status": "not_evaluated",
            "promotion_reason": "not_evaluated",
            "run_owner": run_owner,
            "promotion_evidence_contract_version": "v1",
            "quality_gate_status": "not_checked_in_train",
        }
    )

    if variant:
        mlflow.set_tags(
            {
                "variant_name": variant.get("variant_name", "unknown"),
                "variant_type": variant.get("variant_type", "unknown"),
                "comparison_type": variant.get("comparison_type", "unknown"),
                "changed_variable": variant.get("changed_variable", "unknown"),
                "controlled_variables": ",".join(
                    variant.get("controlled_variables", [])
                ),
                "model_family": variant.get("model_family", "unknown"),
            }
        )

    if feature_pipeline:
        mlflow.set_tags(
            {
                "feature_pipeline_name": feature_pipeline.get(
                    "feature_pipeline_name",
                    "unknown",
                ),
                "vectorizer_name": feature_pipeline.get(
                    "vectorizer_name",
                    "unknown",
                ),
                "feature_pipeline_sha256": feature_pipeline.get(
                    "feature_pipeline_sha256",
                    "unknown",
                ),
            }
        )

    if evaluation_code:
        mlflow.set_tags(
            {
                "evaluation_code_sha256": evaluation_code.get(
                    "evaluation_code_sha256",
                    "unknown",
                ),
                "evaluation_code_files": ",".join(
                    file_info["path"]
                    for file_info in evaluation_code.get("source_files", [])
                ),
            }
        )

    if experiment_config:
        mlflow.set_tags(
            {
                "experiment_name": experiment_config.get(
                    "experiment_name",
                    "unknown",
                ),
                "experiment_config_path": experiment_config.get(
                    "experiment_config_path",
                    "unknown",
                ),
                "experiment_config_sha256": experiment_config.get(
                    "experiment_config_sha256",
                    "unknown",
                ),
            }
        )

    split_evidence = artifact.get("evaluation", {}).get("split", {})

    if split_evidence:
        comparison_group_id = split_evidence.get(
            "comparison_group_id",
            split_evidence["split_sha256"],
        )
        mlflow.set_tags(
            {
                "train_split_sha256": split_evidence["train_split_sha256"],
                "test_split_sha256": split_evidence["test_split_sha256"],
                "split_sha256": split_evidence["split_sha256"],
                "comparison_group_id": comparison_group_id,
                "fair_comparison_group_id": comparison_group_id,
            }
        )

    training_warning_count = int(
        artifact["metrics"].get("training_warning_count", 0)
    )
    convergence_warning_count = int(
        artifact["metrics"].get("convergence_warning_count", 0)
    )

    mlflow.set_tags(
        {
            "has_training_warnings": str(training_warning_count > 0).lower(),
            "has_convergence_warning": str(convergence_warning_count > 0).lower(),
        }
    )

    combined_evaluation = artifact.get("evaluation", {}).get("combined", {})
    if combined_evaluation:
        mlflow.set_tag(
            "weakest_class_by_recall",
            combined_evaluation["weakest_class_by_recall"],
        )

    offline_latency = artifact.get("evaluation", {}).get("offline_latency", {})

    if offline_latency:
        mlflow.set_tags(
            {
                "latency_benchmark_scope": offline_latency["scope"],
                "latency_benchmark_includes": ",".join(offline_latency["includes"]),
                "latency_benchmark_excludes": ",".join(offline_latency["excludes"]),
            }
        )

    mlflow.log_dict(
        artifact.get("training_warnings", {}),
        "diagnostics/training_warnings.json",
    )

    mlflow.log_artifacts(str(artifact_dir), artifact_path="decision_artifact")

    active_run = mlflow.active_run()
    if active_run is not None:
        promotion_contract = load_promotion_evidence_contract()
        promotion_report_text = build_default_promotion_report_text(
            run_id=active_run.info.run_id,
            variant_name=variant.get("variant_name", "unknown"),
            candidate_status="not_evaluated",
            promotion_reason="not_evaluated",
            run_owner=run_owner,
            contract=promotion_contract,
        )
        mlflow.log_text(
            promotion_report_text,
            artifact_file="promotion_report.md",
        )

        print(f"\nMLflow run logged: {active_run.info.run_id}")
        print(f"MLflow artifact path: decision_artifact/{artifact_dir.name}")


def build_manifest(
    artifact_dir,
    model_path,
    vectorizer_path,
    known_actions_path,
    artifact: dict,
) -> dict:
    artifact_id = artifact_dir.name
    policy_config_snapshot = load_yaml_snapshot(POLICY_CONFIG_PATH)

    parsed = parse_versioned_artifact_dir(artifact_id)
    version = None
    if parsed is not None:
        version = {
            "name": parsed["name"],
            "major": parsed["major"],
            "minor": parsed["minor"],
            "patch": parsed["patch"],
            "label": f"v{parsed['major']}.{parsed['minor']}.{parsed['patch']}",
            "timestamp": parsed["timestamp"],
        }

    manifest = {
        "artifact_id": artifact_id,
        "run_id": artifact["run_id"],
        "created_at": artifact["created_at"],
        "git_sha": get_git_sha(),
        "schema_version": artifact["schema_version"],
        "artifact_type": artifact["artifact_type"],
        "files": {
            "model": "model.pkl",
            "vectorizer": "vectorizer.pkl",
            "known_actions": "known_actions.json",
            "training_config": "training.yaml",
            "policy": "policy.yaml",
            "combined_evaluation": "evaluation/combined_evaluation.json",
            "confusion_matrix": "evaluation/confusion_matrix.json",
            "confusion_matrix_markdown": "evaluation/confusion_matrix.md",
            "confusion_matrix_text": "evaluation/confusion_matrix.txt",
            "decision_quality": "evaluation/decision_quality.json",
            "offline_latency": "evaluation/offline_latency.json",
            "split_evidence": "evaluation/split_evidence.json",
        },
        "hashes": build_artifact_file_hashes(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            known_actions_path=known_actions_path,
            artifact=artifact,
        ),
        "training_config": artifact["training_config"],
        "policy_config": {
            "source": "config/policy.yaml",
            "config": policy_config_snapshot,
        },
        "decision_contract": artifact["decision_contract"],
        "metrics": artifact["metrics"],
        "variant": artifact.get("variant", {}),
        "feature_pipeline": artifact.get("feature_pipeline", {}),
        "evaluation_code": artifact.get("evaluation_code", {}),
        "experiment_config": artifact.get("experiment_config", {}),
    }

    if version is not None:
        manifest["version"] = version
        manifest["model_version"] = version["label"]

    return manifest


def save_decision_artifact(artifact: dict) -> Path:
    artifact_dir = create_artifact_dir(artifact)

    model_path = artifact_dir / "model.pkl"
    vectorizer_path = artifact_dir / "vectorizer.pkl"
    known_actions_path = artifact_dir / "known_actions.json"
    evaluation_dir = artifact_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    combined_evaluation_path = evaluation_dir / "combined_evaluation.json"
    confusion_matrix_path = evaluation_dir / "confusion_matrix.json"
    confusion_matrix_markdown_path = evaluation_dir / "confusion_matrix.md"
    confusion_matrix_text_path = evaluation_dir / "confusion_matrix.txt"
    decision_quality_path = evaluation_dir / "decision_quality.json"
    offline_latency_path = evaluation_dir / "offline_latency.json"
    split_evidence_path = evaluation_dir / "split_evidence.json"

    joblib.dump(artifact["model"], model_path)
    joblib.dump(artifact["vectorizer"], vectorizer_path)

    with open(known_actions_path, "w", encoding="utf-8") as file:
        json.dump(artifact["known_actions"], file, indent=2)

    combined_evaluation = artifact.get("evaluation", {}).get("combined", {})

    with open(combined_evaluation_path, "w", encoding="utf-8") as file:
        json.dump(combined_evaluation, file, indent=2)

    with open(confusion_matrix_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "labels": combined_evaluation.get("labels", []),
                "label_ids": combined_evaluation.get("label_ids", []),
                "confusion_matrix": combined_evaluation.get("confusion_matrix", []),
            },
            file,
            indent=2,
        )

    with open(confusion_matrix_markdown_path, "w", encoding="utf-8") as file:
        file.write(build_confusion_matrix_markdown(combined_evaluation))

    with open(confusion_matrix_text_path, "w", encoding="utf-8") as file:
        file.write(build_confusion_matrix_text(combined_evaluation))

    with open(decision_quality_path, "w", encoding="utf-8") as file:
        json.dump(
            artifact.get("evaluation", {}).get("decision_quality", {}),
            file,
            indent=2,
        )

    with open(offline_latency_path, "w", encoding="utf-8") as file:
        json.dump(
            artifact.get("evaluation", {}).get("offline_latency", {}),
            file,
            indent=2,
        )

    with open(split_evidence_path, "w", encoding="utf-8") as file:
        json.dump(
            artifact.get("evaluation", {}).get("split", {}),
            file,
            indent=2,
        )

    shutil.copy2(TRAINING_CONFIG_PATH, artifact_dir / "training.yaml")
    shutil.copy2(POLICY_CONFIG_PATH, artifact_dir / "policy.yaml")

    manifest = build_manifest(
        artifact_dir=artifact_dir,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        known_actions_path=known_actions_path,
        artifact=artifact,
    )

    manifest_path = artifact_dir / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"\nArtifact saved to {artifact_dir}")
    print(f"Model: {model_path}")
    print(f"Vectorizer: {vectorizer_path}")
    print(f"Known actions: {known_actions_path}")
    print(f"Manifest: {manifest_path}")

    return artifact_dir


def build_rows(raw_logs):
    """Parse raw log lines once and keep text, manual features, and label aligned."""
    rows = []
    skipped = []

    for line in raw_logs:
        try:
            event = parse_log_line(line)
            manual_features = extract_features(event)
            label = extract_label(event)

            rows.append(
                {
                    "raw": line,
                    "message": event["message"],
                    "manual_features": manual_features,
                    "label": label,
                }
            )

        except ValueError as error:
            skipped.append({"line": line, "error": str(error)})

    return rows, skipped


def split_rows(rows):
    labels = [row["label"] for row in rows]

    return train_test_split(
        rows,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def unpack_rows(rows):
    messages = [row["message"] for row in rows]
    manual_features = [row["manual_features"] for row in rows]
    labels = [row["label"] for row in rows]

    return messages, manual_features, labels


def print_mistakes(title, test_rows, predictions, actual_labels):
    print(f"\n--- {title} Mistakes ---")

    mistake_count = 0

    for row, predicted, actual in zip(test_rows, predictions, actual_labels):
        predicted_label = decode_label(predicted)
        actual_label = decode_label(actual)

        if predicted_label != actual_label:
            mistake_count += 1
            print(f"\nMistake #{mistake_count}")
            print("text:", row["message"])
            print("manual_features:", dict(zip(FEATURE_NAMES, row["manual_features"])))
            print("predicted:", predicted_label)
            print("actual:   ", actual_label)

    if mistake_count == 0:
        print("No mistakes found on this test set.")


def evaluate_model(title, model, X_test, y_test, test_rows):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0  # type: ignore[arg-type]
    )

    print(f"\n=== {title} ===")
    print("accuracy:", accuracy)
    print("f1_macro:", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=list(range(len(KNOWN_ACTIONS)))))

    print("\nLabel order:")
    for index, action in enumerate(KNOWN_ACTIONS):
        print(index, "->", action)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(KNOWN_ACTIONS))),
            target_names=KNOWN_ACTIONS,
            zero_division=0,  # type: ignore[arg-type]
        )
    )

    print_mistakes(title, test_rows, predictions, y_test)

    evaluation = build_classification_evaluation(
        y_true=y_test,
        predictions=predictions,
        labels=list(range(len(KNOWN_ACTIONS))),
        target_names=KNOWN_ACTIONS,
    )

    return accuracy, f1, predictions, evaluation


def build_decision_object(model, X_single, original_text):
    probabilities = model.predict_proba(X_single)[0]
    best_index = probabilities.argmax()

    predicted_label = int(model.classes_[best_index])
    raw_action = decode_label(predicted_label)
    original_prediction = ACTION_DISPLAY_NAME[raw_action]
    confidence = float(probabilities[best_index])

    if confidence < MIN_CONFIDENCE:
        return build_decision(
            strategy_used="manual_features_plus_tfidf",
            predicted_action="needs_more_context",
            original_prediction=original_prediction,
            confidence=round(confidence, 4),
            risk_level=ACTION_RISK["needs_more_context"],
            requires_approval=REQUIRES_APPROVAL["needs_more_context"],
            reason=(
                f"Low confidence prediction. Model suggested "
                f"{original_prediction}, but confidence is below {MIN_CONFIDENCE}."
            ),
            similar_incidents=[],
            input_text=original_text,
        )

    return build_decision(
        strategy_used="manual_features_plus_tfidf",
        predicted_action=original_prediction,
        confidence=round(confidence, 4),
        risk_level=ACTION_RISK[original_prediction],
        requires_approval=REQUIRES_APPROVAL[original_prediction],
        reason=(
            f"Predicted {original_prediction} based on combined "
            "manual features and TF-IDF text signals."
        ),
        similar_incidents=[],
        input_text=original_text,
    )


def fit_model_with_warning_capture(
    *,
    model_name: str,
    model,
    X_train,
    y_train,
    warning_registry: dict,
):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)

    warning_registry[model_name] = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in caught_warnings
    ]

    return model


def count_warnings(training_warnings: dict) -> int:
    return sum(len(items) for items in training_warnings.values())


def count_convergence_warnings(training_warnings: dict) -> int:
    return sum(
        1
        for items in training_warnings.values()
        for item in items
        if item["category"] == ConvergenceWarning.__name__
    )


def stable_rows_sha256(rows: list[dict]) -> str:
    payload = [
        {
            "raw": row["raw"],
            "message": row["message"],
            "label": int(row["label"]),
        }
        for row in rows
    ]

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def build_split_evidence(
    *,
    train_rows: list[dict],
    test_rows: list[dict],
    comparison_group_id: str,
) -> dict:
    train_split_sha256 = stable_rows_sha256(train_rows)
    test_split_sha256 = stable_rows_sha256(test_rows)

    split_sha256 = hashlib.sha256(
        json.dumps(
            {
                "train_split_sha256": train_split_sha256,
                "test_split_sha256": test_split_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "train_split_sha256": train_split_sha256,
        "test_split_sha256": test_split_sha256,
        "split_sha256": split_sha256,
        "comparison_group_id": comparison_group_id,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "random_state": RANDOM_STATE,
        "test_size_config": TEST_SIZE,
    }


def train_evaluate_and_log_variant(
    *,
    variant: VariantDefinition,
    feature_pipeline: dict,
    evaluation_code: dict,
    experiment_config: dict,
    vectorizer,
    X_train_combined,
    X_test_combined,
    y_train,
    y_test,
    test_rows,
    split_evidence: dict | None = None,
) -> Path:
    with mlflow_run_context(run_name=variant.variant_name):
        training_warnings: dict = {}
        model = build_model_for_variant(variant)

        fit_model_with_warning_capture(
            model_name=variant.variant_name,
            model=model,
            X_train=X_train_combined,
            y_train=y_train,
            warning_registry=training_warnings,
        )

        accuracy, f1, _, evaluation = evaluate_model(
            variant.variant_name,
            model,
            X_test_combined,
            y_test,
            test_rows,
        )

        print("\n--- Decision Objects ---")

        decision_records = []

        for index, (row, x_single) in enumerate(zip(test_rows, X_test_combined)):
            started_at = perf_counter()

            decision = build_decision_object(
                model=model,
                X_single=x_single,
                original_text=row["message"],
            )

            policy_result = validate_policy(decision)
            latency_ms = (perf_counter() - started_at) * 1000

            decision_records.append(
                {
                    "decision": decision,
                    "policy_result": policy_result,
                    "latency_ms": round(float(latency_ms), 4),
                }
            )

            if index < 5:
                print(
                    json.dumps(
                        policy_result["modified_decision"],
                        indent=2,
                        sort_keys=True,
                    )
                )

        training_warning_count = count_warnings(training_warnings)
        convergence_warning_count = count_convergence_warnings(training_warnings)

        decision_quality_evaluation = build_decision_quality_evaluation(
            decision_records=decision_records,
            min_confidence=MIN_CONFIDENCE,
        )

        decision_quality_metrics = flatten_decision_quality_metrics(
            decision_quality_evaluation
        )

        offline_latency_evaluation = build_offline_latency_evaluation(
            decision_records=decision_records,
        )

        offline_latency_metrics = flatten_offline_latency_metrics(
            offline_latency_evaluation
        )

        per_class_metrics = flatten_per_class_metrics(
            prefix="combined",
            evaluation=evaluation,
        )

        print("\n--- Variant Summary ---")
        print("variant:", variant.variant_name)
        print("accuracy:", round(float(accuracy), 4))
        print("f1_macro:", round(float(f1), 4))

        print("\n--- Decision Quality Metrics ---")
        for key, value in decision_quality_metrics.items():
            print(f"{key}: {value}")

        print("\n--- Offline Artifact Smoke Latency ---")
        for key, value in offline_latency_metrics.items():
            print(f"{key}: {value}")

        evaluation_payload = {
            "combined": evaluation,
            "decision_quality": decision_quality_evaluation,
            "offline_latency": offline_latency_evaluation,
        }

        if split_evidence:
            evaluation_payload["split"] = split_evidence

        artifact = build_decision_artifact(
            model=model,
            vectorizer=vectorizer,
            manual_feature_names=MANUAL_FEATURE_NAMES,
            known_actions=KNOWN_ACTIONS,
            primary_model_accuracy=accuracy,
            test_size=len(y_test),
            training_warnings=training_warnings,
            evaluation=evaluation_payload,
            variant=variant_to_dict(variant),
            feature_pipeline=feature_pipeline,
            evaluation_code=evaluation_code,
            experiment_config=experiment_config,
            metrics={
                "combined_accuracy": round(float(accuracy), 4),
                "f1_macro": round(float(f1), 4),
                "combined_f1_macro": round(float(f1), 4),
                "manual_feature_count": len(MANUAL_FEATURE_NAMES),
                "tfidf_vocabulary_size": len(vectorizer.get_feature_names_out()),
                "training_warning_count": training_warning_count,
                "convergence_warning_count": convergence_warning_count,
                **per_class_metrics,
                **decision_quality_metrics,
                **offline_latency_metrics,
            },
        )

        artifact_dir = save_decision_artifact(artifact)
        log_mlflow_outputs(artifact=artifact, artifact_dir=artifact_dir)

        return artifact_dir


def main():
    experiment_config = load_experiment_config_for_run()

    validation_errors = validate_experiment_config(
        experiment_config=experiment_config,
        training_config=TRAINING_CONFIG,
    )

    if validation_errors:
        message = "\n".join(f"- {error}" for error in validation_errors)
        raise ValueError(f"Invalid experiment config:\n{message}")

    experiment_config_payload = experiment_config_to_dict(experiment_config)
    comparison_group_id = build_comparison_group_id(experiment_config)

    print("\n--- Experiment Config ---")
    print("experiment_name:", experiment_config.experiment_name)
    print("comparison_type:", experiment_config.comparison_type)
    print("changed_variable:", experiment_config.changed_variable)
    print("experiment_config_path:", experiment_config.experiment_config_path)
    print(
        "experiment_config_sha256:",
        experiment_config.experiment_config_sha256,
    )
    print("comparison_group_id:", comparison_group_id)

    raw_logs = load_raw_logs()
    rows, skipped = build_rows(raw_logs)

    print("Dataset size:", len(rows))
    print("Skipped examples:", len(skipped))

    if skipped:
        print("\nSkipped:")
        for item in skipped:
            print(item)

    train_rows, test_rows = split_rows(rows)

    split_evidence = build_split_evidence(
        train_rows=train_rows,
        test_rows=test_rows,
        comparison_group_id=comparison_group_id,
    )

    print("\n--- Split Evidence ---")
    print("split_sha256:", split_evidence["split_sha256"])
    print("train_split_sha256:", split_evidence["train_split_sha256"])
    print("test_split_sha256:", split_evidence["test_split_sha256"])
    print("comparison_group_id:", split_evidence["comparison_group_id"])

    X_train_text, X_train_manual, y_train = unpack_rows(train_rows)
    X_test_text, X_test_manual, y_test = unpack_rows(test_rows)

    X_train_manual_sparse = csr_matrix(X_train_manual)
    X_test_manual_sparse = csr_matrix(X_test_manual)

    vectorizer_kwargs = build_vectorizer_kwargs_from_experiment_config(
        experiment_config
    )

    feature_pipeline_cfg = experiment_config.feature_pipeline

    feature_pipeline_identity = build_feature_pipeline_identity(
        feature_pipeline_name=feature_pipeline_cfg["feature_pipeline_name"],
        vectorizer_name=feature_pipeline_cfg["vectorizer_name"],
        vectorizer_params=dict(feature_pipeline_cfg.get("vectorizer_params", {})),
        manual_feature_names=MANUAL_FEATURE_NAMES,
    )

    feature_pipeline = feature_pipeline_identity_to_dict(feature_pipeline_identity)

    vectorizer = TfidfVectorizer(**vectorizer_kwargs)

    X_train_tfidf = cast(csr_matrix, vectorizer.fit_transform(X_train_text))
    X_test_tfidf = cast(csr_matrix, vectorizer.transform(X_test_text))

    X_train_combined = hstack([X_train_manual_sparse, X_train_tfidf])
    X_test_combined = hstack([X_test_manual_sparse, X_test_tfidf])

    print("\n--- Feature Shapes ---")
    print("X_train_manual shape:", X_train_manual_sparse.shape)
    print("X_train_tfidf shape: ", X_train_tfidf.shape)
    print("X_train_combined shape:", X_train_combined.shape)
    print("manual feature count:", len(FEATURE_NAMES))
    print("tfidf vocabulary size:", len(vectorizer.get_feature_names_out()))

    variants = select_variants_for_run(experiment_config)

    print("\n--- Variants Selected ---")
    for variant in variants:
        print("-", variant.variant_name)

    print("feature_pipeline_name:", feature_pipeline["feature_pipeline_name"])
    print("feature_pipeline_sha256:", feature_pipeline["feature_pipeline_sha256"])

    evaluation_code = evaluation_code_identity_to_dict(
        build_evaluation_code_identity()
    )

    print("\n--- Evaluation Code Identity ---")
    print("evaluation_code_sha256:", evaluation_code["evaluation_code_sha256"])
    print(
        "evaluation_code_files:",
        ",".join(file_info["path"] for file_info in evaluation_code["source_files"]),
    )

    artifact_dirs = []

    for variant in variants:
        artifact_dir = train_evaluate_and_log_variant(
            variant=variant,
            feature_pipeline=feature_pipeline,
            evaluation_code=evaluation_code,
            experiment_config=experiment_config_payload,
            vectorizer=vectorizer,
            X_train_combined=X_train_combined,
            X_test_combined=X_test_combined,
            y_train=y_train,
            y_test=y_test,
            test_rows=test_rows,
            split_evidence=split_evidence,
        )

        artifact_dirs.append(str(artifact_dir))

    print("\n--- Artifacts Created ---")
    for artifact_dir in artifact_dirs:
        print(artifact_dir)


if __name__ == "__main__":
    main()
