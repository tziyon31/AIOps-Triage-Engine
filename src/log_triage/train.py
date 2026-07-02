from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

import joblib  # type: ignore[import-not-found]

from scipy.sparse import csr_matrix, hstack  # type: ignore[import-not-found]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]

from src.log_triage.artifact_version import (
    build_content_hashes,
    build_versioned_artifact_dir_name,
    find_latest_artifact_for_version,
    parse_versioned_artifact_dir,
    resolve_patch,
    sha256_json,
)
from src.log_triage.config import (
    ACTION_RISK,
    ALLOWED_ACTIONS,
    ARTIFACT_TYPE,
    FORBIDDEN_ACTIONS,
    MIN_CONFIDENCE,
    MODEL_VERSION,
    REQUIRES_APPROVAL,
    SCHEMA_VERSION,
    SIMILARITY_THRESHOLD,
    load_training_config,
)
from src.log_triage.data import load_raw_logs
from src.log_triage.features import MANUAL_FEATURE_NAMES
from src.log_triage.pipeline import (
    FEATURE_NAMES,
    KNOWN_ACTIONS,
    extract_features,
    extract_label,
    parse_log_line,
)

TRAINING_CONFIG = load_training_config()
TEST_SIZE = TRAINING_CONFIG["test_size"]
RANDOM_STATE = TRAINING_CONFIG["random_seed"]
MODEL_MAX_ITER = TRAINING_CONFIG["model"]["max_iter"]

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


def build_decision_artifact(
    model,
    vectorizer,
    manual_feature_names: list[str],
    known_actions: list[str],
    primary_model_accuracy: float,
    test_size: int,
    metrics: dict,
) -> dict:
    return {
        "model": model,
        "vectorizer": vectorizer,
        "manual_feature_names": manual_feature_names,
        "known_actions": known_actions,
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": TRAINING_CONFIG["model"]["type"],
        "text_representation": "manual_features_plus_tfidf",
        "decision_contract": build_decision_contract(),
        "metrics": {
            # `accuracy` is the primary quality metric for the saved artifact.
            "accuracy": round(float(primary_model_accuracy), 4),
            "test_size": int(test_size),
            **metrics,
        },
        "training_config": build_training_config(),
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


def sha256_file(path) -> str:
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_manifest(
    artifact_dir,
    model_path,
    vectorizer_path,
    known_actions_path,
    artifact: dict,
) -> dict:
    artifact_id = artifact_dir.name

    config_snapshot = {
        "training_config": artifact["training_config"],
        "decision_contract": artifact["decision_contract"],
    }

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
        "created_at": artifact["created_at"],
        "git_sha": get_git_sha(),
        "schema_version": artifact["schema_version"],
        "model_version": artifact["model_version"],
        "artifact_type": artifact["artifact_type"],
        "files": {
            "model": "model.pkl",
            "vectorizer": "vectorizer.pkl",
            "known_actions": "known_actions.json",
        },
        "hashes": {
            "model_sha256": sha256_file(model_path),
            "vectorizer_sha256": sha256_file(vectorizer_path),
            "known_actions_sha256": sha256_file(known_actions_path),
            "config_sha256": sha256_json(config_snapshot),
        },
        "training_config": artifact["training_config"],
        "decision_contract": artifact["decision_contract"],
        "metrics": artifact["metrics"],
    }

    if version is not None:
        manifest["version"] = version

    return manifest


def save_decision_artifact(artifact: dict) -> Path:
    artifact_dir = create_artifact_dir(artifact)

    model_path = artifact_dir / "model.pkl"
    vectorizer_path = artifact_dir / "vectorizer.pkl"
    known_actions_path = artifact_dir / "known_actions.json"

    joblib.dump(artifact["model"], model_path)
    joblib.dump(artifact["vectorizer"], vectorizer_path)

    with open(known_actions_path, "w", encoding="utf-8") as file:
        json.dump(artifact["known_actions"], file, indent=2)

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

    print(f"\n=== {title} ===")
    print("accuracy:", accuracy)

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
            zero_division=0,
        )
    )

    print_mistakes(title, test_rows, predictions, y_test)

    return accuracy, predictions


def build_decision_object(model, X_single, original_text):
    probabilities = model.predict_proba(X_single)[0]
    best_index = probabilities.argmax()

    predicted_label = int(model.classes_[best_index])
    raw_action = decode_label(predicted_label)
    original_prediction = ACTION_DISPLAY_NAME[raw_action]
    confidence = float(probabilities[best_index])

    if confidence < MIN_CONFIDENCE:
        return {
            "strategy_used": "manual_features_plus_tfidf",
            "predicted_action": "needs_more_context",
            "original_prediction": original_prediction,
            "confidence": round(confidence, 4),
            "risk_level": ACTION_RISK["needs_more_context"],
            "requires_approval": REQUIRES_APPROVAL["needs_more_context"],
            "reason": (
                f"Low confidence prediction. Model suggested "
                f"{original_prediction}, but confidence is below {MIN_CONFIDENCE}."
            ),
            "input_text": original_text,
        }

    predicted_action = original_prediction
    risk_level = ACTION_RISK[predicted_action]
    requires_approval = REQUIRES_APPROVAL[predicted_action]

    return {
        "strategy_used": "manual_features_plus_tfidf",
        "predicted_action": predicted_action,
        "confidence": round(confidence, 4),
        "risk_level": risk_level,
        "requires_approval": requires_approval,
        "reason": (
            f"Predicted {predicted_action} based on combined "
            "manual features and TF-IDF text signals."
        ),
        "input_text": original_text,
    }


def main():
    raw_logs = load_raw_logs()
    rows, skipped = build_rows(raw_logs)

    print("Dataset size:", len(rows))
    print("Skipped examples:", len(skipped))

    if skipped:
        print("\nSkipped:")
        for item in skipped:
            print(item)

    train_rows, test_rows = split_rows(rows)

    X_train_text, X_train_manual, y_train = unpack_rows(train_rows)
    X_test_text, X_test_manual, y_test = unpack_rows(test_rows)

    X_train_manual_sparse = csr_matrix(X_train_manual)
    X_test_manual_sparse = csr_matrix(X_test_manual)

    tfidf_cfg = TRAINING_CONFIG.get("tfidf", {})
    vectorizer_kwargs = {}
    if tfidf_cfg.get("max_features") is not None:
        vectorizer_kwargs["max_features"] = tfidf_cfg["max_features"]
    if "ngram_range" in tfidf_cfg:
        vectorizer_kwargs["ngram_range"] = tuple(tfidf_cfg["ngram_range"])

    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    X_train_combined = hstack([X_train_manual_sparse, X_train_tfidf])
    X_test_combined = hstack([X_test_manual_sparse, X_test_tfidf])

    print("\n--- Feature Shapes ---")
    print("X_train_manual shape:", X_train_manual_sparse.shape)
    print("X_train_tfidf shape: ", X_train_tfidf.shape)
    print("X_train_combined shape:", X_train_combined.shape)
    print("manual feature count:", len(FEATURE_NAMES))
    print("tfidf vocabulary size:", len(vectorizer.get_feature_names_out()))

    manual_model = LogisticRegression(max_iter=MODEL_MAX_ITER, random_state=RANDOM_STATE)
    manual_model.fit(X_train_manual, y_train)

    tfidf_model = LogisticRegression(max_iter=MODEL_MAX_ITER)
    tfidf_model.fit(X_train_tfidf, y_train)

    combined_model = LogisticRegression(max_iter=MODEL_MAX_ITER)
    combined_model.fit(X_train_combined, y_train)

    manual_accuracy, _ = evaluate_model(
        "Manual Features Only",
        manual_model,
        X_test_manual,
        y_test,
        test_rows,
    )

    tfidf_accuracy, _ = evaluate_model(
        "TF-IDF Only",
        tfidf_model,
        X_test_tfidf,
        y_test,
        test_rows,
    )

    combined_accuracy, _ = evaluate_model(
        "Manual Features + TF-IDF",
        combined_model,
        X_test_combined,
        y_test,
        test_rows,
    )

    print("\n--- Summary ---")
    print("approach | accuracy")
    print(f"manual   | {manual_accuracy:.3f}")
    print(f"tfidf    | {tfidf_accuracy:.3f}")
    print(f"combined | {combined_accuracy:.3f}")

    print("\n--- Decision Objects ---")
    for row, x_single in zip(test_rows[:5], X_test_combined[:5]):
        decision = build_decision_object(
            model=combined_model,
            X_single=x_single,
            original_text=row["message"],
        )
        print(decision)

    artifact = build_decision_artifact(
        model=combined_model,
        vectorizer=vectorizer,
        manual_feature_names=MANUAL_FEATURE_NAMES,
        known_actions=KNOWN_ACTIONS,
        primary_model_accuracy=combined_accuracy,
        test_size=len(y_test),
        metrics={
            "manual_accuracy": round(float(manual_accuracy), 4),
            "tfidf_accuracy": round(float(tfidf_accuracy), 4),
            "combined_accuracy": round(float(combined_accuracy), 4),
            "manual_feature_count": len(MANUAL_FEATURE_NAMES),
            "tfidf_vocabulary_size": len(vectorizer.get_feature_names_out()),
        },
    )

    save_decision_artifact(artifact)


if __name__ == "__main__":
    main()
