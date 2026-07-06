import json
from pathlib import Path

import joblib  # type: ignore[import-not-found]
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]

from src.log_triage.artifact_version import find_latest_artifact_dir
from src.log_triage.config import load_training_config


def _latest_versioned_artifact_dir() -> Path:
    artifact_dir = find_latest_artifact_dir(load_training_config())
    if artifact_dir is None:
        pytest.skip("no versioned artifacts found")
    return artifact_dir


def test_latest_artifact_dir_has_required_files():
    artifact_dir = _latest_versioned_artifact_dir()

    assert (artifact_dir / "model.pkl").exists()
    assert (artifact_dir / "vectorizer.pkl").exists()
    assert (artifact_dir / "known_actions.json").exists()
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "training.yaml").exists()
    assert (artifact_dir / "policy.yaml").exists()
    assert (artifact_dir / "evaluation" / "combined_evaluation.json").exists()
    assert (artifact_dir / "evaluation" / "confusion_matrix.json").exists()
    assert (artifact_dir / "evaluation" / "confusion_matrix.md").exists()
    assert (artifact_dir / "evaluation" / "confusion_matrix.txt").exists()
    assert (artifact_dir / "evaluation" / "decision_quality.json").exists()
    assert (artifact_dir / "evaluation" / "offline_latency.json").exists()


def test_model_pkl_contains_only_model():
    artifact_dir = _latest_versioned_artifact_dir()
    model_obj = joblib.load(artifact_dir / "model.pkl")

    assert not isinstance(model_obj, dict)
    assert isinstance(model_obj, LogisticRegression)


def test_vectorizer_pkl_contains_only_vectorizer():
    artifact_dir = _latest_versioned_artifact_dir()
    vectorizer = joblib.load(artifact_dir / "vectorizer.pkl")

    assert isinstance(vectorizer, TfidfVectorizer)
    assert hasattr(vectorizer, "transform")


def test_known_actions_json_is_action_mapping_only():
    artifact_dir = _latest_versioned_artifact_dir()

    with (artifact_dir / "known_actions.json").open("r", encoding="utf-8") as file:
        known_actions = json.load(file)

    assert isinstance(known_actions, list)
    assert all(isinstance(action, str) for action in known_actions)


def test_manifest_holds_metadata_not_model_objects():
    artifact_dir = _latest_versioned_artifact_dir()

    with (artifact_dir / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    required_keys = {
        "artifact_id",
        "run_id",
        "created_at",
        "git_sha",
        "schema_version",
        "model_version",
        "artifact_type",
        "files",
        "hashes",
        "training_config",
        "decision_contract",
        "metrics",
        "version",
    }
    assert required_keys.issubset(manifest.keys())

    assert set(manifest["files"].keys()) == {
        "model",
        "vectorizer",
        "known_actions",
        "training_config",
        "policy",
        "combined_evaluation",
        "confusion_matrix",
        "confusion_matrix_markdown",
        "confusion_matrix_text",
        "decision_quality",
        "offline_latency",
    }
    assert set(manifest["hashes"].keys()) == {
        "model_sha256",
        "vectorizer_sha256",
        "known_actions_sha256",
        "training_config_sha256",
        "policy_sha256",
        "config_sha256",
        "training_data_sha256",
    }

    for forbidden_key in ("model", "vectorizer", "known_actions"):
        assert forbidden_key not in manifest

    assert manifest["model_version"] == manifest["version"]["label"]


def test_manifest_contains_traceability_hashes():
    artifact_dir = _latest_versioned_artifact_dir()
    manifest_path = artifact_dir / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "run_id" in manifest
    assert isinstance(manifest["run_id"], str)
    assert manifest["run_id"]

    assert "git_sha" in manifest
    assert isinstance(manifest["git_sha"], str)
    assert manifest["git_sha"]

    hashes = manifest["hashes"]

    assert "config_sha256" in hashes
    assert len(hashes["config_sha256"]) == 64

    assert "training_config_sha256" in hashes
    assert len(hashes["training_config_sha256"]) == 64

    assert "policy_sha256" in hashes
    assert len(hashes["policy_sha256"]) == 64

    assert "training_data_sha256" in hashes
    assert len(hashes["training_data_sha256"]) == 64

    assert "model_sha256" in hashes
    assert len(hashes["model_sha256"]) == 64
