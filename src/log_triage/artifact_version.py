from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-not-found]

VERSIONED_ARTIFACT_DIR_PATTERN = re.compile(
    r"^(?P<name>.+)-v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)-(?P<date>\d{8})-(?P<time>\d{6})$"
)

CONTENT_HASH_KEYS = [
    "model_sha256",
    "vectorizer_sha256",
    "known_actions_sha256",
    "config_sha256",
    "training_data_sha256",
]


def parse_versioned_artifact_dir(dir_name: str) -> dict[str, Any] | None:
    match = VERSIONED_ARTIFACT_DIR_PATTERN.match(dir_name)
    if not match:
        return None

    return {
        "name": match.group("name"),
        "major": int(match.group("major")),
        "minor": int(match.group("minor")),
        "patch": int(match.group("patch")),
        "timestamp": f"{match.group('date')}-{match.group('time')}",
    }


def sha256_joblib_object(obj: Any) -> str:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def sha256_known_actions(known_actions: list[str]) -> str:
    encoded = json.dumps(known_actions, indent=2).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_json(data: dict) -> str:
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_content_hashes(artifact: dict) -> dict[str, str]:
    config_snapshot = {
        "training_config": artifact["training_config"],
        "decision_contract": artifact["decision_contract"],
    }

    raw_logs_path = artifact["training_config"]["config"]["raw_logs_path"]

    return {
        "model_sha256": sha256_joblib_object(artifact["model"]),
        "vectorizer_sha256": sha256_joblib_object(artifact["vectorizer"]),
        "known_actions_sha256": sha256_known_actions(artifact["known_actions"]),
        "config_sha256": sha256_json(config_snapshot),
        "training_data_sha256": sha256_file(raw_logs_path),
    }


def find_latest_artifact_for_version(
    output_dir: str | Path,
    name: str,
    major: int,
    minor: int,
) -> tuple[int, dict[str, str] | None]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return -1, None

    best_patch = -1
    best_timestamp = ""
    best_hashes: dict[str, str] | None = None

    for path in output_path.iterdir():
        if not path.is_dir():
            continue

        parsed = parse_versioned_artifact_dir(path.name)
        if parsed is None:
            continue
        if parsed["name"] != name:
            continue
        if parsed["major"] != major or parsed["minor"] != minor:
            continue

        patch = parsed["patch"]
        timestamp = parsed["timestamp"]
        manifest_hashes = None
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            manifest_hashes = manifest.get("hashes")

        if patch > best_patch or (patch == best_patch and timestamp > best_timestamp):
            best_patch = patch
            best_timestamp = timestamp
            best_hashes = manifest_hashes

    return best_patch, best_hashes


def content_hashes_changed(
    current_hashes: dict[str, str],
    last_hashes: dict[str, str] | None,
) -> bool:
    if last_hashes is None:
        return True

    for key in CONTENT_HASH_KEYS:
        if current_hashes.get(key) != last_hashes.get(key):
            return True

    return False


def resolve_patch(
    current_hashes: dict[str, str],
    last_patch: int,
    last_hashes: dict[str, str] | None,
) -> int:
    if last_patch < 0:
        return 0

    if content_hashes_changed(current_hashes, last_hashes):
        return last_patch + 1

    return last_patch


def build_versioned_artifact_dir_name(
    name: str,
    major: int,
    minor: int,
    patch: int,
    timestamp: str,
) -> str:
    return f"{name}-v{major}.{minor}.{patch}-{timestamp}"


def find_latest_artifact_model_path(training_config: dict[str, Any]) -> Path | None:
    artifact_cfg = training_config["artifact"]
    output_path = Path(artifact_cfg["output_dir"])
    if not output_path.exists():
        return None

    name = artifact_cfg["name"]
    major = artifact_cfg["major"]
    minor = artifact_cfg["minor"]

    candidates: list[tuple[int, str, Path]] = []

    for path in output_path.iterdir():
        if not path.is_dir():
            continue

        parsed = parse_versioned_artifact_dir(path.name)
        if parsed is None:
            continue
        if parsed["name"] != name:
            continue
        if parsed["major"] != major or parsed["minor"] != minor:
            continue

        model_path = path / "model.pkl"
        if not model_path.exists():
            continue

        candidates.append((parsed["patch"], parsed["timestamp"], model_path))

    if not candidates:
        return None

    _, _, model_path = max(candidates)
    return model_path


def find_latest_artifact_dir(training_config: dict[str, Any]) -> Path | None:
    model_path = find_latest_artifact_model_path(training_config)

    if model_path is None:
        return None

    return model_path.parent


def load_artifact_bundle(artifact_path: str | Path) -> dict[str, Any]:
    artifact_path = Path(artifact_path)

    if artifact_path.name == "model.pkl" and (artifact_path.parent / "vectorizer.pkl").exists():
        vectorizer_path = artifact_path.parent / "vectorizer.pkl"
        known_actions_path = artifact_path.parent / "known_actions.json"

        with known_actions_path.open("r", encoding="utf-8") as file:
            known_actions = json.load(file)

        manifest_path = artifact_path.parent / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Artifact manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        return {
            "model": joblib.load(artifact_path),
            "vectorizer": joblib.load(vectorizer_path),
            "known_actions": known_actions,
            "manifest": manifest,
            "artifact_dir": str(artifact_path.parent.resolve()),
        }

    loaded = joblib.load(artifact_path)
    if isinstance(loaded, dict) and "model" in loaded and "vectorizer" in loaded:
        return loaded

    raise ValueError(f"Unsupported artifact format: {artifact_path}")
