from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.log_triage.artifact_version import find_latest_artifact_model_path

# Runtime / secrets (not policy or training hyperparameters)
def is_llm_disabled() -> bool:
    return os.getenv("LOG_TRIAGE_DISABLE_LLM") == "1"


BITWARDEN_SECRET_NAME = "OpenAIKey-MLOps"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"
_LEGACY_ARTIFACT_PATH = "artifacts/log_triage_decision_engine_v1.pkl"


def _read_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")

    return data


@lru_cache
def load_training_config() -> dict[str, Any]:
    return _read_yaml("training.yaml")


@lru_cache
def load_policy_config() -> dict[str, Any]:
    return _read_yaml("policy.yaml")


def _apply_policy_exports(policy: dict[str, Any]) -> None:
    global MIN_CONFIDENCE, SIMILARITY_THRESHOLD
    global ALLOWED_ACTIONS, FORBIDDEN_ACTIONS, ACTION_RISK, REQUIRES_APPROVAL

    MIN_CONFIDENCE = policy["min_confidence"]
    SIMILARITY_THRESHOLD = policy["similarity_threshold"]
    ALLOWED_ACTIONS = set(policy["allowed_actions"])
    FORBIDDEN_ACTIONS = set(policy["forbidden_actions"])
    ACTION_RISK = policy["action_risk"]
    REQUIRES_APPROVAL = policy["requires_approval"]


def _apply_training_exports(training: dict[str, Any]) -> None:
    global ARTIFACT_PATH, SCHEMA_VERSION, ARTIFACT_TYPE

    artifact_cfg = training["artifact"]
    latest_model_path = find_latest_artifact_model_path(training)
    if latest_model_path is not None:
        ARTIFACT_PATH = str(latest_model_path)
    elif Path(_LEGACY_ARTIFACT_PATH).exists():
        ARTIFACT_PATH = _LEGACY_ARTIFACT_PATH
    else:
        ARTIFACT_PATH = str(
            Path(artifact_cfg["output_dir"])
            / f"{artifact_cfg['name']}-v{artifact_cfg['major']}.{artifact_cfg['minor']}.0"
            / "model.pkl"
        )

    SCHEMA_VERSION = training["schema_version"]
    ARTIFACT_TYPE = training["artifact_type"]


_policy = load_policy_config()
_training = load_training_config()

MIN_CONFIDENCE: float
SIMILARITY_THRESHOLD: float
ALLOWED_ACTIONS: set[str]
FORBIDDEN_ACTIONS: set[str]
ACTION_RISK: dict[str, str]
REQUIRES_APPROVAL: dict[str, bool]
ARTIFACT_PATH: str
SCHEMA_VERSION: str
ARTIFACT_TYPE: str

_apply_policy_exports(_policy)
_apply_training_exports(_training)
