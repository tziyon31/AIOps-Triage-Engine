from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_decision_trace(artifact: dict[str, Any]) -> dict[str, Any]:
    manifest = artifact["manifest"]
    hashes = manifest["hashes"]

    return {
        "decision_id": str(uuid4()),
        "created_at": utc_now(),
        "artifact_id": manifest["artifact_id"],
        "artifact_path": artifact["artifact_dir"],
        "run_id": manifest["run_id"],
        "git_sha": manifest["git_sha"],
        "model_sha256": hashes["model_sha256"],
        "vectorizer_sha256": hashes["vectorizer_sha256"],
        "known_actions_sha256": hashes["known_actions_sha256"],
        "config_sha256": hashes["config_sha256"],
        "training_data_sha256": hashes["training_data_sha256"],
    }


def attach_trace(decision: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    traced_decision = dict(decision)
    traced_decision["trace"] = build_decision_trace(artifact)
    return traced_decision
