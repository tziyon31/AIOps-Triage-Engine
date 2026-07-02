from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_TRACE_PATH = Path("traces") / "prediction_trace.jsonl"


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


def persist_prediction_trace(
    raw_log: str,
    decision: dict[str, Any],
    trace_path: str | Path = DEFAULT_TRACE_PATH,
) -> None:
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trace = decision.get("trace")
    if not trace:
        raise ValueError("Cannot persist prediction trace without decision.trace")

    record = {
        "decision_id": trace["decision_id"],
        "created_at": trace["created_at"],
        "request": {
            "raw_log": raw_log,
        },
        "decision": decision,
    }

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")
