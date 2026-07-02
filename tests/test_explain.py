import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_explain_returns_error_for_missing_decision_id(tmp_path):
    trace_path = tmp_path / "prediction_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "explain.py",
            "missing-decision-id",
            "--trace-path",
            str(trace_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Decision id not found" in completed.stderr


def test_explain_prints_decision_metadata(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    manifest = {
        "artifact_id": "artifact-123",
        "run_id": "run-123",
        "git_sha": "git-123",
        "files": {
            "model": "model.pkl",
            "vectorizer": "vectorizer.pkl",
            "known_actions": "known_actions.json",
        },
        "hashes": {
            "model_sha256": "m" * 64,
            "vectorizer_sha256": "v" * 64,
            "known_actions_sha256": "k" * 64,
            "config_sha256": "c" * 64,
            "training_data_sha256": "d" * 64,
        },
        "training_config": {
            "config": {
                "raw_logs_path": str(tmp_path / "raw_logs.txt"),
            }
        },
        "decision_contract": {},
    }

    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    trace_record = {
        "decision_id": "decision-123",
        "created_at": "2026-07-02T15:37:38Z",
        "request": {"raw_log": "test log"},
        "decision": {
            "strategy_used": "test",
            "predicted_action": "open_ticket",
            "confidence": 0.9,
            "risk_level": "low",
            "requires_approval": False,
            "reason": "test",
            "similar_incidents": [],
            "trace": {
                "decision_id": "decision-123",
                "created_at": "2026-07-02T15:37:38Z",
                "artifact_id": "artifact-123",
                "artifact_path": str(artifact_dir.resolve()),
                "run_id": "run-123",
                "git_sha": "git-123",
                "model_sha256": "m" * 64,
                "vectorizer_sha256": "v" * 64,
                "known_actions_sha256": "k" * 64,
                "config_sha256": "c" * 64,
                "training_data_sha256": "d" * 64,
            },
        },
    }

    trace_path = tmp_path / "prediction_trace.jsonl"
    trace_path.write_text(json.dumps(trace_record) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "explain.py",
            "decision-123",
            "--trace-path",
            str(trace_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "model.pkl" in completed.stderr
