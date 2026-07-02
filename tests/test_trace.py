import json

from src.log_triage.trace import persist_prediction_trace


def test_persist_prediction_trace_writes_jsonl_record(tmp_path):
    trace_path = tmp_path / "prediction_trace.jsonl"

    decision = {
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
            "artifact_path": "/tmp/artifact-123",
            "run_id": "run-123",
            "git_sha": "abc",
            "model_sha256": "m" * 64,
            "vectorizer_sha256": "v" * 64,
            "known_actions_sha256": "k" * 64,
            "config_sha256": "c" * 64,
            "training_data_sha256": "d" * 64,
        },
    }

    persist_prediction_trace(
        raw_log="test raw log",
        decision=decision,
        trace_path=trace_path,
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["decision_id"] == "decision-123"
    assert record["request"]["raw_log"] == "test raw log"
    assert record["decision"]["trace"]["artifact_id"] == "artifact-123"
