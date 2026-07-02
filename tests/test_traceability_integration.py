import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_train_predict_explain_end_to_end(tmp_path):
    trace_path = tmp_path / "prediction_trace.jsonl"

    env = {
        **os.environ,
        "LOG_TRIAGE_DISABLE_LLM": "1",
        "LOG_TRIAGE_TRACE_PATH": str(trace_path),
    }

    subprocess.run(
        [sys.executable, "-m", "src.log_triage.train"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    raw_log = (
        "2026-05-03 09:12:11 ERROR payments "
        "db timeout after retries cpu 93 memory 84"
    )

    predict_result = subprocess.run(
        [sys.executable, "-m", "src.log_triage.predict", raw_log],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    decision = json.loads(predict_result.stdout)
    decision_id = decision["trace"]["decision_id"]

    assert trace_path.exists()

    explain_result = subprocess.run(
        [
            sys.executable,
            "explain.py",
            decision_id,
            "--trace-path",
            str(trace_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Decision ID:" in explain_result.stdout
    assert decision_id in explain_result.stdout
    assert "Artifact ID:" in explain_result.stdout
    assert "Git SHA:" in explain_result.stdout
    assert "config_sha256" in explain_result.stdout
    assert "training_data_sha256" in explain_result.stdout
    assert "OK: trace hashes match manifest hashes" in explain_result.stdout
    assert "OK: trace hashes match live artifact/data hashes" in explain_result.stdout
