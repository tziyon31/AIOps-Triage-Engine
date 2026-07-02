import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.log_triage.policy import validate
from src.log_triage.schemas import DecisionObject

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.llm_integration
def test_llm_integration_returns_valid_decision_object():
    if os.getenv("LOG_TRIAGE_ENABLE_LLM_INTEGRATION") != "1":
        pytest.skip("LLM integration test is disabled")

    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is required"

    raw_log = (
        "2026-05-03 09:12:11 WARNING unknown-service "
        "unclear intermittent issue with partial symptoms cpu 41 memory 44"
    )

    completed = subprocess.run(
        [sys.executable, "-m", "src.log_triage.predict", raw_log],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "LOG_TRIAGE_DISABLE_LLM": "0",
            "LOG_TRIAGE_ENABLE_LLM_INTEGRATION": "1",
        },
    )

    decision = json.loads(completed.stdout)

    DecisionObject.model_validate(decision)

    policy_result = validate(decision)

    assert "allowed" in policy_result
    assert "reason" in policy_result
    assert "modified_decision" in policy_result
