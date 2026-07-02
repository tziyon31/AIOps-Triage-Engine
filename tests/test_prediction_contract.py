import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.log_triage.schemas import DecisionObject
from src.log_triage.secrets import require_bitwarden_session

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DECISION_FIELDS = {
    "strategy_used",
    "predicted_action",
    "confidence",
    "risk_level",
    "requires_approval",
    "reason",
    "similar_incidents",
}

# These are currently allowed metadata fields.
# If you add a new output field later, update this list intentionally.
OPTIONAL_DECISION_FIELDS = {
    "original_prediction",
    "input_text",
    "raw_log",
    "router_reason",
    "summary",
    "root_cause",
    "missing_context",
    "machine_context",
}

ALLOWED_DECISION_FIELDS = REQUIRED_DECISION_FIELDS | OPTIONAL_DECISION_FIELDS


def run_predict(args: list[str], expected_returncode: int = 0) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "src.log_triage.predict", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == expected_returncode, (
        f"Unexpected return code.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    assert completed.stdout.strip(), "predict.py returned empty stdout"

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"predict.py did not return valid JSON.\nstdout:\n{completed.stdout}"
        ) from error


def assert_decision_contract(decision: dict) -> None:
    print("\nParsed Decision Object:")
    print(json.dumps(decision, indent=2, sort_keys=True))

    missing_fields = REQUIRED_DECISION_FIELDS - set(decision)
    assert not missing_fields, f"Missing required fields: {missing_fields}"

    unexpected_fields = set(decision) - ALLOWED_DECISION_FIELDS
    assert not unexpected_fields, (
        f"Unexpected fields found: {unexpected_fields}. "
        "If this is intentional, update the schema/contract test."
    )

    DecisionObject.model_validate(decision)

    assert isinstance(decision["requires_approval"], bool)
    assert 0.0 <= decision["confidence"] <= 1.0
    assert decision["risk_level"] in {"low", "medium", "high"}
    assert isinstance(decision["similar_incidents"], list)


@pytest.mark.contract
def test_predict_normal_input_returns_valid_decision_object():
    try:
        require_bitwarden_session()
    except RuntimeError as error:
        pytest.fail(str(error))

    raw_log = (
        "2026-05-03 09:12:11 ERROR payments "
        "db timeout after retries cpu 93 memory 84"
    )

    print("\nInput log:", raw_log)

    decision = run_predict([raw_log])

    assert_decision_contract(decision)
