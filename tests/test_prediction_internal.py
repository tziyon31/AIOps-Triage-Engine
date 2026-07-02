from pathlib import Path

import pytest

from src.log_triage.artifact_version import load_artifact_bundle
from src.log_triage.config import ARTIFACT_PATH
from src.log_triage.predict import build_classifier_decision
from src.log_triage.schemas import DecisionObject, build_error_decision
from src.log_triage.strategy_router import route_decision

REQUIRED_FIELDS = set(DecisionObject.model_fields.keys())
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}

SUCCESS_LOG = (
    "2026-05-03 09:12:11 ERROR payments db timeout after retries cpu 93 memory 84"
)


def assert_decision_schema(decision: dict) -> None:
    for field in REQUIRED_FIELDS:
        assert field in decision, f"missing required field: {field}"

    confidence = decision["confidence"]
    assert isinstance(confidence, (int, float))
    assert 0.0 <= confidence <= 1.0

    assert isinstance(decision["requires_approval"], bool)
    assert decision["risk_level"] in ALLOWED_RISK_LEVELS
    assert "similar_incidents" in decision
    assert isinstance(decision["similar_incidents"], list)


@pytest.fixture
def artifact():
    artifact_path = Path(ARTIFACT_PATH)
    if not artifact_path.exists():
        pytest.skip(f"artifact not found: {artifact_path}")
    return load_artifact_bundle(artifact_path)


def test_success_decision_has_all_required_fields(artifact):
    classifier_decision = build_classifier_decision(artifact, SUCCESS_LOG)

    decision = route_decision(
        classifier_decision=classifier_decision,
        incident_memory=[],
        embedding_client=None,
        llm_client=None,
    )

    assert_decision_schema(decision)


def test_error_decision_has_all_required_fields():
    decision = build_error_decision("Missing log input.")

    assert_decision_schema(decision)
    assert decision["strategy_used"] == "error_handler"
    assert decision["predicted_action"] == "needs_more_context"


def test_classifier_decision_includes_similar_incidents(artifact):
    decision = build_classifier_decision(artifact, SUCCESS_LOG)

    assert_decision_schema(decision)
    assert decision["similar_incidents"] == []


def test_higher_min_confidence_changes_prediction(artifact, monkeypatch):
    baseline = build_classifier_decision(artifact, SUCCESS_LOG)
    if baseline["predicted_action"] != "open_ticket":
        pytest.skip("log does not produce high-confidence open_ticket with default policy")

    monkeypatch.setattr("src.log_triage.predict.MIN_CONFIDENCE", 0.95)

    stricter = build_classifier_decision(artifact, SUCCESS_LOG)

    assert stricter["predicted_action"] == "needs_more_context"
    assert stricter["confidence"] == baseline["confidence"]
    assert "Low confidence" in stricter["reason"]
