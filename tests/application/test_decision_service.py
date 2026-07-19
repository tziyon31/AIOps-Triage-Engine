from __future__ import annotations

from typing import Any

from src.log_triage.application import decision_service as decision_service_module
from src.log_triage.application.decision_service import (
    DecisionService,
    FakeDecisionService,
)


def make_decision(
    *,
    strategy_used: str = "manual_features_plus_tfidf",
    predicted_action: str = "open_ticket",
    confidence: float = 0.91,
    risk_level: str = "low",
    requires_approval: bool = False,
    reason: str = "Test decision.",
) -> dict[str, Any]:
    return {
        "strategy_used": strategy_used,
        "predicted_action": predicted_action,
        "confidence": confidence,
        "risk_level": risk_level,
        "requires_approval": requires_approval,
        "reason": reason,
        "similar_incidents": [],
        "trace": None,
    }


def test_decision_service_delegates_to_existing_prediction_flow(monkeypatch):
    runtime = object()
    expected = make_decision()
    calls = {}

    def fake_load_runtime():
        calls["loaded"] = calls.get("loaded", 0) + 1
        return runtime

    def fake_predict_with_runtime(raw_log, loaded_runtime):
        calls["raw_log"] = raw_log
        calls["runtime"] = loaded_runtime
        return expected

    monkeypatch.setattr(decision_service_module, "load_runtime", fake_load_runtime)
    monkeypatch.setattr(
        decision_service_module,
        "predict_with_runtime",
        fake_predict_with_runtime,
    )

    service = DecisionService()

    assert service.decide("ERROR payments timeout", trace_id="trace-123") == expected
    assert calls["loaded"] == 1
    assert calls["raw_log"] == "ERROR payments timeout"
    assert calls["runtime"] is runtime


def test_decision_service_reuses_runtime(monkeypatch):
    runtime = object()
    calls = {"loaded": 0}

    def fake_load_runtime():
        calls["loaded"] += 1
        return runtime

    def fake_predict_with_runtime(raw_log, loaded_runtime):
        return make_decision(reason=f"decision for {raw_log}")

    monkeypatch.setattr(decision_service_module, "load_runtime", fake_load_runtime)
    monkeypatch.setattr(
        decision_service_module,
        "predict_with_runtime",
        fake_predict_with_runtime,
    )

    service = DecisionService()

    service.decide("first log")
    service.decide("second log")

    assert calls["loaded"] == 1


def test_decision_service_preserves_llm_fallback_output(monkeypatch):
    runtime = object()
    expected = make_decision(
        strategy_used="llm_fallback",
        predicted_action="needs_more_context",
        confidence=0.44,
        risk_level="low",
        requires_approval=True,
        reason="LLM fallback requested more context.",
    )

    monkeypatch.setattr(decision_service_module, "load_runtime", lambda: runtime)
    monkeypatch.setattr(
        decision_service_module,
        "predict_with_runtime",
        lambda raw_log, loaded_runtime: expected,
    )

    assert DecisionService().decide("ambiguous log") == expected


def test_decision_service_preserves_policy_block_output(monkeypatch):
    runtime = object()
    expected = make_decision(
        strategy_used="manual_features_plus_tfidf",
        predicted_action="suggest_scale_up",
        confidence=0.86,
        risk_level="medium",
        requires_approval=True,
        reason="Predicted suggest_scale_up. [Action requires approval.]",
    )

    monkeypatch.setattr(decision_service_module, "load_runtime", lambda: runtime)
    monkeypatch.setattr(
        decision_service_module,
        "predict_with_runtime",
        lambda raw_log, loaded_runtime: expected,
    )

    assert DecisionService().decide("cpu 98 memory 91") == expected


def test_fake_decision_service_returns_stable_decision():
    service = FakeDecisionService()

    decision = service.decide("anything", trace_id="api-test-trace")

    assert decision["strategy_used"] == "fake_decision_service"
    assert decision["predicted_action"] == "needs_more_context"
    assert decision["confidence"] == 0.0
    assert decision["requires_approval"] is False
