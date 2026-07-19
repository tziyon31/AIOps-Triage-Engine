from __future__ import annotations

from typing import Any, Protocol

from src.log_triage.predict import DecisionRuntime, load_runtime, predict_with_runtime
from src.log_triage.schemas import build_decision


class DecisionServiceProtocol(Protocol):
    def decide(self, input_text: str, trace_id: str | None = None) -> dict[str, Any]:
        """Return a DecisionObject-compatible dict for one input event."""
        ...


class DecisionService:
    """
    Stable Application/Core facade for runtime decisions.

    Stage 6 FastAPI should depend on this class instead of importing predict.py,
    train.py, MLflow, sklearn, or experiment scripts directly.

    Current boundary:
    - wraps the existing predict.py runtime flow
    - does not duplicate decision logic
    - does not change behavior
    """

    def __init__(self, runtime: DecisionRuntime | None = None) -> None:
        self._runtime = runtime

    def _get_runtime(self) -> DecisionRuntime:
        if self._runtime is None:
            self._runtime = load_runtime()
        return self._runtime

    def decide(self, input_text: str, trace_id: str | None = None) -> dict[str, Any]:
        """
        Return the same DecisionObject produced by the existing prediction flow.

        trace_id is accepted now to define the future API contract.
        It is not wired into the existing trace implementation yet, because this
        module must not change runtime behavior.
        """
        _ = trace_id
        return predict_with_runtime(input_text, self._get_runtime())


class FakeDecisionService:
    """
    Test double for future API tests.

    FastAPI tests should be able to validate request/response behavior without
    loading artifacts, OpenAI clients, MLflow, sklearn, or filesystem state.
    """

    def __init__(self, decision: dict[str, Any] | None = None) -> None:
        self._decision = decision or build_decision(
            strategy_used="fake_decision_service",
            predicted_action="needs_more_context",
            confidence=0.0,
            risk_level="low",
            requires_approval=False,
            reason="Fake decision service response.",
            similar_incidents=[],
        )

    def decide(self, input_text: str, trace_id: str | None = None) -> dict[str, Any]:
        _ = input_text
        _ = trace_id
        return dict(self._decision)
