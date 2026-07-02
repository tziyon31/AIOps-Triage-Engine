from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high"]


class DecisionObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    strategy_used: str
    predicted_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    requires_approval: bool
    reason: str
    similar_incidents: list[dict[str, Any]] = Field(default_factory=list)


class PolicyResult(BaseModel):
    allowed: bool
    reason: str
    modified_decision: DecisionObject


class ApprovalPolicy(BaseModel):
    min_confidence: float = Field(ge=0.0, le=1.0)


class PolicyEngineConfig(BaseModel):
    """Subset of policy.yaml used by policy.validate()."""

    model_config = ConfigDict(extra="ignore")

    forbidden_actions: list[str]
    approval: ApprovalPolicy


def build_decision(**kwargs) -> dict:
    return DecisionObject(**kwargs).model_dump()


def build_error_decision(reason: str) -> dict:
    return build_decision(
        strategy_used="error_handler",
        predicted_action="needs_more_context",
        confidence=0.0,
        risk_level="low",
        requires_approval=False,
        reason=reason,
        similar_incidents=[],
    )
