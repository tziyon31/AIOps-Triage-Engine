from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.log_triage.schemas import (
    DecisionObject,
    PolicyEngineConfig,
    PolicyResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "policy.yaml"


class PolicyConfigError(ValueError):
    """Raised when policy.yaml is missing or invalid."""


def load_policy_config(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise PolicyConfigError(f"Policy file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    try:
        parsed = PolicyEngineConfig.model_validate(raw_config)
    except ValidationError as error:
        raise PolicyConfigError(str(error)) from error

    return {
        "forbidden_actions": set(parsed.forbidden_actions),
        "approval_min_confidence": parsed.approval.min_confidence,
    }


def validate(
    decision: dict[str, Any] | DecisionObject,
    policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate a Decision Object against policy rules.

    This function does not mutate the original decision.
    It returns a PolicyResult dict containing:
    - allowed
    - reason
    - modified_decision
    """
    validated_decision = DecisionObject.model_validate(decision)
    modified_decision = validated_decision.model_dump()

    config = policy_config or load_policy_config()

    forbidden_actions: set[str] = config["forbidden_actions"]
    approval_min_confidence: float = config["approval_min_confidence"]

    action = modified_decision["predicted_action"]
    confidence = modified_decision["confidence"]
    risk_level = modified_decision["risk_level"]

    if action in forbidden_actions:
        modified_decision["requires_approval"] = True

        result = PolicyResult(
            allowed=False,
            reason=f"Action '{action}' is forbidden by policy.",
            modified_decision=modified_decision,
        )
        return result.model_dump()

    approval_reasons = []

    if risk_level == "high":
        modified_decision["requires_approval"] = True
        approval_reasons.append("High-risk decision requires human approval.")

    if confidence < approval_min_confidence:
        modified_decision["requires_approval"] = True
        approval_reasons.append(
            f"Confidence {confidence:.2f} is below approval threshold "
            f"{approval_min_confidence:.2f}."
        )

    if approval_reasons:
        result = PolicyResult(
            allowed=True,
            reason=" ".join(approval_reasons),
            modified_decision=modified_decision,
        )
        return result.model_dump()

    result = PolicyResult(
        allowed=True,
        reason="Decision allowed by policy.",
        modified_decision=modified_decision,
    )
    return result.model_dump()
