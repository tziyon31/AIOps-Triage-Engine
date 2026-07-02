import copy

import pytest

from src.log_triage.policy import PolicyConfigError, load_policy_config, validate


def base_decision(**overrides):
    decision = {
        "strategy_used": "test",
        "predicted_action": "open_ticket",
        "confidence": 0.95,
        "risk_level": "low",
        "requires_approval": False,
        "reason": "test decision",
        "similar_incidents": [],
    }

    decision.update(overrides)
    return decision


def test_load_policy_config_uses_pydantic_validation():
    config = load_policy_config()

    assert "delete_database" in config["forbidden_actions"]
    assert config["approval_min_confidence"] == 0.70


def test_validate_forbidden_action():
    decision = base_decision(
        predicted_action="delete_database",
        confidence=0.95,
        risk_level="low",
        requires_approval=False,
    )

    result = validate(decision)

    assert result["allowed"] is False
    assert "forbidden" in result["reason"]
    assert "delete_database" in result["reason"]
    assert result["modified_decision"]["requires_approval"] is True


def test_validate_high_risk_forces_approval_but_allows_decision():
    decision = base_decision(
        predicted_action="scale_up",
        confidence=0.95,
        risk_level="high",
        requires_approval=False,
    )

    result = validate(decision)

    assert result["allowed"] is True
    assert result["modified_decision"]["requires_approval"] is True
    assert "high" in result["reason"].lower()
    assert "approval" in result["reason"].lower()


def test_validate_low_confidence_forces_approval_but_allows_decision():
    decision = base_decision(
        predicted_action="open_ticket",
        confidence=0.42,
        risk_level="low",
        requires_approval=False,
    )

    result = validate(decision)

    assert result["allowed"] is True
    assert result["modified_decision"]["requires_approval"] is True
    assert "confidence" in result["reason"].lower()


def test_validate_safe_decision_remains_allowed_without_approval():
    decision = base_decision(
        predicted_action="open_ticket",
        confidence=0.95,
        risk_level="low",
        requires_approval=False,
    )

    result = validate(decision)

    assert result["allowed"] is True
    assert result["modified_decision"]["requires_approval"] is False
    assert result["reason"] == "Decision allowed by policy."


def test_validate_does_not_mutate_original_decision():
    decision = base_decision(
        predicted_action="open_ticket",
        confidence=0.42,
        risk_level="low",
        requires_approval=False,
    )
    original_decision = copy.deepcopy(decision)

    result = validate(decision)

    assert result["modified_decision"]["requires_approval"] is True
    assert decision == original_decision


def test_load_policy_config_rejects_invalid_yaml(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "forbidden_actions: not-a-list\napproval:\n  min_confidence: 2.5\n",
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigError):
        load_policy_config(policy_path)
