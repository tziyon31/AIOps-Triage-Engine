from src.log_triage.config import (
    ACTION_RISK,
    ALLOWED_ACTIONS,
    ARTIFACT_PATH,
    MIN_CONFIDENCE,
    REQUIRES_APPROVAL,
    load_policy_config,
    load_training_config,
)
from src.log_triage.train import build_training_config


def test_training_yaml_loads_required_keys():
    training = load_training_config()

    assert "artifact" in training
    assert training["artifact"]["name"] == "log-triage"
    assert training["artifact"]["major"] == 1
    assert training["artifact"]["minor"] == 0
    assert training["artifact"]["output_dir"] == "artifacts"
    assert "random_seed" in training
    assert "test_size" in training
    assert "model" in training
    assert "tfidf" in training
    assert ARTIFACT_PATH.endswith("model.pkl") or ARTIFACT_PATH.endswith(".pkl")


def test_policy_yaml_loads_required_keys():
    policy = load_policy_config()

    assert policy["min_confidence"] == MIN_CONFIDENCE
    assert set(policy["allowed_actions"]) == ALLOWED_ACTIONS
    assert policy["action_risk"] == ACTION_RISK
    assert policy["requires_approval"] == REQUIRES_APPROVAL


def test_build_training_config_snapshots_full_yaml():
    snapshot = build_training_config()

    assert snapshot["source"] == "config/training.yaml"
    assert snapshot["config"]["artifact"]["name"] == "log-triage"
    assert snapshot["config"]["model"]["max_iter"] == snapshot["derived"]["model_max_iter"]
    assert snapshot["config"]["tfidf"]["ngram_range"] == [1, 1]
    assert snapshot["derived"]["text_representation"] == "manual_features_plus_tfidf"
