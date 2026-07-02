"""Route classifier output through embeddings and/or LLM to a single Decision Object."""

from collections import Counter

from src.log_triage.config import (
    ACTION_RISK,
    MIN_CONFIDENCE,
    REQUIRES_APPROVAL,
    SIMILARITY_THRESHOLD,
)
from src.log_triage.llm_fallback import analyze_with_llm
from src.log_triage.similarity_search import find_similar_incidents


def collect_machine_context(log_text: str) -> dict:
    return {
        "recent_deployment": "unknown",
        "pod_restart_count": "unknown",
        "error_rate": "unknown",
        "latency_p95": "unknown",
        "related_logs": [],
        "context_quality": "low",
    }


def similar_incidents_are_strong_and_consistent(similar_incidents: list[dict]) -> bool:
    if not similar_incidents:
        return False

    top_similarity = similar_incidents[0]["similarity"]

    if top_similarity < SIMILARITY_THRESHOLD:
        return False

    top_actions = [incident["known_action"] for incident in similar_incidents[:3]]
    action_counts = Counter(top_actions)

    most_common_action, count = action_counts.most_common(1)[0]

    return count >= 2


def build_similarity_decision(
    original_decision: dict,
    similar_incidents: list[dict],
) -> dict:
    top_action = similar_incidents[0]["known_action"]
    top_similarity = similar_incidents[0]["similarity"]
    if top_similarity >= 0.95:
        router_reason = "Top similar incident was an exact or near-exact match."
    else:
        router_reason = "Top similar incidents showed consistent action agreement."

    return {
        **original_decision,
        "strategy_used": "embedding_similarity",
        "predicted_action": top_action,
        "confidence": round(float(top_similarity), 4),
        "risk_level": ACTION_RISK[top_action],
        "requires_approval": REQUIRES_APPROVAL[top_action],
        "reason": (
            "Classifier decision required additional evidence. "
            "Similar past incidents strongly supported this action."
        ),
        "similar_incidents": similar_incidents,
        "router_reason": router_reason,
    }


def route_decision(
    classifier_decision: dict,
    incident_memory: list[dict],
    embedding_client,
    llm_client,
) -> dict:
    confidence = float(classifier_decision["confidence"])
    risk_level = classifier_decision["risk_level"]
    input_text = classifier_decision["input_text"]

    if confidence >= MIN_CONFIDENCE and risk_level == "low":
        return {
            **classifier_decision,
            "router_reason": "High confidence and low risk. Classifier decision accepted.",
        }

    similar_incidents = find_similar_incidents(
        client=embedding_client,
        new_log=input_text,
        incident_memory=incident_memory,
        top_k=3,
    )

    if similar_incidents_are_strong_and_consistent(similar_incidents):
        return build_similarity_decision(
            original_decision=classifier_decision,
            similar_incidents=similar_incidents,
        )

    machine_context = collect_machine_context(input_text)

    llm_decision = analyze_with_llm(
        client=llm_client,
        log_text=input_text,
    )

    return {
        **llm_decision,
        "similar_incidents": similar_incidents,
        "machine_context": machine_context,
        "router_reason": (
            "Classifier was not sufficient and similar incidents were not strong enough. "
            "LLM fallback was used with available context."
        ),
    }


def main() -> None:
    from src.log_triage.llm_fallback import create_openai_client
    from src.log_triage.similarity_search import build_incident_memory

    client = create_openai_client()
    incident_memory = build_incident_memory(client)

    scenarios = [
        (
            "Case 1: high confidence, low risk -> accept classifier",
            {
                "strategy_used": "manual_features_plus_tfidf",
                "predicted_action": "open_ticket",
                "confidence": 0.92,
                "risk_level": "low",
                "requires_approval": False,
                "reason": "Example",
                "input_text": "db timeout during payout",
            },
        ),
        (
            "Case 2: low confidence -> embeddings then possibly LLM",
            {
                "strategy_used": "manual_features_plus_tfidf",
                "predicted_action": "needs_more_context",
                "original_prediction": "open_ticket",
                "confidence": 0.53,
                "risk_level": "low",
                "requires_approval": False,
                "reason": "Low confidence",
                "input_text": "unknown backend delay during payment",
            },
        ),
        (
            "Case 3: high confidence but medium risk -> verify with embeddings",
            {
                "strategy_used": "manual_features_plus_tfidf",
                "predicted_action": "suggest_scale_up",
                "confidence": 0.91,
                "risk_level": "medium",
                "requires_approval": True,
                "reason": "Example",
                "input_text": "shard lag on reads",
            },
        ),
    ]

    for title, classifier_decision in scenarios:
        print("\n" + "=" * 60)
        print(title)
        print("Classifier in:", classifier_decision)
        out = route_decision(
            classifier_decision=classifier_decision,
            incident_memory=incident_memory,
            embedding_client=client,
            llm_client=client,
        )
        print("Router out:", out)


if __name__ == "__main__":
    main()
