import math

from openai import OpenAI

from src.log_triage.config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD
from src.log_triage.secrets import create_openai_client


PAST_INCIDENTS = [
    {
        "id": "inc-001",
        "log": "db timeout during payout",
        "known_root_cause": "payment database timeout",
        "known_action": "open_ticket",
    },
    {
        "id": "inc-002",
        "log": "database did not respond during payment",
        "known_root_cause": "payment database unavailable",
        "known_action": "open_ticket",
    },
    {
        "id": "inc-003",
        "log": "connection pool exhausted in checkout",
        "known_root_cause": "database connection pool exhausted",
        "known_action": "open_ticket",
    },
    {
        "id": "inc-004",
        "log": "shard lag on reads",
        "known_root_cause": "read replica lag",
        "known_action": "suggest_scale_up",
    },
    {
        "id": "inc-005",
        "log": "queue depth alert",
        "known_root_cause": "worker queue backlog",
        "known_action": "suggest_scale_up",
    },
    {
        "id": "inc-006",
        "log": "batch export completed",
        "known_root_cause": "normal batch completion",
        "known_action": "ignore",
    },
]


def create_embedding(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def build_incident_memory(client: OpenAI) -> list[dict]:
    memory = []

    for incident in PAST_INCIDENTS:
        memory.append(
            {
                **incident,
                "embedding": create_embedding(client, incident["log"]),
            }
        )

    return memory


def find_similar_incidents(
    client: OpenAI,
    new_log: str,
    incident_memory: list[dict],
    top_k: int = 3,
) -> list[dict]:
    new_embedding = create_embedding(client, new_log)

    results = []

    for incident in incident_memory:
        similarity = cosine_similarity(new_embedding, incident["embedding"])

        results.append(
            {
                "id": incident["id"],
                "log": incident["log"],
                "known_root_cause": incident["known_root_cause"],
                "known_action": incident["known_action"],
                "similarity": round(similarity, 4),
                "is_similar_enough": similarity >= SIMILARITY_THRESHOLD,
            }
        )

    results.sort(key=lambda item: item["similarity"], reverse=True)

    return results[:top_k]


def main() -> None:
    client = create_openai_client()
    incident_memory = build_incident_memory(client)

    test_logs = [
        "db timeout during payout",
        "payment database did not respond",
        "checkout cannot reach postgres",
        "shard lag on reads",
        "unknown backend delay during payment",
    ]

    for log in test_logs:
        print("\nInput log:", log)
        print("Top similar incidents:")

        similar_incidents = find_similar_incidents(
            client=client,
            new_log=log,
            incident_memory=incident_memory,
            top_k=3,
        )

        for incident in similar_incidents:
            print(incident)


if __name__ == "__main__":
    main()
