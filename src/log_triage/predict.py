"""
Purpose:
Run the full log triage decision engine flow.

Supports:
1. Single log:
   python -m src.log_triage.predict "2026-05-03 09:12:11 ERROR payments db timeout cpu 93 memory 84"

2. Multiple logs from CLI:
   python -m src.log_triage.predict \
     --log "2026-05-03 09:12:11 ERROR payments db timeout cpu 93 memory 84" \
     --log "2026-05-03 09:13:02 INFO api request completed successfully cpu 12 memory 24"

3. Logs from file:
   python -m src.log_triage.predict --file data/test_logs.txt

Important:
Runtime resources are loaded once:
- model artifact
- OpenAI client
- incident memory
"""

import argparse
import json
from dataclasses import dataclass

from scipy.sparse import csr_matrix, hstack  # type: ignore[import-not-found]

from src.log_triage.artifact_version import load_artifact_bundle
from src.log_triage.config import (
    ACTION_RISK,
    ARTIFACT_PATH,
    MIN_CONFIDENCE,
    REQUIRES_APPROVAL,
    is_llm_disabled,
)
from src.log_triage.llm_fallback import create_openai_client
from src.log_triage.pipeline import (
    extract_features,
    parse_log_line,
)
from src.log_triage.schemas import build_decision, build_error_decision
from src.log_triage.similarity_search import build_incident_memory
from src.log_triage.strategy_router import route_decision
from src.log_triage.trace import attach_trace

@dataclass
class DecisionRuntime:
    artifact: dict
    client: object
    incident_memory: list[dict]


def load_runtime() -> DecisionRuntime:
    """
    Input:
        None.

    Output:
        DecisionRuntime containing artifact, OpenAI client, and incident memory.

    Called by:
        main()

    Why:
        Expensive resources should be loaded once, not once per log.
    """
    artifact = load_artifact_bundle(ARTIFACT_PATH)

    if is_llm_disabled():
        return DecisionRuntime(
            artifact=artifact,
            client=None,
            incident_memory=[],
        )

    client = create_openai_client()
    incident_memory = build_incident_memory(client)

    return DecisionRuntime(
        artifact=artifact,
        client=client,
        incident_memory=incident_memory,
    )


def decode_label(known_actions: list[str], encoded_label: int) -> str:
    """
    Input:
        known_actions:
            List of action labels from the artifact.

        encoded_label:
            Numeric model label.

    Output:
        Decoded action string.

    Called by:
        build_classifier_decision()
    """
    return known_actions[int(encoded_label)]


def build_classifier_decision(
    artifact: dict,
    raw_log: str,
) -> dict:
    """
    Input:
        artifact:
            Trained model artifact.

        raw_log:
            Raw log string without action label.

    Output:
        Initial classifier Decision Object.

    Called by:
        predict_with_runtime()
    """
    model = artifact["model"]
    vectorizer = artifact["vectorizer"]
    known_actions = artifact["known_actions"]

    event = parse_log_line(raw_log)

    manual_features = extract_features(event)
    message = event["message"]

    X_manual = csr_matrix([manual_features])
    X_text = vectorizer.transform([message])
    X_combined = hstack([X_manual, X_text])

    probabilities = model.predict_proba(X_combined)[0]
    best_index = probabilities.argmax()

    predicted_label = int(model.classes_[best_index])
    raw_action = decode_label(known_actions, predicted_label)

    action_display_name = {
        "open_ticket": "open_ticket",
        "ignore": "ignore",
        "scale_up": "suggest_scale_up",
    }

    original_prediction = action_display_name[raw_action]
    confidence = float(probabilities[best_index])

    if confidence < MIN_CONFIDENCE:
        return build_decision(
            strategy_used="manual_features_plus_tfidf",
            predicted_action="needs_more_context",
            original_prediction=original_prediction,
            confidence=round(confidence, 4),
            risk_level=ACTION_RISK["needs_more_context"],
            requires_approval=REQUIRES_APPROVAL["needs_more_context"],
            reason=(
                f"Low confidence prediction. Model suggested "
                f"{original_prediction}, but confidence is below {MIN_CONFIDENCE}."
            ),
            similar_incidents=[],
            input_text=message,
            raw_log=raw_log,
        )

    return build_decision(
        strategy_used="manual_features_plus_tfidf",
        predicted_action=original_prediction,
        confidence=round(confidence, 4),
        risk_level=ACTION_RISK[original_prediction],
        requires_approval=REQUIRES_APPROVAL[original_prediction],
        reason=(
            f"Predicted {original_prediction} based on combined "
            "manual features and TF-IDF text signals."
        ),
        similar_incidents=[],
        input_text=message,
        raw_log=raw_log,
    )


def predict_with_runtime(raw_log: str, runtime: DecisionRuntime) -> dict:
    """
    Input:
        raw_log:
            One raw log string.

        runtime:
            Loaded DecisionRuntime.

    Output:
        Final Decision Object after classifier + router.

    Called by:
        run_predictions()
    """
    classifier_decision = build_classifier_decision(
        artifact=runtime.artifact,
        raw_log=raw_log,
    )

    return attach_trace(
        route_decision(
            classifier_decision=classifier_decision,
            incident_memory=runtime.incident_memory,
            embedding_client=runtime.client,
            llm_client=runtime.client,
        ),
        runtime.artifact,
    )


def read_logs_from_file(file_path: str) -> list[str]:
    """
    Input:
        file_path:
            Path to a text file. One log per line.

    Output:
        List of non-empty log lines.

    Called by:
        collect_input_logs()
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def collect_input_logs(args: argparse.Namespace) -> list[str]:
    """
    Input:
        Parsed CLI arguments.

    Output:
        List of logs from --file, --log, or positional input.

    Called by:
        main()
    """
    logs = []

    if args.file:
        logs.extend(read_logs_from_file(args.file))

    if args.logs:
        logs.extend(args.logs)

    if args.positional_log:
        logs.append(" ".join(args.positional_log))

    return logs


def run_predictions(logs: list[str], runtime: DecisionRuntime) -> list[dict]:
    """
    Input:
        logs:
            List of raw log strings.

        runtime:
            Loaded DecisionRuntime.

    Output:
        List of results. Each item includes status and decision/error.

    Called by:
        main()
    """
    results = []

    for index, raw_log in enumerate(logs, start=1):
        try:
            decision = predict_with_runtime(raw_log, runtime)

            results.append(
                {
                    "index": index,
                    "status": "ok",
                    "raw_log": raw_log,
                    "decision": decision,
                }
            )

        except Exception as error:
            error_decision = attach_trace(
                build_error_decision(f"Prediction failed for this log: {error}"),
                runtime.artifact,
            )
            results.append(
                {
                    "index": index,
                    "status": "error",
                    "raw_log": raw_log,
                    "error": str(error),
                    "decision": error_decision,
                }
            )

    return results


def parse_args() -> argparse.Namespace:
    """
    Input:
        CLI arguments.

    Output:
        Parsed argparse Namespace.

    Called by:
        main()
    """
    parser = argparse.ArgumentParser(
        description="Run Decision Engine prediction for one or many logs."
    )

    parser.add_argument(
        "positional_log",
        nargs="*",
        help="Single raw log passed as positional text.",
    )

    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        help="Raw log. Can be repeated multiple times.",
    )

    parser.add_argument(
        "--file",
        help="Path to file containing one raw log per line.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = collect_input_logs(args)
    runtime = load_runtime()

    if not logs:
        error_response = attach_trace(
            build_error_decision(
                "Missing log input. Use positional log, --log, or --file."
            ),
            runtime.artifact,
        )
        print(json.dumps(error_response, indent=2))
        raise SystemExit(1)

    results = run_predictions(logs, runtime)

    if len(results) == 1:
        print(json.dumps(results[0]["decision"], indent=2))
        return

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
