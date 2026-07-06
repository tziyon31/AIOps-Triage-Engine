from __future__ import annotations

from typing import Any

from sklearn.metrics import classification_report, confusion_matrix  # type: ignore[import-not-found]


def sanitize_metric_label(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def build_classification_evaluation(
    *,
    y_true,
    predictions,
    labels: list[int],
    target_names: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,  # type: ignore[arg-type]
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=labels,
    ).tolist()

    per_class_metrics: dict[str, dict[str, float | int]] = {}

    for name in target_names:
        class_report = report[name]

        per_class_metrics[name] = {
            "precision": round(float(class_report["precision"]), 4),
            "recall": round(float(class_report["recall"]), 4),
            "f1": round(float(class_report["f1-score"]), 4),
            "support": int(class_report["support"]),
        }

    weakest_class_by_recall, weakest_values = min(
        per_class_metrics.items(),
        key=lambda item: item[1]["recall"],
    )

    return {
        "labels": target_names,
        "label_ids": labels,
        "confusion_matrix": matrix,
        "per_class_metrics": per_class_metrics,
        "weakest_class_by_recall": weakest_class_by_recall,
        "weakest_class_recall": weakest_values["recall"],
    }


def flatten_per_class_metrics(
    *,
    prefix: str,
    evaluation: dict[str, Any],
) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}

    for label, values in evaluation["per_class_metrics"].items():
        safe_label = sanitize_metric_label(label)

        flattened[f"{prefix}_{safe_label}_precision"] = values["precision"]
        flattened[f"{prefix}_{safe_label}_recall"] = values["recall"]
        flattened[f"{prefix}_{safe_label}_f1"] = values["f1"]
        flattened[f"{prefix}_{safe_label}_support"] = values["support"]

    flattened[f"{prefix}_weakest_class_recall"] = evaluation[
        "weakest_class_recall"
    ]

    return flattened


def build_confusion_matrix_markdown(evaluation: dict) -> str:
    labels = evaluation["labels"]
    matrix = evaluation["confusion_matrix"]

    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    separator = "|---|" + "|".join(["---:"] * len(labels)) + "|"

    rows = []

    for actual_label, row in zip(labels, matrix):
        values = " | ".join(str(value) for value in row)
        rows.append(f"| {actual_label} | {values} |")

    return "\n".join([header, separator, *rows]) + "\n"


def build_confusion_matrix_text(evaluation: dict) -> str:
    labels = evaluation["labels"]
    matrix = evaluation["confusion_matrix"]

    column_width = max(
        14,
        max(len(label) for label in labels) + 2,
    )

    lines = []
    header = "actual \\ predicted".ljust(column_width)
    header += "".join(label.rjust(column_width) for label in labels)
    lines.append(header)
    lines.append("-" * len(header))

    for actual_label, row in zip(labels, matrix):
        line = actual_label.ljust(column_width)
        line += "".join(str(value).rjust(column_width) for value in row)
        lines.append(line)

    return "\n".join(lines) + "\n"


def build_decision_quality_evaluation(
    *,
    decision_records: list[dict],
    min_confidence: float,
) -> dict:
    decisions = []

    low_confidence_count = 0
    approval_required_count = 0
    policy_block_count = 0
    llm_fallback_count = 0
    invalid_decision_schema_count = 0

    for record in decision_records:
        decision = record["decision"]
        policy_result = record["policy_result"]
        modified_decision = policy_result["modified_decision"]

        confidence = float(modified_decision["confidence"])

        if confidence < min_confidence:
            low_confidence_count += 1

        if modified_decision["requires_approval"]:
            approval_required_count += 1

        if not policy_result["allowed"]:
            policy_block_count += 1

        if modified_decision.get("strategy_used") == "llm_fallback":
            llm_fallback_count += 1

        decisions.append(
            {
                "input_text": modified_decision.get("input_text"),
                "predicted_action": modified_decision["predicted_action"],
                "confidence": confidence,
                "risk_level": modified_decision["risk_level"],
                "requires_approval": modified_decision["requires_approval"],
                "policy_allowed": policy_result["allowed"],
                "policy_reason": policy_result["reason"],
            }
        )

    decision_count = len(decision_records)

    low_confidence_rate = (
        round(low_confidence_count / decision_count, 4)
        if decision_count
        else 0.0
    )

    approval_required_rate = (
        round(approval_required_count / decision_count, 4)
        if decision_count
        else 0.0
    )

    return {
        "thresholds": {
            "min_confidence": min_confidence,
        },
        "metrics": {
            "decision_count": decision_count,
            "low_confidence_count": low_confidence_count,
            "low_confidence_rate": low_confidence_rate,
            "approval_required_count": approval_required_count,
            "approval_required_rate": approval_required_rate,
            "policy_block_count": policy_block_count,
            "llm_fallback_count": llm_fallback_count,
            "invalid_decision_schema_count": invalid_decision_schema_count,
        },
        "decisions": decisions,
    }


def flatten_decision_quality_metrics(evaluation: dict) -> dict:
    return dict(evaluation["metrics"])
