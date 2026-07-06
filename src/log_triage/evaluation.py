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
