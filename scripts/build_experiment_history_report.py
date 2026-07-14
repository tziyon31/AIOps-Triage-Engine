from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_runs import (
    build_comparison_contract,
    build_tradeoff_summary,
    filter_complete_runs,
    normalize_run,
    select_best_runs,
    unique_tag_values,
    validate_controlled_variables,
)


DEFAULT_OUTPUT_DIR = "evidence/experiment_history"

MEANINGFUL_F1_DELTA = 0.02
LOW_CONFIDENCE_RATE_THRESHOLD = 0.20
WEAKEST_CLASS_RECALL_THRESHOLD = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an experiment history report from MLflow comparison groups."
        )
    )

    parser.add_argument(
        "--experiment-name",
        default="log-triage-decision-engine",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for experiment history outputs.",
    )

    return parser.parse_args()


def group_runs_by_comparison_group(
    runs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run in runs:
        group_id = run["tags"].get("comparison_group_id", "unknown")

        if group_id in {"", "unknown", "missing", None}:
            continue

        grouped[str(group_id)].append(run)

    return dict(grouped)


def summarize_best_runs(best_runs: dict[str, Any]) -> dict[str, Any]:
    summary = {}

    for key, run in best_runs.items():
        if run is None:
            summary[key] = None
            continue

        metric_name = key.replace("best_", "")

        summary[key] = {
            "run_id": run["run_id"],
            "variant_name": run["tags"]["variant_name"],
            "metric_name": metric_name,
            "metric_value": run["metrics"].get(metric_name),
        }

    return summary


def build_blocked_reason(
    *,
    comparison_contract: dict[str, Any],
    controlled_variable_validation: dict[str, Any],
) -> str | None:
    if comparison_contract["status"] == "valid":
        return None

    failed_contract_checks = [
        check_name
        for check_name, passed in comparison_contract["checks"].items()
        if not passed
    ]

    failed_controlled_checks = [
        check
        for check in controlled_variable_validation.get("checks", [])
        if check.get("status") == "failed"
    ]

    reasons = []

    if failed_contract_checks:
        reasons.append(
            "failed contract checks: "
            + ", ".join(failed_contract_checks)
        )

    if failed_controlled_checks:
        failed_names = sorted(
            {
                check.get("controlled_variable", "unknown")
                for check in failed_controlled_checks
            }
        )
        reasons.append(
            "failed controlled variables: "
            + ", ".join(failed_names)
        )

    if not reasons:
        reasons.append("comparison contract invalid")

    return "; ".join(reasons)


def read_metric(run: dict[str, Any], metric_name: str) -> float | None:
    value = run.get("metrics", {}).get(metric_name)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_spread(
    runs: list[dict[str, Any]],
    metric_name: str,
) -> float | None:
    values = [
        value
        for run in runs
        if (value := read_metric(run, metric_name)) is not None
    ]

    if len(values) < 2:
        return None

    return max(values) - min(values)


def build_next_experiment_recommendation(
    *,
    comparison_contract: dict[str, Any],
    controlled_variable_validation: dict[str, Any],
    runs: list[dict[str, Any]],
    best_runs: dict[str, Any],
    blocked_reason: str | None,
) -> dict[str, Any]:
    if comparison_contract["status"] != "valid":
        return {
            "recommendation_type": "rerun_invalid_comparison",
            "recommended_next_experiment": "rerun_same_experiment_after_fixing_contract",
            "reason": blocked_reason or "comparison contract is invalid",
            "evidence": [
                f"comparison_status={comparison_contract['status']}",
                (
                    "controlled_variable_status="
                    f"{controlled_variable_validation['status']}"
                ),
            ],
        }

    best_f1_run = best_runs.get("best_f1_macro")
    best_latency_run = best_runs.get("best_latency_p95")
    best_low_confidence_run = best_runs.get("best_low_confidence_rate")

    if best_f1_run:
        weakest_class_recall = read_metric(
            best_f1_run,
            "combined_weakest_class_recall",
        )

        if (
            weakest_class_recall is not None
            and weakest_class_recall < WEAKEST_CLASS_RECALL_THRESHOLD
        ):
            return {
                "recommendation_type": "data_or_feature_pipeline_experiment",
                "recommended_next_experiment": (
                    "weakest_class_data_or_feature_pipeline"
                ),
                "reason": (
                    "The best f1 variant still has weak recall on at least one "
                    "class, so another model-family comparison is unlikely to be "
                    "the highest-leverage next step."
                ),
                "evidence": [
                    f"best_variant={best_f1_run['tags']['variant_name']}",
                    f"combined_weakest_class_recall={weakest_class_recall}",
                ],
            }

        low_confidence_rate = read_metric(
            best_f1_run,
            "low_confidence_rate",
        )

        if (
            low_confidence_rate is not None
            and low_confidence_rate > LOW_CONFIDENCE_RATE_THRESHOLD
        ):
            return {
                "recommendation_type": "confidence_or_feature_experiment",
                "recommended_next_experiment": (
                    "confidence_threshold_or_feature_pipeline"
                ),
                "reason": (
                    "The best f1 variant still produces too many low-confidence "
                    "decisions. The next experiment should target confidence, "
                    "features, or threshold behavior."
                ),
                "evidence": [
                    f"best_variant={best_f1_run['tags']['variant_name']}",
                    f"low_confidence_rate={low_confidence_rate}",
                ],
            }

    f1_spread = metric_spread(runs, "f1_macro")

    if f1_spread is not None and f1_spread < MEANINGFUL_F1_DELTA:
        return {
            "recommendation_type": "change_experiment_direction",
            "recommended_next_experiment": "feature_pipeline_experiment",
            "reason": (
                "The compared variants are too close on f1_macro. Continuing "
                "to test similar variants is unlikely to produce a meaningful "
                "quality gain."
            ),
            "evidence": [
                f"f1_macro_spread={round(f1_spread, 4)}",
                f"meaningful_delta_threshold={MEANINGFUL_F1_DELTA}",
            ],
        }

    if (
        best_f1_run
        and best_latency_run
        and best_f1_run["run_id"] != best_latency_run["run_id"]
    ):
        return {
            "recommendation_type": "quality_latency_tradeoff_experiment",
            "recommended_next_experiment": "latency_or_cost_tradeoff_experiment",
            "reason": (
                "The best quality variant is not the fastest variant. The next "
                "experiment should clarify whether the quality gain is worth "
                "the latency/cost tradeoff."
            ),
            "evidence": [
                f"best_f1_variant={best_f1_run['tags']['variant_name']}",
                f"best_latency_variant={best_latency_run['tags']['variant_name']}",
            ],
        }

    if best_f1_run:
        return {
            "recommendation_type": "candidate_selection",
            "recommended_next_experiment": "candidate_selection_policy",
            "reason": (
                "The comparison is valid and no blocking quality/confidence "
                "issue was detected. The next step is candidate selection, "
                "not another experiment."
            ),
            "evidence": [
                f"best_f1_variant={best_f1_run['tags']['variant_name']}",
                f"comparison_type={comparison_contract['comparison_type']}",
                f"changed_variable={comparison_contract['changed_variable']}",
            ],
        }

    return {
        "recommendation_type": "insufficient_metrics",
        "recommended_next_experiment": "rerun_with_required_metrics",
        "reason": "No usable metric winner was found.",
        "evidence": ["missing best_f1_macro"],
    }


def summarize_comparison_group(
    *,
    comparison_group_id: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    complete_runs = filter_complete_runs(runs)
    controlled_variable_validation = validate_controlled_variables(complete_runs)

    comparison_contract = build_comparison_contract(
        runs=complete_runs,
        controlled_variable_validation=controlled_variable_validation,
    )

    best_runs = select_best_runs(complete_runs)

    blocked_reason = build_blocked_reason(
        comparison_contract=comparison_contract,
        controlled_variable_validation=controlled_variable_validation,
    )

    next_experiment_recommendation = build_next_experiment_recommendation(
        comparison_contract=comparison_contract,
        controlled_variable_validation=controlled_variable_validation,
        runs=complete_runs,
        best_runs=best_runs,
        blocked_reason=blocked_reason,
    )

    return {
        "comparison_group_id": comparison_group_id,
        "status": comparison_contract["status"],
        "ready_for_candidate_selection": (
            comparison_contract["status"] == "valid"
        ),
        "run_count": len(complete_runs),
        "raw_run_count": len(runs),
        "comparison_type": comparison_contract["comparison_type"],
        "changed_variable": comparison_contract["changed_variable"],
        "experiment_config_paths": unique_tag_values(
            complete_runs,
            "experiment_config_path",
        ),
        "experiment_config_sha256_values": unique_tag_values(
            complete_runs,
            "experiment_config_sha256",
        ),
        "variants": sorted(
            {
                run["tags"].get("variant_name", "unknown")
                for run in complete_runs
            }
        ),
        "model_families": unique_tag_values(complete_runs, "model_family"),
        "feature_pipeline_names": unique_tag_values(
            complete_runs,
            "feature_pipeline_name",
        ),
        "controlled_variable_status": controlled_variable_validation["status"],
        "comparison_contract": comparison_contract,
        "controlled_variable_validation": controlled_variable_validation,
        "best_runs": summarize_best_runs(best_runs),
        "tradeoff_summary": build_tradeoff_summary(best_runs),
        "blocked_reason": blocked_reason,
        "next_experiment_recommendation": next_experiment_recommendation,
    }


def build_overall_next_step(
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    if not groups:
        return {
            "recommendation_type": "no_experiments_found",
            "recommended_next_experiment": "run_first_config_driven_comparison",
            "reason": "No MLflow comparison groups were found.",
            "evidence": [],
            "source_group_id": None,
        }

    invalid_groups = [
        group
        for group in groups
        if group["status"] != "valid"
    ]

    if invalid_groups:
        group = invalid_groups[0]
        recommendation = group["next_experiment_recommendation"]

        return {
            **recommendation,
            "source_group_id": group["comparison_group_id"],
        }

    non_candidate_recommendations = [
        group
        for group in groups
        if group["next_experiment_recommendation"]["recommendation_type"]
        != "candidate_selection"
    ]

    if non_candidate_recommendations:
        group = non_candidate_recommendations[0]
        recommendation = group["next_experiment_recommendation"]

        return {
            **recommendation,
            "source_group_id": group["comparison_group_id"],
        }

    ready_groups = [
        group
        for group in groups
        if group["ready_for_candidate_selection"]
    ]

    if ready_groups:
        group = ready_groups[0]
        recommendation = group["next_experiment_recommendation"]

        return {
            **recommendation,
            "source_group_id": group["comparison_group_id"],
        }

    return {
        "recommendation_type": "no_ready_groups",
        "recommended_next_experiment": "rerun_comparison_with_complete_metadata",
        "reason": "No valid or ready comparison group was found.",
        "evidence": [],
        "source_group_id": None,
    }


def build_experiment_history_report(
    *,
    experiment_name: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = group_runs_by_comparison_group(runs)

    groups = [
        summarize_comparison_group(
            comparison_group_id=group_id,
            runs=group_runs,
        )
        for group_id, group_runs in sorted(grouped.items())
    ]

    valid_groups = [
        group
        for group in groups
        if group["status"] == "valid"
    ]

    ready_groups = [
        group
        for group in groups
        if group["ready_for_candidate_selection"]
    ]

    invalid_groups = [
        group
        for group in groups
        if group["status"] != "valid"
    ]

    return {
        "experiment_name": experiment_name,
        "total_comparison_groups": len(groups),
        "valid_comparison_groups": len(valid_groups),
        "invalid_comparison_groups": len(invalid_groups),
        "ready_for_candidate_selection_groups": len(ready_groups),
        "overall_next_step": build_overall_next_step(groups),
        "comparison_groups": groups,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment History Report",
        "",
        f"- MLflow experiment: `{report['experiment_name']}`",
        f"- Total comparison groups: `{report['total_comparison_groups']}`",
        f"- Valid comparison groups: `{report['valid_comparison_groups']}`",
        f"- Invalid comparison groups: `{report['invalid_comparison_groups']}`",
        "- Ready for candidate selection groups: "
        f"`{report['ready_for_candidate_selection_groups']}`",
        "",
        "## Recommended Next Step",
        "",
        f"- Type: `{report['overall_next_step']['recommendation_type']}`",
        "- Recommended next experiment: "
        f"`{report['overall_next_step']['recommended_next_experiment']}`",
        f"- Source group: `{report['overall_next_step']['source_group_id']}`",
        f"- Reason: {report['overall_next_step']['reason']}",
        "",
        "Evidence:",
    ]

    for item in report["overall_next_step"].get("evidence", []):
        lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## Comparison Groups",
            "",
            "| Group | Status | Ready | Changed variable | Variants | Recommendation | Blocked reason |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for group in report["comparison_groups"]:
        variants = ", ".join(group["variants"])
        blocked_reason = group["blocked_reason"] or ""
        recommendation = group["next_experiment_recommendation"]

        lines.append(
            "| {group_id} | {status} | {ready} | {changed_variable} | {variants} | {recommendation} | {blocked_reason} |".format(
                group_id=group["comparison_group_id"],
                status=group["status"],
                ready=str(group["ready_for_candidate_selection"]).lower(),
                changed_variable=group["changed_variable"],
                variants=variants,
                recommendation=recommendation["recommended_next_experiment"],
                blocked_reason=blocked_reason,
            )
        )

    lines.extend(
        [
            "",
            "## Valid Groups",
            "",
        ]
    )

    valid_groups = [
        group
        for group in report["comparison_groups"]
        if group["status"] == "valid"
    ]

    if not valid_groups:
        lines.append("No valid comparison groups found.")
    else:
        for group in valid_groups:
            lines.extend(
                [
                    f"### `{group['comparison_group_id']}`",
                    "",
                    f"- Comparison type: `{group['comparison_type']}`",
                    f"- Changed variable: `{group['changed_variable']}`",
                    f"- Variants: `{', '.join(group['variants'])}`",
                    f"- Experiment config: `{', '.join(group['experiment_config_paths'])}`",
                    "",
                    "Tradeoffs:",
                ]
            )

            for item in group["tradeoff_summary"]:
                lines.append(f"- {item}")

            recommendation = group["next_experiment_recommendation"]

            lines.extend(
                [
                    "",
                    "Recommendation:",
                    f"- Type: `{recommendation['recommendation_type']}`",
                    "- Recommended next experiment: "
                    f"`{recommendation['recommended_next_experiment']}`",
                    f"- Reason: {recommendation['reason']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "This report summarizes experiment history across MLflow comparison groups.",
            "It does not select or promote a production candidate.",
            "Candidate selection should be handled by a separate policy step.",
            "",
        ]
    )

    return "\n".join(lines)


def load_runs_from_mlflow(experiment_name: str) -> list[dict[str, Any]]:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise SystemExit(f"Experiment not found: {experiment_name}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )

    if runs_df.empty:
        return []

    return [
        normalize_run(row)
        for _, row in cast_dataframe(runs_df).iterrows()
    ]


def cast_dataframe(value: Any) -> pd.DataFrame:
    return value


def main() -> None:
    args = parse_args()

    runs = load_runs_from_mlflow(args.experiment_name)

    report = build_experiment_history_report(
        experiment_name=args.experiment_name,
        runs=runs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "experiment_history.json"
    md_path = output_dir / "experiment_history.md"

    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    md_path.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    print(f"Experiment history JSON written to: {json_path}")
    print(f"Experiment history Markdown written to: {md_path}")
    print(f"Total comparison groups: {report['total_comparison_groups']}")
    print(f"Valid comparison groups: {report['valid_comparison_groups']}")
    print(
        "Ready for candidate selection groups: "
        f"{report['ready_for_candidate_selection_groups']}"
    )


if __name__ == "__main__":
    main()
