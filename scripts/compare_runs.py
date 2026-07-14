from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


DEFAULT_METRICS = [
    "combined_accuracy",
    "f1_macro",
    "combined_f1_macro",
    "combined_weakest_class_recall",
    "low_confidence_count",
    "low_confidence_rate",
    "approval_required_count",
    "approval_required_rate",
    "policy_block_count",
    "llm_fallback_count",
    "invalid_decision_schema_count",
    "offline_decision_latency_p50_ms",
    "offline_decision_latency_p95_ms",
    "offline_decision_latency_max_ms",
]


CONTROLLED_VARIABLE_TAGS = {
    "raw_data": ["training_data_sha256"],
    "train_split": ["train_split_sha256"],
    "test_split": ["test_split_sha256"],
    "feature_pipeline": ["feature_pipeline_sha256"],
    "policy": ["policy_sha256"],
    "evaluation_code": ["evaluation_code_sha256"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MLflow runs that belong to the same experiment comparison group."
    )

    parser.add_argument(
        "--experiment-name",
        default="log-triage-decision-engine",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--comparison-group-id",
        required=True,
        help="The shared comparison_group_id tag for comparable runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="evidence/run_comparison",
        help="Directory for comparison report outputs.",
    )

    return parser.parse_args()


def read_cell(row: pd.Series, column: str, default: Any = None) -> Any:
    if column not in row:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def get_run_value(row: pd.Series, kind: str, key: str, default: Any = None) -> Any:
    return read_cell(row, f"{kind}.{key}", default)


def normalize_run(row: pd.Series) -> dict[str, Any]:
    tags = {
        "variant_name": get_run_value(row, "tags", "variant_name", "unknown"),
        "variant_type": get_run_value(row, "tags", "variant_type", "unknown"),
        "comparison_type": get_run_value(row, "tags", "comparison_type", "unknown"),
        "changed_variable": get_run_value(row, "tags", "changed_variable", "unknown"),
        "controlled_variables": get_run_value(
            row,
            "tags",
            "controlled_variables",
            "",
        ),
        "comparison_group_id": get_run_value(
            row,
            "tags",
            "comparison_group_id",
            get_run_value(row, "tags", "fair_comparison_group_id", "unknown"),
        ),
        "experiment_name": get_run_value(
            row,
            "tags",
            "experiment_name",
            "unknown",
        ),
        "experiment_config_path": get_run_value(
            row,
            "tags",
            "experiment_config_path",
            "unknown",
        ),
        "experiment_config_sha256": get_run_value(
            row,
            "tags",
            "experiment_config_sha256",
            "unknown",
        ),
        "split_sha256": get_run_value(row, "tags", "split_sha256", "unknown"),
        "train_split_sha256": get_run_value(
            row,
            "tags",
            "train_split_sha256",
            "unknown",
        ),
        "test_split_sha256": get_run_value(
            row,
            "tags",
            "test_split_sha256",
            "unknown",
        ),
        "feature_pipeline_name": get_run_value(
            row,
            "tags",
            "feature_pipeline_name",
            "unknown",
        ),
        "feature_pipeline_sha256": get_run_value(
            row,
            "tags",
            "feature_pipeline_sha256",
            "unknown",
        ),
        "vectorizer_name": get_run_value(row, "tags", "vectorizer_name", "unknown"),
        "model_family": get_run_value(row, "tags", "model_family", "unknown"),
        "policy_sha256": get_run_value(row, "tags", "policy_sha256", "unknown"),
        "training_data_sha256": get_run_value(
            row,
            "tags",
            "training_data_sha256",
            "unknown",
        ),
        "evaluation_code_sha256": get_run_value(
            row,
            "tags",
            "evaluation_code_sha256",
            "unknown",
        ),
    }

    metrics = {
        metric_name: get_run_value(row, "metrics", metric_name)
        for metric_name in DEFAULT_METRICS
        if get_run_value(row, "metrics", metric_name) is not None
    }

    params = {
        column.replace("params.", ""): read_cell(row, column)
        for column in row.index
        if column.startswith("params.") and read_cell(row, column) is not None
    }

    return {
        "run_id": row["run_id"],
        "run_name": read_cell(row, "tags.mlflow.runName", "unknown"),
        "status": read_cell(row, "status", "unknown"),
        "start_time": str(read_cell(row, "start_time", "")),
        "tags": tags,
        "params": params,
        "metrics": metrics,
    }


def parse_controlled_variables(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_complete_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop runs that are missing required controlled-variable tags.

    Older runs from the same split may share comparison_group_id but lack
    newer identity tags such as evaluation_code_sha256. Those runs are not
    fair members of the current controlled-variable contract.
    """
    if not runs:
        return runs

    controlled_variables = parse_controlled_variables(
        runs[0]["tags"].get("controlled_variables", "")
    )

    required_tags: list[str] = []
    for variable in controlled_variables:
        required_tags.extend(CONTROLLED_VARIABLE_TAGS.get(variable, []))

    if not required_tags:
        return runs

    complete_runs = []
    incomplete_runs = []

    for run in runs:
        tags = run["tags"]
        missing = [
            tag_name
            for tag_name in required_tags
            if tags.get(tag_name) in (None, "", "unknown", "missing")
        ]

        if missing:
            incomplete_runs.append((run, missing))
        else:
            complete_runs.append(run)

    if incomplete_runs:
        print(
            f"Excluded {len(incomplete_runs)} incomplete run(s) missing "
            "controlled-variable tags."
        )

    if len(complete_runs) >= 2:
        return complete_runs

    return runs


def validate_controlled_variables(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"status": "failed", "reason": "no runs found"}

    first_tags = runs[0]["tags"]
    controlled_variables = parse_controlled_variables(
        first_tags.get("controlled_variables", "")
    )

    checks = []

    for variable in controlled_variables:
        tag_names = CONTROLLED_VARIABLE_TAGS.get(variable, [])

        if not tag_names:
            checks.append(
                {
                    "controlled_variable": variable,
                    "status": "not_checked",
                    "reason": "no tag mapping defined",
                }
            )
            continue

        for tag_name in tag_names:
            values = {
                run["tags"].get(tag_name, "missing")
                for run in runs
            }

            if len(values) == 1 and "missing" not in values and "unknown" not in values:
                status = "passed"
            else:
                status = "failed"

            checks.append(
                {
                    "controlled_variable": variable,
                    "tag": tag_name,
                    "status": status,
                    "values": sorted(values),
                }
            )

    overall_status = (
        "passed"
        if checks and all(check["status"] == "passed" for check in checks)
        else "failed"
    )

    return {
        "status": overall_status,
        "controlled_variables": controlled_variables,
        "checks": checks,
    }


def unique_tag_values(runs: list[dict[str, Any]], tag_name: str) -> list[str]:
    return sorted(
        {
            str(run["tags"].get(tag_name, "unknown"))
            for run in runs
        }
    )


def build_comparison_contract(
    runs: list[dict[str, Any]],
    controlled_variable_validation: dict[str, Any],
) -> dict[str, Any]:
    comparison_types = unique_tag_values(runs, "comparison_type")
    changed_variables = unique_tag_values(runs, "changed_variable")
    variant_types = unique_tag_values(runs, "variant_type")
    comparison_group_ids = unique_tag_values(runs, "comparison_group_id")

    is_single_comparison_type = len(comparison_types) == 1
    is_single_changed_variable = len(changed_variables) == 1
    is_single_group = len(comparison_group_ids) == 1

    is_valid = (
        is_single_comparison_type
        and is_single_changed_variable
        and is_single_group
        and controlled_variable_validation["status"] == "passed"
        and len(runs) >= 2
    )

    return {
        "status": "valid" if is_valid else "invalid",
        "comparison_group_id": comparison_group_ids[0]
        if is_single_group
        else "mixed",
        "comparison_type": comparison_types[0]
        if is_single_comparison_type
        else "mixed",
        "changed_variable": changed_variables[0]
        if is_single_changed_variable
        else "mixed",
        "variant_types": variant_types,
        "run_count": len(runs),
        "minimum_run_count_met": len(runs) >= 2,
        "controlled_variable_status": controlled_variable_validation["status"],
        "checks": {
            "single_comparison_group_id": is_single_group,
            "single_comparison_type": is_single_comparison_type,
            "single_changed_variable": is_single_changed_variable,
            "controlled_variables_passed": controlled_variable_validation["status"]
            == "passed",
            "at_least_two_runs": len(runs) >= 2,
        },
    }


def build_candidate_selection_input(
    *,
    runs: list[dict[str, Any]],
    comparison_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ready_for_candidate_selection": comparison_contract["status"] == "valid",
        "reason": (
            "comparison_contract_valid"
            if comparison_contract["status"] == "valid"
            else "comparison_contract_invalid"
        ),
        "candidate_pool": [
            {
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "variant_name": run["tags"]["variant_name"],
                "variant_type": run["tags"]["variant_type"],
                "comparison_type": run["tags"]["comparison_type"],
                "changed_variable": run["tags"]["changed_variable"],
                "model_family": run["tags"]["model_family"],
                "feature_pipeline_name": run["tags"]["feature_pipeline_name"],
                "metrics": run["metrics"],
            }
            for run in runs
        ],
    }


def select_best_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def best_by(metric_name: str, reverse: bool = True):
        candidates = [
            run for run in runs
            if metric_name in run["metrics"]
        ]

        if not candidates:
            return None

        return sorted(
            candidates,
            key=lambda run: run["metrics"][metric_name],
            reverse=reverse,
        )[0]

    return {
        "best_f1_macro": best_by("f1_macro", reverse=True),
        "best_combined_accuracy": best_by("combined_accuracy", reverse=True),
        "best_low_confidence_rate": best_by("low_confidence_rate", reverse=False),
        "best_approval_required_rate": best_by(
            "approval_required_rate",
            reverse=False,
        ),
        "best_latency_p95": best_by(
            "offline_decision_latency_p95_ms",
            reverse=False,
        ),
    }


def build_tradeoff_summary(best_runs: dict[str, Any]) -> list[str]:
    lines = []

    best_f1 = best_runs.get("best_f1_macro")
    best_latency = best_runs.get("best_latency_p95")
    best_low_conf = best_runs.get("best_low_confidence_rate")

    if best_f1 and best_latency:
        if best_f1["run_id"] == best_latency["run_id"]:
            lines.append(
                "The same variant currently wins both f1_macro and p95 offline latency."
            )
        else:
            lines.append(
                "There is a quality/latency tradeoff: the best f1_macro variant is not the fastest p95 offline latency variant."
            )

    if best_f1 and best_low_conf:
        if best_f1["run_id"] == best_low_conf["run_id"]:
            lines.append(
                "The same variant currently wins both f1_macro and low_confidence_rate."
            )
        else:
            lines.append(
                "There is a quality/confidence tradeoff: the best f1_macro variant is not the variant with the lowest low_confidence_rate."
            )

    if not lines:
        lines.append("No clear tradeoff could be inferred from the available metrics.")

    return lines


def build_markdown_report(report: dict[str, Any]) -> str:
    contract = report["comparison_contract"]
    candidate_input = report["candidate_selection_input"]

    lines = [
        "# Run Comparison Report",
        "",
        f"- Experiment: `{report['experiment_name']}`",
        f"- Comparison group: `{report['comparison_group_id']}`",
        f"- Run count: `{report['run_count']}`",
        "",
        "## Summary",
        "",
        f"- Comparison status: `{contract['status']}`",
        f"- Ready for candidate selection: `{str(candidate_input['ready_for_candidate_selection']).lower()}`",
        f"- Comparison type: `{contract['comparison_type']}`",
        f"- Changed variable: `{contract['changed_variable']}`",
        f"- Controlled variable status: `{contract['controlled_variable_status']}`",
        "",
        "## Comparison Contract",
        "",
        "| Check | Passed |",
        "|---|---|",
    ]

    for check_name, passed in contract["checks"].items():
        lines.append(f"| `{check_name}` | `{str(passed).lower()}` |")

    lines.extend(
        [
            "",
            "This report validates the experiment comparison group. It does not promote a model or variant by itself.",
            "",
            "## Comparison Metadata",
            "",
        ]
    )

    for run in report["runs"]:
        tags = run["tags"]
        lines.extend(
            [
                f"### `{tags['variant_name']}`",
                "",
                f"- Run ID: `{run['run_id']}`",
                f"- Run name: `{run['run_name']}`",
                f"- Variant type: `{tags['variant_type']}`",
                f"- Comparison type: `{tags['comparison_type']}`",
                f"- Changed variable: `{tags['changed_variable']}`",
                f"- Controlled variables: `{tags['controlled_variables']}`",
                f"- Model family: `{tags['model_family']}`",
                f"- Feature pipeline: `{tags['feature_pipeline_name']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Controlled Variable Validation",
            "",
            f"Overall status: `{report['controlled_variable_validation']['status']}`",
            "",
            "| Controlled variable | Tag | Status | Values |",
            "|---|---|---|---|",
        ]
    )

    for check in report["controlled_variable_validation"]["checks"]:
        lines.append(
            "| {controlled_variable} | {tag} | {status} | {values} |".format(
                controlled_variable=check.get("controlled_variable", ""),
                tag=check.get("tag", ""),
                status=check.get("status", ""),
                values=", ".join(check.get("values", [])),
            )
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
        ]
    )

    metric_names = sorted(
        {
            metric_name
            for run in report["runs"]
            for metric_name in run["metrics"].keys()
        }
    )

    header = "| Variant | " + " | ".join(metric_names) + " |"
    separator = "|---" * (len(metric_names) + 1) + "|"

    lines.append(header)
    lines.append(separator)

    for run in report["runs"]:
        variant_name = run["tags"]["variant_name"]
        values = [
            str(run["metrics"].get(metric_name, ""))
            for metric_name in metric_names
        ]
        lines.append("| " + " | ".join([variant_name, *values]) + " |")

    lines.extend(
        [
            "",
            "## Tradeoff Summary",
            "",
        ]
    )

    for item in report["tradeoff_summary"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Candidate Selection Input",
            "",
            f"- Ready: `{str(report['candidate_selection_input']['ready_for_candidate_selection']).lower()}`",
            f"- Reason: `{report['candidate_selection_input']['reason']}`",
            "",
            "| Variant | Run ID | Model family | Feature pipeline |",
            "|---|---|---|---|",
        ]
    )

    for candidate in report["candidate_selection_input"]["candidate_pool"]:
        lines.append(
            "| {variant_name} | {run_id} | {model_family} | {feature_pipeline_name} |".format(
                variant_name=candidate["variant_name"],
                run_id=candidate["run_id"],
                model_family=candidate["model_family"],
                feature_pipeline_name=candidate["feature_pipeline_name"],
            )
        )

    lines.extend(
        [
            "",
            "## Production Interpretation",
            "",
            "This report does not promote a candidate by itself. It only proves whether the compared runs belong to a valid comparison group and shows the metric tradeoffs. Candidate selection should be handled by a promotion policy in the next stage.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {args.experiment_name}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.comparison_group_id = '{args.comparison_group_id}'",
        output_format="pandas",
    )

    if runs_df.empty:
        raise SystemExit(
            f"No runs found for comparison_group_id={args.comparison_group_id}"
        )

    runs = [normalize_run(row) for _, row in runs_df.iterrows()]
    runs = filter_complete_runs(runs)

    if len(runs) < 2:
        raise SystemExit(
            "Expected at least two comparable runs after filtering incomplete "
            f"runs for comparison_group_id={args.comparison_group_id}, "
            f"found {len(runs)}."
        )

    controlled_variable_validation = validate_controlled_variables(runs)
    best_runs = select_best_runs(runs)

    comparison_contract = build_comparison_contract(
        runs=runs,
        controlled_variable_validation=controlled_variable_validation,
    )

    candidate_selection_input = build_candidate_selection_input(
        runs=runs,
        comparison_contract=comparison_contract,
    )

    report = {
        "experiment_name": args.experiment_name,
        "comparison_group_id": args.comparison_group_id,
        "run_count": len(runs),
        "comparison_contract": comparison_contract,
        "controlled_variable_validation": controlled_variable_validation,
        "candidate_selection_input": candidate_selection_input,
        "best_runs": {
            key: None
            if value is None
            else {
                "run_id": value["run_id"],
                "variant_name": value["tags"]["variant_name"],
                "metric_value": value["metrics"].get(
                    key.replace("best_", ""),
                ),
            }
            for key, value in best_runs.items()
        },
        "tradeoff_summary": build_tradeoff_summary(best_runs),
        "runs": runs,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "run_comparison.json"
    md_path = output_dir / "run_comparison.md"

    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    md_path.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    print(f"Comparison JSON written to: {json_path}")
    print(f"Comparison Markdown written to: {md_path}")
    print(f"Controlled variable status: {controlled_variable_validation['status']}")
    print(f"Comparison contract status: {comparison_contract['status']}")
    print(
        "Ready for candidate selection: "
        f"{candidate_selection_input['ready_for_candidate_selection']}"
    )

    if comparison_contract["status"] != "valid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
