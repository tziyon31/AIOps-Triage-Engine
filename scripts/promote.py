from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_runs import (
    build_comparison_contract,
    filter_complete_runs,
    normalize_run,
    validate_controlled_variables,
)
from src.log_triage.candidate_selection import (
    DEFAULT_CANDIDATE_SELECTION_POLICY_PATH,
    compare_to_baseline,
    evaluate_candidate_run,
    load_candidate_selection_policy,
    read_metric,
)


DEFAULT_OUTPUT_DIR = "evidence/candidate_selection"

DEFAULT_CURRENT_CANDIDATE_STATE_PATH = (
    "evidence/candidate_selection/current_candidate.json"
)

DEFAULT_CANDIDATE_SELECTION_EVENTS_PATH = (
    "evidence/candidate_selection/candidate_selection_events.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a candidate run from a valid MLflow comparison group."
    )

    parser.add_argument(
        "--experiment-name",
        default="log-triage-decision-engine",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--comparison-group-id",
        required=True,
        help="comparison_group_id to evaluate.",
    )
    parser.add_argument(
        "--baseline-run-id",
        required=True,
        help="Baseline MLflow run id. Required to avoid promoting without baseline.",
    )
    parser.add_argument(
        "--candidate-policy-path",
        default=DEFAULT_CANDIDATE_SELECTION_POLICY_PATH,
        help="Candidate selection policy YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for candidate selection report outputs.",
    )
    parser.add_argument(
        "--current-candidate-state-path",
        default=DEFAULT_CURRENT_CANDIDATE_STATE_PATH,
        help="Path to write the current candidate state file when --apply is used.",
    )
    parser.add_argument(
        "--candidate-selection-events-path",
        default=DEFAULT_CANDIDATE_SELECTION_EVENTS_PATH,
        help="Path to append candidate lifecycle events when --apply is used.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually tag the selected MLflow run. Default is dry-run.",
    )

    return parser.parse_args()


def get_experiment_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise SystemExit(f"Experiment not found: {experiment_name}")

    return experiment.experiment_id


def load_comparison_group_runs(
    *,
    experiment_id: str,
    comparison_group_id: str,
) -> list[dict[str, Any]]:
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.comparison_group_id = '{comparison_group_id}'",
        output_format="pandas",
    )

    if runs_df.empty:
        return []

    return [
        normalize_run(row)
        for _, row in runs_df.iterrows()
    ]


def normalize_mlflow_run_object(run: Any) -> dict[str, Any]:
    tags = dict(run.data.tags)
    metrics = dict(run.data.metrics)
    params = dict(run.data.params)

    def tag(name: str, default: str = "unknown") -> str:
        return str(tags.get(name, default))

    return {
        "run_id": run.info.run_id,
        "run_name": tag("mlflow.runName"),
        "status": run.info.status,
        "start_time": str(run.info.start_time),
        "tags": {
            "variant_name": tag("variant_name"),
            "variant_type": tag("variant_type"),
            "comparison_type": tag("comparison_type"),
            "changed_variable": tag("changed_variable"),
            "controlled_variables": tag("controlled_variables", ""),
            "comparison_group_id": tag("comparison_group_id"),
            "experiment_name": tag("experiment_name"),
            "experiment_config_path": tag("experiment_config_path"),
            "experiment_config_sha256": tag("experiment_config_sha256"),
            "split_sha256": tag("split_sha256"),
            "train_split_sha256": tag("train_split_sha256"),
            "test_split_sha256": tag("test_split_sha256"),
            "feature_pipeline_name": tag("feature_pipeline_name"),
            "feature_pipeline_sha256": tag("feature_pipeline_sha256"),
            "vectorizer_name": tag("vectorizer_name"),
            "model_family": tag("model_family"),
            "policy_sha256": tag("policy_sha256"),
            "training_data_sha256": tag("training_data_sha256"),
            "evaluation_code_sha256": tag("evaluation_code_sha256"),
        },
        "params": params,
        "metrics": metrics,
    }


def load_baseline_run(baseline_run_id: str) -> dict[str, Any]:
    try:
        run = mlflow.get_run(baseline_run_id)
    except Exception as error:
        raise SystemExit(
            f"Baseline run not found: {baseline_run_id}. Error: {error}"
        ) from error

    return normalize_mlflow_run_object(run)


def result_passed(result: dict[str, Any], policy: dict[str, Any]) -> bool:
    return (
        result.get("decision") == policy["decisions"]["pass"]
        or result.get("status") == "passed"
    )


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "run_name": run.get("run_name", "unknown"),
        "variant_name": run["tags"].get("variant_name", "unknown"),
        "model_family": run["tags"].get("model_family", "unknown"),
        "metrics": {
            "f1_macro": read_metric(run, "f1_macro"),
            "combined_weakest_class_recall": read_metric(
                run,
                "combined_weakest_class_recall",
            ),
            "low_confidence_rate": read_metric(run, "low_confidence_rate"),
            "approval_required_rate": read_metric(
                run,
                "approval_required_rate",
            ),
            "offline_decision_latency_p95_ms": read_metric(
                run,
                "offline_decision_latency_p95_ms",
            ),
        },
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple:
    run = candidate["run"]

    f1_macro = read_metric(run, "f1_macro")
    latency_p95 = read_metric(run, "offline_decision_latency_p95_ms")
    low_confidence_rate = read_metric(run, "low_confidence_rate")
    approval_required_rate = read_metric(run, "approval_required_rate")

    return (
        -(f1_macro if f1_macro is not None else float("-inf")),
        latency_p95 if latency_p95 is not None else float("inf"),
        low_confidence_rate if low_confidence_rate is not None else float("inf"),
        approval_required_rate if approval_required_rate is not None else float("inf"),
        str(run.get("start_time", "")),
        run["run_id"],
    )


def evaluate_candidates(
    *,
    candidate_runs: list[dict[str, Any]],
    baseline_run: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    evaluations = []

    for run in candidate_runs:
        if run["run_id"] == baseline_run["run_id"]:
            evaluations.append(
                {
                    "run": run,
                    "run_summary": summarize_run(run),
                    "eligible": False,
                    "decision": "candidate_rejected",
                    "reason": "candidate_is_baseline_run",
                    "policy_result": None,
                    "baseline_result": None,
                }
            )
            continue

        policy_result = evaluate_candidate_run(run=run, policy=policy)
        baseline_result = compare_to_baseline(
            candidate_run=run,
            baseline_run=baseline_run,
            policy=policy,
        )

        policy_passed = result_passed(policy_result, policy)
        baseline_passed = result_passed(baseline_result, policy)

        eligible = policy_passed and baseline_passed

        reasons = []
        if not policy_passed:
            reasons.append(policy_result.get("reason", "policy_failed"))
        if not baseline_passed:
            reasons.append(baseline_result.get("reason", "baseline_failed"))

        evaluations.append(
            {
                "run": run,
                "run_summary": summarize_run(run),
                "eligible": eligible,
                "decision": (
                    "candidate_ready"
                    if eligible
                    else "candidate_rejected"
                ),
                "reason": (
                    "candidate_policy_and_baseline_passed"
                    if eligible
                    else "; ".join(reasons)
                ),
                "policy_result": policy_result,
                "baseline_result": baseline_result,
            }
        )

    return evaluations


def select_candidate(
    candidate_evaluations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible_candidates = [
        candidate
        for candidate in candidate_evaluations
        if candidate["eligible"]
    ]

    if not eligible_candidates:
        return None

    return sorted(eligible_candidates, key=candidate_sort_key)[0]


def find_current_candidate_run_ids(*, experiment_id: str) -> list[str]:
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.current_candidate = 'true'",
        output_format="pandas",
    )

    if runs_df.empty or "run_id" not in runs_df.columns:
        return []

    return [
        str(run_id)
        for run_id in runs_df["run_id"].tolist()
    ]


def build_candidate_lifecycle_plan(
    *,
    selected_run_id: str | None,
    current_candidate_run_ids: list[str],
) -> dict[str, Any]:
    if selected_run_id is None:
        return {
            "new_current_candidate_run_id": None,
            "previous_current_candidate_run_ids": current_candidate_run_ids,
            "superseded_run_ids": [],
            "already_current": False,
        }

    superseded_run_ids = [
        run_id
        for run_id in current_candidate_run_ids
        if run_id != selected_run_id
    ]

    return {
        "new_current_candidate_run_id": selected_run_id,
        "previous_current_candidate_run_ids": current_candidate_run_ids,
        "superseded_run_ids": superseded_run_ids,
        "already_current": (
            selected_run_id in current_candidate_run_ids
            and not superseded_run_ids
        ),
    }


def build_current_candidate_state(
    *,
    report: dict[str, Any],
    lifecycle_plan: dict[str, Any],
) -> dict[str, Any]:
    selected = report["selected_candidate"]

    return {
        "updated_at": report["generated_at"],
        "current_candidate_run_id": None if selected is None else selected["run_id"],
        "current_candidate_variant_name": None
        if selected is None
        else selected["variant_name"],
        "previous_current_candidate_run_ids": lifecycle_plan[
            "previous_current_candidate_run_ids"
        ],
        "superseded_run_ids": lifecycle_plan["superseded_run_ids"],
        "baseline_run_id": report["baseline_run"]["run_id"],
        "comparison_group_id": report["comparison_group_id"],
        "experiment_name": report["experiment_name"],
        "candidate_policy_name": report["candidate_selection_policy"]["policy_name"],
        "candidate_policy_version": report["candidate_selection_policy"][
            "policy_version"
        ],
        "selection_status": report["selection_status"],
        "selection_mode": report["mode"],
        "selection_report_path": (
            "evidence/candidate_selection/candidate_selection_report.json"
        ),
    }


def write_current_candidate_state(
    *,
    state: dict[str, Any],
    state_path: str,
) -> Path:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(state, indent=2, default=str),
        encoding="utf-8",
    )

    return path


def append_candidate_selection_event(
    *,
    event: dict[str, Any],
    events_path: str,
) -> Path:
    path = Path(events_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, default=str) + "\n")

    return path


def build_candidate_selection_report(
    *,
    experiment_name: str,
    comparison_group_id: str,
    comparison_contract: dict[str, Any],
    controlled_variable_validation: dict[str, Any],
    baseline_run: dict[str, Any],
    candidate_runs: list[dict[str, Any]],
    policy: dict[str, Any],
    apply: bool,
    current_candidate_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    candidate_evaluations = evaluate_candidates(
        candidate_runs=candidate_runs,
        baseline_run=baseline_run,
        policy=policy,
    )

    selected = None
    if comparison_contract["status"] == "valid":
        selected = select_candidate(candidate_evaluations)

    selection_status = (
        "selected"
        if selected is not None
        else "no_candidate_selected"
    )

    lifecycle_plan = build_candidate_lifecycle_plan(
        selected_run_id=None if selected is None else selected["run"]["run_id"],
        current_candidate_run_ids=current_candidate_run_ids or [],
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "experiment_name": experiment_name,
        "comparison_group_id": comparison_group_id,
        "selection_status": selection_status,
        "comparison_contract": comparison_contract,
        "controlled_variable_validation": controlled_variable_validation,
        "candidate_selection_policy": {
            "policy_name": policy["policy_name"],
            "policy_version": policy["policy_version"],
            "policy_path": policy.get("policy_path", "unknown"),
            "status": policy.get("status", "unknown"),
            "baseline": policy.get("baseline", {}),
            "thresholds": policy.get("thresholds", {}),
            "tie_breakers": policy.get("tie_breakers", []),
        },
        "baseline_run": summarize_run(baseline_run),
        "selected_candidate": None
        if selected is None
        else {
            "run_id": selected["run"]["run_id"],
            "variant_name": selected["run"]["tags"].get(
                "variant_name",
                "unknown",
            ),
            "reason": selected["reason"],
            "metrics": summarize_run(selected["run"])["metrics"],
        },
        "candidate_evaluations": [
            {
                key: value
                for key, value in candidate.items()
                if key != "run"
            }
            for candidate in candidate_evaluations
        ],
        "candidate_lifecycle": lifecycle_plan,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    selected = report["selected_candidate"]

    lines = [
        "# Candidate Selection Report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Experiment: `{report['experiment_name']}`",
        f"- Comparison group: `{report['comparison_group_id']}`",
        f"- Selection status: `{report['selection_status']}`",
        "",
        "## Policy",
        "",
        f"- Name: `{report['candidate_selection_policy']['policy_name']}`",
        f"- Version: `{report['candidate_selection_policy']['policy_version']}`",
        f"- Status: `{report['candidate_selection_policy']['status']}`",
        f"- Path: `{report['candidate_selection_policy']['policy_path']}`",
        "",
        "## Baseline",
        "",
        f"- Run ID: `{report['baseline_run']['run_id']}`",
        f"- Variant: `{report['baseline_run']['variant_name']}`",
        f"- f1_macro: `{report['baseline_run']['metrics']['f1_macro']}`",
        "",
        "## Selected Candidate",
        "",
    ]

    if selected is None:
        lines.append("No candidate selected.")
    else:
        lines.extend(
            [
                f"- Run ID: `{selected['run_id']}`",
                f"- Variant: `{selected['variant_name']}`",
                f"- Reason: `{selected['reason']}`",
                f"- f1_macro: `{selected['metrics']['f1_macro']}`",
                f"- p95 latency: `{selected['metrics']['offline_decision_latency_p95_ms']}`",
            ]
        )

    lifecycle = report.get("candidate_lifecycle", {})

    lines.extend(
        [
            "",
            "## Candidate Lifecycle",
            "",
            f"- New current candidate: `{lifecycle.get('new_current_candidate_run_id')}`",
            f"- Previous current candidates: `{lifecycle.get('previous_current_candidate_run_ids', [])}`",
            f"- Superseded candidates on apply: `{lifecycle.get('superseded_run_ids', [])}`",
            f"- Already current: `{lifecycle.get('already_current')}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Candidate Evaluations",
            "",
            "| Variant | Run ID | Eligible | Decision | Reason | f1_macro | p95 latency |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for item in report["candidate_evaluations"]:
        summary = item["run_summary"]
        metrics = summary["metrics"]

        lines.append(
            "| {variant} | {run_id} | {eligible} | {decision} | {reason} | {f1} | {latency} |".format(
                variant=summary["variant_name"],
                run_id=summary["run_id"],
                eligible=str(item["eligible"]).lower(),
                decision=item["decision"],
                reason=item["reason"],
                f1=metrics["f1_macro"],
                latency=metrics["offline_decision_latency_p95_ms"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report selects a candidate only according to the offline candidate-selection policy.",
            "It does not deploy the model and does not claim production readiness.",
            "MLflow tags are written only when the script runs with `--apply`.",
            "",
        ]
    )

    return "\n".join(lines)


def apply_mlflow_candidate_tags(
    *,
    selected_candidate: dict[str, Any],
    report: dict[str, Any],
) -> None:
    client = MlflowClient()
    selected_run_id = selected_candidate["run_id"]
    policy = report["candidate_selection_policy"]
    lifecycle = report["candidate_lifecycle"]
    generated_at = report["generated_at"]

    for old_run_id in lifecycle["superseded_run_ids"]:
        client.set_tag(old_run_id, "current_candidate", "false")
        client.set_tag(old_run_id, "candidate_status", "superseded")
        client.set_tag(old_run_id, "superseded_by_run_id", selected_run_id)
        client.set_tag(old_run_id, "superseded_at", generated_at)

    client.set_tag(selected_run_id, "candidate", "true")
    client.set_tag(selected_run_id, "current_candidate", "true")
    client.set_tag(selected_run_id, "candidate_status", "selected")
    client.set_tag(selected_run_id, "promotion_reason", selected_candidate["reason"])
    client.set_tag(selected_run_id, "candidate_policy_name", policy["policy_name"])
    client.set_tag(selected_run_id, "candidate_policy_version", policy["policy_version"])
    client.set_tag(selected_run_id, "candidate_policy_status", policy["status"])
    client.set_tag(selected_run_id, "baseline_run_id", report["baseline_run"]["run_id"])
    client.set_tag(selected_run_id, "comparison_group_id", report["comparison_group_id"])
    client.set_tag(selected_run_id, "candidate_selected_at", generated_at)
    client.set_tag(
        selected_run_id,
        "previous_current_candidate_run_ids",
        ",".join(lifecycle["previous_current_candidate_run_ids"]),
    )


def write_report(
    *,
    report: dict[str, Any],
    output_dir: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "candidate_selection_report.json"
    md_path = output_path / "candidate_selection_report.md"

    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    md_path.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    return json_path, md_path


def main() -> None:
    args = parse_args()

    policy = load_candidate_selection_policy(args.candidate_policy_path)
    experiment_id = get_experiment_id(args.experiment_name)

    raw_runs = load_comparison_group_runs(
        experiment_id=experiment_id,
        comparison_group_id=args.comparison_group_id,
    )

    if not raw_runs:
        raise SystemExit(
            f"No runs found for comparison_group_id={args.comparison_group_id}"
        )

    candidate_runs = filter_complete_runs(raw_runs)

    controlled_variable_validation = validate_controlled_variables(candidate_runs)
    comparison_contract = build_comparison_contract(
        runs=candidate_runs,
        controlled_variable_validation=controlled_variable_validation,
    )

    baseline_run = load_baseline_run(args.baseline_run_id)

    current_candidate_run_ids = find_current_candidate_run_ids(
        experiment_id=experiment_id
    )

    report = build_candidate_selection_report(
        experiment_name=args.experiment_name,
        comparison_group_id=args.comparison_group_id,
        comparison_contract=comparison_contract,
        controlled_variable_validation=controlled_variable_validation,
        baseline_run=baseline_run,
        candidate_runs=candidate_runs,
        policy=policy,
        apply=args.apply,
        current_candidate_run_ids=current_candidate_run_ids,
    )

    json_path, md_path = write_report(
        report=report,
        output_dir=args.output_dir,
    )

    print(f"Candidate selection JSON written to: {json_path}")
    print(f"Candidate selection Markdown written to: {md_path}")
    print(f"Selection status: {report['selection_status']}")

    if report["comparison_contract"]["status"] != "valid":
        print("Candidate selection blocked: comparison contract is invalid.")
        raise SystemExit(2)

    if report["selected_candidate"] is None:
        print("No candidate selected.")
        raise SystemExit(2)

    if args.apply:
        apply_mlflow_candidate_tags(
            selected_candidate=report["selected_candidate"],
            report=report,
        )
        print(
            "MLflow tags applied to selected candidate: "
            f"{report['selected_candidate']['run_id']}"
        )

        state = build_current_candidate_state(
            report=report,
            lifecycle_plan=report["candidate_lifecycle"],
        )

        state_path = write_current_candidate_state(
            state=state,
            state_path=args.current_candidate_state_path,
        )

        event_path = append_candidate_selection_event(
            event={
                "event_type": "candidate_selected",
                "generated_at": report["generated_at"],
                "selected_candidate": report["selected_candidate"],
                "candidate_lifecycle": report["candidate_lifecycle"],
                "baseline_run_id": report["baseline_run"]["run_id"],
                "comparison_group_id": report["comparison_group_id"],
                "policy_name": report["candidate_selection_policy"]["policy_name"],
                "policy_version": report["candidate_selection_policy"][
                    "policy_version"
                ],
            },
            events_path=args.candidate_selection_events_path,
        )

        print(f"Current candidate state written to: {state_path}")
        print(f"Candidate selection event appended to: {event_path}")
    else:
        print("Dry-run only. Re-run with --apply to tag MLflow.")


if __name__ == "__main__":
    main()
