from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_runs import (
    build_comparison_contract,
    filter_complete_runs,
    normalize_run,
    validate_controlled_variables,
)


DEFAULT_OUTPUT_DIR = "evidence/tuning_plan"
PLACEHOLDER_PREFIX = "<"


class TuningPlanError(ValueError):
    """Raised when a tuning plan YAML is missing or invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tuning plan report from parameter experiment winners."
        )
    )

    parser.add_argument(
        "--tuning-plan-path",
        required=True,
        help="Path to tuning plan YAML.",
    )
    parser.add_argument(
        "--experiment-name",
        default="log-triage-decision-engine",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tuning plan report outputs.",
    )

    return parser.parse_args()


def load_tuning_plan(path: Path | str) -> dict[str, Any]:
    plan_path = Path(path)

    if not plan_path.exists():
        raise TuningPlanError(f"Tuning plan not found: {plan_path}")

    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}

    required_keys = {
        "tuning_plan_name",
        "winner_metric",
        "experiments",
        "final_validation",
    }
    missing = sorted(required_keys - set(plan))
    if missing:
        raise TuningPlanError(
            "Tuning plan missing keys: " + ", ".join(missing)
        )

    if not plan.get("experiments"):
        raise TuningPlanError("Tuning plan must include at least one experiment.")

    return plan


def is_placeholder_comparison_group_id(comparison_group_id: str) -> bool:
    value = str(comparison_group_id).strip()
    return not value or value.startswith(PLACEHOLDER_PREFIX)


def read_metric(run: dict[str, Any], metric_name: str) -> float | None:
    value = run.get("metrics", {}).get(metric_name)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_winner_run(
    runs: list[dict[str, Any]],
    winner_metric: str,
) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs
        if read_metric(run, winner_metric) is not None
    ]

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda run: read_metric(run, winner_metric) or float("-inf"),
        reverse=True,
    )[0]


def summarize_parameter_experiment(
    *,
    label: str,
    comparison_group_id: str,
    changed_parameter: str,
    notes: str,
    runs: list[dict[str, Any]],
    winner_metric: str,
    require_valid_comparison_groups: bool,
) -> dict[str, Any]:
    base = {
        "label": label,
        "comparison_group_id": comparison_group_id,
        "changed_parameter": changed_parameter,
        "notes": notes,
        "raw_run_count": len(runs),
    }

    if is_placeholder_comparison_group_id(comparison_group_id):
        return {
            **base,
            "status": "blocked",
            "comparison_status": "blocked",
            "blocked_reason": "comparison_group_id is a placeholder",
            "winner_run_id": None,
            "winner_variant_name": None,
            "winner_metric_value": None,
            "winner_params": {},
        }

    if not runs:
        return {
            **base,
            "status": "blocked",
            "comparison_status": "blocked",
            "blocked_reason": "no runs found for comparison_group_id",
            "winner_run_id": None,
            "winner_variant_name": None,
            "winner_metric_value": None,
            "winner_params": {},
        }

    complete_runs = filter_complete_runs(runs)
    controlled_variable_validation = validate_controlled_variables(complete_runs)
    comparison_contract = build_comparison_contract(
        runs=complete_runs,
        controlled_variable_validation=controlled_variable_validation,
    )

    comparison_status = comparison_contract["status"]

    if require_valid_comparison_groups and comparison_status != "valid":
        failed_checks = [
            check_name
            for check_name, passed in comparison_contract["checks"].items()
            if not passed
        ]
        blocked_reason = (
            "comparison contract invalid: " + ", ".join(failed_checks)
            if failed_checks
            else "comparison contract invalid"
        )

        return {
            **base,
            "status": "blocked",
            "comparison_status": comparison_status,
            "blocked_reason": blocked_reason,
            "run_count": len(complete_runs),
            "comparison_contract": comparison_contract,
            "controlled_variable_validation": controlled_variable_validation,
            "winner_run_id": None,
            "winner_variant_name": None,
            "winner_metric_value": None,
            "winner_params": {},
        }

    winner = select_winner_run(complete_runs, winner_metric)

    if winner is None:
        return {
            **base,
            "status": "blocked",
            "comparison_status": comparison_status,
            "blocked_reason": f"missing winner metric: {winner_metric}",
            "run_count": len(complete_runs),
            "comparison_contract": comparison_contract,
            "controlled_variable_validation": controlled_variable_validation,
            "winner_run_id": None,
            "winner_variant_name": None,
            "winner_metric_value": None,
            "winner_params": {},
        }

    return {
        **base,
        "status": "valid",
        "comparison_status": comparison_status,
        "blocked_reason": None,
        "run_count": len(complete_runs),
        "comparison_contract": comparison_contract,
        "controlled_variable_validation": controlled_variable_validation,
        "winner_run_id": winner["run_id"],
        "winner_variant_name": winner["tags"].get("variant_name", "unknown"),
        "winner_metric_value": read_metric(winner, winner_metric),
        "winner_params": dict(winner.get("params", {})),
    }


def build_combined_candidate_draft(
    *,
    experiment_winners: list[dict[str, Any]],
) -> dict[str, Any]:
    draft_params = {}

    for winner in experiment_winners:
        changed_parameter = winner["changed_parameter"]
        draft_params[changed_parameter] = winner.get("winner_params", {})

    return {
        "source": "best winner per parameter experiment",
        "promotable": False,
        "reason": (
            "combined params must be validated together in final tournament"
        ),
        "draft_params": draft_params,
        "winner_run_ids": [
            winner["winner_run_id"]
            for winner in experiment_winners
            if winner.get("winner_run_id")
        ],
    }


def build_recommended_next_step(
    *,
    plan: dict[str, Any],
    blocked_experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    final_validation = plan.get("final_validation", {})

    if blocked_experiments:
        return {
            "type": "fix_blocked_parameter_experiments",
            "reason": (
                "One or more parameter experiments are blocked. Fix comparison "
                "groups before running the final candidate tournament."
            ),
            "blocked_labels": [
                experiment["label"] for experiment in blocked_experiments
            ],
            "recommended_experiment_name": final_validation.get(
                "recommended_experiment_name",
                "final_candidate_tournament",
            ),
        }

    return {
        "type": "run_final_candidate_tournament",
        "reason": (
            "individual parameter winners may interact differently when combined"
        ),
        "recommended_experiment_name": final_validation.get(
            "recommended_experiment_name",
            "final_candidate_tournament",
        ),
        "recommended_changed_variable": final_validation.get(
            "recommended_changed_variable",
            "candidate_config",
        ),
        "baseline_required": final_validation.get("baseline_required", True),
    }


def build_tuning_plan_report(
    *,
    plan: dict[str, Any],
    experiment_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    winner_metric = plan["winner_metric"]
    require_valid = plan.get("require_valid_comparison_groups", True)

    experiment_winners = [
        summary
        for summary in experiment_summaries
        if summary["status"] == "valid"
    ]
    blocked_experiments = [
        summary
        for summary in experiment_summaries
        if summary["status"] == "blocked"
    ]

    combined_candidate_draft = build_combined_candidate_draft(
        experiment_winners=experiment_winners,
    )

    if blocked_experiments:
        status = "blocked"
    elif experiment_winners:
        status = "requires_final_validation"
    else:
        status = "no_valid_experiments"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tuning_plan_name": plan["tuning_plan_name"],
        "policy_version": plan.get("policy_version", "unknown"),
        "plan_status": plan.get("status", "unknown"),
        "status": status,
        "winner_metric": winner_metric,
        "require_valid_comparison_groups": require_valid,
        "require_final_validation": plan.get("require_final_validation", True),
        "description": plan.get("description", ""),
        "experiment_summaries": experiment_summaries,
        "experiment_winners": experiment_winners,
        "blocked_experiments": blocked_experiments,
        "combined_candidate_draft": combined_candidate_draft,
        "recommended_next_step": build_recommended_next_step(
            plan=plan,
            blocked_experiments=blocked_experiments,
        ),
        "final_validation": plan.get("final_validation", {}),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Tuning Plan Report",
        "",
        f"- Tuning plan: `{report['tuning_plan_name']}`",
        f"- Status: `{report['status']}`",
        f"- Winner metric: `{report['winner_metric']}`",
        f"- Require valid comparison groups: `{report['require_valid_comparison_groups']}`",
        f"- Require final validation: `{report['require_final_validation']}`",
        "",
        "## Recommended Next Step",
        "",
        f"- Type: `{report['recommended_next_step']['type']}`",
        f"- Reason: {report['recommended_next_step']['reason']}",
        "",
        "## Combined Candidate Draft",
        "",
        f"- Source: `{report['combined_candidate_draft']['source']}`",
        f"- Promotable: `{report['combined_candidate_draft']['promotable']}`",
        f"- Reason: {report['combined_candidate_draft']['reason']}",
        "",
        "Parameter winners are not promotion candidates by themselves.",
        "",
        "## Experiment Winners",
        "",
    ]

    if not report["experiment_winners"]:
        lines.append("No valid experiment winners found.")
    else:
        for winner in report["experiment_winners"]:
            lines.extend(
                [
                    f"### `{winner['label']}`",
                    "",
                    f"- Comparison group: `{winner['comparison_group_id']}`",
                    f"- Changed parameter: `{winner['changed_parameter']}`",
                    f"- Winner run: `{winner['winner_run_id']}`",
                    f"- Winner variant: `{winner['winner_variant_name']}`",
                    f"- Winner {report['winner_metric']}: `{winner['winner_metric_value']}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Blocked Experiments",
            "",
        ]
    )

    if not report["blocked_experiments"]:
        lines.append("No blocked experiments.")
    else:
        for blocked in report["blocked_experiments"]:
            lines.extend(
                [
                    f"### `{blocked['label']}`",
                    "",
                    f"- Comparison group: `{blocked['comparison_group_id']}`",
                    f"- Reason: {blocked['blocked_reason']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "This report collects parameter experiment winners into a draft.",
            "It does not tag MLflow and does not call promote.py.",
            "Run the final candidate tournament before candidate selection.",
            "",
        ]
    )

    return "\n".join(lines)


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


def write_report(
    *,
    report: dict[str, Any],
    output_dir: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "tuning_plan_report.json"
    md_path = output_path / "tuning_plan_report.md"

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

    plan = load_tuning_plan(args.tuning_plan_path)
    experiment_id = get_experiment_id(args.experiment_name)

    experiment_summaries = []

    for experiment in plan["experiments"]:
        comparison_group_id = str(experiment["comparison_group_id"])

        if is_placeholder_comparison_group_id(comparison_group_id):
            runs: list[dict[str, Any]] = []
        else:
            runs = load_comparison_group_runs(
                experiment_id=experiment_id,
                comparison_group_id=comparison_group_id,
            )

        experiment_summaries.append(
            summarize_parameter_experiment(
                label=str(experiment["label"]),
                comparison_group_id=comparison_group_id,
                changed_parameter=str(experiment["changed_parameter"]),
                notes=str(experiment.get("notes", "")),
                runs=runs,
                winner_metric=str(plan["winner_metric"]),
                require_valid_comparison_groups=bool(
                    plan.get("require_valid_comparison_groups", True)
                ),
            )
        )

    report = build_tuning_plan_report(
        plan=plan,
        experiment_summaries=experiment_summaries,
    )

    json_path, md_path = write_report(
        report=report,
        output_dir=args.output_dir,
    )

    print(f"Tuning plan JSON written to: {json_path}")
    print(f"Tuning plan Markdown written to: {md_path}")
    print(f"Report status: {report['status']}")
    print(f"Valid experiment winners: {len(report['experiment_winners'])}")
    print(f"Blocked experiments: {len(report['blocked_experiments'])}")
    print(
        "Combined candidate draft promotable: "
        f"{report['combined_candidate_draft']['promotable']}"
    )


if __name__ == "__main__":
    main()
