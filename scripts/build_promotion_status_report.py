from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.promote import (
    list_mlflow_artifact_paths,
    load_full_run_for_evidence,
)
from src.log_triage.promotion_evidence import (
    DEFAULT_PROMOTION_EVIDENCE_CONTRACT_PATH,
    load_promotion_evidence_contract,
    validate_promotion_evidence,
)


DEFAULT_OUTPUT_DIR = "evidence/promotion_status"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a promotion status report from MLflow runs."
    )

    parser.add_argument(
        "--experiment-name",
        default="log-triage-decision-engine",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--promotion-evidence-contract-path",
        default=DEFAULT_PROMOTION_EVIDENCE_CONTRACT_PATH,
        help="Promotion evidence contract YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    return parser.parse_args()


def get_experiment_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise SystemExit(f"Experiment not found: {experiment_name}")

    return experiment.experiment_id


def tag_value(run: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = run.get("tags", {}).get(key)

    if value in {None, ""}:
        return default

    return str(value)


def metric_value(run: dict[str, Any], key: str) -> float | None:
    value = run.get("metrics", {}).get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_run_status(
    *,
    run: dict[str, Any],
    evidence_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "candidate_status": tag_value(run, "candidate_status", "missing"),
        "candidate": tag_value(run, "candidate", "false"),
        "current_candidate": tag_value(run, "current_candidate", "false"),
        "promotion_reason": tag_value(run, "promotion_reason", "missing"),
        "promotion_rejection_reason": tag_value(
            run,
            "promotion_rejection_reason",
            "",
        ),
        "promotion_evaluated_at": tag_value(
            run,
            "promotion_evaluated_at",
            "",
        ),
        "run_owner": tag_value(run, "run_owner", "missing"),
        "variant_name": tag_value(run, "variant_name", "unknown"),
        "comparison_group_id": tag_value(
            run,
            "comparison_group_id",
            "unknown",
        ),
        "metrics": {
            "f1_macro": metric_value(run, "f1_macro"),
            "offline_decision_latency_p95_ms": metric_value(
                run,
                "offline_decision_latency_p95_ms",
            ),
            "low_confidence_rate": metric_value(
                run,
                "low_confidence_rate",
            ),
            "approval_required_rate": metric_value(
                run,
                "approval_required_rate",
            ),
        },
        "promotion_evidence": evidence_result,
    }


def build_status_policy_checks(
    *,
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    current_candidates = [
        item for item in summaries
        if item["current_candidate"] == "true"
    ]

    selected_runs = [
        item for item in summaries
        if item["candidate_status"] == "selected"
    ]

    selected_with_failed_evidence = [
        item["run_id"] for item in selected_runs
        if item["promotion_evidence"]["status"] != "passed"
    ]

    rejected_without_reason = [
        item["run_id"] for item in summaries
        if item["candidate_status"] == "rejected"
        and not item["promotion_reason"]
        and not item["promotion_rejection_reason"]
    ]

    checks = {
        "single_current_candidate": len(current_candidates) <= 1,
        "selected_runs_have_passed_evidence": not selected_with_failed_evidence,
        "rejected_runs_have_reason": not rejected_without_reason,
    }

    status = "passed" if all(checks.values()) else "failed"

    return {
        "status": status,
        "checks": checks,
        "current_candidate_run_ids": [
            item["run_id"] for item in current_candidates
        ],
        "selected_with_failed_evidence": selected_with_failed_evidence,
        "rejected_without_reason": rejected_without_reason,
    }


def build_mlflow_filter_examples() -> dict[str, str]:
    return {
        "selected": "tags.candidate_status = 'selected'",
        "eligible": "tags.candidate_status = 'eligible'",
        "rejected": "tags.candidate_status = 'rejected'",
        "superseded": "tags.candidate_status = 'superseded'",
        "not_evaluated": "tags.candidate_status = 'not_evaluated'",
        "current_candidate": "tags.current_candidate = 'true'",
        "candidate_history": "tags.candidate = 'true'",
    }


def build_promotion_status_report(
    *,
    experiment_name: str,
    run_summaries: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(
        item["candidate_status"] for item in run_summaries
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "promotion_evidence_contract": {
            "contract_name": contract["contract_name"],
            "contract_version": contract["contract_version"],
            "contract_path": contract["contract_path"],
            "status": contract.get("status", "unknown"),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "status_policy_checks": build_status_policy_checks(
            summaries=run_summaries,
        ),
        "mlflow_filter_examples": build_mlflow_filter_examples(),
        "runs": run_summaries,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Promotion Status Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Experiment: `{report['experiment_name']}`",
        "",
        "## Status Counts",
        "",
        "| candidate_status | count |",
        "|---|---:|",
    ]

    for status, count in report["status_counts"].items():
        lines.append(f"| `{status}` | {count} |")

    checks = report["status_policy_checks"]

    lines.extend(
        [
            "",
            "## Status Policy Checks",
            "",
            f"- Overall status: `{checks['status']}`",
            f"- Single current candidate: `{checks['checks']['single_current_candidate']}`",
            f"- Selected runs have passed evidence: `{checks['checks']['selected_runs_have_passed_evidence']}`",
            f"- Rejected runs have reason: `{checks['checks']['rejected_runs_have_reason']}`",
            f"- Current candidate run IDs: `{checks['current_candidate_run_ids']}`",
            "",
            "## MLflow Filter Examples",
            "",
        ]
    )

    for label, filter_query in report["mlflow_filter_examples"].items():
        lines.append(f"- `{label}`: `{filter_query}`")

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Status | Variant | Run ID | Candidate | Current | Evidence | Reason |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for run in report["runs"]:
        lines.append(
            "| {status} | {variant} | {run_id} | {candidate} | {current} | {evidence} | {reason} |".format(
                status=run["candidate_status"],
                variant=run["variant_name"],
                run_id=run["run_id"],
                candidate=run["candidate"],
                current=run["current_candidate"],
                evidence=run["promotion_evidence"]["status"],
                reason=run["promotion_reason"],
            )
        )

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report summarizes promotion status visibility in MLflow.",
            "It does not select, reject, or promote runs.",
            "",
        ]
    )

    return "\n".join(lines)


def load_run_summaries_from_mlflow(
    *,
    experiment_id: str,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas",
    )

    if runs_df.empty:
        return []

    summaries = []

    for run_id in runs_df["run_id"].tolist():
        full_run = load_full_run_for_evidence(str(run_id))
        artifact_paths = list_mlflow_artifact_paths(run_id=str(run_id))

        # Include basenames so nested MLflow paths such as
        # decision_artifact/<id>/manifest.json satisfy required_artifacts.
        artifact_paths_for_validation = list(
            {
                *artifact_paths,
                *[Path(path).name for path in artifact_paths],
            }
        )

        evidence_result = validate_promotion_evidence(
            run=full_run,
            artifact_paths=artifact_paths_for_validation,
            contract=contract,
        )

        summaries.append(
            summarize_run_status(
                run=full_run,
                evidence_result=evidence_result,
            )
        )

    return summaries


def write_report(
    *,
    report: dict[str, Any],
    output_dir: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "promotion_status_report.json"
    md_path = output_path / "promotion_status_report.md"

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

    contract = load_promotion_evidence_contract(
        args.promotion_evidence_contract_path
    )

    experiment_id = get_experiment_id(args.experiment_name)

    summaries = load_run_summaries_from_mlflow(
        experiment_id=experiment_id,
        contract=contract,
    )

    report = build_promotion_status_report(
        experiment_name=args.experiment_name,
        run_summaries=summaries,
        contract=contract,
    )

    json_path, md_path = write_report(
        report=report,
        output_dir=args.output_dir,
    )

    print(f"Promotion status JSON written to: {json_path}")
    print(f"Promotion status Markdown written to: {md_path}")
    print(f"Status policy checks: {report['status_policy_checks']['status']}")

    if report["status_policy_checks"]["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
