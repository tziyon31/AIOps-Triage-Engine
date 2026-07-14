from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_SELECTION_POLICY_PATH = (
    PROJECT_ROOT / "config" / "candidate_selection.yaml"
)

METRIC_CHECKS = (
    ("f1_macro", "minimum_f1_macro", "gte"),
    ("combined_weakest_class_recall", "minimum_weakest_class_recall", "gte"),
    ("low_confidence_rate", "maximum_low_confidence_rate", "lte"),
    ("approval_required_rate", "maximum_approval_required_rate", "lte"),
    (
        "offline_decision_latency_p95_ms",
        "maximum_offline_decision_latency_p95_ms",
        "lte",
    ),
)


class CandidateSelectionPolicyError(ValueError):
    """Raised when candidate_selection.yaml is missing or invalid."""


def load_candidate_selection_policy(
    path: Path | str = DEFAULT_CANDIDATE_SELECTION_POLICY_PATH,
) -> dict[str, Any]:
    policy_path = Path(path)

    if not policy_path.exists():
        raise CandidateSelectionPolicyError(
            f"Candidate selection policy not found: {policy_path}"
        )

    raw_config = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}

    required_keys = {
        "policy_name",
        "policy_version",
        "baseline",
        "thresholds",
        "decisions",
    }
    missing = sorted(required_keys - set(raw_config))
    if missing:
        raise CandidateSelectionPolicyError(
            "Candidate selection policy missing keys: "
            + ", ".join(missing)
        )

    baseline = raw_config.get("baseline") or {}
    if "required" not in baseline:
        raise CandidateSelectionPolicyError(
            "Candidate selection policy missing baseline.required"
        )
    if "minimum_f1_improvement_over_baseline" not in baseline:
        raise CandidateSelectionPolicyError(
            "Candidate selection policy missing "
            "baseline.minimum_f1_improvement_over_baseline"
        )

    thresholds = raw_config.get("thresholds") or {}
    for _, threshold_key, _ in METRIC_CHECKS:
        if threshold_key not in thresholds:
            raise CandidateSelectionPolicyError(
                f"Candidate selection policy missing thresholds.{threshold_key}"
            )

    decisions = raw_config.get("decisions") or {}
    for decision_key in ("pass", "fail", "review"):
        if decision_key not in decisions:
            raise CandidateSelectionPolicyError(
                f"Candidate selection policy missing decisions.{decision_key}"
            )

    return raw_config


def read_metric(run: dict[str, Any], metric_name: str) -> float | None:
    value = run.get("metrics", {}).get(metric_name)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_passes(*, value: float, threshold: float, comparison: str) -> bool:
    if comparison == "gte":
        return value >= threshold
    if comparison == "lte":
        return value <= threshold
    raise ValueError(f"Unsupported comparison: {comparison}")


def evaluate_candidate_run(
    run: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate absolute threshold checks for one candidate run."""
    decisions = policy["decisions"]
    thresholds = policy["thresholds"]

    checks: list[dict[str, Any]] = []
    missing_metrics: list[str] = []
    failed_checks: list[str] = []

    for metric_name, threshold_key, comparison in METRIC_CHECKS:
        threshold = float(thresholds[threshold_key])
        value = read_metric(run, metric_name)

        if value is None:
            missing_metrics.append(metric_name)
            checks.append(
                {
                    "metric_name": metric_name,
                    "threshold_key": threshold_key,
                    "comparison": comparison,
                    "threshold": threshold,
                    "value": None,
                    "status": "missing",
                }
            )
            continue

        passed = _check_passes(
            value=value,
            threshold=threshold,
            comparison=comparison,
        )
        status = "passed" if passed else "failed"
        if not passed:
            failed_checks.append(metric_name)

        checks.append(
            {
                "metric_name": metric_name,
                "threshold_key": threshold_key,
                "comparison": comparison,
                "threshold": threshold,
                "value": value,
                "status": status,
            }
        )

    if missing_metrics:
        decision = decisions["review"]
        reason = (
            "missing_required_metrics: " + ", ".join(sorted(missing_metrics))
        )
    elif failed_checks:
        decision = decisions["fail"]
        reason = "failed_threshold_checks: " + ", ".join(sorted(failed_checks))
    else:
        decision = decisions["pass"]
        reason = "all_threshold_checks_passed"

    return {
        "run_id": run.get("run_id"),
        "decision": decision,
        "reason": reason,
        "checks": checks,
        "missing_metrics": missing_metrics,
        "failed_checks": failed_checks,
    }


def compare_to_baseline(
    *,
    candidate_run: dict[str, Any],
    baseline_run: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Compare candidate f1_macro improvement against the baseline run."""
    decisions = policy["decisions"]
    baseline_policy = policy["baseline"]
    required_improvement = float(
        baseline_policy["minimum_f1_improvement_over_baseline"]
    )

    candidate_f1 = read_metric(candidate_run, "f1_macro")
    baseline_f1 = read_metric(baseline_run, "f1_macro")

    if candidate_f1 is None or baseline_f1 is None:
        missing = []
        if candidate_f1 is None:
            missing.append("candidate.f1_macro")
        if baseline_f1 is None:
            missing.append("baseline.f1_macro")

        return {
            "decision": decisions["review"],
            "reason": "missing_f1_macro_for_baseline_comparison: "
            + ", ".join(missing),
            "candidate_run_id": candidate_run.get("run_id"),
            "baseline_run_id": baseline_run.get("run_id"),
            "candidate_f1_macro": candidate_f1,
            "baseline_f1_macro": baseline_f1,
            "required_improvement": required_improvement,
            "actual_improvement": None,
            "minimum_required_f1_macro": None,
        }

    minimum_required_f1 = baseline_f1 + required_improvement
    actual_improvement = candidate_f1 - baseline_f1
    passed = candidate_f1 >= minimum_required_f1

    if passed:
        decision = decisions["pass"]
        reason = "sufficient_improvement_over_baseline"
    else:
        decision = decisions["fail"]
        reason = "insufficient_improvement_over_baseline"

    return {
        "decision": decision,
        "reason": reason,
        "candidate_run_id": candidate_run.get("run_id"),
        "baseline_run_id": baseline_run.get("run_id"),
        "candidate_f1_macro": candidate_f1,
        "baseline_f1_macro": baseline_f1,
        "required_improvement": required_improvement,
        "actual_improvement": actual_improvement,
        "minimum_required_f1_macro": minimum_required_f1,
    }
