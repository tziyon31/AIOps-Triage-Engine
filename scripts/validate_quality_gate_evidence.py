"""Validate generated Quality Gate evidence files."""

from __future__ import annotations

import json
from pathlib import Path

EVIDENCE_DIR = Path("evidence/quality_gate")

REQUIRED_FILES = [
    "quality_gate_report.json",
    "quality_gate_report.md",
    "sample_decision.json",
    "manifest_hashes.json",
]

REQUIRED_CHECKS = [
    "policy_tests",
    "artifact_tests",
    "quality_gate_report_tests",
    "prediction_contract_tests",
    "smoke_prediction",
    "traceability_integration",
    "deterministic_test_suite",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not (EVIDENCE_DIR / filename).exists()
    ]

    if missing_files:
        raise SystemExit(
            f"Missing required Quality Gate evidence files: {missing_files}"
        )

    report = load_json(EVIDENCE_DIR / "quality_gate_report.json")

    required_top_level_fields = [
        "quality_gate_report_version",
        "report_generated_at",
        "pipeline_run_id",
        "git_sha",
        "status",
        "artifact",
        "checks",
        "sample_decision",
        "limitations",
    ]

    missing_fields = [
        field for field in required_top_level_fields if not report.get(field)
    ]

    if missing_fields:
        raise SystemExit(
            f"Missing required Quality Gate report fields: {missing_fields}"
        )

    if report["status"] != "passed":
        raise SystemExit(f"Quality Gate status is not passed: {report['status']}")

    artifact = report["artifact"]

    required_artifact_fields = [
        "artifact_id",
        "model_version",
        "created_at",
        "run_id",
        "manifest_path",
        "hashes",
    ]

    missing_artifact_fields = [
        field for field in required_artifact_fields if not artifact.get(field)
    ]

    if missing_artifact_fields:
        raise SystemExit(
            f"Missing required artifact fields: {missing_artifact_fields}"
        )

    checks = report["checks"]

    missing_or_failed_checks = [
        check for check in REQUIRED_CHECKS if checks.get(check) != "passed"
    ]

    if missing_or_failed_checks:
        raise SystemExit(
            f"Missing or failed Quality Gate checks: {missing_or_failed_checks}"
        )

    sample_decision = load_json(EVIDENCE_DIR / "sample_decision.json")

    if not sample_decision.get("predicted_action"):
        raise SystemExit("sample_decision.json is missing predicted_action")

    if "confidence" not in sample_decision:
        raise SystemExit("sample_decision.json is missing confidence")

    manifest_hashes = load_json(EVIDENCE_DIR / "manifest_hashes.json")

    required_hashes = [
        "model_sha256",
        "vectorizer_sha256",
        "known_actions_sha256",
        "config_sha256",
        "training_data_sha256",
    ]

    missing_hashes = [
        key for key in required_hashes if not manifest_hashes.get(key)
    ]

    if missing_hashes:
        raise SystemExit(f"Missing manifest hashes: {missing_hashes}")

    print("Quality Gate evidence validation passed.")


if __name__ == "__main__":
    main()
