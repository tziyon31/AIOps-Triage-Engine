from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LOG = (
    "2026-05-03 09:12:11 ERROR payments "
    "db timeout after retries cpu 93 memory 84"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.log_triage.quality_gate_report import (
    build_quality_gate_report,
    write_quality_gate_report,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


RUN_ID = build_run_id()


def log(message: str) -> None:
    print(f"[{utc_now()}] [run_id={RUN_ID}] {message}", flush=True)


def run_command(
    command: list[str],
    description: str,
    *,
    evidence_file: Path | None = None,
) -> None:
    log(f"START: {description}")
    log(f"COMMAND: {shlex.join(command)}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.stdout:
        print(result.stdout, end="")

    if evidence_file is not None:
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_text(result.stdout or "", encoding="utf-8")

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
        )

    log(f"END: {description}")


def run_command_capture_json(
    command: list[str],
    description: str,
    *,
    evidence_file: Path | None = None,
) -> dict[str, Any]:
    log(f"START: {description}")
    log(f"COMMAND: {shlex.join(command)}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.stdout:
        print(result.stdout, end="")

    if evidence_file is not None:
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_text(result.stdout or "", encoding="utf-8")

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
        )

    stdout = (result.stdout or "").strip()
    if not stdout:
        raise RuntimeError(f"{description} returned empty stdout")

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{description} did not return valid JSON. Output was:\n{result.stdout}"
        ) from exc

    log(f"{description} JSON output:")
    print(json.dumps(parsed, indent=2, sort_keys=True), flush=True)

    log(f"END: {description}")
    return parsed


def get_latest_artifact_dir() -> Path | None:
    from src.log_triage.artifact_version import find_latest_artifact_dir
    from src.log_triage.config import load_training_config

    training_config = load_training_config()
    return find_latest_artifact_dir(training_config)


def assert_fresh_artifact_created(before: Path | None, after: Path | None) -> None:
    if after is None:
        raise RuntimeError("Training did not produce any artifact")

    if before is not None and before == after:
        raise RuntimeError(
            f"Training did not produce a fresh artifact. "
            f"Latest artifact is still: {after}"
        )


def validate_smoke_decision(decision: dict[str, Any]) -> None:
    from src.log_triage.schemas import DecisionObject

    DecisionObject.model_validate(decision)
    log("Smoke Decision Object passed schema validation")


def validate_smoke_policy(decision: dict[str, Any]) -> None:
    from src.log_triage.policy import validate

    policy_result = validate(decision)

    required_fields = {"allowed", "reason", "modified_decision"}
    missing_fields = required_fields - set(policy_result)
    if missing_fields:
        raise RuntimeError(f"PolicyResult missing fields: {missing_fields}")

    if policy_result["allowed"] is False:
        raise RuntimeError(
            f"Smoke prediction was blocked by policy: {policy_result['reason']}"
        )

    log("Smoke PolicyResult:")
    print(json.dumps(policy_result, indent=2, sort_keys=True), flush=True)


def main() -> int:
    os.environ.setdefault("LOG_TRIAGE_DISABLE_LLM", "1")

    python = sys.executable
    run_id = RUN_ID
    checks: dict[str, str] = {}
    sample_decision: dict[str, Any] | None = None

    evidence_dir = Path("evidence/quality_gate")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    log("Pipeline started")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Python: {python}")

    before_artifact = get_latest_artifact_dir()
    if before_artifact is None:
        log("Latest artifact before training: none")
    else:
        log(f"Latest artifact before training: {before_artifact}")

    run_command(
        [python, "-m", "pytest", "tests/test_policy.py", "-v"],
        "Policy unit tests",
        evidence_file=evidence_dir / "policy_tests.txt",
    )
    checks["policy_tests"] = "passed"

    run_command(
        [python, "-m", "src.log_triage.train"],
        "Train model and build artifact",
    )
    checks["training_and_artifact_build"] = "passed"

    after_artifact = get_latest_artifact_dir()
    log(f"Latest artifact after training: {after_artifact}")

    assert_fresh_artifact_created(before_artifact, after_artifact)
    checks["fresh_artifact_created"] = "passed"

    run_command(
        [python, "-m", "pytest", "tests/test_artifact.py", "-v"],
        "Artifact tests",
        evidence_file=evidence_dir / "artifact_tests.txt",
    )
    checks["artifact_tests"] = "passed"

    run_command(
        [python, "-m", "pytest", "tests/test_quality_gate_report.py", "-v"],
        "Quality Gate report tests",
        evidence_file=evidence_dir / "quality_gate_report_tests.txt",
    )
    checks["quality_gate_report_tests"] = "passed"

    run_command(
        [python, "-m", "pytest", "tests/test_prediction_contract.py", "-v"],
        "Prediction contract tests",
        evidence_file=evidence_dir / "prediction_contract_tests.txt",
    )
    checks["prediction_contract_tests"] = "passed"

    log(f"Smoke input: {SAMPLE_LOG}")

    smoke_decision = run_command_capture_json(
        [python, "-m", "src.log_triage.predict", SAMPLE_LOG],
        "Smoke prediction",
        evidence_file=evidence_dir / "smoke_prediction_output.json",
    )
    sample_decision = smoke_decision
    checks["smoke_prediction"] = "passed"

    validate_smoke_decision(smoke_decision)
    checks["smoke_decision_schema_validation"] = "passed"

    validate_smoke_policy(smoke_decision)
    checks["smoke_policy_validation"] = "passed"

    run_command(
        [
            python,
            "-m",
            "pytest",
            "tests/test_explain.py",
            "tests/test_traceability_integration.py",
            "-v",
        ],
        "Traceability integration tests",
        evidence_file=evidence_dir / "traceability_integration_tests.txt",
    )
    checks["traceability_integration"] = "passed"

    run_command(
        [
            python,
            "-m",
            "pytest",
            "-v",
            "-m",
            "not integration and not llm_integration",
        ],
        "Full deterministic test suite",
        evidence_file=evidence_dir / "deterministic_test_suite.txt",
    )
    checks["deterministic_test_suite"] = "passed"

    if after_artifact is None:
        raise RuntimeError("Cannot write Quality Gate report without a fresh artifact")

    report = build_quality_gate_report(
        pipeline_run_id=run_id,
        artifact_dir=after_artifact,
        checks=checks,
        sample_decision=sample_decision,
        status="passed",
    )
    write_quality_gate_report(report=report)
    log("Quality Gate report written to evidence/quality_gate/")

    run_command(
        [python, "scripts/validate_quality_gate_evidence.py"],
        "Validate Quality Gate evidence",
    )

    log("Pipeline completed successfully")
    log(f"Fresh artifact: {after_artifact}")
    log("Exit code: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
