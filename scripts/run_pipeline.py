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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


RUN_ID = build_run_id()


def log(message: str) -> None:
    print(f"[{utc_now()}] [run_id={RUN_ID}] {message}", flush=True)


def run_command(name: str, command: list[str]) -> None:
    log(f"START: {name}")
    log("COMMAND: " + shlex.join(command))

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    log(f"END: {name}")


def run_command_capture_json(name: str, command: list[str]) -> dict[str, Any]:
    log(f"START: {name}")
    log("COMMAND: " + shlex.join(command))

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if stderr:
        log(f"{name} stderr:\n{stderr}")

    if not stdout:
        raise RuntimeError(f"{name} returned empty stdout")

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{name} did not return valid JSON.\nstdout:\n{stdout}"
        ) from error

    log(f"{name} JSON output:")
    print(json.dumps(parsed, indent=2, sort_keys=True), flush=True)

    log(f"END: {name}")
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

    log("Pipeline started")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Python: {python}")

    before_artifact = get_latest_artifact_dir()
    if before_artifact is None:
        log("Latest artifact before training: none")
    else:
        log(f"Latest artifact before training: {before_artifact}")

    run_command(
        "Policy unit tests",
        [python, "-m", "pytest", "tests/test_policy.py", "-v"],
    )

    run_command(
        "Train model and build artifact",
        [python, "-m", "src.log_triage.train"],
    )

    after_artifact = get_latest_artifact_dir()
    log(f"Latest artifact after training: {after_artifact}")

    assert_fresh_artifact_created(before_artifact, after_artifact)

    run_command(
        "Artifact tests",
        [python, "-m", "pytest", "tests/test_artifact.py", "-v"],
    )

    run_command(
        "Prediction contract tests",
        [python, "-m", "pytest", "-m", "contract", "-v", "-s"],
    )

    log(f"Smoke input: {SAMPLE_LOG}")

    smoke_decision = run_command_capture_json(
        "Smoke prediction",
        [python, "-m", "src.log_triage.predict", SAMPLE_LOG],
    )

    validate_smoke_decision(smoke_decision)
    validate_smoke_policy(smoke_decision)

    run_command(
        "Full test suite",
        [python, "-m", "pytest", "-v", "-m", "not llm_integration"],
    )

    log("Pipeline completed successfully")
    log(f"Fresh artifact: {after_artifact}")
    log("Exit code: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
