from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.log_triage.artifact_version import (
    sha256_file,
    sha256_json,
)

DEFAULT_TRACE_PATH = Path("traces") / "prediction_trace.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_trace_record(decision_id: str, trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    with trace_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("decision_id") == decision_id:
                return record

    raise LookupError(f"Decision id not found in trace file: {decision_id}")


def load_manifest_from_trace(trace: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifact_path = Path(trace["artifact_path"])

    if not artifact_path.is_absolute():
        raise ValueError(
            f"artifact_path must be absolute for reliable audit: {artifact_path}"
        )

    manifest_path = artifact_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    return manifest_path, manifest


def recompute_live_hashes(
    artifact_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    files = manifest["files"]

    model_path = artifact_dir / files["model"]
    vectorizer_path = artifact_dir / files["vectorizer"]
    known_actions_path = artifact_dir / files["known_actions"]

    config_snapshot = {
        "training_config": manifest["training_config"],
        "decision_contract": manifest["decision_contract"],
    }

    raw_logs_path = manifest["training_config"]["config"]["raw_logs_path"]

    return {
        "model_sha256": sha256_file(model_path),
        "vectorizer_sha256": sha256_file(vectorizer_path),
        "known_actions_sha256": sha256_file(known_actions_path),
        "config_sha256": sha256_json(config_snapshot),
        "training_data_sha256": sha256_file(raw_logs_path),
    }


def compare_hashes(
    expected: dict[str, str],
    actual: dict[str, str],
) -> list[str]:
    mismatches = []

    for key, expected_value in expected.items():
        actual_value = actual.get(key)

        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value} actual={actual_value}"
            )

    return mismatches


def print_explanation(record: dict[str, Any]) -> None:
    decision = record["decision"]
    trace = decision["trace"]

    manifest_path, manifest = load_manifest_from_trace(trace)
    artifact_dir = manifest_path.parent

    trace_hashes = {
        "model_sha256": trace["model_sha256"],
        "vectorizer_sha256": trace["vectorizer_sha256"],
        "known_actions_sha256": trace["known_actions_sha256"],
        "config_sha256": trace["config_sha256"],
        "training_data_sha256": trace["training_data_sha256"],
    }

    manifest_hashes = manifest["hashes"]
    live_hashes = recompute_live_hashes(artifact_dir, manifest)

    trace_vs_manifest_mismatches = compare_hashes(trace_hashes, manifest_hashes)
    trace_vs_live_mismatches = compare_hashes(trace_hashes, live_hashes)

    print("\n=== Decision Explanation ===")
    print(f"Decision ID:      {trace['decision_id']}")
    print(f"Decision time:    {trace['created_at']}")
    print(f"Predicted action: {decision['predicted_action']}")
    print(f"Confidence:       {decision['confidence']}")
    print(f"Risk level:       {decision['risk_level']}")
    print(f"Requires approval:{decision['requires_approval']}")
    print(f"Reason:           {decision['reason']}")

    print("\n=== Artifact Trace ===")
    print(f"Artifact ID:      {trace['artifact_id']}")
    print(f"Artifact path:    {trace['artifact_path']}")
    print(f"Manifest path:    {manifest_path}")
    print(f"Run ID:           {trace['run_id']}")
    print(f"Git SHA:          {trace['git_sha']}")

    print("\n=== Hashes ===")
    print(f"model_sha256:          {trace['model_sha256']}")
    print(f"vectorizer_sha256:     {trace['vectorizer_sha256']}")
    print(f"known_actions_sha256:  {trace['known_actions_sha256']}")
    print(f"config_sha256:         {trace['config_sha256']}")
    print(f"training_data_sha256:  {trace['training_data_sha256']}")

    print("\n=== Verification ===")

    if trace_vs_manifest_mismatches:
        print("ALERT: trace does not match manifest hashes")
        for mismatch in trace_vs_manifest_mismatches:
            print(f"- {mismatch}")
    else:
        print("OK: trace hashes match manifest hashes")

    if trace_vs_live_mismatches:
        print("ALERT: trace does not match live artifact/data hashes")
        for mismatch in trace_vs_live_mismatches:
            print(f"- {mismatch}")
    else:
        print("OK: trace hashes match live artifact/data hashes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a saved prediction decision by decision_id."
    )

    parser.add_argument(
        "decision_id",
        help="Decision ID from traces/prediction_trace.jsonl",
    )

    parser.add_argument(
        "--trace-path",
        default=str(DEFAULT_TRACE_PATH),
        help="Path to prediction trace JSONL file.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        record = load_trace_record(
            decision_id=args.decision_id,
            trace_path=Path(args.trace_path),
        )
        print_explanation(record)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
