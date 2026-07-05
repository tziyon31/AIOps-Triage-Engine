import argparse
import hashlib
import json
import os
from pathlib import Path

from mlflow.tracking import MlflowClient


def compute_artifact_sha256_from_manifest(manifest: dict) -> str:
    artifact_identity = {
        "schema_version": manifest.get("schema_version"),
        "artifact_type": manifest.get("artifact_type"),
        "files": manifest.get("files", {}),
        "hashes": manifest.get("hashes", {}),
        "decision_contract": manifest.get("decision_contract", {}),
    }

    encoded = json.dumps(
        artifact_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "log-triage-decision-engine"),
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set")

    client = MlflowClient(tracking_uri=tracking_uri)

    if args.run_id:
        run = client.get_run(args.run_id)
    else:
        experiment = client.get_experiment_by_name(args.experiment_name)
        if experiment is None:
            raise SystemExit(f"Experiment not found: {args.experiment_name}")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise SystemExit("No MLflow runs found")

        run = runs[0]

    tags = run.data.tags
    artifact_id = tags["artifact_id"]

    manifest_path = Path("artifacts") / artifact_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest["hashes"]

    expected_artifact_sha256 = compute_artifact_sha256_from_manifest(manifest)

    checks = {
        "git_sha": (tags["git_sha"], manifest["git_sha"]),
        "config_sha256": (tags["config_sha256"], hashes["config_sha256"]),
        "training_config_sha256": (
            tags["training_config_sha256"],
            hashes["training_config_sha256"],
        ),
        "policy_sha256": (tags["policy_sha256"], hashes["policy_sha256"]),
        "training_data_sha256": (
            tags["training_data_sha256"],
            hashes["training_data_sha256"],
        ),
        "model_sha256": (tags["model_sha256"], hashes["model_sha256"]),
        "vectorizer_sha256": (
            tags["vectorizer_sha256"],
            hashes["vectorizer_sha256"],
        ),
        "known_actions_sha256": (
            tags["known_actions_sha256"],
            hashes["known_actions_sha256"],
        ),
        "artifact_id": (tags["artifact_id"], manifest["artifact_id"]),
        "artifact_sha256": (tags["artifact_sha256"], expected_artifact_sha256),
    }

    failed = []

    print(f"Checking MLflow run: {run.info.run_id}")
    print(f"Manifest: {manifest_path}")
    print()

    for name, (mlflow_value, manifest_value) in checks.items():
        ok = mlflow_value == manifest_value
        status = "OK" if ok else "MISMATCH"
        print(f"{status:8} {name}")

        if not ok:
            failed.append((name, mlflow_value, manifest_value))

    if failed:
        print("\nMismatches:")
        for name, mlflow_value, manifest_value in failed:
            print(f"- {name}")
            print(f"  mlflow:   {mlflow_value}")
            print(f"  manifest: {manifest_value}")
        raise SystemExit(1)

    print("\nAll MLflow hash tags match the manifest.")


if __name__ == "__main__":
    main()
