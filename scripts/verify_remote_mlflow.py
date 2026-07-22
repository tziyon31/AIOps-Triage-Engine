from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def main() -> None:
    tracking_uri = require_env("MLFLOW_TRACKING_URI")

    # Default to a post-proxy experiment name. Experiments created before
    # --serve-artifacts keep their old s3:// artifact root forever.
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "stage56_remote_mlflow_smoke_proxy",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment not found after set_experiment: {experiment_name}")

    print(f"experiment_id={experiment.experiment_id}")
    print(f"experiment_artifact_location={experiment.artifact_location}")

    if experiment.artifact_location.startswith("s3://"):
        raise RuntimeError(
            "Experiment artifact_location is still s3:// so the local client will "
            "try to upload directly to S3 and require boto3.\n"
            "This usually means the experiment was created before artifact proxy "
            "mode. Create/use a NEW experiment name, for example:\n"
            "  export MLFLOW_EXPERIMENT_NAME=stage56_remote_mlflow_smoke_proxy_v2\n"
            f"Current experiment: {experiment_name}\n"
            f"Current artifact_location: {experiment.artifact_location}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "remote_mlflow_smoke.txt"
        artifact_path.write_text(
            "Stage 5.6 remote MLflow smoke verification.\n",
            encoding="utf-8",
        )

        with mlflow.start_run(run_name="stage56_remote_mlflow_smoke") as run:
            print(f"run_artifact_uri={run.info.artifact_uri}")

            if run.info.artifact_uri.startswith("s3://"):
                raise RuntimeError(
                    "Run artifact_uri is s3:// — client-side S3 upload will fail "
                    "without local boto3. Use a new experiment created under "
                    "mlflow-artifacts:/ proxy mode."
                )

            mlflow.set_tags(
                {
                    "stage": "5.6",
                    "module": "3",
                    "run_source": "manual_remote_smoke",
                    "official": "false",
                    "verification_target": "remote_mlflow_postgres_s3",
                }
            )
            mlflow.log_param("verification_type", "remote_mlflow_smoke")
            mlflow.log_metric("smoke_success", 1.0)
            mlflow.log_artifact(str(artifact_path), artifact_path="verification")

            print("Remote MLflow smoke verification completed")
            print(f"tracking_uri={tracking_uri}")
            print(f"experiment_name={experiment_name}")
            print(f"run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
