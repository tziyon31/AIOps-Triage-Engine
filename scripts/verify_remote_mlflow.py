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

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "stage56_remote_mlflow_smoke",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "remote_mlflow_smoke.txt"
        artifact_path.write_text(
            "Stage 5.6 remote MLflow smoke verification.\n",
            encoding="utf-8",
        )

        with mlflow.start_run(run_name="stage56_remote_mlflow_smoke") as run:
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
