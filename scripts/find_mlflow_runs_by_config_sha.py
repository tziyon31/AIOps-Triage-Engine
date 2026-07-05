import argparse
import os

from mlflow.tracking import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-sha", default=None)
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "log-triage-decision-engine"),
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set")

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {args.experiment_name}")

    config_sha = args.config_sha

    if config_sha is None:
        latest_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )

        if not latest_runs:
            raise SystemExit("No runs found")

        config_sha = latest_runs[0].data.tags["config_sha256"]

    matching_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.config_sha256 = '{config_sha}'",
        order_by=["attributes.start_time DESC"],
    )

    print("config_sha256:", config_sha)
    print("matching_run_count:", len(matching_runs))
    print()

    for run in matching_runs:
        tags = run.data.tags
        metrics = run.data.metrics

        print("run_id:", run.info.run_id)
        print("artifact_id:", tags.get("artifact_id"))
        print("artifact_version:", tags.get("artifact_version"))
        print("policy_sha256:", tags.get("policy_sha256"))
        print("training_data_sha256:", tags.get("training_data_sha256"))
        print("candidate_status:", tags.get("candidate_status"))
        print("accuracy:", metrics.get("accuracy"))
        print("macro_f1:", metrics.get("f1_macro"))
        print("-" * 80)


if __name__ == "__main__":
    main()
