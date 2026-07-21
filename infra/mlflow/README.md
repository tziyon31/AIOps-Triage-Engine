# MLflow Tracking Server Container

## Purpose

This directory provides a dedicated infrastructure container for the Stage 5.6 remote MLflow evidence store.

The container runs the MLflow Tracking Server that will later be deployed to Render Starter and backed by:

- Aiven PostgreSQL `mlflow_db` for metadata
- S3 for artifact storage

This module is infrastructure only. It does not change application runtime behavior.

## Why dedicated Dockerfile

The MLflow Tracking Server is an Infrastructure Adapter, not Application/Core logic.

A dedicated Dockerfile keeps:

- infrastructure dependencies isolated from application runtime code
- deployment configuration separate from `train.py`, `predict.py`, and `DecisionService`
- future Render deployment simple and reviewable

## Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `MLFLOW_BACKEND_STORE_URI` | Yes | Postgres/SQLite backend store URI |
| `MLFLOW_ARTIFACT_ROOT` | Yes | Artifact root, e.g. `s3://bucket/prefix` |
| `AWS_ACCESS_KEY_ID` | Yes for S3 | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | Yes for S3 | S3 secret key |
| `AWS_DEFAULT_REGION` | Yes for S3 | AWS region |
| `PORT` | Optional | Render provides this at runtime |
| `HOST` | Optional | Defaults to `0.0.0.0` |

Expected production values:

- `MLFLOW_BACKEND_STORE_URI` should point to Aiven PostgreSQL / `mlflow_db`
- `MLFLOW_ARTIFACT_ROOT` should point to `s3://<bucket>/<prefix>`

Secrets must be configured in Render or GitHub Actions secrets. They must never be committed to the repository.

## Local build

```bash
docker build \
  -f infra/mlflow/Dockerfile \
  -t aiops-mlflow-server:local .
```

## Local smoke run

```bash
docker run --rm -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI="sqlite:////tmp/mlflow.db" \
  -e MLFLOW_ARTIFACT_ROOT="/tmp/mlflow-artifacts" \
  aiops-mlflow-server:local
```

This local smoke command verifies that the container starts and exposes the MLflow server process.

It does **not** verify:

- Aiven PostgreSQL connectivity
- S3 artifact upload
- Render deployment
- Basic Auth/token enforcement

Remote verification comes later after secrets and infrastructure exist.

Stop the container manually after confirming startup.

## Render deployment target

Initial deployment target: **Render Starter**

Rationale:

- simple to operate
- low maintenance for Stage 5.6
- suitable for learning/portfolio evidence store

Render will inject `PORT`. The container entrypoint reads it through the standard environment variable.

## Aiven PostgreSQL backend

Use the existing Aiven PostgreSQL service with a dedicated database:

```text
mlflow_db
```

Rules:

- do not mix MLflow metadata into `app_db`
- Postgres stores metadata only
- model files do not live in Postgres

## S3 artifact store

S3 stores MLflow artifacts and durable evidence files such as:

- model artifacts
- `manifest.json`
- `promotion_report.md`
- comparison and evaluation reports

Selected candidate evidence and durable promotion evidence must not expire with temporary debug artifacts.

## Secrets policy

No secrets belong in the repository.

Configure secrets in Render and GitHub Actions, for example:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_BACKEND_STORE_URI`
- `MLFLOW_ARTIFACT_ROOT`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- Aiven Postgres connection string

Avoid committed `.env` files.

## Access model note

Stage 5.6 Module #1 documented Basic Auth or token-based access as acceptable for learning/portfolio use.

This module does **not** implement enterprise RBAC.

Auth enforcement may be handled through Render, reverse proxy, or token layer in a later hardening step (Stage 5.6 Module #5 / Stage 16).

## What this module does not do

This module does **not**:

- deploy Render
- create the S3 bucket
- create Aiven `mlflow_db`
- modify GitHub Actions workflows
- add secrets
- change `train.py`, `predict.py`, or `DecisionService`
- import MLflow into Application/Core code

## Definition of Done

- [x] `infra/mlflow/Dockerfile` exists
- [x] `infra/mlflow/start.sh` exists and validates required env vars
- [x] `infra/mlflow/requirements.txt` exists
- [x] `infra/mlflow/README.md` exists
- [x] `docker build` succeeds
- [x] local sqlite smoke container starts
- [x] no secrets committed
- [x] no runtime behavior changed
- [x] no GitHub Actions workflow changed
- [x] no Application/Core coupling introduced

## Stage 5.6 Module #3

Module #3 adds deployment readiness on top of this container:

- Basic Auth support through `MLFLOW_ENABLE_BASIC_AUTH=true`
- Remote deployment instructions in `docs/remote_mlflow_deployment_runbook.md`
- Remote smoke verification via `scripts/verify_remote_mlflow.py`
- Official experiment workflow is **not** implemented yet
