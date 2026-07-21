# Remote MLflow Deployment Runbook

## Purpose

Stage 5.6 Module #2 created the dedicated MLflow Tracking Server Docker image under `infra/mlflow/`.

Module #3 prepares and verifies remote deployment readiness.

Target topology:

```text
Render MLflow Tracking Server
        |
        +--> Aiven PostgreSQL / mlflow_db
        |       (experiments, runs, params, metrics, tags)
        |
        +--> S3 Artifact Store
                (model artifacts, reports, manifests)
```

This module does **not** create remote resources from code and does **not** implement the official experiment GitHub Actions workflow.

## Prerequisites

- PR #31 merged (architecture plan + smoke config)
- PR #32 merged (MLflow Docker image)
- Aiven PostgreSQL service available
- dedicated `mlflow_db` created
- S3 bucket created
- Render service created from `infra/mlflow/Dockerfile`
- no secrets committed to the repository

## Render Service Configuration

| Setting | Value |
|---|---|
| Runtime | Docker |
| Dockerfile path | `infra/mlflow/Dockerfile` |
| Start command | default Docker `ENTRYPOINT` |
| Plan | Render Starter |
| Branch | `main` |

### Required environment variables

| Variable | Purpose |
|---|---|
| `MLFLOW_BACKEND_STORE_URI` | Aiven PostgreSQL / `mlflow_db` URI |
| `MLFLOW_ARTIFACT_ROOT` | S3 root, e.g. `s3://<bucket>/<prefix>` |
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |
| `AWS_DEFAULT_REGION` | AWS region |
| `MLFLOW_ENABLE_BASIC_AUTH` | Set to `true` for Basic Auth |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` | Flask secret for auth sessions |
| `MLFLOW_AUTH_ADMIN_USERNAME` | Admin username |
| `MLFLOW_AUTH_ADMIN_PASSWORD` | Admin password |

### Optional environment variables

| Variable | Purpose |
|---|---|
| `MLFLOW_AUTH_DATABASE_URI` | Auth DB URI; defaults to backend store URI |
| `HOST` | Defaults to `0.0.0.0` |
| `PORT` | Render provides this |

Notes:

- Render injects `PORT`.
- Do not echo the full Postgres URI.
- Do not commit `.env` files.
- `MLFLOW_BACKEND_STORE_URI` should point to Aiven PostgreSQL / `mlflow_db`.
- `MLFLOW_ARTIFACT_ROOT` should be `s3://<bucket>/<prefix>`.

## Aiven PostgreSQL

- Use a dedicated database: `mlflow_db`
- Keep it separate from `app_db`
- Require SSL if the Aiven URI requires it
- Store MLflow metadata only
- Do not store model files in Postgres

Postgres is an Infrastructure Adapter for MLflow metadata.

## S3 Artifact Store

S3 stores MLflow artifacts such as:

- model artifacts
- `manifest.json`
- `promotion_report.md`
- comparison and evaluation reports
- candidate selection / promotion status reports

Recommended prefix:

```text
s3://<bucket>/mlflow
```

Lifecycle by evidence type:

| Evidence Type | Retention guidance |
|---|---|
| temporary/debug artifacts | short retention (e.g. 2 days) |
| official experiment artifacts | medium retention (e.g. 30 days) |
| selected candidate / promotion evidence | durable retention (e.g. 12 months+) |

Do **not** apply short deletion policies to durable promotion evidence such as:

- `promotion_report.md`
- `manifest.json`
- candidate selection reports
- promotion status reports

## Auth

Basic Auth mode is enabled with:

```text
MLFLOW_ENABLE_BASIC_AUTH=true
```

Required when auth is enabled:

- `MLFLOW_FLASK_SERVER_SECRET_KEY`
- `MLFLOW_AUTH_ADMIN_USERNAME`
- `MLFLOW_AUTH_ADMIN_PASSWORD`

Rules:

- configure credentials as Render secrets
- do not use default/admin passwords long-term
- local-dev passwords are only for local smoke
- this is learning-correct / portfolio-grade access, not enterprise RBAC

Stronger identity and governance belong to later hardening (Stage 5.6 Module #5 / Stage 16).

## Local Container Smoke

```bash
docker build \
  -f infra/mlflow/Dockerfile \
  -t aiops-mlflow-server:local .

docker run --rm -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI="sqlite:////tmp/mlflow.db" \
  -e MLFLOW_ARTIFACT_ROOT="/tmp/mlflow-artifacts" \
  aiops-mlflow-server:local
```

This verifies the container starts. It does not verify Aiven or S3.

## Local Basic Auth Smoke

```bash
docker run --rm -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI="sqlite:////tmp/mlflow.db" \
  -e MLFLOW_ARTIFACT_ROOT="/tmp/mlflow-artifacts" \
  -e MLFLOW_ENABLE_BASIC_AUTH="true" \
  -e MLFLOW_FLASK_SERVER_SECRET_KEY="local-dev-secret" \
  -e MLFLOW_AUTH_ADMIN_USERNAME="admin" \
  -e MLFLOW_AUTH_ADMIN_PASSWORD="local-dev-password" \
  aiops-mlflow-server:local
```

`local-dev-password` is only for local smoke. Never use it remotely.

## Remote Smoke Verification

```bash
export MLFLOW_TRACKING_URI="https://<render-service-url>"
export MLFLOW_TRACKING_USERNAME="<admin-or-operator-user>"
export MLFLOW_TRACKING_PASSWORD="<password>"
export MLFLOW_EXPERIMENT_NAME="stage56_remote_mlflow_smoke"

python scripts/verify_remote_mlflow.py
```

Expected results:

- experiment created or reused
- run created
- param logged
- metric logged
- artifact uploaded
- run tagged `official=false`
- artifact visible from the MLflow UI

This script is a **manual remote smoke**, not an official experiment.

## Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| cannot connect to Postgres | bad URI / SSL / network | Aiven URI, SSL mode, Render egress, `mlflow_db` exists |
| S3 AccessDenied | IAM/credentials | access key, secret, bucket policy, prefix permissions |
| missing AWS region | unset region | `AWS_DEFAULT_REGION` |
| artifact upload fails | wrong artifact root or IAM | `MLFLOW_ARTIFACT_ROOT`, S3 permissions |
| Render service starts then exits | missing required env | `MLFLOW_BACKEND_STORE_URI`, `MLFLOW_ARTIFACT_ROOT`, auth vars if enabled |
| auth login fails | wrong credentials / auth config | username/password secrets, `MLFLOW_ENABLE_BASIC_AUTH=true` |
| PORT binding issue | host/port mismatch | Render `PORT`, container `--port`, local `-p` mapping |

## Definition of Done

- [x] PR #32 merged first
- [x] MLflow Docker image still builds
- [x] `start.sh` supports optional Basic Auth
- [x] no secrets committed
- [x] local non-auth smoke works
- [x] local auth smoke works
- [x] remote runbook exists
- [x] `verify_remote_mlflow.py` exists
- [x] remote smoke script is ready
- [x] no runtime/Application/Core changes
- [x] no official experiment workflow yet
