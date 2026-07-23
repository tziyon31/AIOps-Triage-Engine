# Remote MLflow Live Verification

## Purpose

This document is the live verification checklist for the Stage 5.6 remote MLflow evidence store.

It records how to verify that:

- the EC2 MLflow Tracking Server is reachable
- Aiven PostgreSQL `mlflow_db` stores metadata
- S3 stores artifacts through the server proxy
- Basic Auth works
- a non-official smoke run can write params, metrics, tags, and artifacts

This is **not** an official experiment workflow.
This does **not** change runtime Application/Core behavior.

## Boundary

This verification:

- does **not** modify `src/log_triage/*` runtime decision paths
- does **not** implement GitHub Actions official writer (Module #10)
- does **not** implement official promotion workflow (Module #11)
- does **not** claim Stage 5.6 complete

## Security Warning

**Do not commit secrets.**

Never commit:

- Aiven PostgreSQL connection strings
- AWS access keys or secret keys
- MLflow admin usernames or passwords
- Flask secret keys
- Elastic IP-bound credentials
- `/etc/aiops-mlflow/mlflow.env` contents
- `.env` files

Use placeholders only in documentation.

---

## Verified Target Topology

```text
Operator laptop
        |
        | MLFLOW_TRACKING_URI=http://<elastic-ip>:5000
        | Basic Auth username/password
        v
EC2 t4g.small + Elastic IP
  Docker: aiops-mlflow-current
  GitOps-lite systemd timer + deploy script
        |
        +--> Aiven PostgreSQL / mlflow_db
        |       metadata: experiments, runs, params, metrics, tags
        |
        +--> S3 artifact destination
                s3://<s3_bucket>/mlflow/
                via --serve-artifacts proxy
```

## Manual Setup Checklist

| Step | Task | Status |
|---|---|---|
| 1 | Architecture doc exists (`docs/remote_mlflow_evidence_store.md`) | PASS |
| 2 | MLflow Docker image exists (`infra/mlflow/`) | PASS |
| 3 | Deployment readiness / Basic Auth support merged | PASS |
| 4 | EC2 t4g.small host available | PASS |
| 5 | Elastic IP attached | PASS |
| 6 | Aiven PostgreSQL service available | PASS |
| 7 | Dedicated `mlflow_db` created | PASS |
| 8 | S3 bucket + `mlflow/` prefix available | PASS |
| 9 | EC2 IAM Role grants S3 access | PASS |
| 10 | No AWS access keys in `/etc/aiops-mlflow/mlflow.env` | PASS |
| 11 | GitOps-lite timer + deploy script configured | PASS |
| 12 | Artifact proxy enabled (`--serve-artifacts`) | PASS |
| 13 | Basic Auth enabled | PASS |
| 14 | Auth DB on persistent EC2 storage (SQLite mount) | PASS |
| 15 | Security Group restricted to operator IP | PASS |
| 16 | Non-official remote smoke verified | PASS |
| 17 | HTTPS / reverse proxy | NOT STARTED (deferred) |
| 18 | GitHub Actions official writer | NOT STARTED (Module #10) |
| 19 | Official promotion workflow | NOT STARTED (Module #11) |

---

## Required Environment Variable Names

Configured on EC2 via `/etc/aiops-mlflow/mlflow.env` (never committed).

| Variable | Placeholder example | Required |
|---|---|---|
| `MLFLOW_BACKEND_STORE_URI` | `<mlflow_db_uri>` | Yes |
| `MLFLOW_ARTIFACT_ROOT` | `s3://<s3_bucket>/mlflow/` | Yes |
| `AWS_DEFAULT_REGION` | `<aws-region>` | Yes |
| `MLFLOW_ENABLE_BASIC_AUTH` | `true` | Yes |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` | `<random-flask-secret>` | Yes |
| `MLFLOW_AUTH_ADMIN_USERNAME` | `<admin-username>` | Yes |
| `MLFLOW_AUTH_ADMIN_PASSWORD` | `<admin-password>` | Yes |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | `<elastic-ip>:5000,<elastic-ip>,localhost,...` | Recommended |
| `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | `http://<elastic-ip>:5000,...` | Recommended |

### Explicitly not stored in env file

| Item | How it is provided |
|---|---|
| AWS access key / secret key | **Not used.** EC2 IAM Role provides S3 credentials |
| Full DB password strings in git | Never. Only on host env file |

Optional:

| Variable | Purpose |
|---|---|
| `MLFLOW_AUTH_DATABASE_URI` | Defaults / overridden to SQLite on persistent mount |
| `HOST` | Defaults to `0.0.0.0` |
| `PORT` | `5000` production / `5001` candidate |

---

## GitOps-lite Deploy Behavior

Deploy script path on host (not necessarily tracked in this repo):

```text
/usr/local/bin/aiops-mlflow-gitops-deploy.sh
```

Observed behavior:

1. Fetch / reset `main`
2. Build `aiops-mlflow-server:<git-sha>`
3. Start candidate container on `localhost:5001`
4. Health-check candidate
5. Switch production container on port `5000`
6. Keep previous image available for rollback if health fails

Known limitation:

- If git SHA already matches `main` but the running image tag is stale, the script may skip redeploy while reporting healthy.
- Workaround: remove the current container once, then rerun deploy; longer-term fix should compare running image tag to expected SHA.

---

## Remote Smoke Command

```bash
export MLFLOW_TRACKING_URI="http://<elastic-ip>:5000"
export MLFLOW_TRACKING_USERNAME="<admin-or-operator-user>"
export MLFLOW_TRACKING_PASSWORD="<password>"
export MLFLOW_EXPERIMENT_NAME="stage56_remote_mlflow_smoke_proxy2"

.venv/bin/python scripts/verify_remote_mlflow.py
```

Rules:

- Use a **new** experiment name if an older experiment still has `s3://...` artifact location.
- Experiments created before artifact proxy keep their old artifact root forever.
- This smoke is tagged `official=false` and is **local exploratory evidence**, not official CI evidence.

---

## Live Verification Table

| Check | Expected | Result | Notes |
|---|---|---|---|
| EC2 Docker service starts | `aiops-mlflow-current` running | PASS | image SHA on `main` |
| Artifact mode | server proxy (`--serve-artifacts`) | PASS | |
| Postgres metadata store | runs/params/metrics/tags writable | PASS | Aiven `mlflow_db` |
| Artifact location | `mlflow-artifacts:/...` | PASS | not direct `s3://` on client |
| Client needs local boto3 | No | PASS | proxy path |
| S3 credentials on client | Not required | PASS | EC2 IAM Role |
| Basic Auth | login required | PASS | |
| Auth DB persistence | SQLite on EC2 mount | PASS | |
| Security Group | operator IP only | PASS | |
| Smoke script completes | prints `run_id` | PASS | |
| Smoke tagged `official=false` | tag present | PASS | non-official |
| Smoke artifact present | `verification/remote_mlflow_smoke.txt` | PASS | |
| HTTPS | TLS termination | FAIL / DEFERRED | HTTP for now |
| Official CI writer | GitHub Actions workflow | NOT STARTED | Module #10 |
| Official promotion apply | GitHub Actions only | NOT STARTED | Module #11 |

### Verified non-official smoke metadata

| Field | Value |
|---|---|
| Verification date | `2026-07-22` |
| Tracking URI shape | `http://<elastic-ip>:5000` |
| Experiment name | `stage56_remote_mlflow_smoke_proxy2` |
| Experiment ID | `3` |
| Run ID | `1a2b942896ed41f986b5c410404ea3cf` |
| Run status | `FINISHED` |
| Metric | `smoke_success=1.0` |
| Param | `verification_type=remote_mlflow_smoke` |
| Tags | `stage=5.6`, `run_source=manual_remote_smoke`, `official=false`, `verification_target=remote_mlflow_postgres_s3` |
| Artifact | `verification/remote_mlflow_smoke.txt` |
| Evidence class | **local exploratory / non-official** |
| Overall result | PASS (infra smoke only) |

---

## Official vs Local Evidence

| Evidence class | Producer | `official` tag | Accepted as promotion evidence? |
|---|---|---|---|
| Local exploratory smoke | operator laptop + `scripts/verify_remote_mlflow.py` | `false` | No |
| Official experiment run | GitHub Actions (Module #10) | `true` | Yes, when Module #10 exists |
| Official promotion apply | GitHub Actions (Module #11) | auditable apply | Yes, when Module #11 exists |

Current verified smoke is **local exploratory only**.

---

## Known Limitations

- HTTP only; HTTPS / reverse proxy deferred
- Aiven connection limit is low on the current plan; watch connection exhaustion
- GitOps skip logic can ignore stale image tags when git SHA already matches
- Old experiments created with `s3://` artifact roots remain non-proxy forever
- No official experiment writer yet (Module #10)
- No official promotion workflow yet (Module #11)

## Related Documents

- `docs/remote_mlflow_evidence_store.md` — architecture decisions
- `docs/remote_mlflow_deployment_runbook.md` — deployment runbook
- `docs/remote_mlflow_verification_report.md` — verified smoke report
- `docs/stage_5_6_status.md` — Stage 5.6 module status board
