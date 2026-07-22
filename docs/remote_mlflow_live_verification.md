# Remote MLflow Live Verification Template

## Purpose

This document is a **verification template only**.

Use it after the remote MLflow Tracking Server is deployed manually on Render with Aiven PostgreSQL and S3. It records whether the live stack works end-to-end.

This is **not** an official experiment. It does **not** change runtime behavior.

## Boundary

This verification template:

- does **not** modify `src/log_triage/*`
- does **not** modify `.github/workflows/*`
- does **not** modify `infra/mlflow/Dockerfile` or `infra/mlflow/start.sh`
- does **not** implement official experiment workflows
- does **not** create remote resources from code

It only documents manual live verification steps and results.

## Security Warning

**Do not commit secrets.**

Never commit:

- Aiven PostgreSQL connection strings
- Render service URLs with embedded credentials
- AWS access keys or secret keys
- MLflow admin usernames or passwords
- Flask secret keys
- `.env` files

Store all secrets in Render environment variables or GitHub Actions secrets only.

Use placeholders in this document and in any copied commands.

---

## Manual Setup Checklist

Complete these steps before running remote smoke verification.

| Step | Task | Status |
|---|---|---|
| 1 | PR #31 merged (`docs/remote_mlflow_evidence_store.md`) | TODO / PASS / FAIL |
| 2 | PR #32 merged (`infra/mlflow/` Docker image) | TODO / PASS / FAIL |
| 3 | PR #33 merged (deployment readiness + Basic Auth) | TODO / PASS / FAIL |
| 4 | Aiven PostgreSQL service available | TODO / PASS / FAIL |
| 5 | Dedicated `mlflow_db` database created | TODO / PASS / FAIL |
| 6 | S3 bucket created for MLflow artifacts | TODO / PASS / FAIL |
| 7 | Render Web Service created from `infra/mlflow/Dockerfile` | TODO / PASS / FAIL |
| 8 | Render plan set to Starter (or documented alternative) | TODO / PASS / FAIL |
| 9 | Required Render environment variables configured | TODO / PASS / FAIL |
| 10 | Render service deploy succeeded | TODO / PASS / FAIL |
| 11 | MLflow UI reachable over HTTPS | TODO / PASS / FAIL |
| 12 | Basic Auth login works (if enabled) | TODO / PASS / FAIL |
| 13 | No secrets committed to repository | TODO / PASS / FAIL |

---

## Required Render Environment Variables

Configure these in the Render service **Environment** section. Use placeholders below — replace with real values only in Render.

| Variable | Example placeholder | Required |
|---|---|---|
| `MLFLOW_BACKEND_STORE_URI` | `postgresql://<user>:<password>@<aiven-host>:<port>/mlflow_db?sslmode=require` | Yes |
| `MLFLOW_ARTIFACT_ROOT` | `s3://<bucket-name>/<prefix>/` | Yes |
| `AWS_ACCESS_KEY_ID` | `<aws-access-key-id>` | Yes |
| `AWS_SECRET_ACCESS_KEY` | `<aws-secret-access-key>` | Yes |
| `AWS_DEFAULT_REGION` | `<aws-region>` | Yes |
| `MLFLOW_ENABLE_BASIC_AUTH` | `true` | Yes (recommended) |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` | `<random-flask-secret>` | Yes when auth enabled |
| `MLFLOW_AUTH_ADMIN_USERNAME` | `<admin-username>` | Yes when auth enabled |
| `MLFLOW_AUTH_ADMIN_PASSWORD` | `<admin-password>` | Yes when auth enabled |

Optional:

| Variable | Example placeholder | Required |
|---|---|---|
| `MLFLOW_AUTH_DATABASE_URI` | same as `MLFLOW_BACKEND_STORE_URI` if omitted | No |
| `HOST` | `0.0.0.0` | No |
| `PORT` | provided by Render | No |

Notes:

- `MLFLOW_BACKEND_STORE_URI` should point to Aiven PostgreSQL / `mlflow_db`, not `app_db`.
- `MLFLOW_ARTIFACT_ROOT` should use an S3 prefix such as `s3://<bucket-name>/mlflow/`.
- Do not paste real credentials into this file.

---

## Remote Smoke Command

Run from a local machine with repository access and network reachability to the Render service.

```bash
export MLFLOW_TRACKING_URI="https://<render-service-hostname>"
export MLFLOW_TRACKING_USERNAME="<admin-or-operator-user>"
export MLFLOW_TRACKING_PASSWORD="<password>"
export MLFLOW_EXPERIMENT_NAME="stage56_remote_mlflow_smoke"

python scripts/verify_remote_mlflow.py
```

Expected script behavior:

- creates or reuses experiment `stage56_remote_mlflow_smoke`
- creates a run named `stage56_remote_mlflow_smoke`
- logs param `verification_type=remote_mlflow_smoke`
- logs metric `smoke_success=1.0`
- uploads artifact under `verification/`
- tags run with `official=false`

This smoke run is **not** official evidence.

---

## Live Verification Table

Fill in after manual deployment and smoke run.

| Check | Expected | Result | Notes |
|---|---|---|---|
| Render service starts | service stays running | TODO / PASS / FAIL | |
| Postgres metadata store | MLflow can read/write runs | TODO / PASS / FAIL | |
| S3 artifact upload | artifact visible in MLflow UI | TODO / PASS / FAIL | |
| Basic Auth (if enabled) | login required for UI/API | TODO / PASS / FAIL | |
| Smoke script completes | prints `run_id=<id>` | TODO / PASS / FAIL | |
| Smoke run tagged `official=false` | tag visible in MLflow UI | TODO / PASS / FAIL | |
| Smoke artifact present | `verification/remote_mlflow_smoke.txt` | TODO / PASS / FAIL | |
| No secrets in git | repository clean of credentials | TODO / PASS / FAIL | |

Record verification metadata (non-secret):

| Field | Value |
|---|---|
| Verification date | `<YYYY-MM-DD>` |
| Verified by | `<operator-name>` |
| Render service name | `<render-service-name>` |
| Render service URL | `https://<render-service-hostname>` |
| Aiven service name | `<aiven-service-name>` |
| S3 bucket name | `<bucket-name>` |
| Smoke run ID | `<mlflow-run-id>` |
| Overall result | TODO / PASS / FAIL |

---

## Troubleshooting Reference

| Symptom | Likely cause | What to check |
|---|---|---|
| Render service exits on start | missing required env var | `MLFLOW_BACKEND_STORE_URI`, `MLFLOW_ARTIFACT_ROOT`, auth vars |
| Postgres connection failure | bad URI / SSL / firewall | Aiven URI, `sslmode`, Render egress |
| S3 AccessDenied | IAM or bucket policy | AWS keys, bucket name, prefix permissions |
| Artifact upload fails | wrong artifact root | `MLFLOW_ARTIFACT_ROOT`, region |
| Auth login fails | wrong credentials | username/password secrets in Render |
| Smoke script cannot connect | wrong tracking URI or auth | `MLFLOW_TRACKING_URI`, username/password |

See also: `docs/remote_mlflow_deployment_runbook.md`

---

## Definition of Done (Live Verification)

- [ ] Manual setup checklist completed
- [ ] Required Render env vars configured (secrets not committed)
- [ ] Remote smoke command executed successfully
- [ ] Verification table filled with PASS/FAIL results
- [ ] Smoke run confirmed `official=false`
- [ ] No runtime code changes made for this verification
- [ ] No GitHub Actions workflow changes made
- [ ] Official experiment workflow not implemented yet
