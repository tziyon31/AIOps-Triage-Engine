# Remote MLflow Verification Report

## Summary

Stage 5.6 remote MLflow evidence-store infrastructure has been **smoke-verified** from an operator laptop against the EC2-hosted tracking server.

This report records the verified non-official smoke run.

It does **not** mark Stage 5.6 complete.
It does **not** create official experiment evidence.
It does **not** authorize promotion.

## Evidence Classification

| Field | Value |
|---|---|
| Evidence class | Local exploratory smoke |
| Official? | **No** (`official=false`) |
| Producer | Operator laptop |
| Script | `scripts/verify_remote_mlflow.py` |
| Runtime behavior changed? | No |
| GitHub Actions involved? | No |
| Promotion apply performed? | No |

Official CI evidence writing belongs to **Module #10**.
Official promotion workflow belongs to **Module #11**.

---

## Verified Infrastructure Snapshot

Placeholders only. No secrets.

| Component | Verified configuration |
|---|---|
| Compute | EC2 `t4g.small` |
| Network identity | Elastic IP (`<elastic-ip>`) |
| Runtime | Docker container `aiops-mlflow-current` |
| Deploy model | GitOps-lite systemd timer + deploy script |
| Deploy flow | pull `main` → build image → candidate `:5001` → health check → production `:5000` |
| Metadata store | Aiven PostgreSQL dedicated `mlflow_db` (`<mlflow_db_uri>`) |
| Artifact destination | `s3://<s3_bucket>/mlflow/` |
| Artifact access mode | Server proxy (`--serve-artifacts`, `mlflow-artifacts:/...`) |
| Client S3 credentials | Not required |
| EC2 S3 access | IAM Role |
| AWS keys in env file | Must not be present |
| Auth | MLflow Basic Auth enabled |
| Auth DB | SQLite on EC2 persistent storage, mounted into container |
| Network control | Security Group restricted to operator IP |
| Transport | HTTP for now; HTTPS deferred |

---

## Verified Smoke Run

| Field | Value |
|---|---|
| Date (UTC context) | 2026-07-22 |
| Experiment name | `stage56_remote_mlflow_smoke_proxy2` |
| Experiment ID | `3` |
| Experiment artifact location | `mlflow-artifacts:/3` |
| Run ID | `1a2b942896ed41f986b5c410404ea3cf` |
| Run name | `stage56_remote_mlflow_smoke` |
| Run status | `FINISHED` |
| Tracking URI shape | `http://<elastic-ip>:5000` |

### Params

| Param | Value |
|---|---|
| `verification_type` | `remote_mlflow_smoke` |

### Metrics

| Metric | Value |
|---|---|
| `smoke_success` | `1.0` |

### Tags

| Tag | Value |
|---|---|
| `stage` | `5.6` |
| `run_source` | `manual_remote_smoke` |
| `official` | `false` |
| `verification_target` | `remote_mlflow_postgres_s3` |

### Artifacts

| Artifact path | Verified |
|---|---|
| `verification/remote_mlflow_smoke.txt` | Yes |

### Why this experiment name

Earlier smoke experiments were created while the server still exposed `s3://...` artifact locations to clients.

Those older experiments remain permanently on direct S3 upload semantics.

`stage56_remote_mlflow_smoke_proxy2` was created after artifact proxy mode and correctly used:

```text
mlflow-artifacts:/3
```

---

## What This Proves

PASS conditions demonstrated:

1. Remote MLflow accepts authenticated client traffic
2. Metadata is stored in Aiven `mlflow_db`
3. New experiments receive `mlflow-artifacts:/...` URIs
4. Artifact upload succeeds through the tracking server proxy
5. Local client does not need `boto3` or AWS keys
6. EC2 IAM Role is sufficient for server-side S3 writes
7. Non-official tagging works (`official=false`)

## What This Does Not Prove

Not proven / not claimed:

1. Official GitHub Actions experiment execution
2. Official promotion apply audit trail
3. HTTPS / reverse-proxy hardening
4. Retention lifecycle enforcement for durable promotion evidence
5. Multi-operator RBAC
6. Production runtime DecisionService traffic
7. Stage 5.6 completion

---

## Replay Command (placeholders)

```bash
export MLFLOW_TRACKING_URI="http://<elastic-ip>:5000"
export MLFLOW_TRACKING_USERNAME="<admin-or-operator-user>"
export MLFLOW_TRACKING_PASSWORD="<password>"
export MLFLOW_EXPERIMENT_NAME="stage56_remote_mlflow_smoke_proxy2"

.venv/bin/python scripts/verify_remote_mlflow.py
```

If replaying for a fresh check after server config changes, prefer a new experiment name to avoid inheriting stale `s3://` artifact roots.

---

## Related Documents

- `docs/remote_mlflow_live_verification.md` — checklist and topology
- `docs/remote_mlflow_evidence_store.md` — architecture
- `docs/stage_5_6_status.md` — module status board
