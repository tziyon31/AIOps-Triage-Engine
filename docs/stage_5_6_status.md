# Stage 5.6 Status

## Stage intent

Stage 5.6 = **Foundation Lock + Remote MLflow Evidence Store**

Goals:

- keep Application/Core boundaries stable
- move evidence storage off the laptop
- make remote MLflow metadata + artifacts reviewable
- prepare for official GitHub Actions execution later

Non-goals for this stage (do not implement here):

- FastAPI serving
- Kubernetes
- agents
- runtime context providers
- production traffic

## Current verdict

**Stage 5.6 is not complete.**

Remote MLflow infrastructure smoke is verified as non-official exploratory evidence.

Official experiment writer and official promotion workflow are still outstanding.

---

## One-line ownership reminder

```text
Experiment Manager owns intent.
GitHub Actions owns official execution.
MLflow stores evidence.
Postgres stores metadata.
S3 stores artifacts.
```

---

## Module status board (#1–#12)

| Module | Scope | Status | Notes |
|---|---|---|---|
| #1 | Architecture plan + `smoke_official_run.yaml` | **done** | `docs/remote_mlflow_evidence_store.md` |
| #2 | MLflow Tracking Server Docker image | **done** | `infra/mlflow/` |
| #3 | Deployment readiness + Basic Auth + runbook | **done** | `docs/remote_mlflow_deployment_runbook.md` |
| #4 | EC2 host + Elastic IP + Security Group baseline | **done** | operator-IP restricted SG; HTTP only |
| #5 | GitOps-lite deploy (systemd timer + candidate/prod switch) | **partial** | works; stale-image skip logic needs hardening/audit |
| #6 | Aiven PostgreSQL `mlflow_db` backend | **done** | dedicated DB, separate from `app_db` |
| #7 | S3 artifact destination under `mlflow/` | **done** | server-side destination |
| #8 | Artifact proxy (`mlflow-artifacts:/` + `--serve-artifacts`) | **done** | verified by smoke |
| #9 | IAM Role S3 access (no AWS keys in env file) | **done** | keys must not live in `mlflow.env` |
| #10 | GitHub Actions official experiment writer | **not_started** | required for `official=true` runs |
| #11 | Official promotion workflow (apply only via Actions) | **not_started** | no official `--apply` automation yet |
| #12 | Hardening: HTTPS, retention, connection limits, GitOps image-tag check | **partial** | HTTP deferred; Aiven connection limit known; retention not enforced |

Status legend:

- `done` — implemented and verified enough for current stage learning goals
- `partial` — exists but incomplete or known gaps remain
- `not_started` — intentionally not built yet
- `blocked` — cannot proceed due to external dependency
- `needs_audit` — exists operationally but should be reviewed/documented further

---

## Verified remote evidence-store lock

| Item | State |
|---|---|
| EC2 `t4g.small` + Elastic IP | verified |
| Dockerized MLflow | verified |
| GitOps-lite deploy path | verified |
| Aiven `mlflow_db` | verified |
| S3 `mlflow/` destination | verified |
| Artifact proxy | verified |
| Basic Auth + persistent auth DB | verified |
| Non-official smoke run | verified |
| Official CI evidence | **not verified / not implemented** |
| Stage complete | **no** |

### Anchor smoke run (non-official)

| Field | Value |
|---|---|
| Experiment | `stage56_remote_mlflow_smoke_proxy2` |
| Experiment ID | `3` |
| Run ID | `1a2b942896ed41f986b5c410404ea3cf` |
| Status | `FINISHED` |
| `official` | `false` |
| Artifact | `verification/remote_mlflow_smoke.txt` |

Details: `docs/remote_mlflow_verification_report.md`

---

## Local exploratory vs official evidence

| Class | Current state |
|---|---|
| Local exploratory smoke | Exists and verified |
| Official GitHub Actions experiment | Not implemented (Module #10) |
| Official promotion apply | Not implemented (Module #11) |

Do not treat laptop smoke runs as promotion source of truth.

---

## Known limitations

1. Transport is HTTP; HTTPS/reverse proxy deferred (Module #12).
2. Aiven plan connection limit is low; connection exhaustion was observed during overlapping containers.
3. GitOps may skip rebuild when git SHA matches even if running image tag is stale (Module #5/12).
4. Experiments created before proxy mode keep `s3://` artifact roots forever.
5. No enterprise RBAC yet; Basic Auth only.
6. No official writer / promotion automation yet.

---

## Next modules

Recommended order:

1. **Module #10** — GitHub Actions official experiment writer
2. **Module #11** — official promotion workflow
3. **Module #12** remaining hardening — HTTPS, retention policy enforcement, GitOps image-tag equality check, Postgres connection hygiene

---

## Related documents

- `docs/remote_mlflow_evidence_store.md`
- `docs/remote_mlflow_deployment_runbook.md`
- `docs/remote_mlflow_live_verification.md`
- `docs/remote_mlflow_verification_report.md`
- `docs/architecture_boundaries.md`
