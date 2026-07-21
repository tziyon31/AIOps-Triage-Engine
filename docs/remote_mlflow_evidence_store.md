# Remote MLflow Evidence Store

## Purpose

Stage 5 produced MLflow evidence, comparison reports, candidate selection reports, and promotion artifacts locally on the operator machine.

Stage 5.6 moves toward remote, reviewable evidence that can be inspected without relying on a laptop filesystem.

This document defines the architecture for:

- remote MLflow tracking
- official experiment execution
- official promotion evidence
- metadata and artifact storage boundaries

**Module #1 is architecture only.**

No infrastructure is deployed in this module. No runtime behavior changes. No GitHub Actions workflow changes. No secrets are added to the repository.

---

## One-Line Rule

```text
Experiment Manager owns intent.
GitHub Actions owns official execution.
MLflow stores evidence.
Postgres stores metadata.
S3 stores artifacts.
```

---

## Target Architecture

```text
Experiment Manager / Operator
        |
        | chooses committed experiment YAML
        | provides run_reason
        v
GitHub Actions workflow_dispatch
        |
        | official=true
        | run_source=github_actions
        | git_commit_sha=<sha>
        | workflow_run_id=<id>
        v
MLflow Tracking Server on Render
        |
        +--> Aiven PostgreSQL / mlflow_db
        |       - experiments
        |       - runs
        |       - params
        |       - metrics
        |       - tags
        |
        +--> S3 Artifact Store
                - model artifacts
                - manifest.json
                - promotion_report.md
                - comparison reports
                - evaluation reports
```

---

## Current State

The repository already supports config-driven experiments.

Existing committed experiment configs:

- `config/experiments/model_family.yaml`
- `config/experiments/final_candidate_tournament.yaml`

Stage 5.6 Module #1 adds:

- `config/experiments/smoke_official_run.yaml` — a minimal config for remote infrastructure verification

Existing CI runs local quality gates and training checks. It is **not yet** the official remote experiment workflow.

Local runs may still write to filesystem artifacts and optional local MLflow tracking. Those runs are useful for development but are not official evidence.

---

## Official Run Definition

An **official run** is not just any MLflow run.

An official run must be:

1. Produced by **GitHub Actions**
2. Logged to **remote MLflow**
3. Able to upload artifacts to **S3**
4. Tagged with traceability metadata

### Required tags / metadata

| Field | Purpose |
|---|---|
| `official=true` | Marks the run as official evidence |
| `run_source=github_actions` | Distinguishes automation from local execution |
| `git_commit_sha` | Links evidence to exact repository state |
| `workflow_run_id` | Links evidence to exact workflow execution |
| `run_owner` | Identifies who triggered the run |
| `repository` | Identifies source repository |
| `experiment_config_path` | Points to committed YAML under `config/experiments/` |
| `run_reason` | Human-readable intent for audit/review |

### What is not official

The following are **not** official evidence:

- local laptop training runs
- ad-hoc MLflow runs without GitHub Actions metadata
- runs without committed experiment config path
- runs without `official=true`
- manual MLflow UI edits
- local `promote.py --apply` mutations

### Why artifact existence alone is insufficient

A model artifact or report file may exist on disk or in S3, but without official tags and workflow metadata it cannot answer:

- who triggered the run
- from which commit
- from which experiment config
- for what reason
- whether promotion decisions were auditable

Official evidence requires **traceability**, not just file presence.

---

## Experiment Manager Role

The Experiment Manager owns:

- experiment intent
- which committed config to run
- `run_reason`
- interpretation of comparison/history/promotion reports
- whether to request promotion dry-run or official apply

The Experiment Manager does **not**:

- write official evidence directly from a local machine
- treat local MLflow runs as source of truth
- bypass GitHub Actions for official promotion apply

### Current flow

1. Experiment Manager edits or selects a committed YAML config under `config/experiments/`
2. Experiment Manager triggers `workflow_dispatch` manually
3. `workflow_dispatch` receives:
   - `experiment_config`
   - `compare_variants`
   - `run_reason`

### Future flow

A UI may eventually generate experiment requests or configs.

However:

- official execution must still go through GitHub Actions or another controlled runner
- the UI must not bypass official execution/evidence rules
- official tags and workflow metadata remain mandatory

---

## GitHub Actions Role

GitHub Actions owns:

- official experiment execution
- official MLflow evidence writing
- injection of commit/workflow/actor metadata
- official promotion apply execution

GitHub Actions is the **only** official promotion apply executor.

Automation may call training and operator scripts because it sits on the infrastructure/automation boundary, not the runtime API boundary.

---

## MLflow Tracking Server

Initial deployment target: **Render Starter**

Rationale:

- simple to operate
- low maintenance for Stage 5.6
- suitable for learning/portfolio evidence store
- avoids spending this stage on EC2 security, OS updates, TLS, and systemd

The MLflow Tracking Server:

- exposes UI/API for evidence review
- uses Aiven PostgreSQL as backend metadata store
- uses S3 as artifact store
- is an **Infrastructure Adapter**
- is not Core/Application logic

Application/Core code must not depend directly on the MLflow SDK for runtime decisions.

---

## Backend Store: Aiven PostgreSQL

Decision: use the existing **Aiven PostgreSQL** service with a dedicated database:

```text
mlflow_db
```

Rules:

- do **not** mix MLflow metadata into `app_db`
- `mlflow_db` stores MLflow metadata only
- model files do **not** live in Postgres

This reuses existing database infrastructure and is expected to avoid additional database service cost initially. Pricing should not be treated as a permanent guarantee.

Postgres is an Infrastructure Adapter.

---

## Artifact Store: S3

S3 stores files and MLflow artifacts:

- model artifacts
- `manifest.json`
- `promotion_report.md`
- comparison reports
- evaluation reports
- candidate selection reports
- promotion status reports

S3 is an Infrastructure Adapter.

Selected candidate evidence and durable promotion evidence must survive temporary artifact expiration policies.

---

## Run Types

| Run Type | Producer | Official? | Purpose |
|---|---|---|---|
| Local Exploratory Run | Operator laptop | No | Debug, learning, dry-run interpretation |
| Official Experiment Run | GitHub Actions | Yes | Controlled experiment evidence in remote MLflow |
| Official Promotion Run | GitHub Actions | Yes | Auditable candidate status/tag mutation |

---

## Promotion Apply Policy

Local usage:

- `scripts/promote.py` dry-run is allowed for review/debug
- local `--apply` is **not** official and must not be treated as production evidence mutation

Official usage:

- promotion apply must run only through GitHub Actions
- official apply must log auditable metadata and use remote MLflow evidence

Rationale:

`--apply` mutates official MLflow candidate tags and current-candidate state. That mutation must be traceable to commit, workflow, actor, and reason.

---

## Retention and Cost Policy

Retention is by evidence type:

| Evidence Type | Initial Retention Target |
|---|---|
| local/debug artifacts | 2 days |
| official experiment artifacts | 30 days |
| selected candidate artifacts | 12 months |
| durable promotion evidence | 12 months or longer |

Durable evidence includes:

- `promotion_report.md`
- `manifest.json`
- candidate selection reports
- promotion status reports

These must **not** expire with temporary debug/experiment artifacts.

Costs are expected to remain low at current project scale, but retention and storage usage should be reviewed as official runs increase.

---

## Access Model

Initial MLflow access control:

- Basic Auth or token-based access

Expected usage:

- operator can manage/trigger/review
- reviewer/client can view evidence operationally

Limitation:

This is acceptable for learning/portfolio use, but it is **not** enterprise-grade RBAC. It does not strongly enforce reviewer read-only access yet.

Stronger identity, RBAC, and governance belong to Stage 16.

---

## Security and Secrets

No secrets belong in the repository.

Use GitHub Actions secrets and environment configuration.

Possible secrets:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (or OIDC later)
- S3 bucket name
- Aiven Postgres connection string for `mlflow_db`

Avoid committed `.env` files and avoid logging secret values in workflow output.

---

## Boundary Classification

| Component | Classification | Notes |
|---|---|---|
| Experiment Manager | Operator role | Owns intent and interpretation |
| GitHub Actions workflows | Infrastructure Adapter / Automation | Official execution writer |
| MLflow Tracking Server | Infrastructure Adapter | Evidence API/UI |
| Aiven PostgreSQL `mlflow_db` | Infrastructure Adapter | Metadata store |
| S3 Artifact Store | Infrastructure Adapter | File/artifact store |
| Promotion policy | Core/Application | Decision rules, not storage |
| Candidate status | Core/Application concept stored through adapter | Must be auditable |
| DecisionService | Core/Application facade | Runtime decision entry point |
| Future FastAPI | Infrastructure Adapter | Must not import `train.py` or MLflow directly |

---

## Forbidden Coupling

Forbidden:

- Application/Core importing MLflow SDK directly
- Runtime API importing `train.py`
- Runtime API importing promotion scripts
- Runtime API importing sklearn training internals
- DecisionService importing FastAPI, MLflow, Kubernetes, Prometheus, Postgres, or S3 SDKs

Allowed:

- GitHub Actions calling training/operator scripts
- operator scripts talking to MLflow as automation/infrastructure boundary
- Experiment Manager interpreting reports without writing official evidence locally

The runtime entry point remains:

```text
DecisionService.decide(input_text, trace_id)
```

---

## Stage 5.6 Planned Modules

| Module | Scope |
|---|---|
| Module #1 | Architecture plan + smoke config |
| Module #2 | Remote MLflow server with Postgres backend and S3 artifact store |
| Module #3 | Official experiment workflow |
| Module #4 | Official promotion workflow |
| Module #5 | Access/retention/cost hardening |
| Module #6 | Remote evidence verification report |

---

## Definition of Done for Module #1

- [x] Architecture document exists
- [x] Official run definition exists
- [x] Experiment Manager role documented
- [x] GitHub Actions official writer documented
- [x] Aiven `mlflow_db` decision documented
- [x] Render Starter target documented
- [x] S3 artifact store decision documented
- [x] Retention policy documented
- [x] Basic Auth/token limitation documented
- [x] Promotion apply policy documented
- [x] `smoke_official_run.yaml` added
- [x] No infrastructure implementation
- [x] No runtime behavior change
