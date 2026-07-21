# Remote MLflow Evidence Store

## Purpose

Stage 5 produced local experiment and promotion evidence, including MLflow runs, metrics, parameters, artifacts, manifests, comparison reports, promotion reports, candidate status, and baseline information.

Stage 5.6 moves this evidence model toward a remote, reviewable evidence store.

Module #1 is architecture only.

This module does not:

- deploy MLflow
- create a Render service
- create an S3 bucket
- modify GitHub Actions workflows
- change runtime behavior
- change model behavior
- change `train.py`, `predict.py`, or `DecisionService`

The purpose is to define the production rules before implementing the infrastructure.

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

The repository already has config-driven experiments.

Existing committed experiment configs include:

- `config/experiments/model_family.yaml`
- `config/experiments/final_candidate_tournament.yaml`

Stage 5.6 adds:

- `config/experiments/smoke_official_run.yaml`

The smoke config exists to verify the infrastructure path later:

```text
GitHub Actions -> remote MLflow -> Aiven Postgres metadata -> S3 artifacts
```

The repository also has existing CI, including model training, tests, `run_pipeline.sh`, and GitHub Actions artifacts.

However, the existing CI is not yet the official remote experiment workflow because it does not yet:

- accept an `experiment_config` input
- accept a `compare_variants` input
- accept a `run_reason` input
- write to remote MLflow
- distinguish local runs from official runs
- inject official run metadata
- write artifacts to a remote S3 artifact store through MLflow

---

## Official Run Definition

An official run is not defined only by the existence of an artifact.

An official run must be:

- produced by GitHub Actions
- logged to remote MLflow
- connected to a committed experiment YAML
- linked to a commit SHA
- linked to a workflow run ID
- linked to a GitHub actor
- tagged as official
- able to upload artifacts

Required tags/metadata:

```text
official=true
run_source=github_actions
git_commit_sha=<github_sha>
workflow_run_id=<github_run_id>
run_owner=<github_actor>
repository=<github_repository>
experiment_config_path=<path>
run_reason=<reason>
```

A run is not official when:

- it was created from a local machine
- it lacks `official=true`
- it lacks `run_source=github_actions`
- it lacks commit/workflow traceability
- it uses an untracked experiment definition
- it cannot be reviewed from remote MLflow

Rationale:

A local machine can create MLflow runs and upload artifacts. That alone is not enough for production-style evidence. Official evidence must prove who ran it, from which commit, through which workflow, with which committed configuration.

---

## Experiment Manager Role

The Experiment Manager owns experiment intent.

Responsibilities:

- choose which committed experiment config to run
- write or select the `run_reason`
- decide whether variant comparison should run
- inspect experiment reports
- interpret candidate selection evidence
- decide whether to trigger promotion dry-run
- decide whether to trigger promotion apply

The Experiment Manager does not own official evidence writing from a local machine.

Current flow:

```text
Edit/select committed YAML under config/experiments/
Trigger GitHub Actions workflow_dispatch
Provide experiment_config, compare_variants, and run_reason
Review remote MLflow evidence and reports
```

Future flow:

```text
A UI may generate experiment requests or YAML configs.
Official execution should still go through GitHub Actions or a controlled runner.
The UI must not bypass official execution and evidence rules.
```

---

## GitHub Actions Role

GitHub Actions owns official execution and official evidence writing.

Responsibilities:

- run official experiments
- run official promotion workflows
- inject commit SHA
- inject workflow run ID
- inject actor metadata
- inject repository metadata
- write official MLflow runs
- upload official artifacts through MLflow
- provide workflow logs for audit

GitHub Actions is the only allowed executor for official promotion apply.

---

## MLflow Tracking Server

The initial MLflow Tracking Server deployment target is Render Starter.

Rationale:

- simple setup
- low maintenance
- suitable for Stage 5.6
- avoids spending this stage on EC2 security, OS updates, TLS, and systemd

MLflow responsibilities:

- expose MLflow UI
- expose MLflow tracking API
- store run metadata through the backend store
- store artifact pointers
- provide reviewer-visible experiment evidence

MLflow is an Infrastructure Adapter.

Application/Core code must not depend directly on the MLflow SDK.

---

## Backend Store: Aiven PostgreSQL

Use the existing Aiven PostgreSQL service with a dedicated database:

```text
mlflow_db
```

Do not mix MLflow metadata into:

```text
app_db
```

The backend store should contain MLflow metadata only:

- experiments
- runs
- params
- metrics
- tags
- run lifecycle state
- artifact locations

The backend store does not store large model files.

Using the existing Aiven service is expected to avoid an additional database service initially, but pricing should not be treated as a permanent architectural guarantee.

Recommended separation:

- separate database: `mlflow_db`
- separate credentials if available
- avoid sharing app-specific tables with MLflow tables
- understand backup and restore behavior

---

## Artifact Store: S3

Use S3 as the MLflow artifact store.

S3 stores files such as:

- model artifacts
- vectorizers
- `manifest.json`
- `promotion_report.md`
- comparison reports
- evaluation reports
- diagnostics

S3 does not decide which model is selected.

Artifact retention must depend on evidence type. Temporary artifacts can expire quickly. Selected candidate artifacts and promotion evidence must be durable.

---

## Run Types

### Local Exploratory Run

Used for:

- debugging
- fast feedback
- learning
- validating config behavior before official execution

Required metadata:

```text
run_source=local
official=false
```

Local runs are not official evidence.

Local runs must not be promoted as official candidates unless an explicit future exception process is designed and documented.

### Official Experiment Run

Created by GitHub Actions.

Required metadata:

```text
official=true
run_source=github_actions
run_owner=<github_actor>
git_commit_sha=<github_sha>
workflow_run_id=<github_run_id>
repository=<github_repository>
experiment_config_path=<path>
run_reason=<reason>
```

Official experiment runs are eligible for:

- comparison reports
- candidate selection
- promotion review
- long-term evidence retention according to policy

### Official Promotion Run

Created by GitHub Actions through a controlled promotion workflow.

Required metadata should include:

```text
promotion_run_source=github_actions
promotion_actor=<github_actor>
promotion_workflow_run_id=<github_run_id>
baseline_run_id=<mlflow_run_id>
comparison_group_id=<comparison_group_id>
candidate_policy_version=<version>
promotion_evidence_contract_version=<version>
```

Promotion dry-run may be run locally for debugging and review.

Promotion apply must be explicit and must run only through GitHub Actions.

---

## Promotion Apply Policy

Local dry-run is allowed.

Official promotion apply is allowed only through GitHub Actions.

Local `promote.py` may be used for review/debug without `--apply`.

Local runs must not mutate official candidate tags in remote MLflow.

Rationale:

Promotion apply changes official evidence state, including:

- `candidate_status`
- `current_candidate`
- `superseded`
- `promotion_reason`
- `promotion_evidence_status`

These mutations must be traceable to:

- GitHub actor
- workflow run ID
- commit SHA
- promotion inputs
- workflow logs

---

## Retention and Cost Policy

Use retention by evidence type.

Initial policy:

| Evidence type | Initial retention target | Notes |
|---|---:|---|
| Local/debug artifacts | 2 days | Fast expiry. Not official evidence. |
| Official experiment artifacts | 30 days | Enough for review and comparison. |
| Selected candidate artifacts | 12 months or longer | Durable evidence. |
| `promotion_report.md` | 12 months or longer | Durable audit evidence. |
| `manifest.json` | 12 months or longer | Durable artifact contract evidence. |
| Candidate selection reports | 12 months or longer | Durable promotion evidence. |
| Promotion status reports | 12 months or longer | Durable promotion evidence. |

Expected S3 costs are low at the current project scale, but lifecycle rules still matter to prevent unbounded artifact growth.

Do not delete durable promotion evidence with temporary artifacts.

---

## Access Model

Initial MLflow access control will use Basic Auth or token-based access.

This is acceptable for learning and portfolio use.

Operational access model:

| Role | Allowed | Not strongly enforced yet |
|---|---|---|
| Operator / maintainer | manage MLflow evidence, run workflows, inspect artifacts | Fine-grained RBAC |
| Reviewer / client | view evidence and reports | Strong read-only enforcement |

Limitation:

This is not enterprise-grade RBAC. It does not strongly enforce reviewer read-only access yet unless enforced operationally through credentials and process.

Stage 16 can handle stronger identity, RBAC, governance, and secure enterprise deployment.

---

## Security and Secrets

Secrets must not be committed to the repository.

Use GitHub Actions secrets or managed environment variables for sensitive values.

Possible secrets and environment values:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_BACKEND_STORE_URI`
- `MLFLOW_ARTIFACT_ROOT`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- S3 bucket name
- Aiven Postgres connection string

Preferred future pattern for AWS is OIDC instead of long-lived static credentials.

Avoid:

- committed `.env` files
- committed connection strings
- committed passwords
- committed access keys

---

## Boundary Classification

| Component | Classification | Notes |
|---|---|---|
| Experiment Manager | Operator role | Owns intent and interpretation. |
| GitHub Actions workflows | Infrastructure Adapter / Automation | Official execution writer. |
| MLflow Tracking Server | Infrastructure Adapter | Evidence API/UI. |
| Aiven PostgreSQL `mlflow_db` | Infrastructure Adapter | Metadata store. |
| S3 Artifact Store | Infrastructure Adapter | File/artifact store. |
| Promotion policy | Core/Application | Decision rules, not storage. |
| Candidate status | Core/Application concept stored through adapter | Must be auditable. |
| DecisionService | Core/Application facade | Runtime decision entry point. |
| Future FastAPI | Infrastructure Adapter | Must not import `train.py` or MLflow directly. |

---

## Forbidden Coupling

Application/Core must not import the MLflow SDK directly.

Runtime API must not import:

- `train.py`
- MLflow
- promotion scripts
- experiment scripts
- sklearn training internals
- Postgres SDKs
- S3 SDKs

GitHub Actions workflows may call training/operator scripts because they are automation boundaries, not runtime API code.

The future FastAPI runtime should depend on `DecisionService`, not on training, experiment, or promotion internals.

---

## Stage 5.6 Planned Modules

1. Module #1: architecture plan + smoke config
2. Module #2: remote MLflow server with Postgres backend and S3 artifact store
3. Module #3: official experiment workflow
4. Module #4: official promotion workflow
5. Module #5: access/retention/cost hardening
6. Module #6: remote evidence verification report

---

## Definition of Done for Module #1

- [x] Architecture document exists.
- [x] Official run definition exists.
- [x] Experiment Manager role documented.
- [x] GitHub Actions official writer role documented.
- [x] Aiven `mlflow_db` decision documented.
- [x] Render Starter target documented.
- [x] S3 artifact store decision documented.
- [x] Retention policy documented.
- [x] Basic Auth/token limitation documented.
- [x] Promotion apply policy documented.
- [x] `smoke_official_run.yaml` added.
- [x] No infrastructure implementation.
- [x] No runtime behavior change.
