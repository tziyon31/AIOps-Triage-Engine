# Architecture Boundaries

## Purpose

This document defines the architectural boundaries for the Log Triage / AIOps Decision Engine before FastAPI, agents, remote MLflow, Kubernetes, Prometheus, PostgreSQL, and S3 are added.

The goal is not to reshuffle packages yet.

The goal is to define which code belongs to:

- Core / Application
- AIOps Domain
- Infrastructure Adapters

and to make future coupling visible before it becomes production risk.

## Current Status

The current repository was built incrementally through training, artifact creation, MLflow tracking, candidate selection, and promotion gates.

Some files are intentionally mixed today.

This document classifies the current state and defines target boundaries for future changes.

No behavior change is introduced by this document.

---

## One-Line Rule

```text
Core decides.
AIOps Domain understands incidents.
Adapters connect to external systems.
```

## High-Level Flow

```text
Future API Adapter
    |
    v
Application / DecisionService
    |
    v
Core Decision + Policy + Approval + Audit Contracts
    |
    v
AIOps Domain Logic
    |
    v
Infrastructure Adapters
```

Expanded view:

```text
FastAPI / CLI / Future Agent
        |
        v
DecisionService.decide(input_text, trace_id)
        |
        v
Decision Engine / Strategy Router / Policy
        |
        v
AIOps Incident + Action Semantics
        |
        v
MLflow / OpenAI / Filesystem / Postgres / S3 / K8s / Prometheus
```

## Layer 1 — Core / Application

### Responsibility

Core / Application owns the stable decision process.

It answers:

- What is the decision contract?
- Is the decision allowed?
- Is approval required?
- What confidence/risk rules apply?
- What trace/audit evidence must exist?
- Which candidate is eligible for promotion?
- Which run is selected, rejected, eligible, or superseded?

### Examples in this project

- DecisionObject
- DecisionTrace
- policy validation mechanism
- approval requirement logic
- risk/confidence handling
- candidate selection mechanism
- promotion gate mechanism
- baseline guard
- promotion evidence contract
- comparison contract
- evaluation contract
- future DecisionService

### Core must not know

Core / Application must not directly depend on:

- FastAPI
- MLflow SDK
- OpenAI SDK
- Kubernetes SDK
- Prometheus client
- PostgreSQL SDK
- S3/boto3 SDK
- GitHub Actions internals

### Important note

Core may define generic fields such as:

- predicted_action
- confidence
- risk_level
- requires_approval
- reason

But Core should avoid hardcoding deep AIOps-specific meanings unless explicitly documented as a boundary leak.

## Layer 2 — AIOps Domain

### Responsibility

AIOps Domain owns the professional meaning of incidents, logs, symptoms, and remediation actions.

It answers:

- What does this operational signal mean?
- Is this an incident?
- What action makes sense in an AIOps context?
- Is this action operationally risky?
- Which runbook/remediation category applies?

### Examples in this project

- log triage semantics
- incident labels
- known operational actions
- similar incidents
- remediation meaning
- action semantics

Current action examples:

- ignore
- open_ticket
- suggest_scale_up
- needs_more_context

Potential future domain examples:

- CrashLoopBackOff
- OOMKilled
- high_latency
- pod_restart_loop
- disk_pressure
- memory_pressure
- deployment_regression
- runbook
- remediation

### Domain may know

AIOps Domain may know concepts like:

- Kubernetes symptoms
- Prometheus alert meaning
- incident categories
- remediation action names
- runbook categories

### Domain must not know

AIOps Domain must not directly depend on:

- FastAPI routing
- MLflow tracking API
- OpenAI client calls
- Kubernetes client implementation
- Prometheus query client implementation
- PostgreSQL/S3 persistence implementation

Important distinction:

- CrashLoopBackOff = Domain concept
- kubernetes.client.CoreV1Api = Infrastructure Adapter

## Layer 3 — Infrastructure Adapters

### Responsibility

Infrastructure Adapters connect the application to external systems.

They answer:

- How do we call an external provider?
- How do we load/save data?
- How do we query external infrastructure?
- How do we expose HTTP?
- How do we persist or retrieve artifacts?

### Examples

- FastAPI routes
- MLflow tracking
- OpenAI API calls
- embedding provider calls
- filesystem loading/saving
- GitHub Actions workflows
- PostgreSQL access
- S3 artifact access
- Kubernetes API client
- Prometheus client
- OpenTelemetry exporter

### Adapter rule

Adapters should be replaceable without changing Core decision logic.

For example:

- OpenAI API -> local OpenAI-compatible endpoint
- local MLflow -> remote MLflow
- filesystem artifact store -> S3 artifact store
- FastAPI -> CLI

The Core/Application contract should remain stable.

## Current Package / File Classification

This table documents the current state. Some files are mixed and will be cleaned gradually only when needed.

| File / Area | Current Classification | Notes |
|---|---|---|
| `src/log_triage/schemas.py` | Core / Application Contract | Defines DecisionObject, DecisionTrace, PolicyResult. Contains a small AIOps action leak through default error decision. |
| `src/log_triage/policy.py` | Core / Application, with config loading | Validates decisions against policy. YAML loading is adapter-ish but acceptable for now. |
| `src/log_triage/promotion_evidence.py` | Core / Application | Defines promotion evidence contract validation. |
| `src/log_triage/candidate_selection.py` | Core / Application | Candidate selection logic should stay independent of MLflow SDK. |
| `src/log_triage/experiments.py` | Application / Experiment Platform | Defines experiment variants and comparison identities. Not runtime API. |
| `src/log_triage/evaluation.py` | Application / Evaluation | Computes evaluation metrics and reports. |
| `src/log_triage/artifact_version.py` | Application / Artifact Management | Handles artifact identity/versioning. May touch filesystem, but not runtime decision entry. |
| `src/log_triage/predict.py` | Current runtime-ish entry point | Useful today, but future API should call DecisionService, not this directly unless wrapped. |
| `src/log_triage/train.py` | Mixed: training orchestration + artifact creation + MLflow adapter + evaluation | Not runtime Core. Future FastAPI must not import this. |
| `src/log_triage/pipeline.py` | Mixed: parsing/features/domain labels | Contains feature/log/action logic. Needs later classification between domain parsing and generic pipeline utilities. |
| `src/log_triage/data.py` | Adapter/Application | Loads data. Filesystem behavior is adapter-like; loaded labels/logs are domain data. |
| `src/log_triage/features.py` | Domain/Application | Manual feature definitions tied to current decision problem. |
| `scripts/promote.py` | Operator/Application tooling, MLflow-adapter-heavy | CLI for promotion. It may talk to MLflow, but runtime API must not depend on it. |
| `scripts/compare_runs.py` | Operator/Application tooling | Validates comparison evidence. Not runtime API. |
| `scripts/build_experiment_history_report.py` | Operator/Application reporting | Produces experiment history and recommendations. |
| `scripts/build_promotion_status_report.py` | Operator/Application reporting | Read-only MLflow status report. |
| `config/policy.yaml` | Core/Application config with Domain values | Generic policy mechanism plus AIOps action names. |
| `config/experiments/*.yaml` | Experiment config | Defines controlled comparisons. Not runtime contract. |
| `config/candidate_selection.yaml` | Core/Application promotion policy | Defines promotion thresholds and candidate rules. |
| `.github/workflows/*.yml` | Infrastructure Adapter / Automation | CI and future official experiment writer. |
| `src/log_triage/application/decision_service.py` | Core / Application facade | Stable runtime entry point for future FastAPI. Currently wraps `predict.py` to preserve behavior. |
| future FastAPI package | Infrastructure Adapter | Must call DecisionService; must not import train.py or MLflow directly. |
| future MLflow adapter | Infrastructure Adapter | Encapsulates MLflow SDK usage. |
| future OpenAI adapter | Infrastructure Adapter | Encapsulates provider API usage. |
| future Kubernetes adapter | Infrastructure Adapter | Encapsulates Kubernetes SDK usage. |
| future Prometheus adapter | Infrastructure Adapter | Encapsulates Prometheus queries. |
| future PostgreSQL/S3 adapters | Infrastructure Adapter | Persistence and artifact storage only. |

## Known Boundary Leaks Today

These are known and acceptable temporarily.

They should not be expanded without explicit reason.

### 1. train.py is mixed

`train.py` currently performs multiple roles:

- training orchestration
- feature/vectorizer setup
- model construction
- artifact creation
- evaluation
- MLflow logging
- experiment config handling
- promotion evidence params

This is acceptable for the current learning stage.

It must not become a runtime dependency.

### 2. AIOps action names appear inside Core contracts

Examples:

- open_ticket
- ignore
- suggest_scale_up
- needs_more_context

These are AIOps Domain values.

They currently appear in schemas/config/policy/labels.

This is acceptable for now, but future portability work should isolate domain-specific action catalogs from generic Core contracts.

### 3. Policy mixes generic mechanism and domain values

The mechanism:

- forbidden action
- minimum confidence
- approval required

is Core/Application.

The concrete action names are AIOps Domain.

### 4. MLflow appears in operator/training scripts

MLflow is an Infrastructure Adapter.

It is acceptable in training/operator scripts today.

It must not leak into future FastAPI route handlers or DecisionService internals.

### 5. DecisionService temporarily wraps `predict.py`

`DecisionService` is now the stable runtime facade, but internally it still delegates to the existing `predict.py` flow.

This is intentional for Stage 5.5.

The goal is to create a stable boundary without changing behavior. Later stages may move runtime internals behind cleaner interfaces.

## Allowed Dependency Direction

Target direction:

```text
Adapters -> Application/Core -> Domain concepts
```

More specifically:

- FastAPI Adapter -> DecisionService -> Decision Engine -> Policy/Domain
- MLflow Adapter -> Application reporting/promotion tooling
- OpenAI Adapter -> Application fallback interface
- K8s/Prometheus Adapters -> Domain context providers

Allowed:

- FastAPI route imports DecisionService
- DecisionService imports schemas/policy/router interfaces
- Policy imports schemas
- Domain logic imports schemas or domain config
- Adapters import external SDKs
- Operator scripts import MLflow adapters or MLflow SDK temporarily

## Forbidden Dependency Direction

Forbidden for runtime code:

- FastAPI route -> train.py
- FastAPI route -> mlflow
- FastAPI route -> sklearn training internals
- FastAPI route -> scripts/promote.py
- FastAPI route -> scripts/compare_runs.py
- DecisionService -> FastAPI
- DecisionService -> MLflow SDK
- DecisionService -> Kubernetes SDK
- DecisionService -> Prometheus client
- schemas.py -> MLflow/OpenAI/K8s/Postgres/S3 SDKs
- policy.py -> Kubernetes SDK
- policy.py -> Prometheus client
- audit/core code -> concrete AIOps action implementation details

The most important rule before Stage 6:

- Future API code must call DecisionService.
- Future API code must not call train.py.

## Ownership Rules

### Experiment Manager

Owns:

- experiment intent
- which config to run
- promotion decision review
- dry-run interpretation
- candidate approval decision

Does not own:

- official evidence source of truth from a laptop

### GitHub Actions

Owns:

- official experiment execution
- official evidence writing
- commit/workflow traceability
- future official promotion workflow

### MLflow

Owns:

- experiment evidence storage
- params
- metrics
- tags
- artifacts
- promotion reports
- candidate status visibility

MLflow is an Infrastructure Adapter, not Application/Core logic.

### Postgres and S3

Future Stage 5.6 classification:

- Postgres = MLflow backend metadata store
- S3 = MLflow artifact store

Both are Infrastructure Adapters.

Application/Core code must not depend directly on Postgres or S3 SDKs.

## Runtime Entry Rule

Stage 6 FastAPI must depend on:

```text
DecisionService.decide(input_text, trace_id) -> DecisionObject
```

Stage 6 FastAPI must not depend on:

- train.py
- MLflow SDK
- experiment scripts
- promotion scripts
- sklearn training internals
- local artifact layout internals

## What Behavior Must Not Change Because of Boundaries

Even if current boundaries are ugly, the following behavior must remain stable:

- DecisionObject output shape
- predicted action semantics
- confidence/risk/approval fields
- policy validation behavior
- existing artifact contract
- existing promotion/candidate evidence behavior
- existing MLflow evidence semantics
- existing tests and pipeline behavior

Boundaries are first documented, then enforced gradually.

No runtime behavior should change in Module 1.

## Stage 5.5 Module 1 Definition of Done

- This document exists.
- Core/Application responsibilities are defined.
- AIOps Domain responsibilities are defined.
- Infrastructure Adapter responsibilities are defined.
- Allowed dependency directions are explicit.
- Forbidden imports are explicit.
- Current files/packages are classified.
- Known boundary leaks are documented.
- No code behavior changed.

---

## Automated Boundary Guard

Stage 5.5 adds an automated dependency guard test:

```text
tests/architecture/test_boundaries.py
```

The guard scans Python imports and fails CI when known forbidden edges appear.

Current enforced rules:

- API code must not import `train.py`
- API code must not import MLflow, sklearn, or operator scripts
- Application/Core code must not import FastAPI, MLflow, OpenAI, Kubernetes, Prometheus, Postgres, or S3 SDKs
- `schemas.py` must remain SDK-free
- `policy.py` must not query Kubernetes, Prometheus, MLflow, OpenAI, Postgres, or S3 directly

### How to add a new boundary rule

Add a new `BoundaryRule` entry in:

```text
tests/architecture/test_boundaries.py
```

Each rule must define:

- `name`
- `path_prefixes`
- `forbidden_imports`
- `reason`

Every rule should protect a real architectural boundary, not a personal style preference.

### Why this guard exists

Code review alone is not enough.

A future developer may accidentally add:

```python
from src.log_triage.train import ...
```

inside a FastAPI route because it works locally.

That must fail immediately in CI.

The runtime entry point for Stage 6 is:

```text
DecisionService.decide(input_text, trace_id)
```

not `train.py`, MLflow, sklearn internals, or experiment scripts.
