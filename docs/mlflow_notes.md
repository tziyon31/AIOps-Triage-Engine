# MLflow Notes — Stage 5 Module #1

## Purpose

MLflow is used in this project as an experiment journal.

Its purpose is not to replace Git, the artifact manifest, or the Quality Gate.
Its purpose is to track and compare training runs over time.

The core question MLflow answers is:

> Across many training runs, which run used which parameters, produced which metrics, generated which artifacts, and should be considered a better candidate?

---

## Mental Model

| Mechanism | Purpose |
|---|---|
| Git | Tracks source code history: commits, diffs, branches, and code changes. |
| Manifest | Tracks the identity and hashes of one specific Decision Artifact. |
| Quality Gate | Proves that one artifact passed the minimum trust checks before upload/promotion. |
| MLflow | Tracks experiments and runs so training attempts can be compared over time. |

Short version:

```text
Git = what code changed
Manifest = what artifact was built
Quality Gate = whether the artifact passed minimum trust checks
MLflow = what happened across many training experiments
```

---

## Experiment vs Run

An experiment is a collection of related ML runs for the same problem or model family.

Example:

```text
log-triage-decision-engine
```

A run is one specific training attempt.

In this project, a run will usually mean:

- one `train.py` execution
- with specific params
- that produces metrics
- and usually creates one main Decision Artifact

A run can produce one artifact, multiple artifacts, or only metrics. In this project, the normal case is one main Decision Artifact per training run.

---

## What MLflow Tracks

For each training run, MLflow should track:

- `run_id`
- experiment name
- start/end time
- status
- params
- metrics
- tags
- artifacts
- `artifact_uri`

For this project, useful params include:

- `model_type`
- `test_size`
- `random_seed`
- `tfidf_max_features`
- `min_confidence`

Useful metrics include:

- `accuracy`
- `precision`
- `recall`

Useful tags include:

- `git_sha`
- `artifact_id`
- `quality_gate_status`
- `stage`
- `training_data_sha256`

Useful artifacts include:

- `manifest.json`
- `quality_gate_report.json`
- `quality_gate_report.md`
- `model.pkl`
- `vectorizer.pkl`
- `known_actions.json`

---

## What MLflow Does Not Replace

MLflow does not replace Git.

Git remains the source of truth for code history.

MLflow does not replace `manifest.json`.

The manifest remains the source of truth for artifact identity and hashes.

MLflow does not replace the Quality Gate.

The Quality Gate remains the mechanism that decides whether an artifact passed minimum trust checks.

MLflow does not prove that a model is production-safe.

It helps compare runs, but production safety still requires tests, policy validation, contract checks, traceability, evaluation, security, and deployment controls.

---

## Manifest and Hashes

The artifact hashes should remain in `manifest.json`.

MLflow may store the manifest as an artifact and may also store important identifiers as tags, such as:

- `artifact_id`
- `git_sha`
- `training_data_sha256`
- `quality_gate_status`

The source of truth for artifact hashes is still the manifest.

MLflow connects:

```text
run -> params -> metrics -> artifact_id -> manifest -> hashes
```

---

## Tracking Data Location

For the production-like learning setup, tracking metadata will live in PostgreSQL.

The intended setup is:

```text
MLflow Tracking Server
  -> PostgreSQL backend store for metadata
  -> S3 artifact store for files
```

Metadata includes:

- experiments
- runs
- params
- metrics
- tags
- status
- timestamps
- `artifact_uri`

Artifacts include:

- models
- manifests
- quality gate reports
- plots
- evaluation outputs

---

## Production-like Setup Decision

For this stage, the production-like target is:

- **MLflow Tracking Server:** Render Starter
- **Backend Store:** existing Aiven PostgreSQL service, separate `mlflow_db`
- **Artifact Store:** AWS S3 bucket
- **Primary Writer:** GitHub Actions
- **Human Access:** controlled/read-only UI access where possible

The Aiven service already exists, so the additional database cost should be close to zero.

The expected incremental cost is:

| Item | Cost |
|---|---|
| Aiven PostgreSQL incremental cost | $0 |
| Render Starter | about $7/month |
| S3 artifacts with short retention | usually less than $1/month for small usage |
| **Estimated total** | **about $8/month** |
| **Planning ceiling** | **$10–$15/month** |

This is production-like for learning and portfolio purposes, not full client production.

---

## CI and MLflow Relationship

The CI already creates:

- Decision Artifact
- Quality Gate Evidence

In a later module, CI should log successful training runs to MLflow.

The correct order should be:

1. Train model
2. Build Decision Artifact
3. Run tests
4. Generate Quality Gate evidence
5. Validate Quality Gate evidence
6. Log the passed run to MLflow
7. MLflow stores artifacts in S3
8. GitHub Actions uploads CI artifacts for CI visibility

The important rule is:

**Do not treat a run as publishable in MLflow before the Quality Gate has passed.**

GitHub Actions artifacts and MLflow artifacts may contain the same files, but they serve different purposes:

| Store | Purpose |
|---|---|
| GitHub Actions Artifacts | evidence of a CI run |
| MLflow Artifacts | artifacts connected to an experiment/run |
| S3 | physical storage for MLflow artifacts |

---

## Access Model

The recommended production-like access model is:

| Actor | Access |
|---|---|
| GitHub Actions / training pipeline | writer |
| Client / reviewer / manager | read-only viewer |
| Admin | manages users, permissions, retention, and deletion |

MLflow should not be exposed publicly without authentication.

For a real client deployment, access should be protected by one of:

- Basic Auth
- SSO/OIDC
- VPN
- Cloudflare Access
- Tailscale
- private network
- reverse proxy auth

---

## Backup and Retention

For this learning stage, the setup is intentionally small.

A real production setup would require:

- PostgreSQL backups/snapshots
- S3 versioning or retention policy
- rules for promoted vs non-promoted artifacts
- deletion policy
- monitoring
- access control

For this stage, short S3 retention is acceptable for non-promoted artifacts.

A better future policy:

- ordinary experiment artifacts: short retention
- candidate/promoted artifacts: longer retention
- production artifacts: protected from automatic deletion

---

## When to Trust Git vs MLflow

Use Git when asking:

- What code changed?
- Which commit introduced this behavior?
- What was the diff?

Use MLflow when asking:

- Which training run performed best?
- Which params produced these metrics?
- Which artifact came from this run?
- How do I compare 30 training attempts?

Use the manifest when asking:

- What exactly is inside this artifact?
- Which hashes identify the model, vectorizer, config, actions, and training data?

Use the Quality Gate when asking:

- Did this artifact pass the minimum trust checks before upload or promotion?

---

## Current Limitations

This stage does not yet implement the full MLflow infrastructure.

Not yet implemented:

- Render deployment
- Aiven `mlflow_db` creation
- S3 artifact bucket
- MLflow auth
- CI logging to MLflow
- MLflow artifact upload
- backup automation
- retention automation
- read-only user model
- monitoring
