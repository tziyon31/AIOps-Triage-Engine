# AIOps Triage Engine

![CI](https://github.com/tziyon31/AIOps-Triage-Engine/actions/workflows/ci.yml/badge.svg)

ML decision engine for DevOps log triage: parse a raw log line, classify it with manual features + TF-IDF, route through embeddings or LLM when needed, and return a structured **Decision Object** JSON.

```
Raw log → parse → features + TF-IDF → classifier
       → strategy router (fast path / embeddings / LLM)
       → Decision Object
       → policy.validate()
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/log_triage/` | Application code (parse, train, predict, router, policy, secrets) |
| `config/` | `training.yaml` and `policy.yaml` |
| `data/` | Raw logs (`raw_logs.txt`, `test_logs.txt`) |
| `artifacts/` | Versioned trained bundles (not in git) |
| `scripts/` | `run_pipeline.py` — full validation pipeline |
| `tests/` | pytest (unit, artifact layout, contract, policy) |
| `legacy/` | Archived scripts; see `legacy/README.md` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### LLM / API keys

By default, `run_pipeline.sh` and CI set `LOG_TRIAGE_DISABLE_LLM=1` so predict uses only the classifier fast path (no OpenAI/Bitwarden).

For full router + LLM locally:

```bash
export BW_SESSION="$(bw unlock --raw)"   # or export OPENAI_API_KEY=...
unset LOG_TRIAGE_DISABLE_LLM
python -m src.log_triage.predict "..."
```

`OPENAI_API_KEY` is used when set; otherwise the key is loaded from Bitwarden (`OpenAIKey-MLOps`). Requires [Bitwarden CLI](https://bitwarden.com/help/cli/) (`bw`) on `PATH` only for the Bitwarden path.

## Train

Trains a combined manual-features + TF-IDF logistic regression model and writes a versioned artifact directory:

```
artifacts/log-triage-v{major}.{minor}.{patch}-{YYYYMMDD-HHMMSS}/
├── model.pkl
├── vectorizer.pkl
├── known_actions.json
└── manifest.json
```

`major` / `minor` come from `config/training.yaml`. `patch` is auto-computed from content hashes when training data or model content changes.

```bash
python -m src.log_triage.train
```

## MLflow Local Tracking

This project supports MLflow tracking for training experiments.

Start a local MLflow Tracking Server:

```bash
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/.mlflow/mlflow.db \
  --default-artifact-root $(pwd)/.mlflow/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

In another terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=log-triage-decision-engine
```

Run training:

```bash
.venv/bin/python -m src.log_triage.train
```

The MLflow UI is available at:

http://127.0.0.1:5000

MLflow tracks experiment runs, params, metrics, tags, warnings, and artifact links.

Local MLflow state is stored under `.mlflow/` and must not be committed to Git.

For deterministic CI / Quality Gate runs, MLflow logging is disabled with:

```bash
LOG_TRIAGE_DISABLE_MLFLOW=1
```

Full notes: [docs/mlflow_notes.md](docs/mlflow_notes.md)

## Predict

```bash
# Single log
python -m src.log_triage.predict "2026-05-03 09:12:11 ERROR payments db timeout cpu 93 memory 84"

# File (one log per line)
python -m src.log_triage.predict --file data/test_logs.txt

# Multiple --log flags
python -m src.log_triage.predict --log "..." --log "..."
```

Output is JSON. Required Decision Object fields:

- `strategy_used`, `predicted_action`, `confidence`, `risk_level`
- `requires_approval`, `reason`, `similar_incidents`

Optional metadata (router / LLM): `input_text`, `raw_log`, `router_reason`, `summary`, `root_cause`, `machine_context`, etc.

## Policy engine

`policy.validate(decision)` checks a Decision Object against `config/policy.yaml`:

- **Forbidden actions** → `allowed: false`, approval required
- **High risk** or **low confidence** (below `approval.min_confidence`) → `allowed: true`, approval required
- Does not mutate the original decision; returns a `PolicyResult` with `modified_decision`

```python
from src.log_triage.policy import validate

result = validate(decision_dict)
# {"allowed": True, "reason": "...", "modified_decision": {...}}
```

Router thresholds (`min_confidence`, `similarity_threshold`) and policy approval thresholds serve different roles — both live in `policy.yaml`.

## Pipeline

Deterministic end-to-end validation (no external API calls):

```bash
chmod +x run_pipeline.sh
LOG_TRIAGE_DISABLE_LLM=1 ./run_pipeline.sh
```

`run_pipeline.sh` sets `LOG_TRIAGE_DISABLE_LLM=1` by default.

Steps (in order):

1. Policy unit tests
2. Train + build fresh artifact
3. Assert a new artifact was created
4. Artifact layout tests
5. Prediction contract tests
6. Smoke prediction via `predict.py`
7. Pydantic + policy validation on smoke output
8. Full test suite

`run_pipeline.sh` is a thin Bash wrapper (`set -euo pipefail`); logic lives in `scripts/run_pipeline.py`.

## Quality Gate

![CI](https://github.com/tziyon31/AIOps-Triage-Engine/actions/workflows/ci.yml/badge.svg)

This project includes an automated Quality Gate for the Decision Artifact pipeline.

The gate ensures that a generated Decision Artifact is not uploaded by CI unless the required production checks pass.

The Quality Gate validates that the artifact is:

- packaged
- identifiable
- loadable
- contract-compatible
- policy-checked
- smoke-tested
- traceable
- supported by generated evidence

Run the deterministic gate locally:

```bash
LOG_TRIAGE_DISABLE_LLM=1 ./run_pipeline.sh
python scripts/validate_quality_gate_evidence.py
```

Generated evidence is written to:

```text
evidence/quality_gate/
```

Main evidence files:

- `quality_gate_report.json`
- `quality_gate_report.md`
- `sample_decision.json`
- `manifest_hashes.json`
- `policy_tests.txt`
- `artifact_tests.txt`
- `quality_gate_report_tests.txt`
- `prediction_contract_tests.txt`
- `traceability_integration_tests.txt`
- `deterministic_test_suite.txt`
- `smoke_prediction_output.json`

CI workflow: https://github.com/tziyon31/AIOps-Triage-Engine/actions/workflows/ci.yml

In CI, the workflow uploads two artifacts **only after** the Quality Gate passes:

- `decision-artifact-<pipeline_run_id>-<model_version>`
- `quality-gate-evidence-<pipeline_run_id>-<model_version>`

Full documentation: [docs/quality_gate.md](docs/quality_gate.md)

**Current limitation:** GitHub Actions artifacts are used as registry-like storage for this learning slice. A full production setup would use an immutable external registry such as MLflow, S3 with versioning, or a dedicated model registry.

## Tests

```bash
# Deterministic tests (default in CI)
LOG_TRIAGE_DISABLE_LLM=1 pytest -v -m "not integration and not llm_integration"

# Public predict CLI contract (no Bitwarden required)
LOG_TRIAGE_DISABLE_LLM=1 pytest -m contract -v

# Real LLM/API integration (optional, main or manual CI job)
LOG_TRIAGE_ENABLE_LLM_INTEGRATION=1 OPENAI_API_KEY=... pytest tests/test_llm_integration.py -v
```

## Configuration

**`config/training.yaml`** — dataset path, split, TF-IDF, model params, artifact versioning.

**`config/policy.yaml`** — allowed/forbidden actions, risk levels, router thresholds, approval rules.

`src/log_triage/config.py` loads YAML at import time and resolves the latest artifact path automatically.

## Contributing

Changes to `main` should go through a pull request. The required CI status check is **`quality`** (deterministic tests + contract). `pipeline-artifact`, `traceability-integration`, and `llm-integration` are informational and not merge blockers.

## Legacy

Older standalone scripts are under `legacy/`. Use the packaged modules instead:

```bash
python -m src.log_triage.train
python -m src.log_triage.predict "..."
```
