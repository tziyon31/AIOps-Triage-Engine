# log_triage — decision engine for DevOps logs

Project layout:

- `src/log_triage/` — application code (parse, features, train, predict, router, LLM, embeddings)
- `data/` — raw logs and sample inputs (`raw_logs.txt`, `test_logs.txt`)
- `artifacts/` — trained bundles (`log_triage_decision_engine_v1.pkl`)
- `config/` — YAML training and policy settings (`training.yaml`, `policy.yaml`)
  - `training.yaml` defines `artifact.major` / `artifact.minor`; patch is auto-computed from content hashes
- `configs/` — legacy training defaults (superseded by `config/`)
- `legacy/` — archived scripts and their small artifacts (`model.pkl`, `model_tfidf.pkl`); see `legacy/README.md`
- `tests/` — pytest
- `scripts/` — smoke scripts

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the combined model (writes the artifact under `artifacts/`):

```bash
python -m src.log_triage.train
```

Predict (requires Bitwarden CLI + OpenAI for router fallbacks):

```bash
python -m src.log_triage.predict --file data/test_logs.txt
```

Smoke:

```bash
chmod +x scripts/run_smoke.sh
./scripts/run_smoke.sh
```

## Legacy

Older standalone scripts live under `legacy/`. Prefer **`python -m src.log_triage.train`** and **`python -m src.log_triage.predict`** so you always hit the packaged decision engine, not accidental root copies.
