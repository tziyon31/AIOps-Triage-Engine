# Stage 5.5 — Experiment Variant Comparison Framework

## Objective

Convert the Stage 5 strategy comparison from a narrow model-vs-model comparison into a general experiment framework.

The framework supports comparing different experiment variants such as:

- model family
- feature pipeline / vectorization
- hyperparameters
- policy thresholds
- training data variants
- fallback behavior

## Core Concepts

### Variant

A `variant` is the specific experiment option being tested.

Examples:

- `manual_tfidf_logistic_regression`
- `manual_tfidf_sgd_log_loss`
- `embeddings_minilm_logistic_regression`
- `min_confidence_085`

### Changed Variable

The single main variable being tested in the comparison.

Examples:

- `model_family`
- `feature_pipeline`
- `model_max_iter`
- `min_confidence`
- `train_data`

### Controlled Variables

The variables that must remain stable for the comparison to be valid.

For the current model-family comparison:

- raw data
- train split
- test split
- feature pipeline
- policy
- evaluation code

Each controlled variable is backed by an MLflow tag/hash such as:

- `training_data_sha256`
- `train_split_sha256`
- `test_split_sha256`
- `feature_pipeline_sha256`
- `policy_sha256`
- `evaluation_code_sha256`

## Current Comparison

The current comparison type is:

```text
comparison_type = model_family
changed_variable = model_family
```

Variants:

- `manual_tfidf_logistic_regression`
- `manual_tfidf_sgd_log_loss`

## Evidence

`scripts/compare_runs.py` produces:

- `evidence/run_comparison/run_comparison.json`
- `evidence/run_comparison/run_comparison.md`

The report includes:

- comparison contract
- controlled variable validation
- run metadata
- metrics table
- tradeoff summary
- candidate selection input

## How to Run

Default (single variant):

```bash
.venv/bin/python -m src.log_triage.train
```

Compare variants:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=log-triage-decision-engine
export LOG_TRIAGE_COMPARE_VARIANTS=1
.venv/bin/python -m src.log_triage.train
```

Generate comparison evidence:

```bash
.venv/bin/python scripts/compare_runs.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id '<COMPARISON_GROUP_ID>'
```

## Important Boundary

This module does not promote a candidate.

It only proves whether a group of runs is valid and ready for Candidate Selection.

Candidate Selection is handled by the next module using promotion policy.
