# Stage 5.6 — Config-driven Experiments

## Objective

Move experiment definitions out of Python and into YAML.

The experiment owner should be able to define:

- experiment name
- comparison type
- changed variable
- controlled variables
- feature pipeline
- variants

without changing `train.py`.

## Current Experiment Config

Current default config:

```text
config/experiments/model_family.yaml
```

This experiment compares model families:

```text
comparison_type = model_family
changed_variable = model_family
```

Current variants:

* `manual_tfidf_logistic_regression`
* `manual_tfidf_sgd_log_loss`

## Source of Truth

### `config/experiments/*.yaml`

Defines what experiment should run.

The YAML controls:

* `experiment_name`
* `comparison_type`
* `changed_variable`
* `controlled_variables`
* `feature_pipeline`
* `variants`

### `config/training.yaml`

Still controls baseline training and artifact settings:

* raw logs path
* random seed
* test size
* artifact name/version/output
* schema/artifact type
* baseline TF-IDF/model defaults

### Validation boundary

If the same value appears in both files, it must be consistent unless that value is the `changed_variable`.

Example:

* In a `model_family` comparison, `model_family` may differ between variants.
* `random_state`, `model_max_iter`, and TF-IDF params must stay consistent.

## Running the default experiment

```bash
LOG_TRIAGE_EXPERIMENT_CONFIG=config/experiments/model_family.yaml \
LOG_TRIAGE_DISABLE_LLM=1 \
LOG_TRIAGE_DISABLE_MLFLOW=1 \
.venv/bin/python -m src.log_triage.train
```

Default mode runs one variant: the first variant in the YAML, unless `LOG_TRIAGE_VARIANT` is set.

## Running one specific variant

```bash
LOG_TRIAGE_EXPERIMENT_CONFIG=config/experiments/model_family.yaml \
LOG_TRIAGE_VARIANT=manual_tfidf_sgd_log_loss \
LOG_TRIAGE_DISABLE_LLM=1 \
LOG_TRIAGE_DISABLE_MLFLOW=1 \
.venv/bin/python -m src.log_triage.train
```

Use this for debugging, quick checks, or generating one artifact.

A single-variant run is not a full comparison by itself.

## Running the full comparison

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=log-triage-decision-engine
export LOG_TRIAGE_EXPERIMENT_CONFIG=config/experiments/model_family.yaml
export LOG_TRIAGE_COMPARE_VARIANTS=1

.venv/bin/python -m src.log_triage.train
```

This runs all variants in the YAML.

Each variant creates:

* one MLflow run
* one Decision Artifact
* variant metadata
* feature pipeline identity
* evaluation code identity
* split evidence

All variants in the same comparison run share the same `comparison_group_id`.

## Comparison group vs split hash

`comparison_group_id` identifies one experiment run group.

`split_sha256` proves which train/test split was used.

They are intentionally different.

If `comparison_group_id` reused `split_sha256`, separate experiment reruns with the same split would be mixed together incorrectly.

## Comparing the runs

After running compare mode, copy the `comparison_group_id` from the train output or MLflow tags.

Then run:

```bash
.venv/bin/python scripts/compare_runs.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id '<COMPARISON_GROUP_ID>'
```

Expected output:

```text
Controlled variable status: passed
Comparison contract status: valid
Ready for candidate selection: True
```

Reports are written to:

```text
evidence/run_comparison/run_comparison.json
evidence/run_comparison/run_comparison.md
```

## What makes a comparison valid?

A comparison is valid only when:

* all runs share one `comparison_group_id`
* all runs share one `comparison_type`
* all runs share one `changed_variable`
* at least two variants exist
* controlled variables pass hash validation

For the current model-family experiment, controlled variables include:

* raw data
* train split
* test split
* feature pipeline
* policy
* evaluation code

## What still requires Python changes?

YAML can define variants for capabilities that already exist in code.

Examples that do not require major Python changes:

* adding another LogisticRegression max_iter variant, if the validator allows it
* adding another SGD parameter variant
* changing experiment name
* selecting a different existing variant

Examples that still require Python changes:

* adding embeddings as a new feature pipeline
* adding a new model family not supported by `build_model_for_variant`
* adding a new evaluation metric
* changing how Decision Objects are built

## Operator mental model

Do not start from “run another model.”

Start from:

1. What question am I testing?
2. What is the changed variable?
3. What variants answer that question?
4. What must remain controlled?
5. What hashes prove the controls stayed fixed?
6. Is the comparison contract valid?
7. Is the candidate pool ready for candidate selection?

## Boundary

This module does not select a production candidate.

It only makes experiments configurable and produces valid comparison evidence.

Candidate Selection belongs to the next stage.
