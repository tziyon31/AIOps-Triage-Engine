# Candidate Selection / Promotion Rule

## Purpose

Select a candidate run from a valid MLflow comparison group.

This step does not deploy a model. It only selects and tags a candidate run.

## Inputs

- MLflow experiment name
- comparison_group_id
- baseline_run_id
- candidate selection policy YAML

## Policy

Policy file:

```text
config/candidate_selection.yaml
```

A candidate can be selected only if:

1. The comparison group is valid.
2. A baseline run is provided.
3. The candidate passes all metric thresholds.
4. The candidate improves over baseline by the configured minimum f1 delta.
5. Tie scores are broken deterministically.

## Baseline protection

Promotion without a baseline is blocked.

This prevents selecting a run that looks good in isolation but is worse than the current baseline.

## Dry-run by default

`scripts/promote.py` does not tag MLflow unless `--apply` is passed.

## Commands

Dry-run:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id "<COMPARISON_GROUP_ID>" \
  --baseline-run-id "<BASELINE_RUN_ID>" \
  --candidate-policy-path config/candidate_selection.yaml
```

Apply MLflow tags:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id "<COMPARISON_GROUP_ID>" \
  --baseline-run-id "<BASELINE_RUN_ID>" \
  --candidate-policy-path config/candidate_selection.yaml \
  --apply
```

## MLflow tags

Only with `--apply`, the selected run receives:

* `candidate=true`
* `candidate_status=selected`
* `promotion_reason`
* `candidate_policy_name`
* `candidate_policy_version`
* `candidate_policy_status`
* `baseline_run_id`
* `comparison_group_id`
* `candidate_selected_at`

## Boundary

This is an offline candidate-selection gate.

It does not mean the model is production ready.
Deployment, service latency, monitoring, rollback, and live validation are separate stages.
