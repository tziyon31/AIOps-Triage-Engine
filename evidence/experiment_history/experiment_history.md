# Experiment History Report

- MLflow experiment: `log-triage-decision-engine`
- Total comparison groups: `5`
- Valid comparison groups: `5`
- Invalid comparison groups: `0`
- Ready for candidate selection groups: `5`

## Recommended Next Step

- Type: `quality_latency_tradeoff_experiment`
- Recommended next experiment: `latency_or_cost_tradeoff_experiment`
- Source group: `1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724`
- Reason: The best quality variant is not the fastest variant. The next experiment should clarify whether the quality gain is worth the latency/cost tradeoff.

Evidence:
- `best_f1_variant=manual_tfidf_logistic_regression`
- `best_latency_variant=manual_tfidf_sgd_log_loss`

Candidate selection dry-run command:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id 1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724 \
  --baseline-run-id <BASELINE_RUN_ID> \
  --candidate-policy-path config/candidate_selection.yaml
```

## Comparison Groups

| Group | Status | Ready | Changed variable | Variants | Recommendation | Blocked reason |
|---|---|---|---|---|---|---|
| 1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724 | valid | true | model_family | manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss | latency_or_cost_tradeoff_experiment |  |
| model_family_baseline-0c4c0f11-20260714-095546-bf3685ca | valid | true | model_family | manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss | latency_or_cost_tradeoff_experiment |  |
| model_family_baseline-0c4c0f11-20260714-095856-a98bebd5 | valid | true | model_family | manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss | latency_or_cost_tradeoff_experiment |  |
| model_family_baseline-0c4c0f11-20260714-113233-30392754 | valid | true | model_family | manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss | latency_or_cost_tradeoff_experiment |  |
| model_family_baseline-0c4c0f11-20260715-080324-49930cac | valid | true | model_family | manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss | candidate_selection_policy |  |

## Valid Groups

### `1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724`

- Comparison type: `model_family`
- Changed variable: `model_family`
- Variants: `manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss`
- Experiment config: `unknown`

Tradeoffs:
- There is a quality/latency tradeoff: the best f1_macro variant is not the fastest p95 offline latency variant.
- The same variant currently wins both f1_macro and low_confidence_rate.

Recommendation:
- Type: `quality_latency_tradeoff_experiment`
- Recommended next experiment: `latency_or_cost_tradeoff_experiment`
- Reason: The best quality variant is not the fastest variant. The next experiment should clarify whether the quality gain is worth the latency/cost tradeoff.

Candidate selection dry-run command:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id 1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724 \
  --baseline-run-id <BASELINE_RUN_ID> \
  --candidate-policy-path config/candidate_selection.yaml
```

### `model_family_baseline-0c4c0f11-20260714-095546-bf3685ca`

- Comparison type: `model_family`
- Changed variable: `model_family`
- Variants: `manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss`
- Experiment config: `config/experiments/model_family.yaml`

Tradeoffs:
- There is a quality/latency tradeoff: the best f1_macro variant is not the fastest p95 offline latency variant.
- The same variant currently wins both f1_macro and low_confidence_rate.

Recommendation:
- Type: `quality_latency_tradeoff_experiment`
- Recommended next experiment: `latency_or_cost_tradeoff_experiment`
- Reason: The best quality variant is not the fastest variant. The next experiment should clarify whether the quality gain is worth the latency/cost tradeoff.

Candidate selection dry-run command:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id model_family_baseline-0c4c0f11-20260714-095546-bf3685ca \
  --baseline-run-id <BASELINE_RUN_ID> \
  --candidate-policy-path config/candidate_selection.yaml
```

### `model_family_baseline-0c4c0f11-20260714-095856-a98bebd5`

- Comparison type: `model_family`
- Changed variable: `model_family`
- Variants: `manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss`
- Experiment config: `config/experiments/model_family.yaml`

Tradeoffs:
- There is a quality/latency tradeoff: the best f1_macro variant is not the fastest p95 offline latency variant.
- The same variant currently wins both f1_macro and low_confidence_rate.

Recommendation:
- Type: `quality_latency_tradeoff_experiment`
- Recommended next experiment: `latency_or_cost_tradeoff_experiment`
- Reason: The best quality variant is not the fastest variant. The next experiment should clarify whether the quality gain is worth the latency/cost tradeoff.

Candidate selection dry-run command:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id model_family_baseline-0c4c0f11-20260714-095856-a98bebd5 \
  --baseline-run-id <BASELINE_RUN_ID> \
  --candidate-policy-path config/candidate_selection.yaml
```

### `model_family_baseline-0c4c0f11-20260714-113233-30392754`

- Comparison type: `model_family`
- Changed variable: `model_family`
- Variants: `manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss`
- Experiment config: `config/experiments/model_family.yaml`

Tradeoffs:
- There is a quality/latency tradeoff: the best f1_macro variant is not the fastest p95 offline latency variant.
- The same variant currently wins both f1_macro and low_confidence_rate.

Recommendation:
- Type: `quality_latency_tradeoff_experiment`
- Recommended next experiment: `latency_or_cost_tradeoff_experiment`
- Reason: The best quality variant is not the fastest variant. The next experiment should clarify whether the quality gain is worth the latency/cost tradeoff.

Candidate selection dry-run command:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id model_family_baseline-0c4c0f11-20260714-113233-30392754 \
  --baseline-run-id <BASELINE_RUN_ID> \
  --candidate-policy-path config/candidate_selection.yaml
```

### `model_family_baseline-0c4c0f11-20260715-080324-49930cac`

- Comparison type: `model_family`
- Changed variable: `model_family`
- Variants: `manual_tfidf_logistic_regression, manual_tfidf_sgd_log_loss`
- Experiment config: `config/experiments/model_family.yaml`

Tradeoffs:
- The same variant currently wins both f1_macro and p95 offline latency.
- The same variant currently wins both f1_macro and low_confidence_rate.

Recommendation:
- Type: `candidate_selection`
- Recommended next experiment: `candidate_selection_policy`
- Reason: The comparison is valid and no blocking quality/confidence issue was detected. The next step is candidate selection, not another experiment.

Candidate selection dry-run command:

```bash
.venv/bin/python scripts/promote.py \
  --experiment-name log-triage-decision-engine \
  --comparison-group-id model_family_baseline-0c4c0f11-20260715-080324-49930cac \
  --baseline-run-id <BASELINE_RUN_ID> \
  --candidate-policy-path config/candidate_selection.yaml
```

## Interpretation

This report summarizes experiment history across MLflow comparison groups.
It does not select or promote a production candidate.
Candidate selection should be handled by a separate policy step.
