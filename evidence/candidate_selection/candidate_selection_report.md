# Candidate Selection Report

- Mode: `apply`
- Experiment: `log-triage-decision-engine`
- Comparison group: `model_family_baseline-0c4c0f11-20260715-113809-b7752fad`
- Selection status: `selected`

## Policy

- Name: `decision_engine_candidate_selection`
- Version: `v1`
- Status: `learning_policy_not_production_gate`
- Path: `config/candidate_selection.yaml`

## Baseline

- Run ID: `1db2300c064843a394f67486cf226e7c`
- Variant: `manual_tfidf_sgd_log_loss`
- f1_macro: `0.3718`

## Baseline Guard

- Status: `passed_with_override`
- Mode: `non_current_baseline_override`
- Reason: `non_current_baseline_allowed_with_explicit_reason`
- Baseline run ID: `1db2300c064843a394f67486cf226e7c`
- Current candidate run IDs: `['7bfd8e17403a4984a625fa4b007063e0']`
- Override allowed: `True`
- Override reason: `module7-step3-status-transitions-live-validation`

## Selected Candidate

- Run ID: `8e36e5abc3df49cfabcc0f1bb3fb5c2c`
- Variant: `manual_tfidf_logistic_regression`
- Reason: `candidate_policy_and_baseline_passed`
- f1_macro: `1.0`
- p95 latency: `1.6678`

## Promotion Evidence Contract

- Name: `promotion_evidence_contract`
- Version: `v1`
- Path: `config/promotion_evidence_contract.yaml`

## Candidate Lifecycle

- New current candidate: `8e36e5abc3df49cfabcc0f1bb3fb5c2c`
- Previous current candidates: `['7bfd8e17403a4984a625fa4b007063e0']`
- Superseded candidates on apply: `['7bfd8e17403a4984a625fa4b007063e0']`
- Already current: `False`

## Candidate Evaluations

| Variant | Run ID | Eligible | Decision | Reason | f1_macro | p95 latency |
|---|---|---|---|---|---|---|
| manual_tfidf_sgd_log_loss | 1db2300c064843a394f67486cf226e7c | false | candidate_rejected | candidate_is_baseline_run | 0.3718 | 0.9941 |
| manual_tfidf_logistic_regression | 8e36e5abc3df49cfabcc0f1bb3fb5c2c | true | candidate_ready | candidate_policy_and_baseline_passed | 1.0 | 1.6678 |

## Candidate Status Transitions

| Variant | Run ID | Action | New status | Reason |
|---|---|---|---|---|
| manual_tfidf_sgd_log_loss | 1db2300c064843a394f67486cf226e7c | skip | None | baseline_run_status_not_changed |
| manual_tfidf_logistic_regression | 8e36e5abc3df49cfabcc0f1bb3fb5c2c | update | selected | candidate_policy_and_baseline_passed |

## Interpretation

This report selects a candidate only according to the offline candidate-selection policy.
It does not deploy the model and does not claim production readiness.
MLflow tags are written only when the script runs with `--apply`.
