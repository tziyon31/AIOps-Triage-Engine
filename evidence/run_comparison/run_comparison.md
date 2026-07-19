# Run Comparison Report

- Experiment: `log-triage-decision-engine`
- Comparison group: `final_candidate_tournament-2d47db4a-20260715-094014-bbb92192`
- Run count: `2`

## Summary

- Comparison status: `valid`
- Ready for candidate selection: `true`
- Comparison type: `candidate_tournament`
- Changed variable: `candidate_config`
- Controlled variable status: `passed`

## Comparison Contract

| Check | Passed |
|---|---|
| `single_comparison_group_id` | `true` |
| `single_comparison_type` | `true` |
| `single_changed_variable` | `true` |
| `controlled_variables_passed` | `true` |
| `at_least_two_runs` | `true` |

This report validates the experiment comparison group. It does not promote a model or variant by itself.

## Comparison Metadata

### `tuned_combined_candidate`

- Run ID: `ca99ebe55172433a9bfa2d20b4a6bd73`
- Run name: `tuned_combined_candidate`
- Variant type: `candidate_config`
- Comparison type: `candidate_tournament`
- Changed variable: `candidate_config`
- Controlled variables: `raw_data,train_split,test_split,policy,evaluation_code`
- Model family: `LogisticRegression`
- Feature pipeline: `manual_features_plus_tfidf`

### `current_baseline_config`

- Run ID: `20e0daf5425440ec832b9b089f86ab6f`
- Run name: `current_baseline_config`
- Variant type: `candidate_config`
- Comparison type: `candidate_tournament`
- Changed variable: `candidate_config`
- Controlled variables: `raw_data,train_split,test_split,policy,evaluation_code`
- Model family: `SGDClassifier`
- Feature pipeline: `manual_features_plus_tfidf`

## Controlled Variable Validation

Overall status: `passed`

| Controlled variable | Tag | Status | Values |
|---|---|---|---|
| raw_data | training_data_sha256 | passed | 8fa110e64b91dc692fad42dd58a8073a0909fe5f619489842f735c1098963414 |
| train_split | train_split_sha256 | passed | a628a9a806058548a801a950e9656c2c9fd92d634d7ef06a9c1f1397b41b87da |
| test_split | test_split_sha256 | passed | fcbbb619856f234b6713c4024d6d3fb2dcd5adade40947acb4d661e7f93ecd6d |
| policy | policy_sha256 | passed | 0c8855fb4be30b3a99b425b2c9df3ae329170681d32263132c2149a7aa366c4a |
| evaluation_code | evaluation_code_sha256 | passed | 323a2bb338feb48f8f3ad083ee759676abb743260d3c8e44a8e78ce4365ef136 |

## Metrics

| Variant | approval_required_count | approval_required_rate | combined_accuracy | combined_f1_macro | combined_weakest_class_recall | f1_macro | invalid_decision_schema_count | llm_fallback_count | low_confidence_count | low_confidence_rate | offline_decision_latency_max_ms | offline_decision_latency_p50_ms | offline_decision_latency_p95_ms | policy_block_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tuned_combined_candidate | 10.0 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 2.0 | 0.1 | 3.3964 | 2.3975 | 2.9333 | 0.0 |
| current_baseline_config | 18.0 | 0.9 | 0.5 | 0.3718 | 0.0 | 0.3718 | 0.0 | 0.0 | 6.0 | 0.3 | 2.8612 | 2.1198 | 2.2933 | 0.0 |

## Tradeoff Summary

- There is a quality/latency tradeoff: the best f1_macro variant is not the fastest p95 offline latency variant.
- The same variant currently wins both f1_macro and low_confidence_rate.

## Candidate Selection Input

- Ready: `true`
- Reason: `comparison_contract_valid`

| Variant | Run ID | Model family | Feature pipeline |
|---|---|---|---|
| tuned_combined_candidate | ca99ebe55172433a9bfa2d20b4a6bd73 | LogisticRegression | manual_features_plus_tfidf |
| current_baseline_config | 20e0daf5425440ec832b9b089f86ab6f | SGDClassifier | manual_features_plus_tfidf |

## Production Interpretation

This report does not promote a candidate by itself. It only proves whether the compared runs belong to a valid comparison group and shows the metric tradeoffs. Candidate selection should be handled by a promotion policy in the next stage.
