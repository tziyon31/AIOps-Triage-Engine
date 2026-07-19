# Strategy Comparison Report

- fair_comparison_group_id: `1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724`
- split_sha256: `1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724`
- train_split_sha256: `a628a9a806058548a801a950e9656c2c9fd92d634d7ef06a9c1f1397b41b87da`
- test_split_sha256: `fcbbb619856f234b6713c4024d6d3fb2dcd5adade40947acb4d661e7f93ecd6d`

## Runs

| strategy | model_family | run_id | accuracy | combined_accuracy | f1_macro | combined_f1_macro | combined_weakest_class_recall | low_confidence_count | approval_required_count | policy_block_count | llm_fallback_count | offline_decision_latency_p50_ms | offline_decision_latency_p95_ms | offline_decision_latency_max_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| manual_tfidf_logistic_regression | LogisticRegression | bbfd273bf3574f3787ec8578fee55524 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 10.0 | 0.0 | 0.0 | 2.7404 | 3.47 | 4.5002 |
| manual_tfidf_sgd_log_loss | SGDClassifier | e06ad25bd3e943e5b63301d3bea02f4c | 0.5 | 0.5 | 0.3718 | 0.3718 | 0.0 | 6.0 | 18.0 | 0.0 | 0.0 | 2.2758 | 2.5339 | 4.3111 |

## Tradeoffs

- Different strategies win on f1_macro and p95 latency: manual_tfidf_logistic_regression has the best f1_macro, while manual_tfidf_sgd_log_loss has the best offline p95 latency.

## Interpretation

This comparison is fair only because all runs share the same train/test split hashes. A strategy should not be promoted from this report alone; promotion should also check policy, risk, latency, fallback behavior, and product constraints.
