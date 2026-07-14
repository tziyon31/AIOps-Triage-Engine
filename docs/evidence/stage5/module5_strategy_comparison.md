# Stage 5 Module 5 — Strategy Comparison

## Objective

Compare multiple Decision Engine strategies as separate MLflow runs under the same experiment.

## Strategies Compared

- `manual_tfidf_logistic_regression`
- `manual_tfidf_sgd_log_loss`

## Fair Comparison Evidence

Both strategies were trained and evaluated using the same split identity:

- `split_sha256`: `1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724`
- `train_split_sha256`: `a628a9a806058548a801a950e9656c2c9fd92d634d7ef06a9c1f1397b41b87da`
- `test_split_sha256`: `fcbbb619856f234b6713c4024d6d3fb2dcd5adade40947acb4d661e7f93ecd6d`
- `fair_comparison_group_id`: `1b82b05d89383af705930097fb3ba580f7cab037e5728d2f855751050e980724`

This means the comparison is fair with respect to train/test split.  
Any metric difference is attributable to the strategy/model choice, not to a different data split.

## Result

`manual_tfidf_logistic_regression` outperformed `manual_tfidf_sgd_log_loss` on accuracy/f1 in this run
(`accuracy` 1.0 vs 0.5, `f1_macro` 1.0 vs ~0.37).

SGD is not a viable candidate in this comparison.
Its accuracy drop is too large to justify any possible latency/cost benefit.

This is still a valid tradeoff outcome: an alternative was evaluated fairly and rejected.

## Production Interpretation

A strategy should not be selected by accuracy alone.  
Promotion should consider:

- per-class recall
- weakest class
- low-confidence count
- approval-required count
- policy blocks
- fallback behavior
- latency
- product risk

## Evidence

- MLflow experiment: `log-triage-decision-engine`
- MLflow comparison screenshot: `docs/evidence/stage5/module5_strategy_comparison_mlflow.png`
  (Model training → select both runs → Compare; not GenAI Evaluation/Traces)
- Generated comparison report: `evidence/strategy_comparison/strategy_comparison.md`
