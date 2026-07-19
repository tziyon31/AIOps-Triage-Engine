# Tuning Plan Report

- Tuning plan: `logistic_regression_tuning`
- Status: `requires_final_validation`
- Winner metric: `f1_macro`
- Require valid comparison groups: `True`
- Require final validation: `True`

## Recommended Next Step

- Type: `run_final_candidate_tournament`
- Reason: individual parameter winners may interact differently when combined

## Combined Candidate Draft

- Source: `best winner per parameter experiment`
- Promotable: `False`
- Reason: combined params must be validated together in final tournament

Parameter winners are not promotion candidates by themselves.

## Experiment Winners

### `model_family_tuning`

- Comparison group: `model_family_baseline-0c4c0f11-20260715-080324-49930cac`
- Changed parameter: `model_family`
- Winner run: `40fa1b4f96544a76a86ac13d82986c33`
- Winner variant: `manual_tfidf_logistic_regression`
- Winner f1_macro: `1.0`

### `model_family_tuning_repeat`

- Comparison group: `model_family_baseline-0c4c0f11-20260714-113233-30392754`
- Changed parameter: `model_family`
- Winner run: `fbf5b49b832a48f1ad057e1601453630`
- Winner variant: `manual_tfidf_logistic_regression`
- Winner f1_macro: `1.0`

## Blocked Experiments

No blocked experiments.
## Interpretation

This report collects parameter experiment winners into a draft.
It does not tag MLflow and does not call promote.py.
Run the final candidate tournament before candidate selection.
