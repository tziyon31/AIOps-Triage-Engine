# Tuning Plan and Final Tournament

## Purpose

A tuning plan collects winners from multiple parameter experiments.

It does not promote candidates.

## Why final validation is required

The best value for one parameter may not remain best when combined with other tuned parameters.

Therefore, the combined candidate must be validated in a final tournament before `promote.py` is used.

## Flow

1. Run parameter experiments.
2. Build tuning plan report.
3. Create combined candidate draft.
4. Run final candidate tournament.
5. Validate comparison group.
6. Run `promote.py` dry-run.
7. Apply only after review.

## Commands

Build a tuning plan report:

```bash
.venv/bin/python scripts/build_tuning_plan_report.py \
  --tuning-plan-path config/tuning_plans/logistic_regression_tuning.yaml \
  --experiment-name log-triage-decision-engine
```

Update `config/experiments/final_candidate_tournament.yaml` with combined params from the draft, then run compare mode:

```bash
export LOG_TRIAGE_EXPERIMENT_CONFIG=config/experiments/final_candidate_tournament.yaml
export LOG_TRIAGE_COMPARE_VARIANTS=1
.venv/bin/python -m src.log_triage.train
```

## Boundary

Tuning winners are not promotion candidates by themselves.
Only a run from a valid final comparison group can be promoted.
