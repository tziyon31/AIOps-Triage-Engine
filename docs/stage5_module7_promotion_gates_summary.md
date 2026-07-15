# Stage 5 · Module 7 — Promotion Gates + Candidate Status

## Objective

Turn MLflow into a promotion-decision system.

## What was built

### 1. Promotion evidence contract

A YAML-backed contract defines the required evidence for promotion eligibility:

- required params
- required metrics
- required tags
- required artifacts
- allowed candidate statuses

File:

- `config/promotion_evidence_contract.yaml`

### 2. Default promotion status during training

Every new MLflow run starts with:

- `candidate_status=not_evaluated`
- `promotion_reason=not_evaluated`
- `run_owner=<owner>`
- `promotion_evidence_contract_version=v1`

Training also logs a default `promotion_report.md` artifact.

### 3. Evidence enforcement inside `promote.py`

`promote.py` validates every candidate against the promotion evidence contract before it can be selected.

A run can only be selected if it passes:

- valid comparison group
- candidate policy
- baseline improvement
- baseline guard
- promotion evidence contract

Incomplete evidence causes:

- `promotion_evidence_contract_failed`
- no candidate selection

### 4. Candidate status transitions

On `--apply`, evaluated non-baseline runs receive explicit status:

- `selected`
- `eligible`
- `rejected`
- `superseded`

Baseline runs are skipped and not overwritten.

### 5. Promotion status reporting

A read-only report summarizes MLflow candidate state:

- counts by `candidate_status`
- MLflow filter examples
- current candidate checks
- selected evidence checks
- rejected reason checks

Files:

- `scripts/build_promotion_status_report.py`
- `evidence/promotion_status/promotion_status_report.json`
- `evidence/promotion_status/promotion_status_report.md`

## Promotion flow

```text
training run
→ candidate_status=not_evaluated
→ compare_runs.py validates comparison group
→ promote.py dry-run validates policy + baseline + evidence
→ report shows selected / rejected reasons
→ promote.py --apply updates candidate statuses
→ promotion status report verifies visibility
```

## MLflow filters

```text
tags.candidate_status = 'selected'
tags.candidate_status = 'eligible'
tags.candidate_status = 'rejected'
tags.candidate_status = 'superseded'
tags.candidate_status = 'not_evaluated'
tags.current_candidate = 'true'
tags.candidate = 'true'
```

## Key files

| Area | Path |
| --- | --- |
| Evidence contract | `config/promotion_evidence_contract.yaml` |
| Contract validator | `src/log_triage/promotion_evidence.py` |
| Training default status | `src/log_triage/train.py` |
| Promotion gate | `scripts/promote.py` |
| Status visibility report | `scripts/build_promotion_status_report.py` |
| Operator docs | `docs/promotion.md` |

## Completion evidence

| Module requirement | Evidence |
| --- | --- |
| Default `candidate_status` at train time | Live runs tagged `not_evaluated` + `promotion_report.md` artifact |
| Evidence contract blocks incomplete runs | `promote.py` reason `promotion_evidence_contract_failed` |
| Selected requires full gate stack | comparison + policy + baseline + baseline guard + evidence |
| Status transitions beyond winner-only | `candidate_status_transitions` in dry-run/apply reports |
| MLflow filter by status | `scripts/build_promotion_status_report.py` filter examples + live status counts |
| At most one current candidate | status report check `single_current_candidate` |
| No silent mutation from status report | report is read-only; only `promote.py --apply` writes tags |

Related PRs:

- #24 — Promotion evidence contract + default train status
- #25 — Enforce evidence contract in `promote.py`
- #26 — Candidate status transitions
- #27 — Promotion status report

## Boundary

Module 7 is an offline promotion-decision and visibility layer.

It does **not**:

- deploy a model
- replace online monitoring
- prove production readiness by itself
- auto-promote without operator review (`--apply` remains explicit)

## Completion questions

1. **What is the difference between `candidate=true` and `current_candidate=true`?**

   - `candidate=true` is historical: the run was selected at some point.
   - `current_candidate=true` is the active pointer: only the current winner should have this.

2. **Why can a strong metric run still fail promotion?**

   - Because selection requires the full gate stack: valid comparison group, policy thresholds, baseline improvement, baseline guard, and complete promotion evidence.

3. **When is a non-current baseline allowed?**

   - Only with explicit override: `--allow-non-current-baseline` plus `--baseline-override-reason`.

4. **What does the promotion status report prove that `promote.py` alone does not?**

   - Cross-run visibility: status counts, filter examples, single-current-candidate health, and selected-evidence checks over the whole experiment — without mutating MLflow.

5. **What should an operator do after dry-run shows `selected`?**

   - Review the candidate selection report and evidence, then re-run with `--apply` only if the decision is intentional.

6. **What status should a brand-new training run have before any promote evaluation?**

   - `candidate_status=not_evaluated` with `promotion_reason=not_evaluated`.
