## Summary

<!-- What changed and why (1–3 bullets). -->

-

## Stage / module

- Stage:
- Module:

## Type of change

- [ ] docs
- [ ] tests
- [ ] refactor (no behavior change)
- [ ] feature
- [ ] bug fix
- [ ] infra / CI
- [ ] experiment / evidence tooling
- [ ] other:

## AI assistance disclosure

Select one:

- [ ] none
- [ ] AI generated draft
- [ ] AI materially modified code/docs
- [ ] AI used for review only

Notes (optional):

<!-- What the AI did / what you changed after review. -->

## Manual author review

The PR author remains accountable for this change.

- Files reviewed:
  -
- Assumptions checked:
  -
- Commands / tests run locally:
  -

## Risk level

Select one:

- [ ] low
- [ ] medium
- [ ] high

Guidance:

- Trivial docs-only / comment-only changes may be **low**.
- Changes touching architecture boundaries, schemas, policy, promotion, MLflow evidence semantics, auth/secrets, or deploy scripts are at least **medium**.
- Broad runtime/decision-behavior changes or irreversible evidence mutations are **high**.

## Contract / policy impact

Select all that apply:

- [ ] architecture boundary
- [ ] schema
- [ ] policy
- [ ] MLflow evidence
- [ ] promotion behavior
- [ ] none

## Evidence

- Tests run:
  -
- CI links:
  -
- MLflow run id (if relevant):
  - `official=false` exploratory only unless produced by the future official Actions writer

## Known limitations

-

## Rollback / revert path

<!-- How to undo this change safely. -->

-

## Checklist

- [ ] I reviewed the diff myself
- [ ] I understand the change and can explain it
- [ ] Meaningful CI gates are expected to pass (`quality` / relevant jobs)
- [ ] No secrets, credentials, private keys, or raw DB URLs committed
- [ ] No claim that AI output alone is evidence
- [ ] Official experiment/promotion workflows were not falsely claimed as implemented
- [ ] If this touches medium/high-risk areas, I documented risk and validation

## References

- Policy: `docs/ai_assisted_change_policy.md`
- Architecture boundaries: `docs/architecture_boundaries.md`
- Stage status: `docs/stage_5_6_status.md`
