# Quality Gate — Stage 4 Module #12

## Purpose

The Quality Gate is the trust boundary for the Decision Artifact pipeline.

Its purpose is to ensure that a bad artifact cannot be treated as publishable unless the required production checks pass.

A green test suite alone is not enough. This gate requires evidence that the artifact is packaged, identifiable, loadable, contract-compatible, policy-checked, smoke-tested, and traceable.

---

## What This Gate Proves

This Quality Gate proves that:

1. The artifact package was created successfully.
2. The artifact includes a `manifest.json`.
3. The manifest records identity and lineage metadata such as:
   - `git_sha`
   - `run_id`
   - model/vectorizer/action hashes
   - config hash
   - training data hash
4. The artifact files can be loaded.
5. The predictor returns a valid Decision Object.
6. The prediction output follows the expected schema and contract.
7. The policy layer blocks forbidden actions and forces approval for risky or low-confidence cases.
8. A smoke prediction can run successfully against the generated artifact.
9. Traceability works: a decision can be connected back to the artifact and manifest.
10. A reviewer can re-run the deterministic gate from one command.

---

## What This Gate Does Not Prove Yet

This Quality Gate does not yet prove that:

1. The model is accurate enough for broad real-world production traffic.
2. The training dataset represents real production diversity.
3. The system is protected against data drift.
4. The system meets production latency, cost, or scale requirements.
5. The LLM fallback produces consistently high-quality answers.
6. The system has full security hardening.
7. The artifact is stored in a full external immutable registry such as MLflow, S3 with versioning, or a production model registry.

For this stage, GitHub Actions artifacts act as the temporary registry-like storage. That is acceptable for the learning slice, but it is not a complete production registry.

---

## Automated Evidence

Each successful pipeline run generates machine-readable and human-readable evidence under:

```text
evidence/quality_gate/
```

Expected files include:

- `quality_gate_report.json`
- `quality_gate_report.md`
- `sample_decision.json`
- `manifest_hashes.json`
- `policy_tests.txt`
- `artifact_tests.txt`
- `quality_gate_report_tests.txt`
- `prediction_contract_tests.txt`
- `traceability_integration_tests.txt`
- `deterministic_test_suite.txt`
- `smoke_prediction_output.json`

The report includes:

- report version
- timestamp
- pipeline run id
- git sha
- artifact id
- artifact version
- manifest hashes
- check status
- sample Decision Object
- limitations

---

## CI Behavior

The GitHub Actions workflow enforces this order:

1. Run deterministic pipeline
2. Generate Quality Gate evidence
3. Validate Quality Gate evidence
4. Upload Decision Artifact
5. Upload Quality Gate evidence
6. Add report to GitHub job summary

The important rule is:

**The Decision Artifact is uploaded only after the Quality Gate evidence has been validated.**

This prevents an unverified artifact from being treated as publishable.

---

## Re-run Instructions

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
LOG_TRIAGE_DISABLE_LLM=1 ./run_pipeline.sh
python scripts/validate_quality_gate_evidence.py
```

Expected result:

```text
Pipeline completed successfully
Quality Gate evidence validation passed.
```

---

## One-Command Gate

The main deterministic gate is:

```bash
LOG_TRIAGE_DISABLE_LLM=1 ./run_pipeline.sh
```

This command trains the model, builds the artifact, runs the required deterministic checks, performs smoke prediction, validates policy behavior, verifies traceability, and writes Quality Gate evidence.

---

## Current Production-Quality Meaning

For this slice, “production-quality” means the artifact is not just produced. It is produced with evidence.

A publishable artifact must be:

- identifiable
- reproducible enough for this stage
- loadable
- contract-compatible
- policy-checked
- smoke-tested
- traceable
- supported by CI evidence

This is not yet a complete production AI platform. It is a controlled trust boundary for the Decision Artifact stage.
