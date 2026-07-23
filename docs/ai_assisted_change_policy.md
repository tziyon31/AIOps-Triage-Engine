# AI-Assisted Change Policy

## Purpose

This policy defines how AI-assisted work is reviewed and merged in this repository.

The goal is accountability, not bureaucracy.

AI may accelerate drafting and implementation.
The human PR author remains responsible for correctness, safety, and honesty of claims.

## Core rules

1. **Author accountability**
   - The person opening the PR owns the change.
   - "AI wrote it" is not a valid excuse for broken contracts, leaked secrets, or false claims.

2. **Explain what changed**
   - Every PR must summarize the change and why it exists.
   - The author must be able to explain the diff in their own words.

3. **Explain what was reviewed**
   - The author must state which files were reviewed.
   - The author must state which assumptions were checked.
   - The author must state which commands/tests were run.

4. **AI output is not evidence**
   - AI-generated text, summaries, or screenshots are not accepted as proof of correctness by themselves.
   - Meaningful validation comes from tests, CI, and (when relevant) MLflow evidence.

5. **Official evidence remains workflow-owned**
   - Local exploratory MLflow runs are not official evidence.
   - Official experiment writing belongs to Stage 5.6 Module #10 (GitHub Actions official writer) and is not implemented yet.
   - Official promotion apply belongs to Stage 5.6 Module #11 and is not implemented yet.

6. **Failed meaningful gates block merge**
   - A red meaningful CI gate blocks merge.
   - Do not bypass required checks for convenience.

## Risk guidance

| Change type | Minimum risk |
|---|---|
| Trivial docs-only / typo / comment | low |
| Tests that lock existing behavior | low / medium |
| Architecture boundaries, schemas, policy | medium |
| MLflow evidence semantics / tags / artifact contracts | medium |
| Promotion behavior / candidate status mutation | medium |
| Auth, secrets handling, IAM, deploy scripts | medium |
| Broad runtime decision behavior changes | high |

AI-assisted changes in medium/high areas require explicit manual review notes in the PR template.

## What this policy does not do

- Does **not** require automatic AI detection.
- Does **not** trust AI-detection tooling as a merge gate.
- Does **not** add fake plagiarism scanners.
- Does **not** add broad fragile grep rules for "AI style".
- Does **not** replace architecture-boundary tests, schema/policy tests, or quality-gate evidence.

## Meaningful gates already expected by CI

Current repository CI (`quality` job and related workflows) already covers meaningful checks such as:

- deterministic tests (`pytest`, including architecture-boundary tests when present)
- schema / policy validation covered by existing unit tests
- basic syntax/compile checks
- on `main`: pipeline / quality-gate evidence and selected integration jobs

This Module #5 adds a **human review contract** on top of those gates.
It does not invent a parallel heavyweight tooling stack.

## PR author obligations

Before requesting review/merge, the author must complete `.github/pull_request_template.md`, including:

- AI assistance disclosure
- manual review notes
- risk level
- contract/policy impact
- evidence (tests/CI/MLflow ids when relevant)
- rollback path
- confirmation that no secrets were committed

## Reviewer focus

Reviewers should prioritize:

- boundary violations
- schema/policy contract drift
- promotion/evidence semantics
- secret leakage
- false claims (for example: "Stage complete", "official workflow implemented", "AI detection is automatic")

Reviewers should not demand ceremony for trivial low-risk docs PRs beyond a short honest template fill-out.

## Relationship to Stage 5.6

- Module #5 = AI-assisted PR quality gate (this document + PR template)
- Module #10 = official experiment workflow (not started)
- Module #11 = official promotion workflow (not started)

Stage 5.6 remains incomplete until later modules are done.
