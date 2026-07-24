# Definition of Done

This document defines the shared bar every Product Backlog Item must meet before it can be marked done. It is referenced from [`working-agreement.md`](working-agreement.md). Acceptance criteria are defined per story in sprint plans and at release level in [`release-summary.md`](release-summary.md); see [Where acceptance criteria live](#where-acceptance-criteria-live) below.

## Where acceptance criteria live

This document defines the shared quality bar. The actual acceptance criteria for a story depend on scope:

- **Release criteria (canonical for 1.0):** [`release-summary.md`](release-summary.md) — US1–US5 outcomes for the shipped product.
- **Sprint criteria (planning history):** [`sprint-*-plan.md`](sprint-1-plan.md) — what each sprint committed to deliver; may differ from the final release.
- **Verification evidence:** [`test-plan-and-report.md`](test-plan-and-report.md) — commands, expected results, and recorded test outcomes.

A story is done when it meets its sprint acceptance criteria **and** this Definition of Done. Release-level criteria in `release-summary.md` describe what the product delivers overall, not every sprint task.

A story or task is done when all of the following hold.

## 1. Functional criteria

- Its acceptance criteria are met.
- Every behavior change is covered by a test that would fail without the change.

## 2. Code quality

- Code and tests are merged to `main`.
- CI is green on all five checks: lint, format, types, imports, tests.
- Branch coverage is at least 80%.

## 3. Review

- At least one teammate has reviewed and approved (marked as a draft, not required reviewer count).

## 4. Documentation

- User-facing docs are updated when behavior or the CLI changes.

## Out of scope for this document

- Style rules (Ruff, Pyright, import-linter, coverage thresholds) live in `CONTRIBUTING.md`.
- Acceptance criteria content — see [Where acceptance criteria live](#where-acceptance-criteria-live) above.