# Definition of Done

This document defines the shared bar every Product Backlog Item must meet before it can be marked done. It is referenced from [`working-agreement.md`](working-agreement.md). Per-story acceptance criteria live separately in sprint plans (`docs/sprint-*-plan.md`) and `docs/release-summary.md`.

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
- Per-story acceptance criteria live in `docs/sprint-*-plan.md` and `docs/release-summary.md`.