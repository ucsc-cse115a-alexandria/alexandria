# Team Working Agreement

**Product:** Alexandria ·
**Team:** Alexandria ·
**Date:** 2026-07-20

| Role | Member |
| --- | --- |
| Product Owner | Masa Ishihara |
| Scrum Master | Matthew Zerner |
| Team | Virinchi Chintala |
| Team | Marc Dylan Tan |

## How we work

We keep written rules short. Style, formatting, typing, and import structure are decided by tool
configuration and enforced by CI, not by prose. Passing CI means the code is in house style, so
review focuses on whether a change is correct. The code conventions and the exact tool rules live in
the root [`CONTRIBUTING.md`](../CONTRIBUTING.md); this document covers the Scrum agreement.

## Definition of Done

A user story or task is done when all of these hold:

- Its acceptance criteria are met.
- Code and tests are merged to `main`.
- CI is green across all five checks (lint, format, types, import layering, tests).
- Branch coverage is at least 80%.
- User-facing docs are updated when behavior or the CLI changes.
- At least one teammate has reviewed and approved. (draft: confirm the required reviewer count.)
- Every behavior change is covered by a test that fails without the change.

## Style guide

The style guide is the tool configuration in `pyproject.toml`, enforced by the five CI checks and
summarized in [`CONTRIBUTING.md`](../CONTRIBUTING.md). Ruff owns lint and formatting, Pyright owns
strict typing, import-linter owns the layering, and coverage holds an 80% branch gate.

## Team process and cadence

- Scrum meetings run three times a week: Monday 5:30pm, Thursday 5:15pm, Saturday 5:00pm.
- Keep pull requests small and reviewable, and do not merge until CI is green.
- Separate structural changes (rename, move, extract) from behavior changes.
- Decide the test cases when you define the task.
- Write a pull request description that says what changed and why.
