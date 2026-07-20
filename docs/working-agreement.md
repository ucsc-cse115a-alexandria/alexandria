# Team Working Agreement

- Product: Alexandria
- Team: Alexandria
- Date: 2026-07-20

| Role | Member |
| --- | --- |
| Product Owner | Masa Ishihara |
| Scrum Master | Matthew Zerner |
| Team | Virinchi Chintala |
| Team | Marc Dylan Tan |

## How we keep the code uniform

We keep written rules short. Style, formatting, typing, and import structure are decided by tool configuration in `pyproject.toml` and enforced by CI, not by prose that contributors have to read and remember. If a check passes, the code is in the house style; if it fails, CI blocks the merge. This frees human review to focus on whether a change is correct and does what it should, not on how it is formatted.

So this document has two jobs: state when a change is done, and point you at the tool config that is the real style guide. You should be able to contribute without asking a maintainer how the team works.

## Definition of Done

A change is done when all of these hold:

- Acceptance criteria for the task are met.
- Code and tests are merged to `main`.
- CI is green across all five checks (lint, format, types, imports, tests).
- Branch coverage is at least 80%.
- User-facing docs are updated when behavior or the CLI changes.
- At least one teammate has reviewed and approved. (draft: confirm the required reviewer count; it is not written down elsewhere.)
- Every behavior change is covered by a test that fails without the change.

## Style guide

The style guide is the tool configuration. CI runs five checks on every push and on every pull request to `main`, in this order, and all must pass before a merge:

1. `uv run ruff check .` for lint
2. `uv run ruff format --check .` for format
3. `uv run pyright` for types (strict)
4. `uv run lint-imports` for import layering
5. `uv run pytest` for tests and coverage

A coverage-comment action posts the coverage number on each pull request. To reproduce a check locally, run the same command without the CI wrapper. To fix formatting instead of only checking it, run `uv run ruff format .`.

The rules those commands enforce, so you know them without opening `pyproject.toml`:

### Ruff

- Line length 119, target Python 3.14, double quotes.
- Lint rule sets: E, F, I (isort), UP (pyupgrade), B (bugbear), SIM (simplify), C4 (comprehensions), TRY (tryceratops), ASYNC, TCH (type-checking imports), FA (future annotations), S (bandit security), ARG (unused args), PERF (perflint), RUF, PIE, RET (returns), SLF (private-member access), PTH (use pathlib).
- Ignored globally: TRY003 (long inline exception messages are allowed), S311 (non-crypto random is allowed), RUF001 (fullwidth characters are allowed for Japanese content).
- Per file: `tests/**` and `*/*_test.py` allow `assert` (S101 off); `scripts/**` allow subprocess and URL-open calls (S603, S310 off).

### Pyright

Strict mode. It checks `src` and `benchmarks`.

### Import layering

`lint-imports` enforces the functional-core / imperative-shell design. Root package is `alexandria`. The contracts:

- Layered architecture: `cli` → `ops` → `utils` → `ir`. `ir` is the shared contract layer that any layer may import.
- CLI reaches `utils` only through `ops`; `cli` must not import `utils` directly.
- `ops.pipe` may import features; features must not import `ops.pipe`.
- Features are independent: a feature may not import a sibling feature. Shared types live in `ir.contracts`.
- The OpenAI client is isolated to the shell: `ir`, `ops`, and `cli` must not import `openai`. Only the `utils` shell constructs OpenAI clients.

### Coverage

Measured over `src/alexandria` with branch coverage on. CI fails below 80%. `__init__.py` and `*_test.py` are omitted.

## Testing conventions

Write code that can be unit-tested without mocking our own code. The pipeline takes its dependencies as arguments: an `Embedder` and a `SentenceMerger`. Tests pass a deterministic offline `HashEmbedder` and a small fake merger instead of patching internals. The OpenAI API is reached only through those injected boundaries, so tests never call the real API.

- Unit tests sit next to the module they cover as `<module>_test.py` (for example `pipe_test.py` beside `pipe.py`). Broader end-to-end tests live in `tests/`.
- Live end-to-end tests that need a real OpenAI key carry the pytest `ai` marker. They are skipped automatically unless a key is resolvable. Run them on purpose with `uv run pytest -m ai`.
- Branch coverage must stay at or above 80%; CI enforces it.
- Decide the test cases when you define the task: know what must pass before you start.

## Team process and cadence

- Scrum meetings run three times a week: Monday 5:30pm, Thursday 5:15pm, Saturday 5:00pm.
- Keep pull requests small and reviewable.
- Do not merge a pull request until CI is green.
- Separate structural changes (rename, move, extract) from behavior changes. Put them in different commits or different pull requests.
- Write a pull request description that says what changed and why.
