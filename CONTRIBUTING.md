# Contributing to Alexandria

This guide covers local setup, the code conventions, and how a change gets merged. The team keeps
written rules short: style, formatting, typing, and import structure are decided by the tool
configuration in [`pyproject.toml`](pyproject.toml) and enforced by CI, not by prose you have to
remember. If the checks pass, the code is in house style, so review can focus on whether the change
is correct.

The Scrum working agreement (roles, Definition of Done, meeting cadence) lives in
[`docs/working-agreement.md`](docs/working-agreement.md).

## Setup

Alexandria needs Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev --frozen
uv run alexandria --help
```

Pipeline commands call the OpenAI API. Set a key with `uv run alexandria config set openai-api-key`
or `export OPENAI_API_KEY=...`. Tests do not need a key unless you run the live ones (see Testing).

## The checks

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request to
`main`. All five must pass before a merge. Run them locally with the same commands:

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format  (apply with: uv run ruff format .)
uv run pyright               # types (strict)
uv run lint-imports          # import layering
uv run pytest                # tests + coverage (80% branch gate)
```

Never silence a check with an inline ignore comment (`# noqa`, `# type: ignore`, `# ruff: noqa`,
and the like). Fix the underlying cause instead.

## Code conventions

The style guide is the tool configuration. The rules below are what those commands enforce, so you
know them without opening `pyproject.toml`.

### Ruff

- Line length 119, target Python 3.14, double quotes.
- Lint rule sets: E, F, I (isort), UP (pyupgrade), B (bugbear), SIM (simplify), C4 (comprehensions),
  TRY (tryceratops), ASYNC, TCH (type-checking imports), FA (future annotations), S (bandit
  security), ARG (unused arguments), PERF (perflint), RUF, PIE, RET (returns), SLF (private-member
  access), PTH (use pathlib).
- Ignored globally: TRY003 (long inline exception messages), S311 (non-crypto random), RUF001
  (fullwidth characters, for Japanese content).
- Per file: `tests/**` and `*/*_test.py` allow `assert` (S101 off); `scripts/**` allow subprocess
  and URL-open calls (S603, S310 off).

### Pyright

Strict mode, over `src` and `benchmarks`.

### Import layering

`lint-imports` enforces a functional-core / imperative-shell design. Root package `alexandria`:

- Layers `cli` -> `ops` -> `utils` -> `ir`. `ir` is the shared contract layer any layer may import.
- `cli` reaches `utils` only through `ops` (no direct `cli` -> `utils` import).
- `ops.pipe` may import features; a feature never imports `ops.pipe`.
- Features are independent: a feature may not import a sibling feature. Shared types live in
  `ir.contracts`.
- Only the `utils` shell constructs OpenAI clients. `ir`, `ops`, and `cli` may not import `openai`.

### Coverage

Measured over `src/alexandria` with branch coverage on. CI fails below 80%. `__init__.py` and
`*_test.py` are omitted.

## Testing

- Write code that can be unit-tested without mocking our own code. The pipeline takes its
  dependencies as arguments (an `Embedder` and a `SentenceMerger`), so tests pass a deterministic
  offline `HashEmbedder` and a small fake merger instead of patching internals.
- Unit tests sit next to the module they cover as `<module>_test.py`. Broader end-to-end tests live
  in [`tests/`](tests/).
- Live end-to-end tests that call OpenAI carry the pytest `ai` marker. They skip unless a key is
  resolvable. Run them on purpose with `uv run pytest -m ai`.
- Decide the test cases when you define the task: know what must pass before you start.

A separate prompt-quality regression report is described in
[`docs/contributing.md`](docs/contributing.md). Its CI workflow is not yet on `main` (see the known
problems in the [Release Summary](docs/release-summary.md)).

## Pull requests

- Keep pull requests small and reviewable, one concern each.
- Separate structural changes (rename, move, extract) from behavior changes.
- Cover every behavior change with a test that fails without the change.
- Do not merge until CI is green.
- Write a description that says what changed and why.
