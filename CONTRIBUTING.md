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

## Prompt-quality regression report

Alexandria has a deterministic quality check that guards against optimization regressions: it runs
`alexandria report` on a fixed prompt and compares the result against a committed baseline. The
report command and its prompt ([`benchmarks/optimization_prompt.txt`](benchmarks/optimization_prompt.txt))
are on `main`; the CI workflow (`.github/workflows/optimization-quality.yml`) and the committed
`benchmarks/optimization_baseline.json` are not on `main` yet (see the known problems in the [Release
Summary](docs/release-summary.md)). Once the baseline is committed, run the same check locally:

```bash
uv run alexandria report benchmarks/optimization_prompt.txt \
  --model deterministic \
  --optimizer greedy_pairwise \
  --selector auto \
  --threshold 0.85 \
  --drift-budget 2.0 \
  --baseline benchmarks/optimization_baseline.json \
  --token-tolerance 0 \
  --quality-tolerance 0.0
```

The command exits with status 1 when the reduced token count rises above the baseline or either
monitored quality score falls below it. Zero tolerance is safe because the deterministic embedder and
locked dependencies make the fixture reproducible.

Update the baseline only when a metric change is intentional and explained by the same pull request
(an approved optimizer or selector change, a report-calculation fix, or a deliberate change to the
prompt or configuration). Do not update it to make an unexplained regression pass. Generate a
candidate, review the diff, then replace it and rerun the check:

```bash
uv run alexandria report benchmarks/optimization_prompt.txt \
  --model deterministic --optimizer greedy_pairwise --selector auto \
  --threshold 0.85 --drift-budget 2.0 > benchmarks/optimization_baseline.json.new
git diff --no-index benchmarks/optimization_baseline.json benchmarks/optimization_baseline.json.new || true
mv benchmarks/optimization_baseline.json.new benchmarks/optimization_baseline.json
```

Commit the baseline together with the change that justified it.

## Pull requests

- Keep pull requests small and reviewable, one concern each.
- Separate structural changes (rename, move, extract) from behavior changes.
- Cover every behavior change with a test that fails without the change.
- Do not merge until CI is green.
- Write a description that says what changed and why.
