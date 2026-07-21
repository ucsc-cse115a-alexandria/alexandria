# Contributing to Alexandria

This guide explains how to set up the repository, run the required checks, write tests, and prepare a pull request. [`pyproject.toml`](pyproject.toml) defines the formatting, lint, typing, test, coverage, and import rules.

The Scrum roles and Definition of Done are in [`docs/working-agreement.md`](docs/working-agreement.md).

## Setup

Alexandria requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev --frozen
uv run alexandria --help
```

Commands that embed or rewrite text need an OpenAI API key. Store one with `uv run alexandria config set openai-api-key`, or set `OPENAI_API_KEY`. Most tests run without a key. Tests marked `ai` call OpenAI and skip when a key is not available.

## Required checks

CI runs these checks on every push and pull request to `main`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run lint-imports
uv run pytest
```

Do not hide an error with `# noqa`, `# type: ignore`, or a similar inline exception. Fix the cause. The repository requires at least 80% branch coverage.

## Code conventions

Ruff owns formatting and lint rules. Pyright runs in strict mode. Import-linter checks the package layers.

The main layer direction is:

```text
cli -> ops -> utils -> ir
```

`ir` contains shared contracts. Features under `ops.features` do not import each other. The OpenAI client is created only behind the utility boundary. See [`pyproject.toml`](pyproject.toml) for the complete contracts and exceptions.

## Testing

Unit tests use the `*_test.py` name and usually sit beside the code they test. Broader end-to-end tests are in [`tests/`](tests/). Tests for repository scripts and benchmark code stay beside those modules.

Pass dependencies into code instead of mocking Alexandria internals. Pipeline tests use `HashEmbedder` and small fake mergers for deterministic offline checks.

Run live tests only when you intend to make API calls:

```bash
uv run pytest -m ai
```

## Benchmark reports

Benchmark result directories contain raw records, a manifest, a JSON summary, and `report.md`. Treat the raw records and manifest as evidence. Do not edit a generated `report.md` directly.

For a retained-target run, rebuild `summary.json` and `report.md` from its saved records:

```bash
uv run python -m scripts.summarize_prompt_compression_benchmark \
  benchmarks/BENCHMARK/results/RUN_DIRECTORY \
  --release-threshold 0.90 --bootstrap-samples 10000 --bootstrap-seed 42
```

If the report format changes, update its generator and tests, then regenerate every affected report. Check that the committed table still matches `summary.json`.

The `alexandria report` command can compare a new optimization report with a compatible baseline supplied by the user. The repository does not commit a baseline or run this comparison in CI.

## Pull requests

- Keep one concern in each pull request.
- Separate structural changes from behavior changes.
- Add a test for every behavior change.
- Update user documentation when behavior or commands change.
- Explain what changed and why.
- Do not merge until CI passes and a teammate approves.
