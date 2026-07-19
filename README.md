# Alexandria

[![Coverage badge](https://github.com/ucsc-cse115a-alexandria/alexandria/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/ucsc-cse115a-alexandria/alexandria/tree/python-coverage-comment-action-data)

Label-free prompt optimization: shorten instruction-heavy prompts while preserving their meaning,
using sentence embeddings. Alexandria finds which instructions overlap (a redundancy score) and
drops the redundant ones — no labels, no training, no target output to compare against.

See [the design specification](docs/spec.md) for the implementation architecture.

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## CLI

Run the full optimization pipeline with one command:

```bash
uv run alexandria reduce prompt.txt > reduced.txt
```

`report` runs the full optimization and always emits machine-readable JSON with token metrics and
quality scores:

```bash
uv run alexandria report prompt.txt --model deterministic --drift-budget 2.0
```

The `tokens` object reports source, reduced, and saved tokens. The `quality` object reports the
token-weighted mean and minimum best-match cosine similarity for every source instruction. To fail
when a report is worse than a committed baseline, pass the baseline file:

```bash
uv run alexandria report benchmarks/optimization_prompt.txt \
  --model deterministic \
  --threshold 0.85 \
  --drift-budget 2.0 \
  --baseline benchmarks/optimization_baseline.json
```

The command exits with status 1 when reduced token count rises or either monitored quality score
falls beyond its tolerance. Use `--token-tolerance` and `--quality-tolerance` for expected numerical
variation.

For phase-by-phase execution, saving and resuming JSON envelopes, options, and offline runs, see
[the CLI guide](docs/cli.md).

## Library

The CLI is a thin wrapper; everything is importable. Call `reduce` directly from Python:

```python
import alexandria

result = alexandria.reduce("Be concise.\nBe concise.\nUse examples.\n")
print(result.text)
```

See [the library guide](docs/library.md) for deterministic and offline embedding, direct phase
composition, and a runnable example in `examples/reduce_prompt.py`.

## How it works

Three pure phases over one intermediate representation (`Document` → `Section` → `Sentence`):

1. **Represent** — split the prompt into instructions, tokenize, and embed each one.
2. **Score** — rate each instruction's redundancy (its cosine similarity to the most similar other).
3. **Optimize** — drop redundant instructions while preserving meaning, keeping the more
   load-bearing instruction of each near-duplicate pair.

## Tech stack

Python 3.14 · Pydantic (the validated IR) · sentence-transformers · NumPy · tiktoken · click.

## Development

```bash
uv run pytest        # tests + coverage
uv run ruff check .  # lint
uv run pyright       # types
```

See [the contributing guide](docs/contributing.md) for the optimization-quality CI command, the
committed report baseline, and the review process for intentional baseline updates.
