# Alexandria

[![Coverage badge](https://github.com/ucsc-cse115a-alexandria/alexandria/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/ucsc-cse115a-alexandria/alexandria/tree/python-coverage-comment-action-data)

Label-free prompt optimization: shorten instruction-heavy prompts while preserving their meaning,
using sentence embeddings. Alexandria finds which instructions overlap (a redundancy score) and
drops the redundant ones — no labels, no training, no target output to compare against.

See [docs/spec.md](docs/spec.md) for the design.

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## CLI

Each verb reads from a `FILE` argument or stdin and writes to stdout; diagnostics go to stderr.

`reduce` runs all four phases end to end — prompt in, reduced prompt out:

```bash
uv run alexandria reduce prompt.txt > reduced.txt
```

The four phases are also separate verbs that compose over a Unix pipe. Each carries the data the next
one needs as a self-contained JSON envelope (`represent` → `score` → `optimize` → `select`):

```bash
uv run alexandria represent < prompt.txt \
  | uv run alexandria score \
  | uv run alexandria optimize \
  | uv run alexandria select > reduced.txt
```

`score --table` prints a per-instruction redundancy report instead of a `ScoredEnvelope`:

```bash
uv run alexandria represent < prompt.txt | uv run alexandria score --table
```

`reduce --json` and `select --json` emit a summary (`text`, `applied`, `source_tokens`, `reduced_tokens`).

By default the embeddings come from `sentence-transformers` (`all-MiniLM-L6-v2`), downloaded on first
use. Pass `--model deterministic` for a fast, offline run with a non-semantic hash embedder:

```bash
$ printf 'Be concise.\nBe concise.\nUse examples.\n' | uv run alexandria reduce --model deterministic
Be concise.
Use examples.
```

`represent`, `select`, and `reduce` take `--model`; `optimize` and `reduce` take `--threshold`;
`select` and `reduce` take `--drift-budget`. Envelopes are JSON today (`schema_version=1`).

## Library

The CLI is a thin wrapper; everything is importable. `reduce` runs all four phases end to end:

```python
from alexandria import reduce
from alexandria.runtime.embedding import build_embedder

embedder = build_embedder("all-MiniLM-L6-v2")
reduced = reduce("Be concise.\nBe concise.\nUse examples.\n", embedder)
```

Each CLI verb maps to a function of the same name, so you can also compose the phases directly —
the same `represent` → `score` → `optimize` → `select` pipeline as the Unix pipe above:

```python
from alexandria import represent, score, optimize, select
from alexandria.runtime.embedding import build_embedder

embedder = build_embedder("all-MiniLM-L6-v2")
document = represent("Be concise.\nBe concise.\nUse examples.\n", embedder)
scores = score(document)
plan = optimize(document, scores)
selection = select(document, plan, embedder)
reduced = selection.document.text
```

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
