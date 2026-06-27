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

Each verb reads a prompt from a `FILE` argument or stdin and writes to stdout.

`reduce` — prompt in, reduced prompt out:

```bash
uv run alexandria reduce prompt.txt > reduced.txt
```

`score` — per-instruction redundancy scores (a table on a terminal, JSON when piped or with `--json`):

```bash
uv run alexandria score prompt.txt --json
```

By default the embeddings come from `sentence-transformers` (`all-MiniLM-L6-v2`), downloaded on first
use. Pass `--model deterministic` for a fast, offline run with a non-semantic hash embedder:

```bash
$ printf 'Be concise.\nBe concise.\nUse examples.\n' | uv run alexandria reduce --model deterministic
Be concise.
Use examples.
```

Options: `reduce` takes `--optimizer` and `--threshold`; both verbs take `--model`.

## Library

The CLI is a thin wrapper; everything is importable.

```python
from alexandria import reduce
from alexandria.runtime.embedding import build_embedder

embedder = build_embedder("all-MiniLM-L6-v2")
reduced = reduce("Be concise.\nBe concise.\nUse examples.\n", embedder)
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
