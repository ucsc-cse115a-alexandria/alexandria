# Alexandria

[![Coverage badge](https://github.com/ucsc-cse115a-alexandria/alexandria/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/ucsc-cse115a-alexandria/alexandria/tree/python-coverage-comment-action-data)

Label-free prompt optimization: shorten instruction-heavy prompts while preserving their meaning,
using sentence embeddings. Alexandria finds which instructions overlap (a redundancy score) and
merges each near-duplicate pair into a single rewritten sentence — instead of dropping instructions,
an LLM fuses their intent — no labels, no training, no target output to compare against.

See [the design specification](docs/spec.md) for the implementation architecture.

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Setup

Alexandria uses OpenAI for embeddings (`text-embedding-3-small`) and merging (`gpt-5.6-luna`), so it
needs an API key. Store it once:

```bash
uv run alexandria config set openai-api-key
```

This prompts with hidden input and saves the key to `~/.config/alexandria/config.toml` (owner-only,
XDG-aware). You can instead `export OPENAI_API_KEY=...`. Resolution order is explicit argument, then
`OPENAI_API_KEY`, then the config file. Without a key, commands fail before any work with:

```text
OpenAI API key not found. Set it with `alexandria config set openai-api-key` or export OPENAI_API_KEY.
```

## CLI

Run the full optimization pipeline with one command:

```bash
uv run alexandria reduce prompt.txt > reduced.txt
```

Use `--save-tokens N` to stop once N tokens are saved and `--drift-budget` to cap the cumulative
whole-document embedding drift the reduction may accept (default: `0.5` = 50%):

```bash
uv run alexandria reduce prompt.txt --save-tokens 200 > reduced.txt
```

Use `--target-reduction P` when the reduction percentage is a requirement rather than a best-effort
budget. It exits with an error if the drift gate prevents reducing by P percent, so an unmet target
cannot be mistaken for success:

```bash
uv run alexandria reduce prompt.txt --target-reduction 10 > reduced.txt
```

Strict targets keep Markdown/XML boundaries fixed and ask the merge model to rewrite the largest content groups to
the exact budget needed by the complete prompt. The result must satisfy the requested token count and stay within
the whole-prompt embedding drift budget. Rejected attempts feed their measured token and drift failures back for
correction. Exact duplicate text in best-effort reduction is still removed without a merge-model call. `--json`
includes `merge_metrics` with call, retry, job, proposal, and applied-edit counts; text mode prints call and retry
counts to stderr. Add `-v`/`--verbose` (automatic reduction only) to stream that progress live to stderr instead of
waiting for the final summary.

`report` runs the full optimization and always emits machine-readable JSON with token metrics and
quality scores:

```bash
uv run alexandria report prompt.txt --drift-budget 2.0
```

The `tokens` object reports source, reduced, and saved tokens. The `quality` object reports the
token-weighted mean and minimum best-match cosine similarity for every source instruction. To fail
when a report is worse than a committed baseline, pass the baseline file:

```bash
uv run alexandria report benchmarks/optimization_prompt.txt \
  --drift-budget 2.0 \
  --baseline benchmarks/optimization_baseline.json
```

The command exits with status 1 when reduced token count rises or either monitored quality score
falls beyond its tolerance. Use `--token-tolerance` and `--quality-tolerance` for expected numerical
variation. Regenerate the baseline manually (it needs an API key and is not committed by CI):

```bash
uv run alexandria report benchmarks/optimization_prompt.txt > benchmarks/optimization_baseline.json
```

For phase-by-phase execution, saving and resuming JSON envelopes, and the full option reference, see
[the CLI guide](docs/cli.md).

## Library

The CLI is a thin wrapper; everything is importable. Call `reduce` directly from Python (it builds the
OpenAI defaults, so a key must be resolvable — pass `api_key=`, export `OPENAI_API_KEY`, or use a
`.env` file):

```python
import alexandria

result = alexandria.reduce("Be concise.\nBe concise.\nUse examples.\n")
print(result.text)
```

See [the library guide](docs/library.md) for injecting your own embedder and merger for offline tests,
direct phase composition, and a runnable example in `examples/reduce_prompt.py`.

## How it works

Four pure phases over one intermediate representation (`Document` → `Section` → `Sentence`):

1. **Represent** — split the prompt into instructions, tokenize, and embed each one.
2. **Score** — rate each instruction's redundancy (its cosine similarity to the most similar other).
3. **Optimize** — for each near-duplicate pair the LLM rewrites both sentences as one minimal-token
   sentence, kept at the first occurrence (the second is removed). Every rewrite is checked by
   applying it and measuring the whole-document embedding drift; if it exceeds the drift budget the
   LLM is re-asked with feedback, up to 3 attempts, then the pair is skipped.
4. **Select** — apply the accepted edits least-drift-first under the cumulative drift budget, stopping
   at the requested token budget. `--target-reduction` additionally fails the command if its percentage
   target is not reached.

## Tech stack

Python 3.14 · Pydantic (the validated IR) · openai · NumPy · tiktoken · click.

## Development

```bash
uv run pytest        # tests + coverage
uv run ruff check .  # lint
uv run pyright       # types
```
