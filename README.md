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

Use `--save-tokens N` to stop once N tokens are saved and `--cos-sim-diff-budget` to cap the cumulative
whole-document `cos_sim_diff` (`1 - cosine_similarity`) the reduction may accept (default: `0.5`):

```bash
uv run alexandria reduce prompt.txt --save-tokens 200 > reduced.txt
```

Use `--target-reduction P` when the reduction percentage is a requirement rather than a best-effort
budget. The returned prompt is always at or below the derived token ceiling. The command fails before
calling the merge model only when protected Markdown/XML structure alone cannot fit:

```bash
uv run alexandria reduce prompt.txt --target-reduction 10 > reduced.txt
```

Strict targets keep Markdown/XML boundaries fixed and, for each content group, fire three generation requests in
parallel with different rewrite strategies (plain compression, extractive deletion, and dense paraphrase), each
capped so a completed response fits the token budget. Alexandria checks every candidate with `cl100k_base` and
deterministically repairs any overshoot. Among the structure-valid candidates within the token ceiling it selects
the one with the lowest whole-prompt `cos_sim_diff`, breaking ties by coverage. Undershooting the target is acceptable: the
guarantee is at most the requested token count. When no candidate meets the `cos_sim_diff` budget, the best target-safe
result is returned and `merge_metrics.cos_sim_diff_budget_met` is `false`. `--json` also reports final `cos_sim_diff`, repaired tokens, calls, and retries.
Exact duplicate text in best-effort reduction is still removed without a merge-model call. Text mode prints call
and retry counts to stderr. Add `-v`/`--verbose` to stream automatic-reduction progress live to stderr instead of
waiting for the final summary.

`report` runs the full optimization and always emits machine-readable JSON with token metrics and
quality scores:

```bash
uv run alexandria report prompt.txt --cos-sim-diff-budget 2.0
```

The `tokens` object reports source, reduced, and saved tokens. The `quality` object reports the
token-weighted mean and minimum best-match cosine similarity for every source instruction. To fail
when a report is worse than a committed baseline, pass the baseline file:

```bash
uv run alexandria report benchmarks/optimization_prompt.txt \
  --cos-sim-diff-budget 2.0 \
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

## Benchmark

The current study measures 50 cases each from BABILong 8k and RULERv2 with seed 42, using `gpt-5.6-luna` for both
compression and answers. It tests best-effort context `cos_sim_diff` budgets from 0.0025 through 0.02. Within this
range, completed prompts were reduced by 0.40%–0.51% on average while mean realized full-prompt `cos_sim_diff`
remained between 0.0019 and 0.0024.

| Configured budget | Average task accuracy | Mean token reduction | Complete case-condition pairs |
|---:|---:|---:|---:|
| Original | 76.0% | 0.00% | 100.0% |
| 0.0025 | 63.3% | 0.40% | 79.0% |
| 0.005 | 60.9% | 0.43% | 80.0% |
| 0.01 | 58.3% | 0.46% | 78.0% |
| 0.015 | 56.0% | 0.48% | 79.0% |
| 0.02 | 63.4% | 0.51% | 79.0% |

`Average` is the equal-weight mean of the two benchmarks. Accuracy uses completed paired cases; completion is
shown alongside it so the effective coverage remains visible.

![Quality and prompt reduction by semantic-change budget](benchmarks/prompt_compression/results/2026-07-19-luna-cos-budget-n50-v1/quality_and_reduction_vs_budget.png)

A separate hard-target study used 50 cases per benchmark to retain 50%, 60%, 70%, 80%, and 90% of each prompt,
with the uncompressed prompt shown at 100%. This provides the broader compression-quality curve independently of
the semantic-change budget above.

![Task accuracy by retained prompt percentage](benchmarks/prompt_compression/results/2026-07-19-luna-keep50-90-n50-v1/accuracy_vs_retained.png)

See the
[detailed benchmark report](benchmarks/prompt_compression/results/2026-07-19-luna-cos-budget-n50-v1/report.md) for
benchmark-specific results, paired-bootstrap intervals and decisions, compliance and completion analysis, exact
reproduction commands, timing and cost, caveats, and links to every append-only raw artifact. The
[benchmark runner guide](benchmarks/prompt_compression/README.md) documents the shared evidence format and how to
execute a new run.

## How it works

Four pure phases over one intermediate representation (`Document` → `Section` → `Sentence`):

1. **Represent** — split the prompt into instructions, tokenize, and embed each one.
2. **Score** — rate each instruction's redundancy (its cosine similarity to the most similar other).
3. **Optimize** — for each near-duplicate pair the LLM rewrites both sentences as one minimal-token
   sentence, kept at the first occurrence (the second is removed). Every rewrite is checked by
   applying it and measuring the whole-document `cos_sim_diff`; if it exceeds the `cos_sim_diff` budget the
   LLM is re-asked with feedback, up to 3 attempts, then the pair is skipped.
4. **Select** — apply accepted edits in ascending `cos_sim_diff` order under the cumulative budget, stopping
   at the requested token budget. `--target-reduction` uses a hard-target path that repairs model overshoot
   deterministically and reports whether the resulting prompt also met the `cos_sim_diff` budget.

## Tech stack

Python 3.14 · Pydantic (the validated IR) · openai · NumPy · tiktoken · click.

## Development

```bash
uv run pytest        # tests + coverage
uv run ruff check .  # lint
uv run pyright       # types
```
