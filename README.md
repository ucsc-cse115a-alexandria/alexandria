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

The current evidence run measures 50 samples each from BABILong 8k and RULERv2 with seed 42. It uses
`gpt-5.6-luna` with reasoning `none` for both compression and answers, plus `text-embedding-3-small`. The runner
measures every original first and only runs compressed conditions when original accuracy is at least 50%.
BABILong (68%) and RULERv2 (74%) both passed that gate.

The table and `Average` curve are the equal-weight mean of the two benchmarks. `Accuracy retention` is calculated
per benchmark relative to its own original accuracy before averaging.

| Prompt retained | Average accuracy | Accuracy retention | Achieved token reduction | Mean `cos_sim_diff` | Measured time |
|---:|---:|---:|---:|---:|---:|
| 50% | 22.0% | 30.6% | 58.2% | 0.090447 | 2,008.4s |
| 60% | 36.0% | 50.7% | 45.3% | 0.054069 | 1,838.2s |
| 70% | 38.0% | 53.1% | 34.5% | 0.036756 | 1,518.4s |
| 80% | 45.0% | 63.3% | 23.9% | 0.025185 | 1,186.8s |
| 90% | 61.0% | 86.7% | 13.5% | 0.016461 | 901.5s |
| 100% (original) | 71.0% | 100.0% | 0.0% | 0.000000 | 116.8s |

![Task accuracy by retained prompt percentage](benchmarks/prompt_compression/results/2026-07-19-luna-keep50-90-n50-v1/accuracy_vs_retained.png)

![Accuracy retention by retained prompt percentage](benchmarks/prompt_compression/results/2026-07-19-luna-keep50-90-n50-v1/accuracy_retention_vs_retained.png)

`cos_sim_diff` (`1 - cosine_similarity` between the original and reduced whole-prompt embeddings)
is the quality knob the CLI exposes as `--cos-sim-diff-budget`. The next two figures connect it to
both sides of the trade-off — how much semantic change each retention level introduces, and how
accuracy falls as semantic change grows — so you can pick a budget instead of guessing:

![Cosine difference by retained prompt percentage](benchmarks/prompt_compression/results/2026-07-19-luna-keep50-90-n50-v1/cos_sim_diff_vs_retained.png)

![Accuracy and retention versus semantic change](benchmarks/prompt_compression/results/2026-07-19-luna-keep50-90-n50-v1/semantic_change_vs_accuracy.png)

All five compressed conditions failed the predeclared paired-bootstrap release rule on both qualified benchmarks.
The two published runs recorded 600 case/condition results and 4,946 metered usage events. Their summed measured
reduction-plus-answer time was 7,570.1 seconds. The per-condition time table sums measured API work across both
benchmarks; it is not the elapsed time of the parallel shell processes.

The raw per-benchmark results, pass/fail decisions, exact commands, and links to the append-only records are in the
[shared benchmark documentation](benchmarks/prompt_compression/README.md).

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
