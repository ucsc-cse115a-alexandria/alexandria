# Alexandria

[![Coverage badge](https://github.com/ucsc-cse115a-alexandria/alexandria/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/ucsc-cse115a-alexandria/alexandria/tree/python-coverage-comment-action-data)

Label-free prompt optimization: shorten instruction-heavy prompts while preserving their meaning.
Alexandria finds near-duplicate instructions with sentence embeddings and has an LLM merge each pair
into a single rewritten sentence — no labels, no training, no target output to compare against.

## Prerequisites and install

Alexandria is currently version `0.1.0`; its pre-1.0 API may still change. It requires CPython 3.14,
[uv](https://docs.astral.sh/uv/), Git, and internet access for installation. The current release path
is the public GitHub repository because `alexandria-prompt` is not yet published on PyPI. No GitHub
authentication is required when installing over HTTPS.

```bash
# Install the CLI in an isolated tool environment.
uv tool install "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"

# Or run one command without keeping the tool installed.
uvx --from "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git" \
  alexandria --help

# Add the importable library to an existing uv project.
uv add "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"
```

The distribution name is `alexandria-prompt`; the command and import name are both `alexandria`.
See the [release installation guide](docs/release-install.md) for clean-environment checks, the
installed CLI and API smoke tests, and Windows notes.

## Quickstart

Normal reductions call OpenAI for embeddings and merging. They require an API key, network access,
and access to the configured models. Store a key once, or export `OPENAI_API_KEY`:

```bash
alexandria config set openai-api-key

cat > prompt.txt <<'EOF'
Keep your answers concise and to the point.
Keep your answers brief and to the point.
Use examples.
EOF

alexandria reduce prompt.txt > reduced.txt
cat reduced.txt
```

The examples below use the installed `alexandria` command. From a development checkout, substitute
`uv run alexandria`.

Common options:

- `--save-tokens N` stops once N tokens are saved.
- `--target-reduction P` treats a P% reduction as a hard requirement; the result never exceeds the
  derived token ceiling.
- `--cos-sim-diff-budget B` caps accepted semantic change (`1 - cosine_similarity`, default `0.5`).
- `--json` emits a machine-readable summary; `-v` streams progress to standard error.

`report` runs the same pipeline and emits JSON with token and quality metrics, optionally failing
against a baseline report saved earlier:

```bash
alexandria report prompt.txt
```

See the [CLI guide](docs/cli.md) for phase-by-phase execution, saved JSON envelopes, baseline
comparisons, and the full option reference.

## Library

The CLI is a thin wrapper; everything is importable. Call `reduce` directly from Python (it builds
the OpenAI defaults, so a key must be resolvable — pass `api_key=`, export `OPENAI_API_KEY`, or use
the saved config file):

```python
import alexandria

result = alexandria.reduce(
    "Keep your answers concise and to the point.\n"
    "Keep your answers brief and to the point.\n"
    "Use examples.\n"
)
print(result.text)
```

See [the library guide](docs/library.md) for injecting your own embedder and merger for offline
tests, and a runnable example in `examples/reduce_prompt.py`. The core library does not load `.env`
files automatically; that example opts into `python-dotenv` for local development.

## Benchmark

Hard-target reduction on BABILong 8k (n=50, seed 42, `gpt-5.6-luna` for compression and answers)
swept retained-prompt targets from 95% down to 75% against the uncompressed 72% baseline. Shallow
cuts keep most of the accuracy — an 8.0% token reduction (95% retained) scored 62.0% and a 13.2%
reduction (90% retained) scored 64.0% — while at 85% retained and below, accuracy falls to 44–48%.

![Task accuracy by retained prompt share](benchmarks/prompt_compression/results/2026-07-20-luna-keep75-95-n50-v1/accuracy_vs_retained.png)

See the [run report](benchmarks/babilong_8k/results/2026-07-20-luna-keep75-95-n50-v1/report.md) for
per-task results, timing, cost, and append-only raw artifacts, earlier studies under
[`benchmarks/prompt_compression/results/`](benchmarks/prompt_compression/results/), and the
[benchmark runner guide](benchmarks/prompt_compression/README.md) for executing a new run.

A `cos_sim_diff_budget` sweep (0.0025–0.02, BABILong 8k, n=50, seed 42) found no operating point
that clears the 90% accuracy-retention release threshold with 95% confidence, including the shipped
default. **Shipped default:** 70.0% accuracy, 0.8% token reduction, n=50. Token reduction and
cost stayed flat across the whole range (~0.7–0.8%, ~$0.58 per run) — BABILong 8k prompts have
little sentence-level redundancy for this compression mode to find. The default remains unchanged.
See the [P0–P6 sweep report](benchmarks/prompt_compression/results/2026-07-20-babilong-p0-p6-sweep-n50-v1/report.md)
for the full comparison table, time/cost breakdown, and evidence.

### Reproduce the published run

The benchmark harness is repository tooling rather than part of the installed CLI wheel. After
installing Alexandria, clone the repository and run the harness from that checkout.
The published run requires an OpenAI key, access to `gpt-5.6-luna`, the pinned BABILong download,
network access, and a cost allowance of up to USD 15. Model output is nondeterministic, so a rerun
reproduces the procedure and evidence format rather than byte-identical responses.

```bash
git clone https://github.com/ucsc-cse115a-alexandria/alexandria.git
cd alexandria
uv sync --frozen
uv run python -m scripts.download_babilong_8k_data
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k \
  --n 50 --seed 42 \
  --reductions 25 20 15 10 5 \
  --data-dir data/babilong/8k \
  --model gpt-5.6-luna \
  --merge-model gpt-5.6-luna \
  --min-original-accuracy 0.50 \
  --max-estimated-cost-usd 15 \
  --out trial_results/babilong_8k/keep75-95-reproduction
```

Run the same command with `--dry-run` first to validate sampling and output setup without model calls.
The committed [manifest](benchmarks/babilong_8k/results/2026-07-20-luna-keep75-95-n50-v1/manifest.json)
records the original command, models, case IDs, provenance, and dirty working-tree status.

## How it works

The composable API and CLI expose four phases over one validated intermediate representation
(`Document` → `Section` → `Sentence`). The default end-to-end reduction does not need a standalone
Score result: `merge_rewrite` ranks pairs directly from their embeddings, so `reduce` skips Score
unless a selected optimizer declares that it needs a scorer.

1. **Represent** — split the prompt into instructions, tokenize, and embed each one.
2. **Score** — optionally rate each instruction's redundancy (cosine similarity to its most similar
   other) when a selected optimizer needs scores.
3. **Optimize** — an LLM rewrites each near-duplicate pair as one minimal-token sentence, re-checked
   against the semantic budget with up to 3 attempts.
4. **Select** — apply accepted edits in ascending `cos_sim_diff` order under the cumulative budget,
   stopping at the requested token budget; `--target-reduction` uses a hard-target path that repairs
   model overshoot deterministically.

See [the design specification](docs/spec.md) for the implementation architecture.

## Development

```bash
git clone https://github.com/ucsc-cse115a-alexandria/alexandria
cd alexandria
uv sync

uv run pytest                          # tests + coverage
python scripts/release_smoke_test.py   # build/install release artifacts; verify CLI + API
uv run ruff check .                    # lint
uv run pyright                         # types
```
