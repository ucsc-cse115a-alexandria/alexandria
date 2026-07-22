# Alexandria

[![Coverage badge](https://github.com/ucsc-cse115a-alexandria/alexandria/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/ucsc-cse115a-alexandria/alexandria/tree/python-coverage-comment-action-data)

Alexandria shortens instruction-heavy prompts without labels, training, or expected outputs. It finds similar instructions with sentence embeddings and asks an LLM to merge each pair into one shorter instruction.

Alexandria is at version `0.1.0`. The API may change before version 1.0. Normal reductions use the OpenAI API and may take several minutes or cost money.

## Install

Alexandria requires Python 3.14 or later, Git, and internet access. It is installed from GitHub because `alexandria-prompt` is not on PyPI yet.

```bash
# Install with pip.
python -m pip install "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"
alexandria --help

# Or install the CLI in an isolated uv environment.
uv tool install "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"

# Add the package to an existing uv project.
uv add "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"
```

The distribution name is `alexandria-prompt`. The command and import name are both `alexandria`. See the [installation guide](docs/release-install.md) for clean-environment checks and Windows notes.

## Quickstart

Store an OpenAI API key, then reduce a prompt:

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

From a development checkout, use `uv run alexandria` instead of `alexandria`.

Common controls:

- `--save-tokens N` requests a specific token saving.
- `--keep P` requests that the result retain P percent of the source tokens.
- `--target-reduction P` requires a P percent reduction.
- `--cos-sim-diff-budget B` limits accepted semantic change.
- `--json` returns a machine-readable summary.

Best-effort controls may stop before their requested target. `--target-reduction` treats the token ceiling as a requirement. Use `--interactive` or `--browser` to review edits before applying them.

See the [CLI guide](docs/cli.md) for all commands, review modes, saved JSON envelopes, and baseline reports.

## Python library

```python
import alexandria

result = alexandria.reduce(
    "Keep your answers concise.\n"
    "Keep your answers brief.\n"
    "Use examples.\n"
)
print(result.text)
```

The default library path also uses OpenAI. You can pass your own embedder and merger for offline use. See the [library guide](docs/library.md) and [`examples/reduce_prompt.py`](examples/reduce_prompt.py).

## Benchmark

A BABILong 8k run used 50 cases, seed 42, and `gpt-5.6-luna` for compression and answers. The original prompts scored 72.0% accuracy. Keeping 95% of the prompt produced an 8.0% token reduction and 62.0% accuracy. Keeping 90% produced a 13.2% reduction and 64.0% accuracy.

![Task accuracy by retained prompt share](benchmarks/prompt_compression/results/2026-07-20-luna-keep75-95-n50-v1/accuracy_vs_retained.png)

The benchmark measures downstream answer accuracy. It does not measure lossless preservation of every detail. Model output is nondeterministic, so another run may produce different values.

See the [run report](benchmarks/babilong_8k/results/2026-07-20-luna-keep75-95-n50-v1/report.md) for the saved measurements. The [benchmark guide](benchmarks/prompt_compression/README.md) explains the method, cost controls, raw artifacts, and reproduction commands.

## How it works

Alexandria uses a validated `Document`, `Section`, and `Sentence` model.

1. Represent splits, tokenizes, and embeds the prompt.
2. Score can measure sentence redundancy when an optimizer needs it.
3. Optimize proposes shorter replacements for similar instructions.
4. Select applies compatible edits within the semantic and token budgets.

The default optimizer ranks pairs directly, so `reduce` skips the optional Score phase. A separate hard-target path repairs model output until it reaches the required token ceiling when possible.

See the [design specification](docs/spec.md) for the full contracts.

## Development

```bash
git clone https://github.com/ucsc-cse115a-alexandria/alexandria
cd alexandria
uv sync

uv run pytest
python scripts/release_smoke_test.py
uv run ruff check .
uv run pyright
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for every required check and the test layout.
