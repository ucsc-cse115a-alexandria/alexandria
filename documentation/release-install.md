# Release installation and benchmark reproduction

This guide verifies the current release path outside a source checkout. Alexandria is not yet on
PyPI, so version 0.1.0 is installed from the GitHub repository. The distribution is named
`alexandria-prompt`; the console command and Python import are both named `alexandria`.

## Environment prerequisites

- CPython 3.14.
- `uv` and Git.
- Internet access while Git and `uv` fetch the public repository and its dependencies.
- An OpenAI API key and access to the configured models for normal reductions and benchmark runs.
- No GPU. OpenAI performs the model work remotely.

The automated installation smoke test described below uses a local deterministic embedding endpoint
and an injected offline Python merger. It does not require an OpenAI key or make external model calls.

## Install the CLI

Install the current Git release into an isolated `uv` tool environment:

```bash
uv tool install "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"
alexandria --help
```

To replace an older installation with the current repository version:

```bash
uv tool install --force "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"
```

On Windows PowerShell, the same commands work after `uv` and Git are on `PATH`.

## Install the Python library

Add the current Git release to a `uv` project:

```bash
mkdir alexandria-api-example
cd alexandria-api-example
uv init --python 3.14
uv add "git+https://github.com/ucsc-cse115a-alexandria/alexandria.git"
```

Confirm that the distribution and import names resolve:

```bash
uv run python - <<'PY'
import importlib.metadata

import alexandria

print(importlib.metadata.version("alexandria-prompt"))
print(alexandria.__file__)
PY
```

The printed module path should be inside the project environment, not inside an Alexandria checkout.

## Installed CLI quickstart

Normal CLI reductions use OpenAI embeddings and merging. Save a key once, or export
`OPENAI_API_KEY`:

```bash
alexandria config set openai-api-key

cat > prompt.txt <<'EOF'
Keep your answers concise and to the point.
Keep your answers brief and to the point.
Use examples.
EOF

alexandria reduce prompt.txt --json
```

A successful run exits with status 0 and emits JSON whose `reduced_tokens` value is lower than
`source_tokens` for this redundant fixture. Model output can vary between runs.

## Installed Python API quickstart

Run this inside the `uv` project created above:

```python
import alexandria

result = alexandria.reduce(
    "Keep your answers concise and to the point.\n"
    "Keep your answers brief and to the point.\n"
    "Use examples.\n"
)
print(result.text)
print(result.source_tokens, result.reduced_tokens)
```

The default API path resolves the key from `api_key=`, `OPENAI_API_KEY`, or the saved Alexandria
configuration. For a deterministic API-only example without network calls, see the
[offline library example](library.md#run-offline).

## Automated clean-package smoke test

From a repository checkout, run:

```bash
python scripts/release_smoke_test.py
```

The script performs the release checks in a temporary directory:

1. Creates a clean Python 3.14 virtual environment.
2. Builds the `alexandria-prompt` wheel and source distribution from release inputs, then installs the wheel.
3. Runs the installed `alexandria` console script against an exact-duplicate fixture, using a local
   deterministic mock for the OpenAI embeddings endpoint.
4. Runs `import alexandria` through the installed interpreter with `HashEmbedder` and an offline
   merger.
5. Fails unless both paths reduce the prompt and the imported module comes from the clean virtual
   environment.

CI runs this smoke test on every pull request to `main` and every push to `main`.

## Reproduce the published benchmark

The published hard-target result is the
[50-case BABILong 8k report](../benchmarks/babilong_8k/results/2026-07-20-luna-keep75-95-n50-v1/report.md).
Its [manifest](../benchmarks/babilong_8k/results/2026-07-20-luna-keep75-95-n50-v1/manifest.json)
records the original implementation revision, case IDs, models, data provenance, and command.

The benchmark harness and dataset adapters are repository tooling and are not bundled into the
installed CLI wheel. After installation, clone the repository to obtain them:

```bash
git clone https://github.com/ucsc-cse115a-alexandria/alexandria.git
cd alexandria
uv sync --frozen
uv run python -m scripts.download_babilong_8k_data
```

Before spending money, validate the selected cases and output directory without model calls:

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k \
  --n 50 --seed 42 \
  --reductions 25 20 15 10 5 \
  --data-dir data/babilong/8k \
  --model gpt-5.6-luna \
  --merge-model gpt-5.6-luna \
  --min-original-accuracy 0.50 \
  --max-estimated-cost-usd 15 \
  --out trial_results/babilong_8k/keep75-95-reproduction \
  --dry-run
```

Remove `--dry-run` to execute the reproduction. It requires `OPENAI_API_KEY`, access to
`gpt-5.6-luna`, network access, and a cost allowance of up to USD 15. The published run spent
about one hour in reduction work, so allow roughly that order of time.
Because the answer and merge model outputs are nondeterministic, the expected result is the same
procedure, case selection, and evidence schema rather than byte-identical responses or scores. The
manifest also records that the original working tree was dirty, so its exact source state cannot be
reconstructed from the commit alone.
