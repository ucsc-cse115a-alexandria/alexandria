# Shared prompt-compression benchmark runner

This package provides one evidence format for BABILong 8k, RULERv2, and LongBench v2. Each case is a large,
static text request with an automatic answer metric, so the same case can be sent unchanged and with only its long
context compressed. Agent-environment benchmarks such as SWE-bench and OSWorld are outside this interface because
their observations and trajectories cannot be represented as one controlled original-versus-compressed prompt
pair.

Run-specific numbers, figures, costs, and caveats live in versioned reports under `results/`; this guide documents
the shared runner and evidence contract.

## Experimental unit and compression boundary

Every adapter turns an upstream dataset row into three lossless prompt parts:

```text
fixed prefix + compressible long context + fixed suffix
```

`original` sends their exact concatenation. Each requested reduction compresses only the context, reconstructs the
complete prompt, and enforces a hard full-prompt token ceiling. For example, `--reductions 50 25 10 5` creates
`keep50`, `keep75`, `keep90`, and `keep95`. Instructions, queries, answer choices, output formats, and structural
delimiters remain fixed where the upstream benchmark provides them.

Adapters supply the expected answer and benchmark-specific verifier. The common runner never asks a model to
grade another model. It records both the benchmark score, which may include partial credit, and strict case
correctness.

## Run interface

Use the same command shape for all benchmarks:

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k \
  --n 5 --seed 42 --reductions 50 40 30 20 10 \
  --out trial_results/babilong_8k/smoke --dry-run
```

Replace the benchmark name with `ruler_v2` or `longbench_v2`, provide `--data-dir` when needed, and remove
`--dry-run` to make model calls. Use `--n 5` for a smoke run and a larger preregistered sample for publication
evidence. `--min-source-tokens` and `--max-source-tokens` constrain eligibility using the complete prompt counted
with `cl100k_base`.

The runner completes every `original` response first. Compression proceeds only when original accuracy is at
least `--min-original-accuracy` (default 0.50), preventing an incapable answer model from producing a misleading
retention curve. `--model` selects the answer model and `--merge-model` independently selects the compression
model; both are recorded in the manifest. Conditions are always compared using the same answer model.

Sampling is deterministic for a fixed eligible dataset, `n`, and seed. It round-robins across task labels before
taking additional cases from a task.

Recompute a summary from saved outcomes without API calls:

```bash
uv run python -m scripts.summarize_prompt_compression_benchmark \
  trial_results/babilong_8k/smoke \
  --release-threshold 0.90 --bootstrap-samples 10000 --bootstrap-seed 42
```

## Published reports

- [Current 50-case Luna report](results/2026-07-19-luna-keep50-90-n50-v1/report.md): complete conditions,
  confidence intervals, figures, time/cost, reproduction commands, caveats, and raw evidence links.
- [Archived five-case aggregate](results/2026-07-18-luna-keep50-90-n5-v1/): machine-readable aggregate and smoke
  figures.
- Benchmark-specific reports and historical pilots:
  [BABILong 8k](../babilong_8k/README.md),
  [RULERv2](../ruler_v2/README.md), and
  [LongBench v2](../longbench_v2/README.md).

## Evidence written for every run

The output directory is resumable and contains:

- `manifest.json`: implementation commit, dataset provenance, selected case IDs, model settings, eligibility
  filters, reductions, token distribution, and cost assumptions;
- `records.jsonl`: append-only responses, verdicts, prompt hashes, token counts, cosine difference, latency, merge
  metrics, API usage, and estimated cost;
- `prompts.jsonl.gz`: exact model-visible original and compressed prompts keyed by case and condition;
- `summary.json`: aggregate and per-task scores, token reduction, time/cost, paired transitions, bootstrap intervals,
  and release decisions;
- `report.md`: the benchmark-specific result table and explicit PASS/FAIL statements; and
- `run.log` for committed evidence: progress, baseline-gate output, interruption traces, resume output, and final
  report.

A completed `(case ID, condition)` pair is skipped on rerun. The prompt SHA-256 is checked against the exact
prompt before a record is appended.

## Accuracy retention and release rule

The summary resamples paired case indices with replacement and calculates compressed accuracy divided by original
accuracy. The default 10,000-sample percentile interval uses seed 42 and the 2.5th and 97.5th percentiles. Resamples
with zero original accuracy are excluded because their retention ratio is undefined.

A condition passes only when the interval's lower endpoint is at least the release threshold, 90% by default. A
point estimate above 90% is not sufficient. Original and compressed accuracy, score change, all four paired
outcome transitions, assumptions, and the decision remain available in the versioned report and raw summary.

## Benchmark adapters

The benchmark-specific guides document upstream data, preparation, scoring, and limitations:

- [BABILong 8k](../babilong_8k/README.md)
- [RULERv2](../ruler_v2/README.md)
- [LongBench v2](../longbench_v2/README.md)
