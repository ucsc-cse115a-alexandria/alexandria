# Shared prompt-compression benchmark runner

This package provides one evidence format for BABILong 8k, RULERv2, and LongBench v2. These benchmarks were chosen because each case is a large, static text request with an automatic answer metric. The same case can therefore be sent unchanged and with only its long context compressed, producing a direct paired accuracy comparison. Agent-environment benchmarks such as SWE-bench and OSWorld are intentionally outside this interface because their observations and trajectories change during execution and cannot be represented as one controlled original-versus-compressed prompt pair.

## Experimental unit and compression boundary

Every adapter turns an upstream dataset row into three lossless prompt parts:

```text
fixed prefix + compressible long context + fixed suffix
```

`original` sends their exact concatenation. Each requested reduction compresses only the context, reconstructs the complete prompt, and enforces a hard full-prompt token ceiling. For example, `--reductions 50 25 10 5` creates `keep50`, `keep75`, `keep90`, and `keep95`. The instruction, query, answer choices, required output format, and structural delimiters remain fixed where the upstream benchmark provides them.

Adapters also supply the benchmark-specific expected answer and verifier. The common runner never asks a model to grade another model. It records both the benchmark score (which may include partial credit) and strict case correctness.

## Run interface

Use the same command shape for all three benchmarks:

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k \
  --n 5 --seed 42 --reductions 50 25 10 5 \
  --out trial_results/babilong_8k/smoke --dry-run
```

Replace the benchmark name with `ruler_v2` or `longbench_v2`, provide `--data-dir` when needed, and remove `--dry-run` to make model calls. A smoke run can use `--n 5`; a release-oriented run can use `--n 100`. `--min-source-tokens` and `--max-source-tokens` constrain eligibility using the actual complete prompt counted with `cl100k_base`, so a run can explicitly require long inputs and remain within the answer model's context window.

Sampling is deterministic for a fixed eligible dataset, `n`, and seed. It round-robins across task labels before taking additional cases from a task, avoiding a small sample that accidentally contains only one task family.

## Evidence written for every run

The output directory is resumable and contains:

- `manifest.json`: implementation commit, pinned dataset/prompt provenance, selected case IDs, model settings, eligibility filters, reductions, complete-prompt token distribution, and cost assumptions;
- `records.jsonl`: append-only per-case and per-condition responses, parsed verdicts, prompt hashes, source/target/sent tokens, whole-prompt embedding cosine difference, compression and answer latency, merge metrics, API usage, and estimated cost;
- `prompts.jsonl.gz`: the exact model-visible original and compressed prompts, keyed by case and condition;
- `summary.json`: aggregate and per-task accuracy, benchmark score, token reduction, time, cost, paired transitions, bootstrap intervals, and the release decision; and
- `report.md`: a compact original-versus-compressed evidence table and a plain PASS or FAIL statement.

A completed `(case ID, condition)` pair is skipped on rerun, so interruption does not require repeating paid calls. The prompt SHA-256 in `records.jsonl` is checked against the exact prompt before it is appended.

## Accuracy retention and release rule

Conditions are paired by case ID. The summary resamples case indices with replacement, preserving the original/compressed pairing, and calculates compressed accuracy divided by original accuracy for each resample. The default 10,000-sample percentile interval uses seed 42 and the 2.5th and 97.5th percentiles. Resamples whose original accuracy is zero are excluded because their retention ratio is undefined.

The default release threshold is 90% accuracy retention. A condition passes only when the lower endpoint of its 95% paired-bootstrap interval is at least 90%; a point estimate above 90% is not sufficient. Original accuracy, compressed accuracy, score change, all four paired outcome transitions, assumptions, and the decision are retained so readers can see when low baseline solvability or a small sample makes retention unstable.

Recompute the summary entirely from saved raw outcomes, without API calls:

```bash
uv run python -m scripts.summarize_prompt_compression_benchmark \
  trial_results/babilong_8k/smoke \
  --release-threshold 0.90 --bootstrap-samples 10000 --bootstrap-seed 42
```

The benchmark-specific READMEs explain the upstream input, metric, preparation, and limitations:

- [`../babilong_8k/README.md`](../babilong_8k/README.md)
- [`../ruler_v2/README.md`](../ruler_v2/README.md)
- [`../longbench_v2/README.md`](../longbench_v2/README.md)
