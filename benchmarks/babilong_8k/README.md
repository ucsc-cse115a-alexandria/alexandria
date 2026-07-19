# BABILong 8k compression benchmark

This harness measures whether Alexandria can aggressively reduce a full long-context model input while preserving
the information needed to solve the downstream task. It uses the official BABILong 8k evaluation data for `qa1`
through `qa5`, the official default prompt text, and an equivalent implementation of the official label checker.

## What enters the model

Every case sends the complete prompt, not just the PG-19 context:

```text
[Task instruction]

[Few-shot examples]

[Required answer format]

<context>
[Mostly irrelevant PG-19 prose]
[A small number of bAbI facts hidden in the prose]
</context>

Question: [question]
```

Across the 500 official 8k cases in `qa1`-`qa5`, this full input has 7,723 `cl100k_base` tokens at p50 and 7,803
tokens at p75. The configured `8k` name and these counts differ because BABILong generation and this harness use
different tokenizers.

## What the result supports

The primary result should be framed as downstream task retention:

> On BABILong 8k, Alexandria reduced input prompts by **n%** while accuracy decreased by only **m percentage
> points**, indicating that it preserved most of the task-relevant information required to answer the questions.

This is evidence that Alexandria retains **task-relevant information** while removing substantial irrelevant
context. It is not evidence that every semantic detail in the PG-19 prose survived, that compression is lossless,
or that every output-format instruction survived. The official checker scores the answer label, not full semantic
equivalence of the compressed prompt.

Report percentage-point change rather than relative percent change. For example, a change from 90% to 86% is
`-4 percentage points`, not `-4%`.

## Verifier

For each task, the checker:

1. lowercases the response and reads its first sentence;
2. extracts labels from the task's finite label set;
3. ignores labels already present in the question; and
4. passes only when the target is the sole remaining label.

Thus both `bathroom` and `The most recent location of Mary is bathroom.` can pass when the target is `bathroom`.
The harness deliberately reports this as `Task accuracy`, not strict format compliance.

## Reproducible experiment

### Common benchmark interface

The shared runner exposes BABILong, RULERv2, and LongBench v2 through the same options and raw-log schema. It keeps BABILong's instruction, examples, required answer format, `<context>` tags, and final question fixed, and compresses only the PG-19-plus-bAbI context inside the tags.

Preview a small paired sample without API calls:

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 5 --seed 42 --reductions 50 25 10 5 \
  --out trial_results/babilong_8k/smoke --dry-run
```

Remove `--dry-run` to evaluate `original`, `keep50`, `keep75`, `keep90`, and `keep95`. Use `--n 100` for a larger task-balanced measurement. The shared run directory records exact prompts, per-case responses and verdicts, token counts, separate execution and reduction time/cost, merge work, API usage, paired confidence intervals, and plain release decisions.

Recompute the summary from the saved raw records without new API calls:

```bash
uv run python -m scripts.summarize_prompt_compression_benchmark \
  trial_results/babilong_8k/smoke --release-threshold 0.90
```

The legacy phase-one commands below remain available for reproducing the previously published keep90 run.

## Shared-run 10-case keep90 pilot

This pilot used ten task-balanced cases (two from each of `qa1`–`qa5`), seed 42, `gpt-5.6-luna`, answer reasoning `none`, and a full-prompt target of at most 90% of each original prompt. The complete original prompts ranged from 7,251 to 7,883 `cl100k_base` tokens. Execution measures only the answer-model call after its prompt is ready. Reduction is separate and includes compression plus the whole-prompt cosine measurement with `text-embedding-3-small`.

| Condition | Mean prompt tokens (total) | Token reduction | Mean whole-prompt cosine diff | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 7,668.5 (76,685) | 0.00% | 0.000000 | 60.0% (6/10) | 11.5s | $0.0768 | 0.0s | $0.0000 |
| keep90 | 6,634.2 (66,342) | 13.49% | 0.006987 | 60.0% (6/10) | 9.2s | $0.0665 | 94.8s | $0.2166 |

Original-to-keep90 reduction and whole-prompt comparison took 94.8 seconds and cost $0.2166. Once each prompt was ready, execution changed from 11.5 to 9.2 seconds and from $0.0768 to $0.0665. One original-only success and one compressed-only success produced the same aggregate accuracy. Accuracy retention was 100%, but its paired 95% percentile-bootstrap interval was 60.0%–175.0%. The lower bound did not clear the 90% release threshold, so this small pilot's decision is **FAIL**.

Reproduce the run:

```bash
uv run python -m scripts.download_babilong_8k_data
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 10 --seed 42 --reductions 10 \
  --data-dir data/babilong/8k \
  --out benchmarks/babilong_8k/results/2026-07-18-keep90-n10-common-v1
```

Evidence: [`manifest.json`](results/2026-07-18-keep90-n10-common-v1/manifest.json), [`records.jsonl`](results/2026-07-18-keep90-n10-common-v1/records.jsonl), [`prompts.jsonl.gz`](results/2026-07-18-keep90-n10-common-v1/prompts.jsonl.gz), [`summary.json`](results/2026-07-18-keep90-n10-common-v1/summary.json), and [`report.md`](results/2026-07-18-keep90-n10-common-v1/report.md).

Cost uses the manifest assumptions per million tokens: $1.00 model input, $0.10 cached input, $6.00 model output, and $0.02 embedding input. This is an exploratory ten-case result and should not replace the existing 100-case result below.

The default phase-one run uses 50 cases selected with seed 42, balanced as 10 cases from each of `qa1`-`qa5`.
It compares the original prompt with a 90%-retained target (at least a 10% token reduction):

```bash
uv run python -m scripts.download_babilong_8k_data
uv run python -m scripts.babilong_8k_phase1
```

Results are written under `trial_results/babilong_8k/`.

The reduction target is strict. Alexandria keeps instruction, example, format, and Markdown/XML boundaries fixed,
then, for each content window, fires three generation requests to Luna in parallel with different rewrite
strategies (plain compression, extractive deletion, and dense paraphrase). It verifies the complete prompt's token
count and deterministically repairs model overshoot. Candidate windows avoid query-linked terms and protect local
semantic outliers. Among target-safe candidates it selects the one with the lowest whole-prompt `cos_sim_diff`,
using local coverage as a tiebreaker. The three requests run concurrently, so each window costs roughly one
generation call. Each result records merge calls, retries, generated candidates, repaired tokens, and final `cos_sim_diff`.

## 100-case hard-target result

The committed run used `gpt-5.6-luna`, answer reasoning `none`, merge reasoning `low`, seed 42, and 20 cases from
each of `qa1`-`qa5`.

| Condition | Mean input tokens | Token reduction | Task accuracy | Accuracy change |
|---|---:|---:|---:|---:|
| Original | 7,540.82 | 0.00% | 66.0% (66/100) | - |
| 90%-retained | 6,716.60 | 10.93% | 65.0% (65/100) | -1.0 pp |

All 100 compressed prompts met their token ceilings. Compression took 1,640.1 seconds total (16.4 seconds/case),
used 99 merge calls with no retries, and had mean whole-prompt `cos_sim_diff` 0.0072. Compression cost an estimated
$1.2002, and compressed-answer generation cost $0.6723. Including the $0.7536 original baseline, the measured API
total was $2.6260 and sequential wall time was 1,898.0 seconds. Prices and raw API usage are recorded in
[`summary.json`](results/2026-07-18-keep90-hard-target-n100-v1/summary.json).

Accuracy retention was 98.48%. Its 95% paired percentile-bootstrap interval was 85.71%-112.90% (10,000 resamples,
seed 42). The release rule was fixed at a 90% retention threshold and requires the confidence interval's lower bound
to be at least 90%. **Decision: FAIL. This run does not clear the release threshold.** The point estimate exceeds
the threshold, but its confidence interval does not.

The bootstrap resamples paired case indices with replacement and computes compressed accuracy divided by original
accuracy in each resample. Ratios can exceed 100% when a resample contains more compressed-only successes than
original-only successes. Resamples with zero original accuracy are discarded. The 100 observed pairs contained 11
regressions, 10 improvements, 55 cases correct in both conditions, and 24 cases wrong in both conditions. Model
generation is stochastic, so this interval describes this paired run and sampling procedure rather than repeat-run
variance.

The confidence calculation is reproducible from the per-case outcomes committed in `summary.json`:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

from benchmarks.babilong_8k.statistics import paired_retention_bootstrap

path = Path("benchmarks/babilong_8k/results/2026-07-18-keep90-hard-target-n100-v1/summary.json")
records = json.loads(path.read_text())["records"]
result = paired_retention_bootstrap(
    [record["original_correct"] for record in records],
    [record["compressed_correct"] for record in records],
    samples=10_000,
    seed=42,
    confidence_level=0.95,
    release_threshold=0.90,
)
print(result.model_dump_json(indent=2))
PY
```

The full compressed prompts and metered API responses are in
[`raw.json`](results/2026-07-18-keep90-hard-target-n100-v1/raw.json); compact original responses are in
[`original.json`](results/2026-07-18-keep90-hard-target-n100-v1/original.json). Given a full original
`ExperimentResult` from the same selected cases, rerun measurement with:

```bash
uv run python -m scripts.babilong_8k_keep90_measure \
  --n 100 --seed 42 --keep 90 \
  --data-dir data/babilong/8k \
  --baseline trial_results/babilong_8k/original_n100/original_luna_reasoning_none.json \
  --baseline-summary trial_results/babilong_8k/original_n100/original_summary.json \
  --out trial_results/babilong_8k/keep90-hard-target-n100
```

The `--baseline-summary` argument is optional; it adds original-run time and cost to the combined totals. Raw
results are checkpointed after each case, so rerunning the same command resumes an interrupted run.

## Reporting format

Use this table format for larger runs:

| Condition | Mean input tokens | Token reduction | Merge calls | Retries | Task accuracy | Accuracy change |
|---|---:|---:|---:|---:|---:|---:|
| original_luna | 7,720.0 | 0.0% | 0 | 0 | 90.0% (45/50) | - |
| keep90_luna | 6,948.0 | 10.0% | 50 | 0 | 86.0% (43/50) | -4.0 pp |

The numbers above illustrate reporting only; they are not benchmark results.

For a publication-quality claim, compare Alexandria against token-matched head/tail truncation and random-deletion
baselines. Similar accuracy from Alexandria but substantially lower accuracy from those controls is stronger
evidence that the retained task information was selected deliberately rather than surviving by chance.

## Provenance

- Dataset: [RMT-team/babilong](https://huggingface.co/datasets/RMT-team/babilong), revision
  `ee0d588794c7ac098062ee0d247c733d62e94fe2`
- Prompt and metric source: [booydar/babilong](https://github.com/booydar/babilong), revision
  `38da79d79519ef87aa46ae804f838e1eab7f86d7`
- Upstream code is Apache-2.0 licensed; its license is preserved in [`UPSTREAM_LICENSE`](UPSTREAM_LICENSE).
