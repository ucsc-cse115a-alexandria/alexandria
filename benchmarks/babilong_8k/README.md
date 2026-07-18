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

The default phase-one run uses 50 cases selected with seed 42, balanced as 10 cases from each of `qa1`-`qa5`.
It compares the original prompt with a 90%-retained target (at least a 10% token reduction):

```bash
uv run python -m scripts.download_babilong_8k_data
uv run python -m scripts.babilong_8k_phase1
```

Results are written under `trial_results/babilong_8k/`.

The reduction target is strict. Alexandria keeps instruction, example, format, and Markdown/XML boundaries fixed,
then asks Luna to rewrite a content window to the remaining token budget. It verifies the complete prompt's token
count and deterministically repairs model overshoot. Candidate windows avoid query-linked terms and protect local
semantic outliers. Target-safe candidates are ranked by local coverage and whole-prompt embedding drift. One merge
call normally returns two candidates; a second call is allowed only for quality refinement. Each result records
merge calls, retries, generated candidates, repaired tokens, and final drift.

## 10-case hard-target validation

The committed validation run used `gpt-5.6-luna`, answer reasoning `none`, merge reasoning `low`, seed 42, and two
cases from each of `qa1`-`qa5`. It is an implementation validation, not a publication-quality release result.

| Condition | Mean input tokens | Token reduction | Task accuracy | Accuracy change |
|---|---:|---:|---:|---:|
| Original | 7,374.3 | 0.00% | 80.0% (8/10) | - |
| 90%-retained | 6,564.5 | 10.98% | 70.0% (7/10) | -10.0 pp |

All 10 compressed prompts met their token ceilings. Compression took 159.5 seconds total (15.9 seconds/case), used
10 merge calls with no retries, and had mean whole-prompt embedding drift 0.0078. The compressed run cost an
estimated $0.1832; including the original answer baseline, the measured total was $0.2569. Prices and raw API usage
are recorded in [`summary.json`](results/2026-07-18-keep90-hard-target-n10-v4/summary.json).

Accuracy retention was 87.5%. Its 95% paired percentile-bootstrap interval was 60.0%-100.0% (10,000 resamples,
seed 42). The release rule was fixed at a 90% retention threshold and requires the confidence interval's lower bound
to be at least 90%. **Decision: FAIL. This run does not clear the release threshold.** The point estimate is below
the threshold, and the small sample leaves substantial uncertainty.

The confidence calculation is reproducible from the per-case outcomes committed in `summary.json`:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

from benchmarks.babilong_8k.statistics import paired_retention_bootstrap

path = Path("benchmarks/babilong_8k/results/2026-07-18-keep90-hard-target-n10-v4/summary.json")
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
[`raw.json`](results/2026-07-18-keep90-hard-target-n10-v4/raw.json); compact original responses are in
[`original.json`](results/2026-07-18-keep90-hard-target-n10-v4/original.json). Rerun measurement with:

```bash
uv run python -m scripts.babilong_8k_keep90_measure \
  --n 10 --seed 42 --keep 90 \
  --data-dir data/babilong/8k \
  --baseline trial_results/babilong_8k/original_gpt56_luna_reasoning_none_n10_20260718/original_luna_reasoning_none.json \
  --baseline-summary trial_results/babilong_8k/original_gpt56_luna_reasoning_none_n10_20260718/measured_summary.json \
  --out trial_results/babilong_8k/keep90-hard-target-n10
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
