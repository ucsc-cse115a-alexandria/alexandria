# LongBench v2 prompt-compression benchmark

This harness measures whether Alexandria can shorten realistic long documents while preserving enough evidence for a model to answer the same multiple-choice question. LongBench v2 contains 503 English questions with contexts ranging from roughly 8,000 to 2,000,000 words across six broad categories.

## What LongBench v2 measures

The dataset covers:

- single-document question answering;
- multi-document question answering;
- long in-context learning;
- understanding long dialogue histories;
- understanding code repositories represented as text; and
- understanding long structured data.

Every item has one correct answer, A through D. Unlike an agent benchmark, the complete evidence is present in one static text request, so the original and compressed requests can be scored with the same deterministic label.

## What enters the answer model

The adapter reproduces the official zero-shot template pinned at `THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e`:

```text
Please read the following text and answer the question below.

<text>
[The long document, dialogue, code repository, demonstrations, or structured data]
</text>

What is the correct answer to this question: [question]
Choices:
(A) [choice A]
(B) [choice B]
(C) [choice C]
(D) [choice D]

Format your response as follows: "The correct answer is (insert answer here)".
```

Only the text inside `<text>` is compressible. The instruction, question, four choices, tags, and output format remain byte-for-byte fixed. The original condition contains the complete official context. The runner never uses the official runner's fallback head-tail truncation: an item that does not fit `--max-source-tokens` is excluded before sampling and its eligibility assumptions are recorded in `manifest.json`.

## How answers are scored

The verifier removes Markdown `*` characters and extracts A, B, C, or D from the official response phrase `The correct answer is (...)`. The extracted letter must exactly equal the dataset's `answer` field. Missing or malformed answer phrases are incorrect and remain visible as `parsed_answer: null` in the raw record.

Accuracy is the number of exact correct letters divided by the number of paired cases. The summary includes top-level domain breakdowns. Sub-domain, difficulty, and length remain attached to every raw record so additional slices can be reproduced from the saved evidence. Because LongBench v2 can be difficult even without compression, the report includes the four paired transitions: both correct, compressed regression, compressed improvement, and both wrong.

## Downloading the pinned dataset

The downloader fetches `zai-org/LongBench-v2@2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`:

```bash
uv run python -m scripts.download_longbench_v2_data
```

The resulting `data/longbench_v2/data.json` is about 465 MB and is ignored by Git.

## Running a small evidence-producing experiment

First inspect token lengths and case selection without API calls:

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark longbench_v2 --n 5 --seed 42 \
  --reductions 50 25 10 5 --min-source-tokens 16000 --max-source-tokens 128000 \
  --out trial_results/longbench_v2/smoke --dry-run
```

Remove `--dry-run` to run original, keep50, keep75, keep90, and keep95. Change `--n 5` to `--n 100` for the standard larger run. Sampling is deterministic and balanced across the six top-level task categories as far as eligible data permits.

## Measured 10-case keep90 pilot

This pilot filtered the pinned dataset to complete prompts between 16,000 and 128,000 `cl100k_base` tokens, then selected ten cases deterministically across the six top-level domains with seed 42. The selected original prompts ranged from 22,676 to 123,954 tokens, with a mean of 68,824.7 tokens.

| Condition | Mean prompt tokens (total) | Token reduction | Mean whole-prompt cosine diff | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 68,824.7 (688,247) | 0.00% | 0.000000 | 60.0% (6/10) | 14.8s | $0.6371 | 0.0s | $0.0000 |
| keep90 | 61,042.0 (610,420) | 11.31% | 0.009065 | 60.0% (6/10) | 12.6s | $0.6021 | 290.0s | $0.2925 |

Original-to-keep90 reduction and whole-prompt comparison took 290.0 seconds and cost $0.2925. Once each prompt was ready, execution changed from 14.8 to 12.6 seconds and from $0.6371 to $0.6021. All six original successes remained correct, and all four original failures remained wrong, so the paired retention estimate and every defined bootstrap resample were 100%. The pilot therefore records **PASS** against the 90% retention threshold. With only ten cases, this does not establish that the same retention will hold across LongBench v2.

Reproduce the run:

```bash
uv run python -m scripts.download_longbench_v2_data
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark longbench_v2 --n 10 --seed 42 --reductions 10 \
  --data-dir data/longbench_v2 \
  --min-source-tokens 16000 --max-source-tokens 128000 \
  --out benchmarks/longbench_v2/results/2026-07-18-keep90-n10-v2
```

Evidence: [`manifest.json`](results/2026-07-18-keep90-n10-v2/manifest.json), [`records.jsonl`](results/2026-07-18-keep90-n10-v2/records.jsonl), [`prompts.jsonl.gz`](results/2026-07-18-keep90-n10-v2/prompts.jsonl.gz), [`summary.json`](results/2026-07-18-keep90-n10-v2/summary.json), and [`report.md`](results/2026-07-18-keep90-n10-v2/report.md).

Cost uses the manifest assumptions per million tokens: $1.00 model input, $0.10 cached input, $6.00 model output, and $0.02 embedding input. The answer and merge model was `gpt-5.6-luna` with answer reasoning `none`; whole-prompt cosine difference used `text-embedding-3-small`.

## Logs, confidence intervals, and release decisions

The run directory contains:

- `manifest.json`: revisions, exact case IDs, model settings, token distribution, reductions, and pricing assumptions;
- `records.jsonl`: append-only case/condition outcomes and usage;
- `prompts.jsonl.gz`: exact original and compressed prompts;
- `summary.json`: aggregate and paired-bootstrap calculations, including separately derived execution and reduction time/cost; and
- `report.md`: the publishable accuracy/token/time/cost table and plain PASS/FAIL decisions.

Recalculate the evidence from raw outputs:

```bash
uv run python -m scripts.summarize_prompt_compression_benchmark \
  trial_results/longbench_v2/smoke --release-threshold 0.90
```

The calculation resamples paired case indices, computes compressed/original accuracy retention for each resample, and uses the 2.5th and 97.5th percentiles as a 95% interval. The default release decision is PASS only when the interval's lower bound is at least 90%.

## Limitations

Multiple-choice accuracy measures whether task-relevant information survived; it does not establish semantic equivalence of the compressed context. Very small pilots have wide intervals. Results also depend on the answer model's original solvability, so original accuracy and the paired transitions must always accompany retention.

Upstream sources: [LongBench repository](https://github.com/THUDM/LongBench), MIT; [LongBench v2 dataset](https://huggingface.co/datasets/zai-org/LongBench-v2), Apache-2.0.
