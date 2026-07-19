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

## Logs, confidence intervals, and release decisions

The run directory contains:

- `manifest.json`: revisions, exact case IDs, model settings, token distribution, reductions, and pricing assumptions;
- `records.jsonl`: append-only case/condition outcomes and usage;
- `prompts.jsonl.gz`: exact original and compressed prompts;
- `summary.json`: aggregate and paired-bootstrap calculations; and
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
