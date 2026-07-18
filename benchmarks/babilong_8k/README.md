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

## Reproducible n=50 experiment

The default phase-one run uses 50 cases selected with seed 42, balanced as 10 cases from each of `qa1`-`qa5`.
It compares the original prompt with a 90% reduction target (10% of source tokens retained):

```bash
uv run python -m scripts.download_babilong_8k_data
uv run python -m scripts.babilong_8k_phase1
```

Results are written under `trial_results/babilong_8k/`. The comparison table uses this format:

The reduction target is strict. Alexandria first computes an optimistic floor from the eligible similarity graph;
if 90% is impossible for the merge-rewrite optimizer, it stops before calling Luna. For reachable targets,
candidate generation stops as soon as the proposed plan has enough token savings. Each result records merge calls,
retries, attempted pairs, proposed edits, and applied edits.

| Condition | Mean input tokens | Token reduction | Merge calls | Retries | Task accuracy | Accuracy change |
|---|---:|---:|---:|---:|---:|---:|
| original_luna | 7,720.0 | 0.0% | 0 | 0 | 90.0% (45/50) | — |
| reduction90_luna | 772.0 | 90.0% | 35 | 4 | 86.0% (43/50) | -4.0 pp |

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
