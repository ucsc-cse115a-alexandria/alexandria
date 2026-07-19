# RULERv2 prompt-compression benchmark

This harness measures whether Alexandria can shorten a synthetic long-context prompt while preserving the facts and reasoning steps needed to answer it. It uses the official RULERv2 data shape and task-level scoring rules pinned in the run manifest.

## What RULERv2 measures

RULERv2 creates prompts at a requested sequence length by mixing a small amount of task-relevant information with a much larger haystack. Its twelve official task configurations cover three families at four difficulty levels:

- **Multi-key needle-in-a-haystack:** find the value associated with one or more keys.
- **Multi-value needle-in-a-haystack:** recover several values associated with a key or intermediate task.
- **Question answering:** find or combine evidence from documents in the long prompt.

The `basic`, `easy`, `medium`, and `hard` variants increase the amount of retrieval, intermediate reasoning, or symbolic answering required. This makes RULERv2 useful for distinguishing simple fact retention from preservation of multi-step task information.

## What enters the answer model

An official generated JSONL row contains a complete `question` string and one or more `expected_answer` values. Conceptually, the request is:

```text
[Instructions and long synthetic haystack]
[Needles, documents, examples, or indexed questions mixed into the haystack]

[Final question or requested operation]
```

The adapter recognizes the pinned templates for all twelve task configurations. It keeps the leading task instruction and the final query or requested operation fixed; for the two few-shot multiple-choice tasks, it also protects the examples. Only the generated haystack of needles, questions, or documents is compressible. The original condition sends the official string unchanged. A compressed condition replaces only that haystack and then verifies the token count of the complete reconstructed prompt. No benchmark-only XML tag or instruction is inserted into the model input.

Generated data is expected in either of these layouts:

```text
data/ruler_v2/<task>/test.jsonl
data/ruler_v2/<task>.jsonl
```

The task directory or filename becomes the task label used for balanced sampling and reporting.

## How answers are scored

The verifier reproduces the task-specific rules in the pinned NeMo Skills RULERv2 preparation and evaluator. Most tasks score every expected string by taking the better of case-insensitive containment and word-error similarity; multi-value cases average those values. QA tasks take the best score across accepted answer variants. `mv_niah_medium` evaluates only the response section after the last blank line, matching its two-step rule. `mk_niah_medium` and `mk_niah_hard` extract an A-D answer and use exact multiple-choice correctness. A case is strictly correct only when its resulting score is 1.0.

The report therefore includes both:

- **Official score:** mean case score, including partial credit for multi-value answers.
- **Strict accuracy:** fraction of cases with a complete score of 1.0.

Original and compressed conditions use the same case, expected values, answer model, and decoding settings, so their results can be joined by case ID and compared directly.

## Preparing pinned official data

The run manifest expects the RULERv2 entry point at `NVIDIA/RULER@4809570a2a40e803bfe341773e561524224c2e7c` and the current dataset implementation at `NVIDIA-NeMo/Skills@74b8649734a6ecc2d3beca89311e1a5e02da48fa`. Use the official NeMo Skills `ruler2` preparation command with the tokenizer used by the answer model, sequence length, dataset size, and seed 42, then place each generated `test.jsonl` under `data/ruler_v2/<task>/`.

RULER data generation targets a tokenizer-specific total sequence length. The common runner always records the actual `cl100k_base` size of the complete request as well; the two counts need not be identical.

## Running a small evidence-producing experiment

Inspect the selected cases without making API calls:

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark ruler_v2 --n 5 --reductions 50 25 10 5 \
  --data-dir data/ruler_v2 --out trial_results/ruler_v2/smoke --dry-run
```

Run the paired experiment:

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark ruler_v2 --n 5 --seed 42 --reductions 50 25 10 5 \
  --data-dir data/ruler_v2 --out trial_results/ruler_v2/smoke
```

Change `--n 5` to `--n 100` for the standard larger run. The four reductions produce `keep50`, `keep75`, `keep90`, and `keep95` conditions alongside `original`.

## Logs, confidence intervals, and release decisions

Each run writes `manifest.json`, append-only `records.jsonl`, exact prompts in `prompts.jsonl.gz`, `summary.json`, and `report.md`. Each condition record contains the response, parsed score, prompt hash, source/sent tokens, latency, merge metrics, API usage, and estimated cost.

Recompute the paired percentile-bootstrap confidence intervals and the release decision without calling a model:

```bash
uv run python -m scripts.summarize_prompt_compression_benchmark \
  trial_results/ruler_v2/smoke --release-threshold 0.90
```

The default release rule passes only when the lower bound of the 95% accuracy-retention interval is at least 90%. Pilot runs with very few cases are evidence that the pipeline works, not strong release evidence.

## Limitations

RULERv2 is synthetic. High retention shows that Alexandria preserved information required by these retrieval and reasoning tasks, not that every detail of the haystack survived. Results must also be reported by task and length because a mean can hide failures on harder configurations.

Upstream sources: [NVIDIA/RULER](https://github.com/NVIDIA/RULER) and [NeMo Skills RULERv2](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/dataset/ruler2), Apache-2.0.
