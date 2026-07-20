# US2: Compression-strength sweep matrix and runbook

Docs-only deliverable for ticket #79. Uses existing `scripts.prompt_compression_benchmark` flags and
library APIs; no repo script changes required.

- Benchmark: BABILong 8k (`benchmarks/babilong_8k/README.md`)
- Prompt under test: official template (`benchmarks/babilong_8k/prompts.py`) and corpus
  (`data/babilong/8k/{qa1..qa5}.json`). Only the `<context>...</context>` block is compressed.
- Library default compression: `threshold = 0.85`, `cos_sim_diff_budget = 0.5`
  (`src/alexandria/ir/contracts.py:27-28`), applied by `reduce(..., params=Params())`.
- Harness limit: `--cos-sim-diff-budgets` values above `0.02` are rejected
  (`scripts/prompt_compression_benchmark.py:250-251`), so library default (`0.5`) is **not** a flag on
  that script. P1 uses a copy-paste inline runner below that calls the same library path.

## Prep (run once, before any sweep point)

```bash
uv run python -m scripts.download_babilong_8k_data
```

## Sweep matrix

Every row has its own standalone command in the runbook section.

| Point | Type | Setting | Mechanism |
|---|---|---|---|
| P0 | Baseline | Uncompressed original prompt | `--min-original-accuracy 1.0` gate (original only) |
| P1 | Baseline | Current default compression (`Params()` defaults) | Inline runner → `reduce(..., params=Params())` |
| P2 | Weaker | Tight semantic budget | `--cos-sim-diff-budgets 0.0025` |
| P3 | Weaker | Slightly tight semantic budget | `--cos-sim-diff-budgets 0.005` |
| P4 | Reference | Mid semantic budget | `--cos-sim-diff-budgets 0.01` |
| P5 | Stronger | Loose semantic budget | `--cos-sim-diff-budgets 0.015` |
| P6 | Stronger | Loosest allowed semantic budget | `--cos-sim-diff-budgets 0.02` |
| P7 | Stronger, hard-target | 50% forced reduction | `--reductions 50` |
| P8 | Stronger, hard-target | 30% forced reduction | `--reductions 30` |
| P9 | Stronger, hard-target | 10% forced reduction | `--reductions 10` |

P2–P6 sweep semantic-budget controls. P7–P9 sweep hard token-reduction targets.

## Runbook

Baselines (P0–P1), one command per row:

```bash
# P0: uncompressed original only. The runner scores every case, writes records/summary, then exits
# because original accuracy is below the impossible 100% gate; compressed conditions are skipped.
# Expect exit code 1; artifacts under --out are still valid P0 evidence.
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 --reductions 50 \
  --min-original-accuracy 1.0 \
  --model gpt-5.6-luna \
  --out benchmarks/babilong_8k/results/2026-07-20-p0-original-n50-v1

# P1: library-default compression (threshold=0.85, cos_sim_diff_budget=0.5) with paired task accuracy.
# Inline runner only; not a new repo file. Uses existing reduce/Params/RunStore APIs.
uv run python - <<'PY'
from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path

from openai import OpenAI

from alexandria.ir.contracts import MergeMetrics, Params
from alexandria.ops.features.compare import compare
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import OPENAI_EMBEDDING_MODEL, default_embedder
from alexandria.utils.merger import MERGE_MODEL, default_merger
from alexandria.utils.tokens import count_tokens
from benchmarks.prompt_compression.adapters import get_adapter
from benchmarks.prompt_compression.runner import benchmark_report, summarize_records
from benchmarks.prompt_compression.store import RunStore
from scripts.prompt_compression_benchmark import DEFAULT_MODEL, _generate, _git_dirty, _git_sha, _record

OUT = Path("benchmarks/babilong_8k/results/2026-07-20-p1-library-default-n50-v1")
N, SEED = 50, 42
MODEL, MERGE_MODEL_NAME = DEFAULT_MODEL, MERGE_MODEL
CONDITION, PARAMS = "default", Params()
adapter = get_adapter("babilong_8k")
cases = adapter.load_cases(N, seed=SEED, data_dir=adapter.default_data_dir)
store = RunStore(OUT)
store.write_manifest(
    {
        "schema_version": 1,
        "implementation_commit": _git_sha(),
        "implementation_dirty": _git_dirty(),
        "command": shlex.join(sys.argv),
        "benchmark": adapter.name,
        "provenance": adapter.provenance,
        "data_dir": str(adapter.default_data_dir),
        "n_cases": len(cases),
        "case_keys": [case.key for case in cases],
        "seed": SEED,
        "experiment_mode": "library_default_inline",
        "library_default_params": {
            "threshold": PARAMS.threshold,
            "cos_sim_diff_budget": PARAMS.cos_sim_diff_budget,
            "max_tokens": PARAMS.max_tokens,
            "require_target": PARAMS.require_target,
        },
        "model": MODEL,
        "compression": {"merge_model": MERGE_MODEL_NAME, "embedding_model": OPENAI_EMBEDDING_MODEL},
    }
)
client = OpenAI(timeout=120.0)
completed = store.completed_keys()
for case in cases:
    source_tokens = count_tokens(case.prompt)
    if (case.key, "original") not in completed:
        started = time.monotonic()
        generation = _generate(client, MODEL, "none", case.prompt)
        record = _record(
            case=case,
            condition="original",
            reduction_percent=0.0,
            prompt=case.prompt,
            target_tokens=source_tokens,
            sent_tokens=source_tokens,
            generation=generation,
            adapter=adapter,
            prompt_cosine_difference=0.0,
            compression_elapsed=0.0,
            answer_elapsed=time.monotonic() - started,
            merge_metrics=MergeMetrics(),
            usage=(),
        )
        store.append(record, case.prompt)
    if (case.key, CONDITION) in completed:
        continue
    parts = case.prompt_parts
    embedder = default_embedder()
    merger = default_merger(model=MERGE_MODEL_NAME)
    compression_started = time.monotonic()
    result = reduce(parts.context, embedder, merger, params=PARAMS)
    compressed = parts.replace_context(result.text)
    prompt_difference = float(compare(case.prompt, compressed.prompt, embedder).cos_sim_diff)
    compression_elapsed = time.monotonic() - compression_started
    sent_tokens = count_tokens(compressed.prompt)
    answer_started = time.monotonic()
    generation = _generate(client, MODEL, "none", compressed.prompt)
    answer_elapsed = time.monotonic() - answer_started
    actual_reduction = max(0.0, (1.0 - sent_tokens / source_tokens) * 100.0)
    record = _record(
        case=case,
        condition=CONDITION,
        reduction_percent=actual_reduction,
        prompt=compressed.prompt,
        target_tokens=source_tokens,
        sent_tokens=sent_tokens,
        generation=generation,
        adapter=adapter,
        prompt_cosine_difference=prompt_difference,
        compression_elapsed=compression_elapsed,
        answer_elapsed=answer_elapsed,
        merge_metrics=result.merge_metrics,
        usage=(),
        configured_cos_sim_diff_budget=PARAMS.cos_sim_diff_budget,
    )
    store.append(record, compressed.prompt)
records = store.load_records()
summary = summarize_records(records, bootstrap_seed=SEED)
report = benchmark_report(summary)
store.write_summary(summary, report)
print(report)
PY
```

Semantic-budget points (P2–P6), one command per row:

```bash
# P2
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 \
  --cos-sim-diff-budgets 0.0025 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p2-budget-0.0025-n50-v1

# P3
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 \
  --cos-sim-diff-budgets 0.005 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p3-budget-0.005-n50-v1

# P4
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 \
  --cos-sim-diff-budgets 0.01 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p4-budget-0.01-n50-v1

# P5
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 \
  --cos-sim-diff-budgets 0.015 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p5-budget-0.015-n50-v1

# P6
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 \
  --cos-sim-diff-budgets 0.02 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p6-budget-0.02-n50-v1
```

Hard-target points (P7–P9), one command per row:

```bash
# P7: 50% forced reduction
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 --reductions 50 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p7-reduction-50-n50-v1

# P8: 30% forced reduction
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 --reductions 30 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p8-reduction-30-n50-v1

# P9: 10% forced reduction
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 50 --seed 42 --reductions 10 \
  --model gpt-5.6-luna --merge-model gpt-5.6-luna --min-original-accuracy 0.50 \
  --out benchmarks/babilong_8k/results/2026-07-20-p9-reduction-10-n50-v1
```

Cheap smoke test before running any real sweep point (`README.md` lines 33-37):

```bash
uv run python -m scripts.prompt_compression_benchmark \
  --benchmark babilong_8k --n 5 --seed 42 --reductions 50 40 30 20 10 \
  --out trial_results/babilong_8k/smoke --dry-run
```

## Test plan

- P0–P9 each have their own saved raw output artifact under their `--out` directory (P0 despite exit code 1).
- P1–P9 record paired `original` and compressed conditions; P0 records `original` only.
- Every summary value traces back to its raw output via the exact single-row command above.
- A reviewer can rerun any individual point by copying only that row's command, with no dependency on
  running any other row first (other than the one-time prep step).
