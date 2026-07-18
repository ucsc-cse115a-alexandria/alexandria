#!/usr/bin/env python3
"""Run 50 balanced BABILong 8k cases, original vs. a 90% token-reduction target."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import default_embedder
from alexandria.utils.merger import default_merger
from alexandria.utils.tokens import count_tokens
from benchmarks.babilong_8k import CompressionResult, ExperimentResult, compare, load_cases, run_experiment
from scripts.inflate_redundancy import build_generate

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL = "gpt-5.6-luna"
TARGET_REDUCTION_PERCENT = 90.0
N_CASES = 50
SEED = 42
OUT_DIR = Path("trial_results/babilong_8k")


def compress_to_reduction(reduction_percent: float) -> Callable[[str], CompressionResult]:
    """Build a transform that requires the requested source-token reduction."""
    embedder = default_embedder()
    merger = default_merger()
    keep_ratio = 1 - reduction_percent / 100

    def transform(prompt: str) -> CompressionResult:
        max_tokens = max(1, math.floor(count_tokens(prompt) * keep_ratio))
        result = reduce(prompt, embedder, merger, params=Params(max_tokens=max_tokens, require_target=True))
        require_target_reduction(result.source_tokens, result.reduced_tokens, reduction_percent)
        return CompressionResult(prompt=result.text, merge_metrics=result.merge_metrics)

    return transform


def require_target_reduction(source_tokens: int, reduced_tokens: int, target_percent: float) -> None:
    actual = 1 - reduced_tokens / source_tokens
    if actual + 1e-12 < target_percent / 100:
        raise RuntimeError(
            f"target reduction {target_percent:g}% was not met: achieved {actual:.1%} "
            f"({source_tokens} -> {reduced_tokens} tokens)"
        )


def main() -> None:
    cases = load_cases(n=N_CASES, seed=SEED)
    compress = compress_to_reduction(TARGET_REDUCTION_PERCENT)
    # Validate and cache every strict compression before spending answer-model calls. Failed
    # target merges therefore record their own retries without also spending answer generations.
    prepared = tuple(compress(case.prompt) for case in cases)
    prepared_iter = iter(prepared)

    def use_prepared(_prompt: str) -> CompressionResult:
        return next(prepared_iter)

    generate = build_generate(MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []
    conditions = [
        ("original_luna", None),
        ("reduction90_luna", use_prepared),
    ]
    for label, transform in conditions:
        print(f"--- {label} ---")
        result = run_experiment(cases, generate, label=label, model=MODEL, transform=transform)
        out_path = OUT_DIR / f"{label}.json"
        out_path.write_text(result.model_dump_json(indent=2))
        print(f"wrote {out_path}")
        results.append(result)
    print(compare(*results))


if __name__ == "__main__":
    main()
