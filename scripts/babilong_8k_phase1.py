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
from benchmarks.babilong_8k import ExperimentResult, compare, load_cases, run_experiment
from scripts.inflate_redundancy import build_generate

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL = "gpt-5.6-luna"
TARGET_REDUCTION_PERCENT = 90.0
N_CASES = 50
SEED = 42
OUT_DIR = Path("trial_results/babilong_8k")


def compress_to_reduction(reduction_percent: float) -> Callable[[str], str]:
    """Build a transform that keeps ``100 - reduction_percent`` of source tokens."""
    embedder = default_embedder()
    merger = default_merger()
    keep_ratio = 1 - reduction_percent / 100

    def transform(prompt: str) -> str:
        max_tokens = max(1, math.floor(count_tokens(prompt) * keep_ratio))
        return reduce(prompt, embedder, merger, params=Params(max_tokens=max_tokens)).text

    return transform


def main() -> None:
    cases = load_cases(n=N_CASES, seed=SEED)
    generate = build_generate(MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []
    conditions = [
        ("original_luna", None),
        ("reduction90_luna", compress_to_reduction(TARGET_REDUCTION_PERCENT)),
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
