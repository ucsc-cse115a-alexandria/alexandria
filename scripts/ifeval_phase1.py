#!/usr/bin/env python3
"""Phase 1 shakedown: the longest IFEval cases, original vs ~95%-budget compression."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import default_embedder
from alexandria.utils.merger import default_merger
from alexandria.utils.tokens import count_tokens
from benchmarks.ifeval import compare, load_cases, run_experiment
from scripts.inflate_redundancy import build_generate

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL = "gpt-5.4-mini"
TARGET_RATIO = 0.95
N_CASES = 50
OUT_DIR = Path("trial_results/ifeval")


def compress_to(target_ratio: float) -> Callable[[str], str]:
    """A transform that asks the pipeline to fit each prompt into target_ratio of its tokens."""
    embedder = default_embedder()
    merger = default_merger()

    def transform(prompt: str) -> str:
        max_tokens = math.floor(count_tokens(prompt) * target_ratio)
        return reduce(prompt, embedder, merger, params=Params(max_tokens=max_tokens)).text

    return transform


def main() -> None:
    cases = load_cases(n=N_CASES)
    generate = build_generate(MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for label, transform in [("original", None), ("c95", compress_to(TARGET_RATIO))]:
        print(f"--- {label} ---")
        result = run_experiment(cases, generate, label=label, model=MODEL, transform=transform)
        out_path = OUT_DIR / f"{label}.json"
        out_path.write_text(result.model_dump_json(indent=2))
        print(f"wrote {out_path}")
        results.append(result)
    print(compare(*results))


if __name__ == "__main__":
    main()
