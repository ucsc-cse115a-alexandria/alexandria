from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.prompt_compression.contracts import BenchmarkAdapter, BenchmarkCase


def balanced_sample(cases: Sequence[BenchmarkCase], n: int | None, *, seed: int) -> tuple[BenchmarkCase, ...]:
    """Take a deterministic round-robin sample across task labels."""
    if n is None:
        return tuple(cases)
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > len(cases):
        raise ValueError(f"requested {n} cases, but only {len(cases)} are eligible")
    groups: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        groups[case.task].append(case)
    ordered_groups: list[list[BenchmarkCase]] = []
    for index, task in enumerate(sorted(groups)):
        group = sorted(groups[task], key=lambda case: case.key)
        random.Random(seed + index).shuffle(group)
        ordered_groups.append(group)
    selected: list[BenchmarkCase] = []
    row = 0
    while len(selected) < n:
        for group in ordered_groups:
            if row < len(group):
                selected.append(group[row])
                if len(selected) == n:
                    return tuple(selected)
        row += 1
    return tuple(selected)


def get_adapter(name: str) -> BenchmarkAdapter:
    if name == "babilong_8k":
        from benchmarks.prompt_compression.babilong import BABILongAdapter

        return BABILongAdapter()
    if name == "ruler_v2":
        from benchmarks.ruler_v2.cases import RULERv2Adapter

        return RULERv2Adapter()
    if name == "longbench_v2":
        from benchmarks.longbench_v2.cases import LongBenchV2Adapter

        return LongBenchV2Adapter()
    raise ValueError(f"unknown benchmark {name!r}")
