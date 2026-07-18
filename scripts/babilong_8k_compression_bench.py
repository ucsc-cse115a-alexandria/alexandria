#!/usr/bin/env python3
"""Measure compression-only cost of the strict BABILong 8k 90% target: calls, failures, wall clock."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from alexandria.ir.contracts import MergeMetrics, Params
from alexandria.ops.pipe import TargetMergeError, reduce
from alexandria.utils.tokens import count_tokens
from benchmarks.babilong_8k import load_cases

if TYPE_CHECKING:
    from collections.abc import Iterable

    from alexandria.ir.contracts import Embedder, SentenceMerger

TARGET_REDUCTION_PERCENT = 90.0


class CompressedCase(BaseModel):
    """A case whose strict-target compression succeeded."""

    model_config = ConfigDict(frozen=True)
    status: Literal["compressed"] = "compressed"
    prompt: str = Field(min_length=1)
    merge_metrics: MergeMetrics


class FailedCase(BaseModel):
    """A case whose strict-target compression raised TargetMergeError."""

    model_config = ConfigDict(frozen=True)
    status: Literal["failed"] = "failed"
    error: str = Field(min_length=1)
    merge_metrics: MergeMetrics


CompressionRecord = Annotated[CompressedCase | FailedCase, Field(discriminator="status")]


class CompressionSummary(BaseModel):
    """The gated aggregates for one compression-only run."""

    model_config = ConfigDict(frozen=True)
    n_cases: int = Field(ge=1)
    seed: int
    target_reduction_percent: float
    success_rate: float = Field(ge=0.0, le=1.0)
    zero_merge_call_share: float = Field(ge=0.0, le=1.0)
    mean_merger_calls: float = Field(ge=0.0)
    merger_call_distribution: dict[int, int]
    total_elapsed_seconds: float = Field(ge=0.0)
    mean_elapsed_seconds: float = Field(ge=0.0)


def compress_case(
    prompt: str, embedder: Embedder | None = None, merger: SentenceMerger | None = None
) -> CompressedCase | FailedCase:
    """Run one strict-target reduction, folding a TargetMergeError into a FailedCase record."""
    keep_ratio = 1 - TARGET_REDUCTION_PERCENT / 100
    max_tokens = max(1, math.floor(count_tokens(prompt) * keep_ratio))
    try:
        result = reduce(prompt, embedder, merger, params=Params(max_tokens=max_tokens, require_target=True))
    except TargetMergeError as error:
        return FailedCase(error=str(error), merge_metrics=error.metrics)
    return CompressedCase(prompt=result.text, merge_metrics=result.merge_metrics)


def summarize(records: Iterable[CompressedCase | FailedCase], *, seed: int) -> CompressionSummary:
    """Aggregate per-case records into the run's gated metrics."""
    materialized = list(records)
    if not materialized:
        raise ValueError("at least one record is required")
    calls = [record.merge_metrics.calls for record in materialized]
    elapsed = [record.merge_metrics.elapsed_seconds for record in materialized]
    return CompressionSummary(
        n_cases=len(materialized),
        seed=seed,
        target_reduction_percent=TARGET_REDUCTION_PERCENT,
        success_rate=sum(record.status == "compressed" for record in materialized) / len(materialized),
        zero_merge_call_share=sum(count == 0 for count in calls) / len(calls),
        mean_merger_calls=sum(calls) / len(calls),
        merger_call_distribution=dict(sorted(Counter(calls).items())),
        total_elapsed_seconds=sum(elapsed),
        mean_elapsed_seconds=sum(elapsed) / len(elapsed),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="number of task-balanced cases")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--out", type=Path, required=True, help="run-labeled output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = load_cases(n=args.n, seed=args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[CompressedCase | FailedCase] = []
    for case in cases:
        record = compress_case(case.prompt)
        (out_dir / f"{case.key.replace(':', '_')}.json").write_text(record.model_dump_json(indent=2))
        records.append(record)
        metrics = record.merge_metrics
        print(f"{case.key}: {record.status} ({metrics.calls} merger calls, {metrics.elapsed_seconds:.1f}s)")
    summary = summarize(records, seed=args.seed)
    (out_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
