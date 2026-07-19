"""Shared contracts and execution helpers for prompt-compression benchmarks."""

from benchmarks.prompt_compression.contracts import (
    BenchmarkCase,
    BenchmarkVerdict,
    ConditionRecord,
    GenerationResult,
    PromptParts,
    UsageRecord,
)
from benchmarks.prompt_compression.runner import benchmark_report, summarize_records
from benchmarks.prompt_compression.statistics import PairedEstimate, paired_score_bootstrap
from benchmarks.prompt_compression.store import RunStore

__all__ = [
    "BenchmarkCase",
    "BenchmarkVerdict",
    "ConditionRecord",
    "GenerationResult",
    "PairedEstimate",
    "PromptParts",
    "RunStore",
    "UsageRecord",
    "benchmark_report",
    "paired_score_bootstrap",
    "summarize_records",
]
