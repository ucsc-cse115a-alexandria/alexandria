from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from benchmarks.prompt_compression.metering import estimate_cost
from benchmarks.prompt_compression.statistics import paired_score_bootstrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.prompt_compression.contracts import ConditionRecord


class _ConditionSummary(TypedDict):
    n_cases: int
    accuracy: float
    correct: int
    mean_sent_tokens: float
    token_reduction: float
    mean_prompt_embedding_cosine_difference: float
    compression_seconds: float
    answer_seconds: float
    reduction_seconds: float
    execution_seconds: float
    estimated_reduction_cost_usd: float
    estimated_answer_cost_usd: float
    estimated_execution_cost_usd: float
    estimated_cost_usd: float


class _ComparisonSummary(TypedDict):
    release_decision: str


def _condition_summary(records: Sequence[ConditionRecord]) -> dict[str, object]:
    source_tokens = sum(record.source_tokens for record in records)
    sent_tokens = sum(record.sent_tokens for record in records)
    correct = sum(record.verdict.correct for record in records)
    reduction_usage = tuple(usage for record in records for usage in record.usage if usage.category != "answer")
    answer_usage = tuple(usage for record in records for usage in record.usage if usage.category == "answer")
    reduction_seconds = sum(record.compression_elapsed_seconds for record in records)
    execution_seconds = sum(record.answer_elapsed_seconds for record in records)
    reduction_cost = estimate_cost(reduction_usage)
    execution_cost = estimate_cost(answer_usage)
    return {
        "n_cases": len(records),
        "accuracy": correct / len(records),
        "official_score": float(np.mean([record.verdict.score for record in records])),
        "correct": correct,
        "mean_source_tokens": source_tokens / len(records),
        "mean_sent_tokens": sent_tokens / len(records),
        "token_reduction": 1.0 - sent_tokens / source_tokens,
        "mean_prompt_embedding_cosine_difference": float(
            np.mean([record.prompt_embedding_cosine_difference for record in records])
        ),
        "p95_prompt_embedding_cosine_difference": float(
            np.quantile([record.prompt_embedding_cosine_difference for record in records], 0.95)
        ),
        # Keep the original field names so older consumers remain compatible.
        "compression_seconds": reduction_seconds,
        "answer_seconds": execution_seconds,
        "reduction_seconds": reduction_seconds,
        "execution_seconds": execution_seconds,
        "estimated_reduction_cost_usd": reduction_cost,
        "estimated_answer_cost_usd": execution_cost,
        "estimated_execution_cost_usd": execution_cost,
        "estimated_cost_usd": sum(record.estimated_cost_usd for record in records),
        "merge_calls": sum(record.merge_metrics.calls for record in records),
        "merge_retries": sum(record.merge_metrics.retries for record in records),
    }


def summarize_records(
    records: Sequence[ConditionRecord],
    *,
    release_threshold: float = 0.90,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Build paired condition summaries and plain PASS/FAIL release decisions."""
    if not records:
        raise ValueError("at least one condition record is required")
    if len({record.benchmark for record in records}) != 1:
        raise ValueError("records from different benchmarks cannot be summarized together")
    identities = [(record.case_key, record.condition) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("records contain duplicate case/condition pairs")
    grouped: defaultdict[str, list[ConditionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.condition].append(record)
    if "original" not in grouped:
        raise ValueError("records must include an original condition")
    for condition_records in grouped.values():
        condition_records.sort(key=lambda record: record.case_key)
    original = grouped["original"]
    original_keys = [record.case_key for record in original]
    conditions: dict[str, object] = {"original": _condition_summary(original)}
    comparisons: dict[str, object] = {}
    for condition, condition_records in sorted(grouped.items()):
        if condition == "original":
            continue
        if [record.case_key for record in condition_records] != original_keys:
            raise ValueError(f"condition {condition!r} does not contain the same paired cases as original")
        conditions[condition] = _condition_summary(condition_records)
        official = paired_score_bootstrap(
            [record.verdict.score for record in original],
            [record.verdict.score for record in condition_records],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            release_threshold=release_threshold,
        )
        accuracy = paired_score_bootstrap(
            [record.verdict.correct for record in original],
            [record.verdict.correct for record in condition_records],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            release_threshold=release_threshold,
        )
        transitions = {"both_correct": 0, "regression": 0, "improvement": 0, "both_wrong": 0}
        for before, after in zip(original, condition_records, strict=True):
            key = (
                "both_correct"
                if before.verdict.correct and after.verdict.correct
                else "regression"
                if before.verdict.correct
                else "improvement"
                if after.verdict.correct
                else "both_wrong"
            )
            transitions[key] += 1
        comparisons[condition] = {
            "official_score_retention": official.model_dump(),
            "accuracy_retention": accuracy.model_dump(),
            "transitions": transitions,
            "release_decision": (
                "PASS: accuracy-retention confidence interval clears the release threshold"
                if accuracy.clears_release_threshold
                else "FAIL: accuracy-retention confidence interval does not clear the release threshold"
            ),
        }
    tasks: dict[str, dict[str, object]] = {}
    for task in sorted({record.task for record in records}):
        task_records = [record for record in records if record.task == task]
        task_conditions: defaultdict[str, list[ConditionRecord]] = defaultdict(list)
        for record in task_records:
            task_conditions[record.condition].append(record)
        tasks[task] = {
            condition: _condition_summary(condition_records)
            for condition, condition_records in sorted(task_conditions.items())
        }
    return {
        "schema_version": 1,
        "benchmark": records[0].benchmark,
        "release_threshold": release_threshold,
        "bootstrap": {
            "method": "paired percentile bootstrap over case indices",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
        },
        "conditions": conditions,
        "tasks": tasks,
        "comparisons": comparisons,
    }


def benchmark_report(summary: dict[str, object]) -> str:
    """Render the user-facing evidence table from a saved summary payload."""
    raw_conditions = summary["conditions"]
    raw_comparisons = summary["comparisons"]
    if not isinstance(raw_conditions, dict) or not isinstance(raw_comparisons, dict):
        raise TypeError("summary conditions and comparisons must be mappings")
    conditions = cast("dict[str, _ConditionSummary]", raw_conditions)
    comparisons = cast("dict[str, _ComparisonSummary]", raw_comparisons)
    lines = [
        "| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | "
        "Execution time | Execution cost | Reduction time | Reduction cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, raw in conditions.items():
        lines.append(
            f"| {condition} | {float(raw['mean_sent_tokens']):,.1f} | "
            f"{float(raw['token_reduction']) * 100:.1f}% | "
            f"{float(raw['mean_prompt_embedding_cosine_difference']):.4f} | "
            f"{float(raw['accuracy']) * 100:.1f}% "
            f"({int(raw['correct'])}/{int(raw['n_cases'])}) | "
            f"{float(raw['execution_seconds']):.1f}s | "
            f"${float(raw['estimated_execution_cost_usd']):.4f} | "
            f"{float(raw['reduction_seconds']):.1f}s | "
            f"${float(raw['estimated_reduction_cost_usd']):.4f} |"
        )
    lines.extend(["", "## Release decisions", ""])
    for condition, raw in comparisons.items():
        lines.append(f"- **{condition}:** {raw['release_decision']}")
    return "\n".join(lines)
