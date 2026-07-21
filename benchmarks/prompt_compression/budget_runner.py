from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np

from benchmarks.prompt_compression.runner import condition_summary
from benchmarks.prompt_compression.statistics import paired_score_bootstrap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from benchmarks.prompt_compression.contracts import ConditionRecord


def summarize_budget_records(
    records: Sequence[ConditionRecord],
    *,
    expected_conditions: Sequence[str],
    expected_case_keys: Sequence[str],
    errors: Sequence[Mapping[str, object]] = (),
    release_threshold: float = 0.90,
    minimum_token_reduction: float = 0.10,
    minimum_budget_compliance: float = 0.95,
    minimum_completion_rate: float = 0.98,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Summarize a best-effort semantic-budget sweep without hiding incomplete conditions."""
    if not records:
        raise ValueError("at least one condition record is required")
    if len({record.benchmark for record in records}) != 1:
        raise ValueError("records from different benchmarks cannot be summarized together")
    originals = sorted(
        (record for record in records if record.condition == "original"), key=lambda record: record.case_key
    )
    if [record.case_key for record in originals] != sorted(expected_case_keys):
        raise ValueError("original condition must contain every expected case")
    original_by_key = {record.case_key: record for record in originals}
    grouped: defaultdict[str, list[ConditionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.condition].append(record)
    conditions: dict[str, object] = {"original": condition_summary(originals)}
    comparisons: dict[str, object] = {}
    errors_by_condition: defaultdict[str, int] = defaultdict(int)
    for error in errors:
        condition = error.get("condition")
        if isinstance(condition, str):
            errors_by_condition[condition] += 1

    for condition in expected_conditions:
        compressed = sorted(grouped.get(condition, []), key=lambda record: record.case_key)
        completed_keys = [record.case_key for record in compressed]
        if len(completed_keys) != len(set(completed_keys)):
            raise ValueError(f"condition {condition!r} contains duplicate cases")
        matched_original = [original_by_key[key] for key in completed_keys]
        completion_rate = len(compressed) / len(expected_case_keys)
        if not compressed:
            conditions[condition] = None
            comparisons[condition] = {
                "completed_cases": 0,
                "expected_cases": len(expected_case_keys),
                "completion_rate": 0.0,
                "errors": errors_by_condition[condition],
                "publication_pass": False,
            }
            continue
        raw = condition_summary(compressed)
        conditions[condition] = raw
        accuracy = paired_score_bootstrap(
            [record.verdict.correct for record in matched_original],
            [record.verdict.correct for record in compressed],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            release_threshold=release_threshold,
        )
        compliance = float(np.mean([record.context_cos_sim_diff_budget_met is True for record in compressed]))
        token_reduction = _number(raw, "token_reduction")
        semantic_safety_pass = accuracy.clears_release_threshold
        useful_reduction_pass = token_reduction >= minimum_token_reduction
        budget_reliability_pass = compliance >= minimum_budget_compliance
        operational_reliability_pass = completion_rate >= minimum_completion_rate
        publication_pass = all(
            (
                semantic_safety_pass,
                useful_reduction_pass,
                budget_reliability_pass,
                operational_reliability_pass,
            )
        )
        comparisons[condition] = {
            "completed_cases": len(compressed),
            "expected_cases": len(expected_case_keys),
            "completion_rate": completion_rate,
            "errors": errors_by_condition[condition],
            "matched_original_accuracy": float(np.mean([record.verdict.correct for record in matched_original])),
            "accuracy_retention": accuracy.model_dump(mode="json"),
            "context_budget_compliance": compliance,
            "semantic_safety_pass": semantic_safety_pass,
            "useful_reduction_pass": useful_reduction_pass,
            "budget_reliability_pass": budget_reliability_pass,
            "operational_reliability_pass": operational_reliability_pass,
            "publication_pass": publication_pass,
        }
    return {
        "schema_version": 1,
        "experiment_mode": "cos_sim_diff_budget",
        "benchmark": records[0].benchmark,
        "expected_case_keys": list(expected_case_keys),
        "expected_conditions": list(expected_conditions),
        "thresholds": {
            "accuracy_retention_lower_bound": release_threshold,
            "minimum_token_reduction": minimum_token_reduction,
            "minimum_context_budget_compliance": minimum_budget_compliance,
            "minimum_completion_rate": minimum_completion_rate,
        },
        "bootstrap": {
            "method": "paired percentile bootstrap over completed case indices",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
        },
        "conditions": conditions,
        "comparisons": comparisons,
    }


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def budget_benchmark_report(summary: Mapping[str, object]) -> str:
    """Render a complete condition table with operational measurements."""
    raw_conditions = summary["conditions"]
    raw_comparisons = summary["comparisons"]
    if not isinstance(raw_conditions, dict) or not isinstance(raw_comparisons, dict):
        raise TypeError("budget summary conditions and comparisons must be mappings")
    conditions = cast("dict[str, object]", raw_conditions)
    comparisons = cast("dict[str, object]", raw_comparisons)
    lines = [
        "| Condition | Complete | Accuracy | Accuracy retention (95% CI) | Token reduction | "
        "Mean full-prompt cos diff | Budget compliance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    original = conditions["original"]
    if not isinstance(original, dict):
        raise TypeError("original summary must be a mapping")
    original_values = cast("dict[str, object]", original)
    lines.append(
        f"| original | {int(_number(original_values, 'n_cases'))}/"
        f"{int(_number(original_values, 'n_cases'))} | "
        f"{_number(original_values, 'accuracy') * 100:.1f}% | "
        "100.0% | 0.0% | 0.000000 | 100.0% |"
    )
    expected = summary["expected_conditions"]
    if not isinstance(expected, list | tuple):
        raise TypeError("expected conditions must be a sequence of strings")
    expected_objects = cast("Sequence[object]", expected)
    if not all(isinstance(item, str) for item in expected_objects):
        raise TypeError("expected conditions must be a sequence of strings")
    expected_conditions = cast("Sequence[str]", expected_objects)
    for condition in expected_conditions:
        raw = conditions[condition]
        comparison = comparisons[condition]
        if not isinstance(comparison, dict):
            raise TypeError("comparison summary must be a mapping")
        comparison_values = cast("dict[str, object]", comparison)
        complete = (
            f"{int(_number(comparison_values, 'completed_cases'))}/{int(_number(comparison_values, 'expected_cases'))}"
        )
        if not isinstance(raw, dict):
            lines.append(f"| {condition} | {complete} | — | — | — | — | — |")
            continue
        raw_values = cast("dict[str, object]", raw)
        raw_retention = comparison_values["accuracy_retention"]
        if not isinstance(raw_retention, dict):
            raise TypeError("accuracy retention summary must be a mapping")
        retention = cast("dict[str, object]", raw_retention)
        lines.append(
            f"| {condition} | {complete} | {_number(raw_values, 'accuracy') * 100:.1f}% | "
            f"{_number(retention, 'retention') * 100:.1f}% "
            f"({_number(retention, 'confidence_low') * 100:.1f}%–"
            f"{_number(retention, 'confidence_high') * 100:.1f}%) | "
            f"{_number(raw_values, 'token_reduction') * 100:.1f}% | "
            f"{_number(raw_values, 'mean_prompt_embedding_cosine_difference'):.6f} | "
            f"{_number(comparison_values, 'context_budget_compliance') * 100:.1f}% |"
        )
    return "\n".join(lines)
