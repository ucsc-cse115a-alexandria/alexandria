from __future__ import annotations

from benchmarks.prompt_compression.budget_runner import budget_benchmark_report, summarize_budget_records
from benchmarks.prompt_compression.contracts import BenchmarkVerdict, ConditionRecord


def _record(case_key: str, condition: str, *, sent_tokens: int, correct: bool) -> ConditionRecord:
    return ConditionRecord(
        case_key=case_key,
        benchmark="fixture",
        task="qa",
        condition=condition,
        reduction_percent=100.0 - sent_tokens,
        source_tokens=100,
        target_tokens=50 if condition != "original" else 100,
        sent_tokens=sent_tokens,
        prompt_sha256="hash",
        response="answer",
        response_model="stub",
        verdict=BenchmarkVerdict(score=float(correct), correct=correct),
        configured_cos_sim_diff_budget=0.02 if condition != "original" else None,
        context_embedding_cosine_difference=0.01 if condition != "original" else 0.0,
        context_cos_sim_diff_budget_met=True if condition != "original" else None,
        prompt_embedding_cosine_difference=0.01 if condition != "original" else 0.0,
    )


def test_budget_summary_reports_incomplete_pairs_without_hiding_them() -> None:
    records = (
        _record("case:1", "original", sent_tokens=100, correct=True),
        _record("case:2", "original", sent_tokens=100, correct=True),
        _record("case:1", "budget0p02", sent_tokens=80, correct=True),
    )

    summary = summarize_budget_records(
        records,
        expected_conditions=("budget0p02",),
        expected_case_keys=("case:1", "case:2"),
        errors=({"condition": "budget0p02"},),
        bootstrap_samples=100,
    )

    comparison = summary["comparisons"]["budget0p02"]  # type: ignore[index]
    assert comparison["completion_rate"] == 0.5  # type: ignore[index]
    assert comparison["errors"] == 1  # type: ignore[index]
    assert comparison["operational_reliability_pass"] is False  # type: ignore[index]
    assert comparison["publication_pass"] is False  # type: ignore[index]
    assert "1/2" in budget_benchmark_report(summary)


def test_budget_summary_pass_requires_accuracy_reduction_budget_and_completion() -> None:
    records = tuple(
        _record(case_key, condition, sent_tokens=80 if condition != "original" else 100, correct=True)
        for condition in ("original", "budget0p02")
        for case_key in ("case:1", "case:2")
    )

    summary = summarize_budget_records(
        records,
        expected_conditions=("budget0p02",),
        expected_case_keys=("case:1", "case:2"),
        bootstrap_samples=100,
    )

    comparison = summary["comparisons"]["budget0p02"]  # type: ignore[index]
    assert comparison["publication_pass"] is True  # type: ignore[index]
    assert "| PASS |" in budget_benchmark_report(summary)
