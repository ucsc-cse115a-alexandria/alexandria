from __future__ import annotations

import pytest

from benchmarks.prompt_compression.contracts import BenchmarkVerdict, ConditionRecord, UsageRecord
from benchmarks.prompt_compression.runner import benchmark_report, summarize_records


def _record(
    key: str,
    condition: str,
    correct: bool,
    sent_tokens: int,
    *,
    usage: tuple[UsageRecord, ...] = (),
    estimated_cost_usd: float = 0.0,
    reduction_seconds: float = 0.0,
    execution_seconds: float = 0.0,
) -> ConditionRecord:
    return ConditionRecord(
        case_key=key,
        benchmark="fixture",
        task="qa",
        condition=condition,
        reduction_percent=0 if condition == "original" else 10,
        source_tokens=100,
        target_tokens=sent_tokens,
        sent_tokens=sent_tokens,
        prompt_sha256="0" * 64,
        response="answer",
        response_model="stub",
        verdict=BenchmarkVerdict(score=float(correct), correct=correct),
        compression_elapsed_seconds=reduction_seconds,
        answer_elapsed_seconds=execution_seconds,
        usage=usage,
        estimated_cost_usd=estimated_cost_usd,
    )


def test_summary_pairs_conditions_and_reports_metrics() -> None:
    records = (
        _record("a", "original", True, 100),
        _record("b", "original", True, 100),
        _record("a", "keep90", True, 90),
        _record("b", "keep90", True, 90),
    )
    summary = summarize_records(records, bootstrap_samples=500, bootstrap_seed=3)
    comparison = summary["comparisons"]["keep90"]  # type: ignore[index]
    assert comparison["accuracy_retention"]["retention"] == 1.0  # type: ignore[index]
    assert summary["tasks"]["qa"]["keep90"]["accuracy"] == 1.0  # type: ignore[index]
    report = benchmark_report(summary)
    assert "| keep90 | 90.0 | 10.0% | 0.0000 | 100.0%" in report
    assert "Release decisions" not in report


def test_summary_qualifies_original_accuracy_before_comparing_compression() -> None:
    records = (
        _record("a", "original", True, 100),
        _record("b", "original", False, 100),
    )

    summary = summarize_records(records, minimum_original_accuracy=0.6, bootstrap_samples=10)

    qualification = summary["baseline_qualification"]  # type: ignore[assignment]
    assert qualification["original_accuracy"] == 0.5  # type: ignore[index]
    assert qualification["qualifies"] is False  # type: ignore[index]
    assert "Baseline qualification" not in benchmark_report(summary)


def test_summary_separates_reduction_from_execution_time_and_cost() -> None:
    answer = UsageRecord(
        category="answer",
        model="answer-model",
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=10,
        total_tokens=110,
        elapsed_seconds=1.0,
    )
    embedding = UsageRecord(
        category="embedding",
        model="embedding-model",
        input_tokens=1_000,
        cached_input_tokens=0,
        output_tokens=0,
        total_tokens=1_000,
        elapsed_seconds=2.0,
    )
    merge = UsageRecord(
        category="compression",
        model="merge-model",
        input_tokens=200,
        cached_input_tokens=0,
        output_tokens=20,
        total_tokens=220,
        elapsed_seconds=3.0,
    )
    total_cost = (100 + 10 * 6 + 1_000 * 0.02 + 200 + 20 * 6) / 1_000_000
    records = (
        _record("a", "original", True, 100, usage=(answer,), estimated_cost_usd=0.00016),
        _record(
            "a",
            "keep90",
            True,
            90,
            usage=(embedding, merge, answer),
            estimated_cost_usd=total_cost,
            reduction_seconds=5.0,
            execution_seconds=1.0,
        ),
    )

    summary = summarize_records(records, bootstrap_samples=10)
    compressed = summary["conditions"]["keep90"]  # type: ignore[index]
    assert compressed["estimated_reduction_cost_usd"] == pytest.approx(0.00034)  # type: ignore[index]
    assert compressed["estimated_execution_cost_usd"] == pytest.approx(0.00016)  # type: ignore[index]
    assert compressed["reduction_seconds"] == 5.0  # type: ignore[index]
    assert compressed["execution_seconds"] == 1.0  # type: ignore[index]
    report = benchmark_report(summary)
    assert "Execution time" in report
    assert "Execution cost" in report
    assert "Reduction time" in report
    assert "Reduction cost" in report


def test_summary_rejects_unpaired_conditions() -> None:
    records = (
        _record("a", "original", True, 100),
        _record("b", "original", True, 100),
        _record("a", "keep90", True, 90),
    )
    try:
        summarize_records(records, bootstrap_samples=10)
    except ValueError as error:
        assert "same paired cases" in str(error)
    else:
        raise AssertionError("unpaired conditions should fail")


def test_summary_rejects_duplicate_case_condition_pairs() -> None:
    record = _record("a", "original", True, 100)
    try:
        summarize_records((record, record), bootstrap_samples=10)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate records should fail")
