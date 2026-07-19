from __future__ import annotations

from benchmarks.prompt_compression.contracts import BenchmarkVerdict, ConditionRecord
from benchmarks.prompt_compression.runner import benchmark_report, summarize_records


def _record(key: str, condition: str, correct: bool, sent_tokens: int) -> ConditionRecord:
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
    )


def test_summary_pairs_conditions_and_reports_release_decision() -> None:
    records = (
        _record("a", "original", True, 100),
        _record("b", "original", True, 100),
        _record("a", "keep90", True, 90),
        _record("b", "keep90", True, 90),
    )
    summary = summarize_records(records, bootstrap_samples=500, bootstrap_seed=3)
    comparison = summary["comparisons"]["keep90"]  # type: ignore[index]
    assert comparison["release_decision"].startswith("PASS")  # type: ignore[index]
    assert summary["tasks"]["qa"]["keep90"]["accuracy"] == 1.0  # type: ignore[index]
    report = benchmark_report(summary)
    assert "| keep90 | 90.0 | 10.0% | 100.0%" in report
    assert "PASS:" in report


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
