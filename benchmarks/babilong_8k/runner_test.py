from __future__ import annotations

from alexandria.ir.contracts import MergeMetrics
from benchmarks.babilong_8k.cases import BABILongCase
from benchmarks.babilong_8k.runner import CompressionResult, compare, run_experiment

CASES = (
    BABILongCase(
        key="qa1:0",
        task="qa1",
        source_index=0,
        prompt="Long irrelevant text. Mary journeyed to the bathroom. Where is Mary?",
        question="Where is Mary?",
        target="bathroom",
    ),
    BABILongCase(
        key="qa1:1",
        task="qa1",
        source_index=1,
        prompt="Long irrelevant text. Mary journeyed to the kitchen. Where is Mary?",
        question="Where is Mary?",
        target="kitchen",
    ),
)


def generate(prompt: str) -> str:
    return "bathroom" if "bathroom" in prompt else "bedroom"


def compress(prompt: str) -> str:
    return prompt.replace("Long irrelevant text. ", "")


def compress_with_metrics(prompt: str) -> CompressionResult:
    return CompressionResult(
        prompt=compress(prompt),
        merge_metrics=MergeMetrics(calls=3, retries=1, pairs_attempted=2, proposed_edits=1, applied_edits=1),
    )


def test_run_experiment_records_tokens_and_accuracy() -> None:
    result = run_experiment(CASES, generate, label="original", model="stub")
    assert result.accuracy == 0.5
    assert result.token_reduction == 0
    assert result.mean_source_tokens == result.mean_sent_tokens


def test_run_experiment_records_compressed_prompt() -> None:
    result = run_experiment(CASES, generate, label="compressed", model="stub", transform=compress_with_metrics)
    assert result.token_reduction > 0
    assert result.records[0].sent_tokens < result.records[0].source_tokens
    assert not result.records[0].prompt.startswith("Long irrelevant")
    assert result.merge_calls == 6
    assert result.merge_retries == 2


def test_compare_reports_percentage_point_change() -> None:
    original = run_experiment(CASES, generate, label="original", model="stub")
    worse = run_experiment(CASES, lambda _prompt: "wrong", label="compressed", model="stub")
    table = compare(original, worse)
    assert table.splitlines()[0] == (
        "| Condition | Mean input tokens | Token reduction | Merge calls | Retries | Task accuracy | Accuracy change |"
    )
    assert "| original |" in table
    assert "50.0% (1/2)" in table
    assert "-50.0 pp" in table


def test_compare_requires_a_result() -> None:
    try:
        compare()
    except ValueError as error:
        assert str(error) == "at least one result is required"
    else:
        raise AssertionError("compare() should reject an empty result list")


def test_compare_rejects_different_case_sets() -> None:
    full = run_experiment(CASES, generate, label="full", model="stub")
    partial = run_experiment(CASES[:1], generate, label="partial", model="stub")
    try:
        compare(full, partial)
    except ValueError as error:
        assert str(error) == "all results must contain the same cases in the same order"
    else:
        raise AssertionError("compare() should reject incomparable case sets")
