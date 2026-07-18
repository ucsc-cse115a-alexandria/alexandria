from __future__ import annotations

from babilong_8k_compression_bench import CompressedCase, FailedCase, compress_case, summarize

from alexandria.ir.contracts import MergeMetrics
from alexandria.utils.embedders import HashEmbedder
from alexandria.utils.tokens import count_tokens


class _OversizedMerger:
    """Offline target merger whose candidates require deterministic target repair."""

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first.strip()

    def merge_candidates_to_target(
        self,
        prompt: str,
        max_tokens: int,
        feedback: str | None = None,
        base_candidate: str | None = None,
    ) -> tuple[str, ...]:
        del prompt, max_tokens, feedback, base_candidate
        return ("far too many tokens remain in this oversized candidate line\n",) * 10


def test_compress_case_repairs_oversized_candidates_into_a_target_safe_record() -> None:
    prompt = "".join(f"source line number {index} with several words\n" for index in range(8))
    record = compress_case(prompt, HashEmbedder(), _OversizedMerger())

    assert isinstance(record, CompressedCase)
    assert count_tokens(record.prompt) <= max(1, count_tokens(prompt) // 10)
    assert record.merge_metrics.calls > 0
    assert record.merge_metrics.repaired_tokens > 0
    assert record.merge_metrics.elapsed_seconds > 0.0


def test_summarize_aggregates_calls_failures_and_elapsed() -> None:
    records = [
        CompressedCase(prompt="a", merge_metrics=MergeMetrics(calls=0, elapsed_seconds=1.0)),
        CompressedCase(prompt="b", merge_metrics=MergeMetrics(calls=1, elapsed_seconds=2.0)),
        FailedCase(error="target merge failed", merge_metrics=MergeMetrics(calls=3, elapsed_seconds=5.0)),
    ]

    summary = summarize(records, seed=42)

    assert summary.n_cases == 3
    assert summary.success_rate == 2 / 3
    assert summary.zero_merge_call_share == 1 / 3
    assert summary.mean_merger_calls == 4 / 3
    assert summary.merger_call_distribution == {0: 1, 1: 1, 3: 1}
    assert summary.total_elapsed_seconds == 8.0
    assert summary.mean_elapsed_seconds == 8.0 / 3
