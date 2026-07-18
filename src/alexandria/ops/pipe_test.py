from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import compare_reports, optimization_report, propose, reduce, score_report
from alexandria.utils.embedders import HashEmbedder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _FirstWinsMerger:
    """Offline merger that returns the first sentence, so every merge lands as a Delete of the second."""

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first.strip()


class _RetryOnceMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second
        if feedback is None:
            return "this rewrite is deliberately longer than both source sentences combined"
        return first.strip()


class _CountingEmbedder:
    @property
    def model_id(self) -> str:
        return "counting-3"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([text.count("a"), text.count("b"), text.count("c")], dtype=np.float32) for text in texts]


def test_score_report_adds_the_redundant_peer() -> None:
    rows = score_report("repeat me\nrepeat me\nunique line\n", HashEmbedder())
    assert rows[0]["most_similar_id"] == rows[1]["id"]
    assert rows[0]["most_similar_text"] == "repeat me"
    redundancy = rows[0]["redundancy"]
    assert isinstance(redundancy, float)
    assert redundancy > 0.99


def test_score_report_single_sentence_has_no_peer() -> None:
    rows = score_report("only one\n", HashEmbedder())
    assert rows[0]["most_similar_id"] is None
    assert rows[0]["most_similar_text"] is None


def test_reduce_records_merge_calls_and_retries() -> None:
    result = reduce(
        "aa bb\naaa bb\ncc\n",
        _CountingEmbedder(),
        _RetryOnceMerger(),
        params=Params(drift_budget=2.0),
    )

    assert result.merge_metrics.calls == 2
    assert result.merge_metrics.retries == 1
    assert result.merge_metrics.pairs_attempted == 1
    assert result.merge_metrics.proposed_edits == 1
    assert result.merge_metrics.applied_edits == 1


def test_optimization_report_includes_compression_and_quality_metrics() -> None:
    report = optimization_report(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(drift_budget=2.0),
    )

    assert report.tokens.reduced < report.tokens.source
    assert report.tokens.saved == report.tokens.source - report.tokens.reduced
    assert report.quality.instruction_coverage == 1.0
    assert report.quality.minimum_instruction_similarity == 1.0


def test_compare_reports_flags_token_and_quality_regressions() -> None:
    baseline = optimization_report(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(drift_budget=2.0),
    )
    current = baseline.model_copy(
        update={
            "tokens": baseline.tokens.model_copy(update={"reduced": baseline.tokens.reduced + 1}),
            "quality": baseline.quality.model_copy(
                update={"instruction_coverage": baseline.quality.instruction_coverage - 0.1}
            ),
        }
    )

    comparison = compare_reports(current, baseline)

    assert not comparison.passed
    assert {regression.metric for regression in comparison.regressions} == {
        "tokens.reduced",
        "quality.instruction_coverage",
    }


def test_propose_returns_the_document_and_one_diff_per_candidate() -> None:
    proposal = propose(
        "# A\nrepeat me\nrepeat me\n# B\necho twice\necho twice\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(drift_budget=2.0),
    )

    assert len(proposal.diffs) == 2
    originals = {diff.spans[0].original for diff in proposal.diffs}
    assert originals == {"repeat me\n", "echo twice\n"}
    assert proposal.document.text.count("repeat me") == 2  # proposing edits does not apply them
