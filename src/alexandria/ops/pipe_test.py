from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import propose, reduce, score_report
from alexandria.utils.embedders import HashEmbedder

if TYPE_CHECKING:
    import pytest
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


def test_default_whole_prompt_cos_sim_diff_budget_is_fifty_percent() -> None:
    assert Params().cos_sim_diff_budget == 0.5


def test_default_reduce_and_propose_skip_the_unused_score_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_score(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("default optimizer must not invoke score")

    pipe_module = importlib.import_module("alexandria.ops.pipe")
    monkeypatch.setattr(pipe_module, "score", unexpected_score)

    result = reduce(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(cos_sim_diff_budget=2.0),
    )
    proposal = propose(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(cos_sim_diff_budget=2.0),
    )

    assert result.applied
    assert proposal.diffs


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
        params=Params(cos_sim_diff_budget=2.0),
    )

    assert result.merge_metrics.calls == 2
    assert result.merge_metrics.retries == 1
    assert result.merge_metrics.jobs_attempted == 1
    assert result.merge_metrics.proposed_edits == 1
    assert result.merge_metrics.applied_edits == 1


def test_reduce_stamps_embedding_and_wall_clock_metrics() -> None:
    result = reduce(
        "aa bb\naaa bb\ncc\n",
        _CountingEmbedder(),
        _RetryOnceMerger(),
        params=Params(cos_sim_diff_budget=2.0),
    )

    metrics = result.merge_metrics
    assert metrics.embed_calls > 0
    assert metrics.embed_texts >= metrics.embed_calls
    assert metrics.elapsed_seconds > 0.0


def test_propose_returns_the_document_and_one_diff_per_candidate() -> None:
    proposal = propose(
        "# A\nrepeat me\nrepeat me\n# B\necho twice\necho twice\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(cos_sim_diff_budget=2.0),
    )

    assert len(proposal.diffs) == 2
    originals = {diff.spans[0].original for diff in proposal.diffs}
    assert originals == {"repeat me\n", "echo twice\n"}
    assert proposal.document.text.count("repeat me") == 2  # proposing edits does not apply them
