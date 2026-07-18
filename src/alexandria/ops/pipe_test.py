from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import (
    MAX_TARGET_MERGE_ROUNDS,
    TARGET_REFINEMENT_ROUNDS,
    TargetMergeError,
    compare_reports,
    optimization_report,
    propose,
    reduce,
    score_report,
)
from alexandria.utils.embedders import HashEmbedder
from alexandria.utils.tokens import count_tokens

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


class _TargetMerger:
    def __init__(self, *outputs: tuple[str, ...]) -> None:
        self._outputs = list(outputs)
        self.feedback: list[str | None] = []
        self.bases: list[str | None] = []
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []

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
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        self.feedback.append(feedback)
        self.bases.append(base_candidate)
        return self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]


class _CountingEmbedder:
    @property
    def model_id(self) -> str:
        return "counting-3"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([text.count("a"), text.count("b"), text.count("c")], dtype=np.float32) for text in texts]


def test_default_whole_prompt_drift_budget_is_fifty_percent() -> None:
    assert Params().drift_budget == 0.5


def test_target_merge_window_scales_with_the_required_saving() -> None:
    prompt = "aaaa\n" * 10
    merger = _TargetMerger(("aaaa\naaaa\n",) * 10)

    result = reduce(
        prompt,
        _CountingEmbedder(),
        merger,
        params=Params(max_tokens=16, require_target=True),
    )

    assert result.source_tokens == 20
    assert result.reduced_tokens == 16
    assert count_tokens(merger.prompts[0]) == 8


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
    assert result.merge_metrics.jobs_attempted == 1
    assert result.merge_metrics.proposed_edits == 1
    assert result.merge_metrics.applied_edits == 1


def test_strict_target_retries_until_token_and_drift_constraints_pass() -> None:
    merger = _TargetMerger(
        ("too many tokens remain here\n",) * 10,
        ("short\n",) * 10,
    )
    result = reduce(
        "source prompt with tokens\nsecond source line\n",
        HashEmbedder(),
        merger,
        params=Params(drift_budget=2.0, max_tokens=3, require_target=True),
    )

    assert result.text == "short\n"
    assert result.reduced_tokens <= 3
    assert result.merge_metrics.calls == 2
    assert result.merge_metrics.retries == 1
    assert merger.feedback[0] is None
    assert merger.feedback[1] is not None and "hard limit" in merger.feedback[1]
    assert merger.max_tokens[1] < merger.max_tokens[0]


def test_strict_target_selects_the_lowest_drift_candidate_from_one_call() -> None:
    merger = _TargetMerger(("aaaa\n", "aa bb\n", "bbbb\n") * 3 + ("cccc\n",))

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(drift_budget=0.01, max_tokens=3, require_target=True),
    )

    assert result.text == "aa bb\n"
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.retries == 0
    assert result.merge_metrics.candidates_generated == 10


def test_strict_target_mutates_one_base_and_only_keeps_a_non_worsening_candidate() -> None:
    merger = _TargetMerger(
        ("aaaa\n",) * 10,
        ("bbbb\n", "ab\n") * 5,
    )

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(drift_budget=0.01, max_tokens=3, require_target=True),
    )

    assert merger.bases == [None, "aaaa\n"]
    assert result.text == "ab\n"
    rounds = result.merge_metrics.target_rounds
    assert len(rounds) == 2
    assert rounds[0].base_drift == 0.0
    assert not rounds[0].improved
    assert rounds[1].base_drift == rounds[0].selected_drift
    assert rounds[1].selected_drift < rounds[1].base_drift
    assert rounds[1].generated_best_drift == rounds[1].selected_drift
    assert rounds[1].selected_tokens <= rounds[1].base_tokens


def test_strict_target_preserves_markup_structure() -> None:
    merger = _TargetMerger(("compressed\n",) * 10)
    result = reduce(
        "<context>\nlong source text here\nsecond source line\n</context>\n",
        HashEmbedder(),
        merger,
        params=Params(drift_budget=2.0, max_tokens=8, require_target=True),
    )

    assert result.text.startswith("<context>")
    assert result.text.rstrip().endswith("</context>")
    assert result.merge_metrics.calls == 1


def test_strict_target_exhausts_retries_when_embedding_drift_stays_over_budget() -> None:
    merger = _TargetMerger(("short\n",) * 10)

    expected_calls = TARGET_REFINEMENT_ROUNDS + 1
    with pytest.raises(TargetMergeError, match=rf"failed after {expected_calls} calls") as captured:
        reduce(
            "source prompt with tokens\nsecond source line\n",
            HashEmbedder(),
            merger,
            params=Params(drift_budget=0.0, max_tokens=3, require_target=True),
        )

    assert len(merger.feedback) == expected_calls
    assert all(feedback is not None for feedback in merger.feedback[1:])
    assert merger.bases[0] is None
    assert merger.bases[1] == "short\n"
    assert captured.value.metrics.calls == expected_calls
    assert captured.value.metrics.retries == expected_calls - 1
    assert [round_metrics.improved for round_metrics in captured.value.metrics.target_rounds] == [False, False, False]
    assert captured.value.best_tokens == 2
    assert captured.value.best_drift > 0


def test_strict_target_reserves_two_refinement_rounds_after_late_target_success() -> None:
    merger = _TargetMerger(
        ("too many tokens remain here\n",) * 10,
        ("still too many tokens here\n",) * 10,
        ("short\n",) * 10,
    )

    with pytest.raises(TargetMergeError) as captured:
        reduce(
            "source prompt with tokens\nsecond source line\n",
            HashEmbedder(),
            merger,
            params=Params(drift_budget=0.0, max_tokens=3, require_target=True),
        )

    assert captured.value.metrics.calls == MAX_TARGET_MERGE_ROUNDS
    assert len(captured.value.metrics.target_rounds) == MAX_TARGET_MERGE_ROUNDS


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
