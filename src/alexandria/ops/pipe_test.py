from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.ir.contracts import Params
from alexandria.ir.document import Sentence, SentenceId
from alexandria.ops.pipe import (
    InfeasibleTargetError,
    _target_merge_window,  # pyright: ignore[reportPrivateUsage]
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

    from alexandria.ir.contracts import ReportedCandidate


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
    """Offline targeted merger: each call returns the next queued candidate set (last one repeats)."""

    def __init__(self, *candidate_sets: tuple[str, ...]) -> None:
        self._candidate_sets = list(candidate_sets)
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first.strip()

    def merge_candidates_to_target(self, prompt: str, max_tokens: int) -> tuple[str, ...]:
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        return self._candidate_sets.pop(0) if len(self._candidate_sets) > 1 else self._candidate_sets[0]


class _CountingEmbedder:
    @property
    def model_id(self) -> str:
        return "counting-3"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([text.count("a"), text.count("b"), text.count("c")], dtype=np.float32) for text in texts]


class _LengthEmbedder:
    """Identical texts embed identically, but whole-document drift grows with deleted length."""

    @property
    def model_id(self) -> str:
        return "length-2"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([1.0, len(text) / 40.0], dtype=np.float32) for text in texts]


class _RecordingReporter:
    """Records every ReductionReporter event so tests can assert what the target merge emitted."""

    def __init__(self) -> None:
        self.target_groups: list[tuple[str, int, int]] = []
        self.target_rounds: list[tuple[int, str | None, tuple[ReportedCandidate, ...], ReportedCandidate, bool]] = []
        self.target_group_done_calls: list[tuple[bool, int]] = []

    def redundant_pair(self, first: str, second: str, similarity: float) -> None:
        pass

    def pair_merged(self, merged: str | None, decision: str) -> None:
        pass

    def target_group(self, source_segment: str, group_tokens: int, required_savings: int) -> None:
        self.target_groups.append((source_segment, group_tokens, required_savings))

    def target_round(
        self,
        round_number: int,
        base: str | None,
        candidates: tuple[ReportedCandidate, ...],
        selected: ReportedCandidate,
        selected_from_generation: bool,
    ) -> None:
        self.target_rounds.append((round_number, base, candidates, selected, selected_from_generation))

    def target_group_done(self, applied: bool, document_tokens: int) -> None:
        self.target_group_done_calls.append((applied, document_tokens))


def test_default_whole_prompt_drift_budget_is_fifty_percent() -> None:
    assert Params().drift_budget == 0.5


def test_target_merge_window_scales_with_the_required_saving() -> None:
    prompt = "".join(f"{letter}\n" for letter in "abcdefghij")  # distinct lines: nothing to prune
    merger = _TargetMerger(("a\nb\n",) * 10)

    result = reduce(
        prompt,
        HashEmbedder(),
        merger,
        params=Params(max_tokens=16, require_target=True, drift_budget=2.0),
    )

    assert result.source_tokens == 20
    assert result.reduced_tokens <= 16
    assert result.merge_metrics.pruned_sentences == 0
    assert count_tokens(merger.prompts[0]) == 8


def test_target_merge_window_does_not_hide_one_semantic_outlier_in_a_good_average() -> None:
    similarities = (1.0, 0.2, 0.59, 0.59)
    group = tuple(
        Sentence(
            id=SentenceId(f"s{index}"),
            text=f"line {index}\n",
            token_count=1,
            embedding=np.array([similarity, np.sqrt(1.0 - similarity**2)], dtype=np.float32),
        )
        for index, similarity in enumerate(similarities)
    )

    selected = _target_merge_window(group, required_savings=1, document_embedding=np.array([1.0, 0.0]))

    assert tuple(sentence.id for sentence in selected) == (SentenceId("s2"), SentenceId("s3"))


def test_target_merge_window_avoids_lines_matching_query_anchor_terms() -> None:
    group = tuple(
        Sentence(
            id=SentenceId(f"s{index}"),
            text=text,
            token_count=1,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        for index, text in enumerate(
            ("generic one\n", "Mary moved to the office.\n", "generic two\n", "generic three\n")
        )
    )

    selected = _target_merge_window(
        group,
        required_savings=1,
        document_embedding=np.array([1.0, 0.0]),
        protected_terms=frozenset({"mary"}),
    )

    assert tuple(sentence.id for sentence in selected) == (SentenceId("s2"), SentenceId("s3"))


def test_strict_target_prunes_exact_duplicates_without_merge_calls() -> None:
    merger = _TargetMerger(("unused\n",) * 10)

    result = reduce(
        "aaaa\n" * 10,
        _CountingEmbedder(),
        merger,
        params=Params(max_tokens=16, require_target=True),
    )

    assert result.text == "aaaa\n" * 8
    assert result.reduced_tokens == 16
    assert result.merge_metrics.calls == 0
    assert result.merge_metrics.pruned_sentences == 2
    assert result.merge_metrics.pruned_tokens == 4


def test_prune_drift_backoff_keeps_the_longest_prefix_within_budget() -> None:
    # Full prune (2 deletions) drifts ~0.006, one deletion ~0.0013: a 0.003 budget keeps exactly one.
    merger = _TargetMerger(("aaaa\n",) * 10)

    result = reduce(
        "aaaa\n" * 10,
        _LengthEmbedder(),
        merger,
        params=Params(max_tokens=16, require_target=True, drift_budget=0.003),
    )

    assert result.reduced_tokens <= 16
    assert result.merge_metrics.pruned_sentences == 1
    assert result.merge_metrics.pruned_tokens == 2
    assert result.merge_metrics.drift_budget_met is False


def test_prune_may_undershoot_the_target_since_undershoot_is_allowed() -> None:
    # Two identical 3-token lines: deleting one drops to 3 tokens, below the 5-token target. With the
    # floor guard gone, undershoot is fine, so pruning takes the duplicate without any merge call.
    merger = _TargetMerger(("aa bb cc\n",) * 10)

    result = reduce(
        "aa bb\naa bb\n",
        _CountingEmbedder(),
        merger,
        params=Params(max_tokens=5, require_target=True),
    )

    assert result.merge_metrics.pruned_sentences == 1
    assert result.merge_metrics.calls == 0
    assert result.reduced_tokens < 5


def test_strict_target_within_window_returns_unchanged_with_only_represent_embeds() -> None:
    merger = _TargetMerger(("unused\n",) * 10)
    prompt = "a\nb\nc\nd\ne\n"

    result = reduce(prompt, HashEmbedder(), merger, params=Params(max_tokens=10, require_target=True))

    assert result.text == prompt
    assert result.merge_metrics.calls == 0
    assert result.merge_metrics.pruned_sentences == 0
    assert result.merge_metrics.embed_calls == 2  # represent's two batches; no prune-verify embed


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


def test_reduce_stamps_embedding_and_wall_clock_metrics() -> None:
    result = reduce(
        "aa bb\naaa bb\ncc\n",
        _CountingEmbedder(),
        _RetryOnceMerger(),
        params=Params(drift_budget=2.0),
    )

    metrics = result.merge_metrics
    assert metrics.embed_calls > 0
    assert metrics.embed_texts >= metrics.embed_calls
    assert metrics.elapsed_seconds > 0.0


def test_strict_target_repairs_an_over_budget_candidate_without_retry() -> None:
    merger = _TargetMerger(("too many tokens remain here\n",) * 2)
    result = reduce(
        "source prompt with tokens\nsecond source line\n",
        HashEmbedder(),
        merger,
        params=Params(drift_budget=2.0, max_tokens=3, require_target=True),
    )

    assert result.reduced_tokens <= 3
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.retries == 0
    assert result.merge_metrics.repaired_tokens > 0
    assert result.merge_metrics.embed_calls == 3  # represent's two batches plus one repaired-candidate batch


def test_strict_target_selects_the_lowest_drift_candidate_from_one_call() -> None:
    merger = _TargetMerger(("aaaa\n", "aa bb\n", "bbbb\n"))

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(drift_budget=0.01, max_tokens=3, require_target=True),
    )

    assert result.text == "aa bb\n"
    assert len(merger.prompts) == 1  # exactly one merger invocation for the single group
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.retries == 0
    assert result.merge_metrics.candidates_generated == 3


def test_strict_target_accepts_a_candidate_that_undershoots_the_budget() -> None:
    # Every candidate is far below the budget; undershoot is allowed, so it applies without penalty.
    merger = _TargetMerger(("a\n", "a\n", "a\n"))

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(drift_budget=2.0, max_tokens=3, require_target=True),
    )

    assert result.text == "a\n"
    assert result.reduced_tokens < 3
    assert len(merger.prompts) == 1
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.candidates_generated == 3


def test_target_merge_reports_group_round_and_completion_events() -> None:
    merger = _TargetMerger(("aaaa\n", "aa bb\n", "bbbb\n"))
    reporter = _RecordingReporter()

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(drift_budget=0.01, max_tokens=3, require_target=True),
        reporter=reporter,
    )

    assert result.text == "aa bb\n"
    assert len(reporter.target_groups) == 1
    source_segment, group_tokens, required_savings = reporter.target_groups[0]
    assert source_segment == "aaaa\nbbbb\n"
    assert group_tokens > 0
    assert required_savings > 0

    round_number, base, candidates, selected, selected_from_generation = reporter.target_rounds[0]
    assert round_number == 1
    assert base is None
    assert len(candidates) == 3
    assert selected.text == "aa bb\n"
    assert selected_from_generation

    assert reporter.target_group_done_calls == [(True, result.reduced_tokens)]


def test_strict_target_records_exactly_one_round_per_group() -> None:
    merger = _TargetMerger(("aaaa\n", "aa bb\n", "bbbb\n"))

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(drift_budget=0.01, max_tokens=3, require_target=True),
    )

    assert result.text == "aa bb\n"
    rounds = result.merge_metrics.target_rounds
    assert len(rounds) == 1
    assert rounds[0].round == 1
    assert rounds[0].base_tokens == result.source_tokens  # no pruning: the pre-merge doc is the source
    assert rounds[0].base_drift == pytest.approx(0.0, abs=1e-6)  # the pre-merge doc equals the source
    assert rounds[0].selected_tokens == result.reduced_tokens
    assert rounds[0].generated_best_drift == rounds[0].selected_drift  # the chosen candidate was generated
    assert rounds[0].generated_best_tokens == rounds[0].selected_tokens


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


def test_strict_target_applies_the_best_candidate_even_when_drift_stays_over_budget() -> None:
    # With the refinement round gone, an over-budget-drift candidate is still applied as the best
    # available: the hard ceiling comes first and drift is only minimized best-effort.
    merger = _TargetMerger(("short\n",) * 3)

    result = reduce(
        "source prompt with tokens\nsecond source line\n",
        HashEmbedder(),
        merger,
        params=Params(drift_budget=0.0, max_tokens=3, require_target=True),
    )

    assert result.reduced_tokens <= 3
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.retries == 0
    assert result.merge_metrics.final_drift is not None and result.merge_metrics.final_drift > 0
    assert result.merge_metrics.drift_budget_met is False


def test_strict_target_result_stamps_embedding_and_wall_clock_metrics() -> None:
    merger = _TargetMerger(("short\n",) * 2)

    result = reduce(
        "source prompt with tokens\nsecond source line\n",
        HashEmbedder(),
        merger,
        params=Params(drift_budget=0.0, max_tokens=3, require_target=True),
    )

    assert result.merge_metrics.embed_calls > 0
    assert result.merge_metrics.elapsed_seconds > 0.0


def test_strict_target_can_rewrite_a_single_long_sentence() -> None:
    merger = _TargetMerger(("one two three four\n",) * 2)

    result = reduce(
        "one two three four five six\n",
        HashEmbedder(),
        merger,
        params=Params(drift_budget=2.0, max_tokens=3, require_target=True),
    )

    assert result.reduced_tokens <= 3
    assert result.merge_metrics.calls == 1


def test_strict_target_rejects_an_immutable_structure_that_exceeds_the_budget_before_merging() -> None:
    merger = _TargetMerger(("unused\n",) * 2)

    with pytest.raises(InfeasibleTargetError, match="protected structure"):
        reduce(
            "<context>\n</context>\n",
            HashEmbedder(),
            merger,
            params=Params(drift_budget=2.0, max_tokens=1, require_target=True),
        )

    assert merger.prompts == []


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
