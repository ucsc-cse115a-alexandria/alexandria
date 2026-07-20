from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.ir.contracts import Params
from alexandria.ir.document import Sentence, SentenceId
from alexandria.ops.features.target import (
    InfeasibleTargetError,
    _target_merge_window,  # pyright: ignore[reportPrivateUsage]
)
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import HashEmbedder
from alexandria.utils.tokens import count_tokens

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.ir.contracts import ReportedCandidate


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
    """Identical texts embed identically, but whole-document cos_sim_diff grows with deleted length."""

    @property
    def model_id(self) -> str:
        return "length-2"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([1.0, len(text) / 40.0], dtype=np.float32) for text in texts]


class _RecordingReporter:
    """Records every ReductionReporter event so tests can assert what the target merge emitted."""

    def __init__(self) -> None:
        self.target_groups: list[tuple[str, int, int]] = []
        self.target_rounds: list[tuple[tuple[ReportedCandidate, ...], ReportedCandidate, bool]] = []
        self.target_group_done_calls: list[tuple[bool, int]] = []

    def redundant_pair(self, first: str, second: str, similarity: float) -> None:
        pass

    def pair_merged(self, merged: str | None, decision: str) -> None:
        pass

    def target_group(self, source_segment: str, group_tokens: int, required_savings: int) -> None:
        self.target_groups.append((source_segment, group_tokens, required_savings))

    def target_round(
        self,
        candidates: tuple[ReportedCandidate, ...],
        selected: ReportedCandidate,
        selected_from_generation: bool,
    ) -> None:
        self.target_rounds.append((candidates, selected, selected_from_generation))

    def target_group_done(self, applied: bool, document_tokens: int) -> None:
        self.target_group_done_calls.append((applied, document_tokens))


def test_target_merge_window_scales_with_the_required_saving() -> None:
    prompt = "".join(f"{letter}\n" for letter in "abcdefghij")  # distinct lines: nothing to prune
    merger = _TargetMerger(("a\nb\n",) * 10)

    result = reduce(
        prompt,
        HashEmbedder(),
        merger,
        params=Params(max_tokens=16, require_target=True, cos_sim_diff_budget=2.0),
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
    assert len(result.applied) == 1
    assert result.applied[0].edit.op == "delete"
    assert len(result.applied[0].edit.targets) == 2
    replayed = result.source.apply(result.applied[0])
    assert replayed is not None and replayed.text == result.text
    assert result.merge_metrics.applied_edits == len(result.applied)


def test_prune_cos_sim_diff_backoff_keeps_the_longest_prefix_within_budget() -> None:
    # Full prune (2 deletions) has cos_sim_diff ~0.006; one deletion has ~0.0013, so 0.003 keeps one.
    merger = _TargetMerger(("aaaa\n",) * 10)

    result = reduce(
        "aaaa\n" * 10,
        _LengthEmbedder(),
        merger,
        params=Params(max_tokens=16, require_target=True, cos_sim_diff_budget=0.003),
    )

    assert result.reduced_tokens <= 16
    assert result.merge_metrics.pruned_sentences == 1
    assert result.merge_metrics.pruned_tokens == 2
    assert result.merge_metrics.cos_sim_diff_budget_met is False
    assert [candidate.edit.op for candidate in result.applied] == ["delete", "replace"]
    replayed = result.source
    for candidate in result.applied:
        next_document = replayed.apply(candidate)
        assert next_document is not None
        replayed = next_document
    assert replayed.text == result.text
    assert result.merge_metrics.proposed_edits == len(result.applied)
    assert result.merge_metrics.applied_edits == len(result.applied)


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


def test_strict_target_repairs_an_over_budget_candidate_without_retry() -> None:
    merger = _TargetMerger(("too many tokens remain here\n",) * 2)
    result = reduce(
        "source prompt with tokens\nsecond source line\n",
        HashEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=2.0, max_tokens=3, require_target=True),
    )

    assert result.reduced_tokens <= 3
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.retries == 0
    assert result.merge_metrics.repaired_tokens > 0
    assert result.merge_metrics.embed_calls == 3  # represent's two batches plus one repaired-candidate batch


def test_strict_target_selects_the_lowest_cos_sim_diff_candidate_from_one_call() -> None:
    merger = _TargetMerger(("aaaa\n", "aa bb\n", "bbbb\n"))

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=0.01, max_tokens=3, require_target=True),
    )

    assert result.text == "aa bb\n"
    assert len(merger.prompts) == 1  # exactly one merger invocation for the single group
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.retries == 0
    assert result.merge_metrics.candidates_generated == 3
    assert len(result.applied) == 1
    assert result.applied[0].edit.op == "replace"
    replayed = result.source.apply(result.applied[0])
    assert replayed is not None and replayed.text == result.text


def test_strict_target_merge_preserves_the_last_target_trailing_whitespace() -> None:
    result = reduce(
        "aaaa\t\nbbbb\n\n",
        _CountingEmbedder(),
        _TargetMerger(("aa bb",) * 3),
        params=Params(cos_sim_diff_budget=2.0, max_tokens=3, require_target=True),
    )

    assert result.text == "aa bb\n\n"
    assert result.applied[0].edit.op == "replace"
    assert result.applied[0].edit.replacement.text == "aa bb\n\n"


@pytest.mark.parametrize(
    ("prompt", "max_tokens", "expected"),
    [
        ("# A\naaaa\nbbbb\n# B\ncccc\nab\n", 10, "# A\na\n# B\na\n"),
        ("<X>\naaaa\nbbbb\n</X>\n<Y>\ncccc\nab\n</Y>\n", 15, "<X>\na\n</X>\n<Y>\na\n</Y>\n"),
    ],
)
def test_strict_target_multiple_sections_preserve_boundaries_and_replay_every_replace(
    prompt: str, max_tokens: int, expected: str
) -> None:
    merger = _TargetMerger(("a\n",) * 3)
    result = reduce(
        prompt,
        _CountingEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=2.0, max_tokens=max_tokens, require_target=True),
    )

    assert result.text == expected
    assert result.reduced_tokens == max_tokens
    assert len(result.applied) == 2
    assert merger.max_tokens == [2, 2]
    assert [candidate.edit.op for candidate in result.applied] == ["replace", "replace"]
    source_text = {sentence.id: sentence.text for sentence in result.source.sentences}
    replayed = result.source
    for candidate in result.applied:
        assert candidate.edit.op == "replace"
        suffix = source_text[candidate.edit.targets[-1]][len(source_text[candidate.edit.targets[-1]].rstrip()) :]
        assert candidate.edit.replacement.text.strip()
        assert candidate.edit.replacement.text.endswith(suffix)
        assert candidate.edit.replacement.token_count == 2
        next_document = replayed.apply(candidate)
        assert next_document is not None
        replayed = next_document
    assert replayed.text == result.text
    assert result.merge_metrics.proposed_edits == 2
    assert result.merge_metrics.applied_edits == 2
    assert len(result.merge_metrics.target_rounds) == 2


def test_strict_target_accepts_a_candidate_that_undershoots_the_budget() -> None:
    # Every candidate is far below the budget; undershoot is allowed, so it applies without penalty.
    merger = _TargetMerger(("a\n", "a\n", "a\n"))

    result = reduce(
        "aaaa\nbbbb\n",
        _CountingEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=2.0, max_tokens=3, require_target=True),
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
        params=Params(cos_sim_diff_budget=0.01, max_tokens=3, require_target=True),
        reporter=reporter,
    )

    assert result.text == "aa bb\n"
    assert len(reporter.target_groups) == 1
    source_segment, group_tokens, required_savings = reporter.target_groups[0]
    assert source_segment == "aaaa\nbbbb\n"
    assert group_tokens > 0
    assert required_savings > 0

    candidates, selected, selected_from_generation = reporter.target_rounds[0]
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
        params=Params(cos_sim_diff_budget=0.01, max_tokens=3, require_target=True),
    )

    assert result.text == "aa bb\n"
    rounds = result.merge_metrics.target_rounds
    assert len(rounds) == 1
    assert rounds[0].base_tokens == result.source_tokens  # no pruning: the pre-merge doc is the source
    assert rounds[0].base_cos_sim_diff == pytest.approx(0.0, abs=1e-6)  # the pre-merge doc equals the source
    assert rounds[0].selected_tokens == result.reduced_tokens


def test_strict_target_preserves_markup_structure() -> None:
    merger = _TargetMerger(("compressed\n",) * 10)
    result = reduce(
        "<context>\nlong source text here\nsecond source line\n</context>\n",
        HashEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=2.0, max_tokens=8, require_target=True),
    )

    assert result.text.startswith("<context>")
    assert result.text.rstrip().endswith("</context>")
    assert result.merge_metrics.calls == 1


def test_strict_target_applies_the_best_candidate_even_when_cos_sim_diff_stays_over_budget() -> None:
    # With the refinement round gone, a candidate over the cos_sim_diff budget is still applied as the best
    # available: the hard ceiling comes first and cos_sim_diff is only minimized best-effort.
    merger = _TargetMerger(("short\n",) * 3)

    result = reduce(
        "source prompt with tokens\nsecond source line\n",
        HashEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=0.0, max_tokens=3, require_target=True),
    )

    assert result.reduced_tokens <= 3
    assert result.merge_metrics.calls == 1
    assert result.merge_metrics.retries == 0
    assert result.merge_metrics.final_cos_sim_diff is not None and result.merge_metrics.final_cos_sim_diff > 0
    assert result.merge_metrics.cos_sim_diff_budget_met is False


def test_strict_target_result_stamps_embedding_and_wall_clock_metrics() -> None:
    merger = _TargetMerger(("short\n",) * 2)

    result = reduce(
        "source prompt with tokens\nsecond source line\n",
        HashEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=0.0, max_tokens=3, require_target=True),
    )

    assert result.merge_metrics.embed_calls > 0
    assert result.merge_metrics.elapsed_seconds > 0.0


def test_strict_target_can_rewrite_a_single_long_sentence() -> None:
    merger = _TargetMerger(("one two three four\n",) * 2)

    result = reduce(
        "one two three four five six\n",
        HashEmbedder(),
        merger,
        params=Params(cos_sim_diff_budget=2.0, max_tokens=3, require_target=True),
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
            params=Params(cos_sim_diff_budget=2.0, max_tokens=1, require_target=True),
        )

    assert merger.prompts == []
