from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.ir.contracts import Params
from alexandria.ops.features.optimize import MAX_MERGE_ATTEMPTS, optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import score
from alexandria.utils.embedders import HashEmbedder
from alexandria.utils.tokens import count_tokens

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.ir.contracts import Scores
    from alexandria.ir.document import Document

_PROMPT = "repeat me and me\nrepeat me and me\nunique line\n"
# HashEmbedder re-embeds edited text to an unrelated vector, so the optimizer's drift gate needs
# a generous budget everywhere the merge should be accepted.
_GENEROUS = Params(drift_budget=2.0)


class _CannedMerger:
    """Offline merger returning a fixed rewrite, so tests control every branch."""

    def __init__(self, merged: str) -> None:
        self._merged = merged
        self.calls = 0

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del first, second, feedback  # required by SentenceMerger; a canned merger ignores them
        self.calls += 1
        return self._merged


class _ScriptedMerger:
    """Returns queued rewrites in order and records the feedback each attempt received."""

    def __init__(self, *rewrites: str) -> None:
        self._rewrites = list(rewrites)
        self.feedback: list[str | None] = []

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del first, second  # required by SentenceMerger; this scripted merger keys off feedback only
        self.feedback.append(feedback)
        return self._rewrites.pop(0)


class _CountingEmbedder:
    """Deterministic Embedder: text embeds to its (a, b, c) letter counts, so every drift is predictable."""

    @property
    def model_id(self) -> str:
        return "counting-3"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([t.count("a"), t.count("b"), t.count("c")], dtype=np.float32) for t in texts]


def _scored_document(embedder: HashEmbedder) -> tuple[Document, Scores]:
    document = represent(_PROMPT, embedder)
    return document, score(document, names=("redundancy",))


def test_a_shorter_merge_becomes_a_replace_kept_at_the_first_occurrence() -> None:
    embedder = _CountingEmbedder()
    document = represent("aa bb\naaa bb\ncc\n", embedder)
    scores = score(document, names=("redundancy",))

    plan = optimize(document, scores, embedder, _CannedMerger("a"), params=_GENEROUS)

    (candidate,) = plan
    first, second = document.sentences[0], document.sentences[1]
    assert candidate.edit.op == "replace"
    assert candidate.edit.targets == (first.id, second.id)
    assert candidate.edit.replacement.text == "a\n"  # first target's trailing newline restored
    assert candidate.edit.replacement.token_count == count_tokens("a\n")


def test_a_merge_equal_to_the_first_sentence_becomes_a_delete_of_the_second() -> None:
    embedder = HashEmbedder()
    document, scores = _scored_document(embedder)
    first = document.sentences[0]

    # HashEmbedder maps identical text to an identical vector, so similarity == 1.0 >= 0.99.
    merger = _CannedMerger(first.text.strip())
    plan = optimize(document, scores, embedder, merger, params=_GENEROUS)

    (candidate,) = plan
    assert candidate.edit.op == "delete"
    assert candidate.edit.targets == (document.sentences[1].id,)
    assert merger.calls == 0


def test_a_triple_duplicate_collapses_to_two_deletes() -> None:
    embedder = HashEmbedder()
    document = represent("dup line\ndup line\ndup line\n", embedder)
    scores = score(document, names=("redundancy",))
    first = document.sentences[0]

    # merged == first, so each pair becomes a Delete and the unchanged first stays pairable
    merger = _CannedMerger(first.text.strip())
    plan = optimize(document, scores, embedder, merger, params=_GENEROUS)

    assert [c.edit.op for c in plan] == ["delete", "delete"]
    assert {c.edit.targets[0] for c in plan} == {document.sentences[1].id, document.sentences[2].id}
    assert merger.calls == 0


def test_markup_boundaries_are_never_sent_to_the_merger() -> None:
    embedder = HashEmbedder()
    prompt = "<example>\nfirst\n</example>\n<example>\nsecond\n</example>\n"
    document = represent(prompt, embedder)
    merger = _CannedMerger("unused")

    plan = optimize(document, score(document, names=("redundancy",)), embedder, merger, params=_GENEROUS)

    assert plan == ()
    assert merger.calls == 0


def test_markup_boundaries_survive_when_duplicate_content_is_removed() -> None:
    embedder = HashEmbedder()
    prompt = "<example>\nsame\n</example>\n<example>\nsame\n</example>\n"
    document = represent(prompt, embedder)

    plan = optimize(
        document, score(document, names=("redundancy",)), embedder, _CannedMerger("unused"), params=_GENEROUS
    )
    reduced = document.apply(plan[0])

    assert reduced is not None
    assert reduced.text.count("<example>") == 2
    assert reduced.text.count("</example>") == 2


def test_token_target_stops_merge_generation_once_the_plan_can_reach_it() -> None:
    embedder = _CountingEmbedder()
    document = represent("aa bb\naaa bb\naaaa bb\naaaaa bb\ncc\n", embedder)
    merger = _CannedMerger("aa bb")
    one_deletion_target = document.token_count - 1

    plan = optimize(
        document,
        score(document, names=("redundancy",)),
        embedder,
        merger,
        params=Params(drift_budget=2.0, max_tokens=one_deletion_target),
    )

    assert len(plan) == 1
    assert merger.calls == 1

    unconstrained_merger = _CannedMerger("aa bb")
    optimize(
        document,
        score(document, names=("redundancy",)),
        embedder,
        unconstrained_merger,
        params=_GENEROUS,
    )
    assert unconstrained_merger.calls > merger.calls


def test_strict_unreachable_target_fails_before_calling_the_merger() -> None:
    embedder = HashEmbedder()
    document = represent("dup line\ndup line\nunique line\n", embedder)
    merger = _CannedMerger("dup line")

    with pytest.raises(ValueError, match=r"unreachable.*no merge model calls"):
        optimize(
            document,
            score(document, names=("redundancy",)),
            embedder,
            merger,
            params=Params(drift_budget=2.0, max_tokens=1, require_target=True),
        )

    assert merger.calls == 0


def test_an_over_budget_rewrite_is_retried_with_drift_feedback() -> None:
    embedder = _CountingEmbedder()
    # The near-duplicate pair is not text-identical, so it exercises the merger retry path.
    # Attempt 1 ("zz") leaves only the c-vector and exceeds the drift budget. Attempt 2
    # equals the first sentence, so it lands as a Delete and fits the budget.
    document = represent("aa bb\naaa bb\ncc\n", embedder)
    scores = score(document, names=("redundancy",))
    merger = _ScriptedMerger("zz", "aa bb")

    plan = optimize(document, scores, embedder, merger, params=Params(drift_budget=0.5))

    (candidate,) = plan
    assert candidate.edit.op == "delete"
    assert merger.feedback[0] is None
    assert merger.feedback[1] is not None and "zz" in merger.feedback[1]  # the rejected rewrite fed back


def test_a_pair_still_over_budget_after_all_attempts_is_skipped() -> None:
    embedder = _CountingEmbedder()
    document = represent("aa bb\naaa bb\ncc\n", embedder)
    scores = score(document, names=("redundancy",))
    merger = _ScriptedMerger("zz", "zz", "zz")

    plan = optimize(document, scores, embedder, merger, params=Params(drift_budget=0.5))

    assert plan == ()
    assert len(merger.feedback) == MAX_MERGE_ATTEMPTS


def test_a_merge_that_saves_no_tokens_is_retried_then_dropped() -> None:
    embedder = _CountingEmbedder()
    document = represent("aa bb\naaa bb\ncc\n", embedder)
    scores = score(document, names=("redundancy",))
    bloated = "this rewrite is far longer than the two originals put together, saving nothing at all"
    merger = _ScriptedMerger(bloated, bloated, bloated)

    plan = optimize(document, scores, embedder, merger, params=_GENEROUS)

    assert plan == ()
    assert len(merger.feedback) == MAX_MERGE_ATTEMPTS
    assert merger.feedback[1] is not None and "shorter" in merger.feedback[1]


def test_optimize_rejects_an_embedder_model_mismatch() -> None:
    document, scores = _scored_document(HashEmbedder())

    with pytest.raises(ValueError, match="does not match document embedding model"):
        optimize(document, scores, HashEmbedder(dim=32), _CannedMerger("x"))


def test_optimize_rejects_missing_required_scores() -> None:
    embedder = HashEmbedder()
    document = represent(_PROMPT, embedder)

    with pytest.raises(ValueError, match="requires scorer"):
        optimize(document, {}, embedder, _CannedMerger("x"))
