from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.ir.contracts import Candidate, Delete, Params
from alexandria.ops.features.optimize import optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import score
from alexandria.ops.features.select import select
from alexandria.utils.embedders import HashEmbedder

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GENEROUS = Params(drift_budget=2.0)


class _CountingEmbedder:
    """Deterministic Embedder: a text embeds to its (a, b, c) letter counts, so each deletion
    drifts the prompt by a predictable, monotonic amount."""

    @property
    def model_id(self) -> str:
        return "counting-3"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([t.count("a"), t.count("b"), t.count("c")], dtype=np.float32) for t in texts]


class _FirstWinsMerger:
    """Offline merger that returns the first sentence, so every merge lands as a Delete of the second."""

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback  # required by SentenceMerger; this merger keeps the first instruction only
        return first.strip()


def test_generous_budget_applies_the_delete() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)
    plan = optimize(document, score(document, names=("redundancy",)), embedder, _FirstWinsMerger(), params=_GENEROUS)

    reduced = select(document, plan, embedder, params=_GENEROUS)

    assert reduced.document.text.count("repeat me") == 1
    assert "unique line" in reduced.document.text
    assert len(reduced.applied) == 1


def test_tight_budget_keeps_everything() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)
    plan = optimize(document, score(document, names=("redundancy",)), embedder, _FirstWinsMerger(), params=_GENEROUS)

    reduced = select(document, plan, embedder, params=Params(drift_budget=0.01))

    assert reduced.document.text == document.text
    assert reduced.applied == ()


def test_empty_plan_returns_document_unchanged() -> None:
    embedder = HashEmbedder()
    document = represent("alpha\nbeta\n", embedder)

    reduced = select(document, (), embedder, params=_GENEROUS)

    assert reduced.document.text == document.text


def test_applies_least_drift_first_regardless_of_confidence() -> None:
    embedder = _CountingEmbedder()
    document = represent("a\naa\nb\n", embedder)
    ids = [s.id for s in document.sentences]
    # Base count-vector is (3, 1, 0). Deleting "a\n" leaves (2, 1, 0) — drift ~0.010;
    # deleting "b\n" leaves (3, 0, 0) — drift ~0.051. The bigger-drift candidate gets the
    # higher confidence to prove ordering is by drift, not confidence.
    drop_a = Candidate(edit=Delete(targets=(ids[0],)), confidence=0.5, source="t", reason="r")
    drop_b = Candidate(edit=Delete(targets=(ids[2],)), confidence=0.9, source="t", reason="r")

    reduced = select(document, (drop_b, drop_a), embedder, params=Params(drift_budget=0.06))

    assert reduced.applied[0] == drop_a


def test_stops_once_the_token_target_is_met() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nrepeat me\nunique line\n", embedder)
    ids = [s.id for s in document.sentences]
    first = Candidate(edit=Delete(targets=(ids[1],)), confidence=0.9, source="t", reason="r")
    second = Candidate(edit=Delete(targets=(ids[2],)), confidence=0.9, source="t", reason="r")
    target = document.token_count - document.sentences[1].token_count  # one deletion is enough

    reduced = select(document, (first, second), embedder, params=Params(drift_budget=2.0, max_tokens=target))

    assert len(reduced.applied) == 1
    assert reduced.document.token_count <= target


def test_select_rejects_an_embedder_model_mismatch() -> None:
    document = represent("a\nb\n", HashEmbedder())
    mismatched = HashEmbedder(dim=32)  # model_id 'hash-32' != the document's 'hash-64'

    with pytest.raises(ValueError, match="does not match document embedding model"):
        select(document, (), mismatched, params=_GENEROUS)
