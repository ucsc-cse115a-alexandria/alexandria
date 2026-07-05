from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.core.protocols import Candidate, Delete, Params
from alexandria.phases.optimize import optimize
from alexandria.phases.represent import represent
from alexandria.phases.score import score
from alexandria.phases.select import select
from alexandria.runtime.embedding import HashEmbedder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _CountingEmbedder:
    """Deterministic Embedder: a text embeds to its (a, b, c) letter counts, so each deletion
    drifts the prompt by a predictable, monotonic amount."""

    @property
    def model_id(self) -> str:
        return "counting-3"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.array([t.count("a"), t.count("b"), t.count("c")], dtype=np.float32) for t in texts]


def test_generous_budget_applies_the_delete() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)
    plan = optimize(document, score(document, names=("redundancy",)))

    reduced = select(document, plan, embedder, params=Params(drift_budget=2.0))

    assert reduced.document.text.count("repeat me") == 1
    assert "unique line" in reduced.document.text
    assert len(reduced.applied) == 1


def test_tight_budget_keeps_everything() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)
    plan = optimize(document, score(document, names=("redundancy",)))

    reduced = select(document, plan, embedder, params=Params(drift_budget=0.01))

    assert reduced.document.text == document.text
    assert reduced.applied == ()


def test_empty_plan_returns_document_unchanged() -> None:
    embedder = HashEmbedder()
    document = represent("alpha\nbeta\n", embedder)

    reduced = select(document, (), embedder, params=Params(drift_budget=2.0))

    assert reduced.document.text == document.text


def test_applies_higher_confidence_first_under_a_tight_budget() -> None:
    embedder = _CountingEmbedder()
    document = represent("a\nb\nc\n", embedder)
    ids = [s.id for s in document.sentences]
    drop_a = Candidate(edit=Delete(targets=(ids[0],)), confidence=0.9, source="t", reason="r")
    drop_b = Candidate(edit=Delete(targets=(ids[1],)), confidence=0.5, source="t", reason="r")

    # One deletion drifts ~0.18, two drift ~0.42; the 0.25 budget admits only the higher-confidence drop.
    reduced = select(document, (drop_b, drop_a), embedder, params=Params(drift_budget=0.25))

    assert reduced.document.text == "b\nc\n"
    assert reduced.applied == (drop_a,)


def test_select_rejects_an_embedder_model_mismatch() -> None:
    document = represent("a\nb\n", HashEmbedder())
    mismatched = HashEmbedder(dim=32)  # model_id 'hash-32' != the document's 'hash-64'

    with pytest.raises(ValueError, match="does not match document embedding model"):
        select(document, (), mismatched, params=Params(drift_budget=2.0))
