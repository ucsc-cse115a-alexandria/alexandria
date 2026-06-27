from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alexandria.core.registry import get_scorer, register_scorer
from alexandria.core.similarity import cosine_similarity_matrix

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Scores

DEFAULT_SCORER = "redundancy"


def most_similar(document: Document) -> list[tuple[str | None, float]]:
    """Each sentence's most-similar peer id and its cosine similarity (None, 0.0 if no peer)."""
    sentences = document.sentences
    if len(sentences) < 2:
        return [(None, 0.0) for _ in sentences]
    embeddings = np.stack([s.embedding for s in sentences])
    similarity = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(similarity, -np.inf)
    peers: list[tuple[str | None, float]] = []
    for row in similarity:
        peer_index = int(row.argmax())
        peers.append((sentences[peer_index].id, float(row[peer_index])))
    return peers


@register_scorer(DEFAULT_SCORER, peers=most_similar)
def redundancy(document: Document) -> list[float]:
    """Score each sentence by its max cosine similarity to any other sentence."""
    return [similarity for _, similarity in most_similar(document)]


def score(document: Document, names: tuple[str, ...] = (DEFAULT_SCORER,)) -> Scores:
    """Run each named scorer and validate its vector length against the Document."""
    bundle: dict[str, tuple[float, ...]] = {}
    expected = len(document.sentences)
    for name in names:
        values = get_scorer(name)(document)
        if len(values) != expected:
            raise ValueError(f"scorer {name!r} returned {len(values)} scores for {expected} sentences")
        bundle[name] = tuple(values)
    return bundle


__all__ = ["DEFAULT_SCORER", "most_similar", "redundancy", "score"]
