"""The redundancy scorer: each sentence's similarity to its most similar peer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alexandria.core.registry import register_scorer
from alexandria.core.similarity import cosine_similarity_matrix

if TYPE_CHECKING:
    from alexandria.core.ir import Document

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
