"""The redundancy scorer: each sentence's similarity to its most similar peer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alexandria.core.registry import register_scorer
from alexandria.core.similarity import cosine_similarity_matrix

if TYPE_CHECKING:
    from alexandria.core.ir import Document


@register_scorer("redundancy")
def redundancy(document: Document) -> list[float]:
    """Score each sentence by its max cosine similarity to any other sentence."""
    sentences = document.sentences
    if len(sentences) < 2:
        return [0.0 for _ in sentences]
    embeddings = np.stack([s.embedding for s in sentences])
    similarity = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(similarity, -np.inf)
    return [float(row.max()) for row in similarity]
