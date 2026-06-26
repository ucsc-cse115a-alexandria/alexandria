"""Cosine similarity over stacked embeddings — shared by score and optimize."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cosine_similarity_matrix(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the (n, n) cosine-similarity matrix of n row vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-12, None)
    return (normalized @ normalized.T).astype(np.float32)
