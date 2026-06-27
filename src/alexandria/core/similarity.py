from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalize along the last axis; works for a single vector or a stack."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    normalized = normalize(embeddings)
    return (normalized @ normalized.T).astype(np.float32)


def cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return 1.0 - float(normalize(a) @ normalize(b))
