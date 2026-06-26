from __future__ import annotations

import numpy as np
import pytest

from alexandria.core.similarity import cosine_similarity_matrix


def test_identical_vectors_have_similarity_one() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert cosine_similarity_matrix(vectors)[0, 1] == pytest.approx(1.0)


def test_orthogonal_vectors_have_zero_similarity() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert cosine_similarity_matrix(vectors)[0, 1] == pytest.approx(0.0)
