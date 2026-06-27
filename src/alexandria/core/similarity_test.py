from __future__ import annotations

import numpy as np
import pytest

from alexandria.core.similarity import cosine_distance, cosine_similarity_matrix, normalize


def test_identical_vectors_have_similarity_one() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert cosine_similarity_matrix(vectors)[0, 1] == pytest.approx(1.0)


def test_orthogonal_vectors_have_zero_similarity() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert cosine_similarity_matrix(vectors)[0, 1] == pytest.approx(0.0)


def test_cosine_distance_is_zero_for_identical_vectors() -> None:
    vector = np.array([3.0, 4.0], dtype=np.float32)
    assert cosine_distance(vector, vector) == pytest.approx(0.0)


def test_cosine_distance_is_one_for_orthogonal_vectors() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 2.0], dtype=np.float32)
    assert cosine_distance(a, b) == pytest.approx(1.0)


def test_normalize_returns_unit_length_vector() -> None:
    vector = np.array([3.0, 4.0], dtype=np.float32)
    assert float(np.linalg.norm(normalize(vector))) == pytest.approx(1.0)
