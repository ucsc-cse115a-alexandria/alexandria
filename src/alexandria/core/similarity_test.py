from __future__ import annotations

import numpy as np
import pytest

from alexandria.core.ir import Document, Section, SectionKind, Sentence, SentenceId
from alexandria.core.similarity import (
    cosine_distance,
    cosine_similarity_matrix,
    normalize,
    similarity_matrix_for,
)


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


def _two_sentence_document() -> Document:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    first = Sentence(id=SentenceId("s1"), text="a\n", token_count=1, embedding=a)
    second = Sentence(id=SentenceId("s2"), text="b\n", token_count=1, embedding=b)
    section = Section(
        kind=SectionKind.PLAIN, header="", children=(first, second), text="a\nb\n", token_count=2, embedding=a
    )
    return Document(embedding_model="test", sections=(section,), text="a\nb\n", token_count=2, embedding=a)


def test_similarity_matrix_is_memoized_per_document() -> None:
    document = _two_sentence_document()

    first = similarity_matrix_for(document)
    second = similarity_matrix_for(document)

    # A fresh computation would build a new array each call; one shared object proves memoization.
    assert second is first
    assert not first.flags.writeable  # read-only, so callers cannot corrupt the shared matrix
