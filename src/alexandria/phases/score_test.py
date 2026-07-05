from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alexandria.ir.registry import register_scorer
from alexandria.phases.represent import represent
from alexandria.phases.score import most_similar, redundancy, score
from alexandria.utils.embedders import HashEmbedder

if TYPE_CHECKING:
    from alexandria.ir.document import Document


def test_duplicate_scores_higher_than_unique() -> None:
    document = represent("repeat me\nrepeat me\nunique line\n", HashEmbedder())
    scores = redundancy(document)
    assert scores[0] > 0.99
    assert scores[2] < 0.5


def test_single_sentence_scores_zero() -> None:
    document = represent("only one\n", HashEmbedder())
    assert redundancy(document) == [0.0]


def test_most_similar_points_to_the_duplicate() -> None:
    document = represent("repeat me\nrepeat me\nunique line\n", HashEmbedder())
    peers = most_similar(document)
    sentences = document.sentences
    peer_id, similarity = peers[0]
    assert peer_id == sentences[1].id
    assert similarity > 0.99


def test_most_similar_single_sentence_has_no_peer() -> None:
    document = represent("only one\n", HashEmbedder())
    assert most_similar(document) == [(None, 0.0)]


def _wrong_length(document: Document) -> list[float]:
    return [0.0] * (len(document.sentences) + 1)


register_scorer("wrong_length_probe")(_wrong_length)


def test_score_rejects_a_wrong_length_vector() -> None:
    document = represent("a\nb\n", HashEmbedder())
    with pytest.raises(ValueError):
        score(document, names=("wrong_length_probe",))
