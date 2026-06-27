from __future__ import annotations

from alexandria.phases.represent import represent
from alexandria.phases.score.redundancy import most_similar, redundancy
from alexandria.runtime.embedding import HashEmbedder


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
