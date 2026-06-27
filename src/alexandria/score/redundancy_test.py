from __future__ import annotations

from alexandria.embedding import HashEmbedder
from alexandria.represent import represent
from alexandria.score.redundancy import redundancy


def test_duplicate_scores_higher_than_unique() -> None:
    document = represent("repeat me\nrepeat me\nunique line\n", HashEmbedder())
    scores = redundancy(document)
    assert scores[0] > 0.99
    assert scores[2] < 0.5


def test_single_sentence_scores_zero() -> None:
    document = represent("only one\n", HashEmbedder())
    assert redundancy(document) == [0.0]
