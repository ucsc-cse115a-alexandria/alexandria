from __future__ import annotations

from alexandria.embedding import HashEmbedder
from alexandria.pipeline import score_report


def test_score_report_adds_the_redundant_peer() -> None:
    rows = score_report("repeat me\nrepeat me\nunique line\n", HashEmbedder())
    assert rows[0]["most_similar_id"] == rows[1]["id"]
    assert rows[0]["most_similar_text"] == "repeat me"
    redundancy = rows[0]["redundancy"]
    assert isinstance(redundancy, float)
    assert redundancy > 0.99


def test_score_report_single_sentence_has_no_peer() -> None:
    rows = score_report("only one\n", HashEmbedder())
    assert rows[0]["most_similar_id"] is None
    assert rows[0]["most_similar_text"] is None
