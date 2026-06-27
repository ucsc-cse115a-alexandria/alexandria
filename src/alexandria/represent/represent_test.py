from __future__ import annotations

import pytest

from alexandria.embedding import HashEmbedder
from alexandria.represent import represent


def test_represent_builds_a_document() -> None:
    document = represent("first\nsecond\n", HashEmbedder())
    assert document.embedding_model == "hash-64"
    assert [s.id for s in document.sentences] == ["s0", "s1"]
    assert document.text == "first\nsecond\n"
    assert document.token_count > 0


def test_represent_rejects_an_empty_prompt() -> None:
    with pytest.raises(ValueError):
        represent("   \n", HashEmbedder())
