from __future__ import annotations

import pytest

from alexandria.ops.features.compare import compare
from alexandria.utils.embedders import HashEmbedder


def test_identical_texts_are_a_perfect_match_with_no_reduction() -> None:
    embedder = HashEmbedder()
    result = compare("Do the thing.\nThen stop.\n", "Do the thing.\nThen stop.\n", embedder)
    assert result.similarity == pytest.approx(1.0)
    assert result.token_reduction == 0.0
    assert result.drift == pytest.approx(0.0)


def test_shortening_a_prompt_gives_a_positive_token_reduction() -> None:
    embedder = HashEmbedder()
    result = compare("Do the thing.\nThen stop.\nAnd rest.\n", "Do the thing.\n", embedder)
    assert result.token_reduction > 0.0
    assert result.edited_tokens < result.original_tokens


def test_inflating_a_prompt_gives_a_negative_token_reduction() -> None:
    embedder = HashEmbedder()
    result = compare("Do the thing.\n", "Do the thing.\nDo the thing, again, restated at length.\n", embedder)
    assert result.token_reduction < 0.0
    assert result.edited_tokens > result.original_tokens


def test_empty_text_is_rejected() -> None:
    embedder = HashEmbedder()
    with pytest.raises(ValueError):
        compare("", "Do the thing.\n", embedder)
