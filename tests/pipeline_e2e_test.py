from __future__ import annotations

from alexandria.embedding import HashEmbedder
from alexandria.pipeline import reduce
from alexandria.represent import represent


def test_reduce_removes_redundant_and_preserves_unique_and_cuts_tokens() -> None:
    embedder = HashEmbedder()
    prompt = "Always answer in English.\nAlways answer in English.\nKeep responses concise.\n"

    reduced = reduce(prompt, embedder)

    assert reduced.count("Always answer in English.") == 1  # one duplicate dropped
    assert "Keep responses concise." in reduced  # unique instruction preserved

    original_tokens = represent(prompt, embedder).token_count
    reduced_tokens = represent(reduced, embedder).token_count
    assert reduced_tokens < original_tokens  # the first concrete number: tokens were cut
