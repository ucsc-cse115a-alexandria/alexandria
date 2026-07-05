from __future__ import annotations

from alexandria.ir.contracts import Params
from alexandria.ops.features.represent import represent
from alexandria.runtime.pipeline import reduce
from alexandria.utils.embedders import HashEmbedder


def test_reduce_removes_redundant_and_preserves_unique_and_cuts_tokens() -> None:
    embedder = HashEmbedder()
    prompt = "Always answer in English.\nAlways answer in English.\nKeep responses concise.\n"

    # deterministic embeddings re-embed to unrelated vectors, so a generous budget is needed to delete.
    result = reduce(prompt, embedder, params=Params(drift_budget=2.0))

    assert result.text.count("Always answer in English.") == 1  # one duplicate dropped
    assert "Keep responses concise." in result.text  # unique instruction preserved

    assert result.reduced_tokens < result.source_tokens  # the first concrete number: tokens were cut
    assert result.source_tokens == represent(prompt, embedder).token_count
