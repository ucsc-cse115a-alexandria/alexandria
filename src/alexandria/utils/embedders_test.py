from __future__ import annotations

# pyright: reportPrivateUsage=false
import numpy as np

from alexandria.utils.embedders import (
    HashEmbedder,
    _pool,
    _request_batches,
    _token_chunks,
)


def test_identical_text_yields_identical_vector() -> None:
    embedder = HashEmbedder()
    assert np.array_equal(embedder.embed(["hello"])[0], embedder.embed(["hello"])[0])


def test_different_text_yields_different_vector() -> None:
    embedder = HashEmbedder()
    assert not np.array_equal(embedder.embed(["hello"])[0], embedder.embed(["world"])[0])


def test_model_id_reports_the_hash_dimension() -> None:
    assert HashEmbedder().model_id == "hash-64"


def test_token_chunks_round_trip_long_text() -> None:
    text = "long prompt sentence. " * 1_000
    chunks = _token_chunks(text, max_tokens=100)
    assert len(chunks) > 1
    assert "".join(chunk for chunk, _tokens in chunks) == text
    assert all(tokens <= 100 for _chunk, tokens in chunks)


def test_request_batches_preserve_order_and_budget() -> None:
    chunks = [("a", 3), ("b", 4), ("c", 5)]
    batches = _request_batches(chunks, max_tokens=7)
    assert batches == ((chunks[0], chunks[1]), (chunks[2],))


def test_pool_preserves_one_vector_and_normalizes_multiple() -> None:
    first = np.asarray([1.0, 0.0], dtype=np.float32)
    second = np.asarray([0.0, 1.0], dtype=np.float32)
    assert _pool([first], [1]) is first
    pooled = _pool([first, second], [3, 1])
    assert np.isclose(np.linalg.norm(pooled), 1.0)
    assert pooled[0] > pooled[1]
