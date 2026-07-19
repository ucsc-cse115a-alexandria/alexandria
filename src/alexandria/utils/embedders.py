"""Embedder implementations — the only place a model is constructed (the imperative shell)."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import tiktoken

from alexandria.ir.similarity import normalize
from alexandria.utils.config import require_api_key

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder

_DIM = 64
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_INPUT_TOKENS = 8_000
_EMBEDDING_REQUEST_TOKENS = 250_000
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_chunks(text: str, max_tokens: int = _EMBEDDING_INPUT_TOKENS) -> tuple[tuple[str, int], ...]:
    tokens = _ENCODING.encode(text)
    if not tokens:
        return (("", 1),)
    return tuple(
        (_ENCODING.decode(tokens[start : start + max_tokens]), min(max_tokens, len(tokens) - start))
        for start in range(0, len(tokens), max_tokens)
    )


def _request_batches(
    chunks: list[tuple[str, int]], max_tokens: int = _EMBEDDING_REQUEST_TOKENS
) -> tuple[tuple[tuple[str, int], ...], ...]:
    batches: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    for chunk in chunks:
        if current and current_tokens + chunk[1] > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk[1]
    if current:
        batches.append(current)
    return tuple(tuple(batch) for batch in batches)


def _pool(vectors: list[NDArray[np.float32]], weights: list[int]) -> NDArray[np.float32]:
    if len(vectors) == 1:
        return vectors[0]
    stacked = np.stack(vectors)
    pooled = np.average(stacked, axis=0, weights=np.asarray(weights, dtype=np.float32))
    return normalize(np.asarray(pooled, dtype=np.float32))


class HashEmbedder:
    # Not semantic: identical text -> identical vector, so exact duplicates score as redundant.
    def __init__(self, dim: int = _DIM) -> None:
        self._dim = dim

    @property
    def model_id(self) -> str:
        return f"hash-{self._dim}"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> NDArray[np.float32]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return normalize(vector)


class OpenAIEmbedder:
    def __init__(self, api_key: str | None = None) -> None:  # pragma: no cover
        from openai import OpenAI

        self._client = OpenAI(api_key=require_api_key(api_key))

    @property
    def model_id(self) -> str:
        return OPENAI_EMBEDDING_MODEL

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:  # pragma: no cover
        from openai import OpenAIError

        chunks_by_text = [_token_chunks(text) for text in texts]
        flat_chunks = [chunk for chunks in chunks_by_text for chunk in chunks]
        flat_vectors: list[NDArray[np.float32]] = []
        try:
            for batch in _request_batches(flat_chunks):
                response = self._client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=[text for text, _tokens in batch],
                )
                flat_vectors.extend(np.asarray(item.embedding, dtype=np.float32) for item in response.data)
        except OpenAIError as error:
            # cli/ops may not import openai (import-linter), so the boundary error type is ValueError.
            raise ValueError(f"OpenAI embeddings request failed: {error}") from error
        pooled: list[NDArray[np.float32]] = []
        offset = 0
        for chunks in chunks_by_text:
            count = len(chunks)
            pooled.append(_pool(flat_vectors[offset : offset + count], [tokens for _text, tokens in chunks]))
            offset += count
        return pooled


@lru_cache(maxsize=2)
def default_embedder(api_key: str | None = None) -> Embedder:  # pragma: no cover
    """The process-wide default embedder (OpenAI text-embedding-3-small), built lazily on first use."""
    return OpenAIEmbedder(api_key)
