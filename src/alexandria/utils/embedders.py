"""Embedder implementations — the only place a model is constructed (the imperative shell)."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from alexandria.ir.similarity import normalize
from alexandria.utils.config import require_api_key

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder

_DIM = 64
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


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

        try:
            response = self._client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
        except OpenAIError as error:
            # cli/ops may not import openai (import-linter), so the boundary error type is ValueError.
            raise ValueError(f"OpenAI embeddings request failed: {error}") from error
        return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]


@lru_cache(maxsize=2)
def default_embedder(api_key: str | None = None) -> Embedder:  # pragma: no cover
    """The process-wide default embedder (OpenAI text-embedding-3-small), built lazily on first use."""
    return OpenAIEmbedder(api_key)
