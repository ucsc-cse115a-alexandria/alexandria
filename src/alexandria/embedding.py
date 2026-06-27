"""Embedder implementations — the only place a model is constructed (the imperative shell)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from alexandria.core.protocols import Embedder

_DIM = 64


class _EncodesText(Protocol):
    def encode(
        self, inputs: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool
    ) -> np.ndarray[Any, np.dtype[Any]]: ...


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
        return vector / float(np.linalg.norm(vector))


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:  # pragma: no cover
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = cast("_EncodesText", SentenceTransformer(model_name))

    @property
    def model_id(self) -> str:  # pragma: no cover
        return self._model_name

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:  # pragma: no cover
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [np.asarray(row, dtype=np.float32) for row in vectors]


def build_embedder(model: str) -> Embedder:
    if model == "deterministic":
        return HashEmbedder()
    return SentenceTransformerEmbedder(model)  # pragma: no cover
