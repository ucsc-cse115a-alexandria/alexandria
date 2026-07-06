from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tiktoken
from pydantic import BaseModel, ConfigDict, computed_field

from alexandria.ir.contracts import Drift
from alexandria.ir.similarity import cosine_distance, normalize
from alexandria.utils.embedders import default_embedder

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder

_ENCODING = tiktoken.get_encoding("cl100k_base")  # duplicates represent._ENCODING; features may not import each other
_CHUNK_TOKENS = 200  # keep chunks under the 256-token window all-MiniLM-L6-v2 truncates at


class CompareResult(BaseModel):
    """Fidelity of an edit: how close ``edited`` stays to ``original`` and how much shorter it got."""

    model_config = ConfigDict(frozen=True)
    similarity: float  # cosine similarity of the two chunk-pooled embeddings
    original_tokens: int  # cl100k_base
    edited_tokens: int
    token_reduction: float  # 1 - edited_tokens / original_tokens (negative when edited is longer)

    @computed_field
    @property
    def drift(self) -> Drift:
        # pydantic does not validate a computed_field's return, so this clamp is the real ge=0 guard.
        return max(0.0, 1.0 - self.similarity)


def compare(original: str, edited: str, embedder: Embedder | None = None) -> CompareResult:
    """Compare two prompts: chunk-pooled cosine similarity and cl100k_base token reduction.

    The caller applies the 99% gate by comparing ``similarity`` against ``1 - Params.drift_budget``.
    When ``embedder`` is omitted, the cached default all-MiniLM-L6-v2 model is used.
    """
    model = embedder if embedder is not None else default_embedder()
    original_ids = _ENCODING.encode(original)
    edited_ids = _ENCODING.encode(edited)
    if not original_ids or not edited_ids:
        raise ValueError("cannot compare an empty text")
    similarity = 1.0 - cosine_distance(_pooled_embedding(original_ids, model), _pooled_embedding(edited_ids, model))
    return CompareResult(
        similarity=similarity,
        original_tokens=len(original_ids),
        edited_tokens=len(edited_ids),
        token_reduction=1.0 - len(edited_ids) / len(original_ids),
    )


def _pooled_embedding(token_ids: list[int], embedder: Embedder) -> NDArray[np.float32]:
    """The length-weighted mean of the normalized ~200-token chunk vectors.

    all-MiniLM-L6-v2 truncates at 256 tokens, so embedding a long text as one vector would blind the
    gate; chunking keeps every token in view. Any Embedder is chunked — the protocol exposes no window.
    """
    chunks = [token_ids[start : start + _CHUNK_TOKENS] for start in range(0, len(token_ids), _CHUNK_TOKENS)]
    weights = np.array([len(chunk) for chunk in chunks], dtype=np.float32)
    vectors = normalize(np.stack(embedder.embed([_ENCODING.decode(chunk) for chunk in chunks])))
    return ((vectors * weights[:, None]).sum(axis=0) / weights.sum()).astype(np.float32)


__all__ = ["CompareResult", "compare"]
