"""Phase 1 — Represent: prompt -> Document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.phases.represent.encode import encode
from alexandria.phases.represent.split import split

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder


def represent(prompt: str, embedder: Embedder) -> Document:
    """Build the Document IR: split losslessly, then tokenize and embed every node."""
    segments = split(prompt)
    if not segments:
        raise ValueError("cannot represent an empty prompt")
    return encode(segments, embedder)


__all__ = ["encode", "represent", "split"]
