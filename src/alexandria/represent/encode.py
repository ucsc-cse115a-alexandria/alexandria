"""Tokenize and embed split segments, then assemble the validated Document tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken

from alexandria.core.ir import Document, Section, Sentence

if TYPE_CHECKING:
    from alexandria.core.protocols import Embedder

_ENCODING = tiktoken.get_encoding("cl100k_base")


def encode(segments: list[str], embedder: Embedder) -> Document:
    """Tokenize and embed each segment, then build the single-section Document IR."""
    vectors = embedder.embed(segments)
    sentences = tuple(
        Sentence(
            id=f"s{i}",
            text=text,
            token_count=len(_ENCODING.encode(text)),
            embedding=vector,
        )
        for i, (text, vector) in enumerate(zip(segments, vectors, strict=True))
    )
    # The single section's text equals the document's, so embed that span once and share the vector.
    whole_text = "".join(s.text for s in sentences)
    whole_vector = embedder.embed([whole_text])[0]
    token_count = sum(s.token_count for s in sentences)
    section = Section(
        header="",
        sentences=sentences,
        text=whole_text,
        token_count=token_count,
        embedding=whole_vector,
    )
    return Document(
        embedding_model=embedder.model_id,
        sections=(section,),
        text=whole_text,
        token_count=token_count,
        embedding=whole_vector,
    )
