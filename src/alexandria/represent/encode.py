"""Tokenize and embed a raw section tree, then assemble the validated Document IR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken

from alexandria.core.ir import Document, Node, Section, Sentence
from alexandria.represent.split import RawSection, RawSentence

if TYPE_CHECKING:
    from collections.abc import Iterator

    from alexandria.core.protocols import Embedder

_ENCODING = tiktoken.get_encoding("cl100k_base")


def encode(sections: tuple[RawSection, ...], embedder: Embedder) -> Document:
    leaves = _leaf_sentences(sections)
    vectors = embedder.embed([leaf.text for leaf in leaves])
    built_sentences = iter(
        Sentence(id=f"s{i}", text=leaf.text, token_count=len(_ENCODING.encode(leaf.text)), embedding=vector)
        for i, (leaf, vector) in enumerate(zip(leaves, vectors, strict=True))
    )
    built = tuple(_build_section(section, embedder, built_sentences) for section in sections)
    text = "".join(section.text for section in built)
    return Document(
        embedding_model=embedder.model_id,
        sections=built,
        text=text,
        token_count=sum(section.token_count for section in built),
        embedding=embedder.embed([text])[0],
    )


def _build_section(raw: RawSection, embedder: Embedder, sentences: Iterator[Sentence]) -> Section:
    children: list[Node] = [
        next(sentences) if isinstance(child, RawSentence) else _build_section(child, embedder, sentences)
        for child in raw.children
    ]
    kept = tuple(children)
    text = "".join(child.text for child in kept)
    return Section(
        kind=raw.kind,
        header=raw.header,
        children=kept,
        text=text,
        token_count=sum(child.token_count for child in kept),
        embedding=embedder.embed([text])[0],
    )


def _leaf_sentences(sections: tuple[RawSection, ...]) -> list[RawSentence]:
    """Every leaf sentence in document (pre-order) order — the order _build_section consumes them."""
    leaves: list[RawSentence] = []

    def walk(section: RawSection) -> None:
        for child in section.children:
            if isinstance(child, RawSentence):
                leaves.append(child)
            else:
                walk(child)

    for section in sections:
        walk(section)
    return leaves
