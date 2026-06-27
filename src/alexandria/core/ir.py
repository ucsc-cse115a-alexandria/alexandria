"""The intermediate representation: a Document -> Section -> Sentence tree, validated on build."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PlainValidator, model_validator


def _as_vector(value: object) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("embedding must be a 1-D vector")
    return array


Embedding = Annotated[NDArray[np.float32], PlainValidator(_as_vector)]
TokenCount = Annotated[int, Field(ge=0)]
SentenceId = str


class SectionKind(StrEnum):
    """Where a Section came from. The Document is the root, so every Section has one of these."""

    MARKDOWN = "markdown"  # a markdown header
    XML = "xml"  # an XML tag block
    PLAIN = "plain"  # body text under no header or tag (a preamble or a structureless prompt)


class Encoded(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    text: str
    token_count: TokenCount
    embedding: Embedding


class Sentence(Encoded):
    node: Literal["sentence"] = "sentence"
    id: SentenceId


class Section(Encoded):
    node: Literal["section"] = "section"
    kind: SectionKind
    header: str
    children: tuple[Node, ...] = Field(min_length=1)

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        """Every descendant Sentence, in document order."""
        flattened: list[Sentence] = []
        for child in self.children:
            if isinstance(child, Sentence):
                flattened.append(child)
            else:
                flattened.extend(child.sentences)
        return tuple(flattened)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.text != "".join(child.text for child in self.children):
            raise ValueError("Section.text must equal the concatenation of its children")
        if self.token_count != sum(child.token_count for child in self.children):
            raise ValueError("Section.token_count must equal the sum of its children")
        _check_one_dimension([*self.children, self])
        return self


Node = Annotated[Sentence | Section, Field(discriminator="node")]


class Document(Encoded):
    embedding_model: str
    sections: tuple[Section, ...] = Field(min_length=1)

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        return tuple(s for section in self.sections for s in section.sentences)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.text != "".join(section.text for section in self.sections):
            raise ValueError("Document.text must equal the concatenation of its sections")
        if self.token_count != sum(section.token_count for section in self.sections):
            raise ValueError("Document.token_count must equal the sum of its sections")
        ids = [s.id for s in self.sentences]
        if len(ids) != len(set(ids)):
            raise ValueError("every Sentence id must be unique within a Document")
        _check_one_dimension([*self.sections, self])
        return self


Section.model_rebuild()


def _check_one_dimension(nodes: list[Encoded]) -> None:
    dims = {node.embedding.shape[0] for node in nodes}
    if len(dims) > 1:
        raise ValueError(f"all embeddings must share one dimension, got {sorted(dims)}")
