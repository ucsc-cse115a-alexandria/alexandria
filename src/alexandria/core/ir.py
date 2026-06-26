"""The intermediate representation: a Document -> Section -> Sentence tree, validated on build."""

from __future__ import annotations

from typing import Annotated, Self

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


class Encoded(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    text: str
    token_count: TokenCount
    embedding: Embedding


class Sentence(Encoded):
    id: SentenceId


class Section(Encoded):
    header: str
    sentences: tuple[Sentence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.text != "".join(s.text for s in self.sentences):
            raise ValueError("Section.text must equal the concatenation of its sentences")
        if self.token_count != sum(s.token_count for s in self.sentences):
            raise ValueError("Section.token_count must equal the sum of its sentences")
        _check_one_dimension([*self.sentences, self])
        return self


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


def _check_one_dimension(nodes: list[Encoded]) -> None:
    dims = {node.embedding.shape[0] for node in nodes}
    if len(dims) > 1:
        raise ValueError(f"all embeddings must share one dimension, got {sorted(dims)}")
