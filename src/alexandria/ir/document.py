"""The intermediate representation: a Document -> Section -> Sentence tree, validated on build.

Document.apply folds one Delete or Replace into a rebuilt tree.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, NewType, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PlainValidator, field_serializer, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from alexandria.ir.contracts import Candidate


def _as_vector(value: object) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("embedding must be a 1-D vector")
    return array


Embedding = Annotated[NDArray[np.float32], PlainValidator(_as_vector)]
TokenCount = Annotated[int, Field(ge=0)]
SentenceId = NewType("SentenceId", str)


class SectionKind(StrEnum):
    MARKDOWN = "markdown"  # a markdown header
    XML = "xml"  # an XML tag block
    PLAIN = "plain"  # body text under no header or tag (a preamble or a structureless prompt)


class Encoded(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    text: str
    token_count: TokenCount
    embedding: Embedding

    @field_serializer("embedding", when_used="json")
    def _serialize_embedding(self, embedding: NDArray[np.float32]) -> list[float]:
        """Emit the vector as a JSON float list; float32 widens to double and narrows back exactly."""
        return embedding.tolist()

    def __eq__(self, other: object) -> bool:
        """Value equality by field; pydantic's default compares embeddings with `==`, which numpy rejects."""
        if not isinstance(other, Encoded) or type(self) is not type(other):
            return False
        if not np.array_equal(self.embedding, other.embedding):
            return False
        return all(
            getattr(self, name) == getattr(other, name) for name in type(self).model_fields if name != "embedding"
        )


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
        return leaves(self.children)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if (self.text, self.token_count) != rollup(self.children):
            raise ValueError("Section.text/token_count must roll up from its children")
        _check_one_dimension([*self.children, self])
        return self


Node = Annotated[Sentence | Section, Field(discriminator="node")]


def rollup(children: Sequence[Encoded]) -> tuple[str, int]:
    """The (text, token_count) a parent must carry: concat of child text, sum of child tokens."""
    return "".join(child.text for child in children), sum(child.token_count for child in children)


def leaves(nodes: Iterable[Node]) -> tuple[Sentence, ...]:
    """Every descendant Sentence under nodes, in document (pre-order) order."""
    out: list[Sentence] = []
    for node in nodes:
        if isinstance(node, Sentence):
            out.append(node)
        else:
            out.extend(leaves(node.children))
    return tuple(out)


class Document(Encoded):
    embedding_model: str
    sections: tuple[Section, ...] = Field(min_length=1)

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        return leaves(self.sections)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if (self.text, self.token_count) != rollup(self.sections):
            raise ValueError("Document.text/token_count must roll up from its sections")
        ids = [s.id for s in self.sentences]
        if len(ids) != len(set(ids)):
            raise ValueError("every Sentence id must be unique within a Document")
        _check_one_dimension([*self.sections, self])
        return self

    def apply(self, candidate: Candidate) -> Document | None:
        """Apply one candidate, returning a rebuilt Document.

        Delete removes whichever targets are still present; Replace applies only when every
        target is present (a partial merge would drop meaning) and is otherwise a no-op.
        Returns the document unchanged when there is nothing to do, and None if the edit
        would empty the Document or a Section.
        """
        surviving = {s.id for s in self.sentences}
        edit = candidate.edit
        if edit.op == "replace":
            if not set(edit.targets).issubset(surviving):
                return self
            remaining = surviving - set(edit.targets[1:])
            if _empties_a_section(self, remaining):
                return None
            return _rebuild(self, remaining, {edit.targets[0]: edit.replacement})
        present = {target for target in edit.targets if target in surviving}
        if not present:
            return self
        remaining = surviving - present
        if not remaining or _empties_a_section(self, remaining):
            return None
        return _rebuild(self, remaining)


Section.model_rebuild()


def _check_one_dimension(nodes: list[Encoded]) -> None:
    dims = {node.embedding.shape[0] for node in nodes}
    if len(dims) > 1:
        raise ValueError(f"all embeddings must share one dimension, got {sorted(dims)}")


def _empties_a_section(document: Document, surviving: set[SentenceId]) -> bool:
    def empty(section: Section) -> bool:
        if all(s.id not in surviving for s in section.sentences):
            return True
        return any(empty(child) for child in section.children if isinstance(child, Section))

    return any(empty(section) for section in document.sections)


def _rebuild(
    document: Document, surviving: set[SentenceId], replacements: dict[SentenceId, Encoded] | None = None
) -> Document:
    sections = tuple(_rebuild_section(section, surviving, replacements or {}) for section in document.sections)
    text, token_count = rollup(sections)
    return Document(
        embedding_model=document.embedding_model,
        sections=sections,
        text=text,
        token_count=token_count,
        embedding=document.embedding,
    )


def _rebuild_section(section: Section, surviving: set[SentenceId], replacements: dict[SentenceId, Encoded]) -> Section:
    kept: list[Node] = []
    for child in section.children:
        if isinstance(child, Sentence):
            if child.id not in surviving:
                continue
            replacement = replacements.get(child.id)
            kept.append(
                child
                if replacement is None
                else Sentence(
                    id=child.id,
                    text=replacement.text,
                    token_count=replacement.token_count,
                    embedding=replacement.embedding,
                )
            )
        else:
            kept.append(_rebuild_section(child, surviving, replacements))
    children = tuple(kept)
    text, token_count = rollup(children)
    return Section(
        kind=section.kind,
        header=section.header,
        children=children,
        text=text,
        token_count=token_count,
        # embedding is the pre-edit vector; re-embedding needs a model, which ir avoids.
        embedding=section.embedding,
    )
