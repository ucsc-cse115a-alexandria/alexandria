"""apply: fold a Plan of Delete edits into a smaller Document. The only place the tree is rewritten."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.ir import Document, Section

if TYPE_CHECKING:
    from alexandria.core.protocols import Plan


def apply(document: Document, plan: Plan) -> Document:
    surviving = {s.id for s in document.sentences}
    for candidate in plan:
        present = tuple(t for t in candidate.edit.targets if t in surviving)
        if not present:
            continue
        remaining = surviving - set(present)
        if not remaining:
            raise ValueError("edit would empty the Document")
        if _empties_a_section(document, remaining):
            raise ValueError("edit would empty a Section")
        surviving = remaining
    return _rebuild(document, surviving)


def _empties_a_section(document: Document, surviving: set[str]) -> bool:
    return any(all(s.id not in surviving for s in section.sentences) for section in document.sections)


def _rebuild(document: Document, surviving: set[str]) -> Document:
    sections: list[Section] = []
    for section in document.sections:
        kept = tuple(s for s in section.sentences if s.id in surviving)
        if not kept:
            continue
        sections.append(
            Section(
                header=section.header,
                sentences=kept,
                text="".join(s.text for s in kept),
                token_count=sum(s.token_count for s in kept),
                # embedding is the pre-edit vector; re-embedding needs a model, which core avoids.
                embedding=section.embedding,
            )
        )
    return Document(
        embedding_model=document.embedding_model,
        sections=tuple(sections),
        text="".join(section.text for section in sections),
        token_count=sum(section.token_count for section in sections),
        embedding=document.embedding,
    )
