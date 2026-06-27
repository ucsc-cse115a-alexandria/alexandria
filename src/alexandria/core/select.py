"""apply: fold a Plan of Delete edits into a smaller Document. The only place the tree is rewritten."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.ir import Document, Node, Section, Sentence

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
    def empty(section: Section) -> bool:
        if all(s.id not in surviving for s in section.sentences):
            return True
        return any(empty(child) for child in section.children if isinstance(child, Section))

    return any(empty(section) for section in document.sections)


def _rebuild(document: Document, surviving: set[str]) -> Document:
    sections = tuple(_rebuild_section(section, surviving) for section in document.sections)
    return Document(
        embedding_model=document.embedding_model,
        sections=sections,
        text="".join(section.text for section in sections),
        token_count=sum(section.token_count for section in sections),
        embedding=document.embedding,
    )


def _rebuild_section(section: Section, surviving: set[str]) -> Section:
    kept: list[Node] = []
    for child in section.children:
        if isinstance(child, Sentence):
            if child.id in surviving:
                kept.append(child)
        else:
            kept.append(_rebuild_section(child, surviving))
    children = tuple(kept)
    return Section(
        kind=section.kind,
        header=section.header,
        children=children,
        text="".join(child.text for child in children),
        token_count=sum(child.token_count for child in children),
        # embedding is the pre-edit vector; re-embedding needs a model, which core avoids.
        embedding=section.embedding,
    )
