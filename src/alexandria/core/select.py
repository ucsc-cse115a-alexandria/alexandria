"""apply: fold a Plan of Delete edits into a smaller Document. The only place the tree is rewritten."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from alexandria.core.ir import Document, Node, Section, Sentence, rollup
from alexandria.core.protocols import Delete

if TYPE_CHECKING:
    from alexandria.core.protocols import Plan


def apply(document: Document, plan: Plan) -> Document:
    surviving = {s.id for s in document.sentences}
    for candidate in plan:
        match candidate.edit:
            case Delete() as edit:
                present = tuple(t for t in edit.targets if t in surviving)
            case _:  # pragma: no cover - unreachable until a second Edit op exists
                assert_never(candidate.edit)
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
    text, token_count = rollup(sections)
    return Document(
        embedding_model=document.embedding_model,
        sections=sections,
        text=text,
        token_count=token_count,
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
    text, token_count = rollup(children)
    return Section(
        kind=section.kind,
        header=section.header,
        children=children,
        text=text,
        token_count=token_count,
        # embedding is the pre-edit vector; re-embedding needs a model, which core avoids.
        embedding=section.embedding,
    )
