"""try_apply: fold one Delete candidate into a smaller Document. The only place the tree is rewritten."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.ir.document import Document, Node, Section, Sentence, rollup

if TYPE_CHECKING:
    from alexandria.ir.contracts import Candidate
    from alexandria.ir.document import SentenceId


def try_apply(document: Document, candidate: Candidate) -> Document | None:
    """Apply one candidate, returning None if it would empty the Document or a Section.

    Returns the document unchanged when the candidate's targets are already gone.
    """
    surviving = {s.id for s in document.sentences}
    present = {target for target in candidate.edit.targets if target in surviving}
    if not present:
        return document
    remaining = surviving - present
    if not remaining or _empties_a_section(document, remaining):
        return None
    return _rebuild(document, remaining)


def _empties_a_section(document: Document, surviving: set[SentenceId]) -> bool:
    def empty(section: Section) -> bool:
        if all(s.id not in surviving for s in section.sentences):
            return True
        return any(empty(child) for child in section.children if isinstance(child, Section))

    return any(empty(section) for section in document.sections)


def _rebuild(document: Document, surviving: set[SentenceId]) -> Document:
    sections = tuple(_rebuild_section(section, surviving) for section in document.sections)
    text, token_count = rollup(sections)
    return Document(
        embedding_model=document.embedding_model,
        sections=sections,
        text=text,
        token_count=token_count,
        embedding=document.embedding,
    )


def _rebuild_section(section: Section, surviving: set[SentenceId]) -> Section:
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
        # embedding is the pre-edit vector; re-embedding needs a model, which ir avoids.
        embedding=section.embedding,
    )
