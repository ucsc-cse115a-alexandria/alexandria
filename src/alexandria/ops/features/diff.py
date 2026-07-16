from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.ir.contracts import Diff, DiffSpan
from alexandria.ir.document import Section, Sentence

if TYPE_CHECKING:
    from collections.abc import Iterator

    from alexandria.ir.contracts import Plan
    from alexandria.ir.document import Document


def diffs(document: Document, plan: Plan) -> tuple[Diff, ...]:
    """Resolve a Plan into displayable Diffs, one per candidate, highest confidence first.

    The order matches what the auto selector applies and what an interactive review presents;
    the sort is stable, so equal-confidence candidates keep their plan order.

    Raises ValueError when a candidate targets a sentence id absent from the document, so a
    mispiped `optimize | diffs` fails at the boundary instead of producing a hollow diff.
    """
    spans = {span.sentence_id: span for span in _spans(document)}
    missing = sorted({target for candidate in plan for target in candidate.edit.targets} - spans.keys())
    if missing:
        raise ValueError(f"plan targets sentence id(s) {missing} not in the document")
    return tuple(
        Diff(
            candidate=candidate,
            spans=tuple(spans[target] for target in candidate.edit.targets),
            replacement="",
        )
        for candidate in sorted(plan, key=lambda candidate: candidate.confidence, reverse=True)
    )


def _spans(document: Document) -> Iterator[DiffSpan]:
    def walk(section: Section, path: tuple[str, ...]) -> Iterator[DiffSpan]:
        for child in section.children:
            if isinstance(child, Sentence):
                yield DiffSpan(sentence_id=child.id, section_path=path, original=child.text)
            else:
                yield from walk(child, (*path, child.header))

    for section in document.sections:
        yield from walk(section, (section.header,))


__all__ = ["diffs"]
