from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.core.ir import Document, Section, SectionKind, Sentence
from alexandria.core.protocols import Candidate, Delete
from alexandria.core.select import apply

if TYPE_CHECKING:
    from numpy.typing import NDArray

_VEC: NDArray[np.float32] = np.ones(4, dtype=np.float32)


def _sentence(sid: str, text: str) -> Sentence:
    return Sentence(id=sid, text=text, token_count=1, embedding=_VEC)


def _plain_section(sentences: tuple[Sentence, ...]) -> Section:
    joined = "".join(s.text for s in sentences)
    return Section(
        kind=SectionKind.PLAIN, header="", children=sentences, text=joined, token_count=len(sentences), embedding=_VEC
    )


def _document(texts: list[str]) -> Document:
    sentences = tuple(_sentence(f"s{i}", t) for i, t in enumerate(texts))
    section = _plain_section(sentences)
    return Document(
        embedding_model="hash-4", sections=(section,), text=section.text, token_count=len(texts), embedding=_VEC
    )


def _delete(target: str) -> Candidate:
    return Candidate(edit=Delete(targets=(target,)), score=1.0, source="t", reason="r")


def test_apply_drops_targeted_sentence() -> None:
    reduced = apply(_document(["a\n", "b\n", "c\n"]), (_delete("s1"),))
    assert [s.id for s in reduced.sentences] == ["s0", "s2"]
    assert reduced.text == "a\nc\n"
    assert reduced.token_count == 2


def test_apply_skips_already_removed_target() -> None:
    reduced = apply(_document(["a\n", "b\n", "c\n"]), (_delete("s1"), _delete("s1")))
    assert [s.id for s in reduced.sentences] == ["s0", "s2"]


def test_apply_refuses_to_empty_the_document() -> None:
    with pytest.raises(ValueError):
        apply(_document(["only\n"]), (_delete("s0"),))


def _two_section_document() -> Document:
    section_a = Section(
        kind=SectionKind.MARKDOWN,
        header="a",
        children=(_sentence("s0", "a0\n"), _sentence("s1", "a1\n")),
        text="a0\na1\n",
        token_count=2,
        embedding=_VEC,
    )
    section_b = Section(
        kind=SectionKind.MARKDOWN,
        header="b",
        children=(_sentence("s2", "b0\n"),),
        text="b0\n",
        token_count=1,
        embedding=_VEC,
    )
    return Document(
        embedding_model="hash-4", sections=(section_a, section_b), text="a0\na1\nb0\n", token_count=3, embedding=_VEC
    )


def test_apply_refuses_to_empty_a_section() -> None:
    with pytest.raises(ValueError):
        apply(_two_section_document(), (_delete("s2"),))


def _nested_document() -> Document:
    inner = Section(
        kind=SectionKind.MARKDOWN,
        header="Database",
        children=(_sentence("s1", "## Database\n"), _sentence("s2", "Run migrations.\n")),
        text="## Database\nRun migrations.\n",
        token_count=2,
        embedding=_VEC,
    )
    outer = Section(
        kind=SectionKind.MARKDOWN,
        header="Setup",
        children=(_sentence("s0", "# Setup\n"), inner),
        text="# Setup\n## Database\nRun migrations.\n",
        token_count=3,
        embedding=_VEC,
    )
    return Document(
        embedding_model="hash-4",
        sections=(outer,),
        text="# Setup\n## Database\nRun migrations.\n",
        token_count=3,
        embedding=_VEC,
    )


def test_apply_drops_sentence_inside_nested_section() -> None:
    reduced = apply(_nested_document(), (_delete("s2"),))
    assert [s.id for s in reduced.sentences] == ["s0", "s1"]
    assert reduced.text == "# Setup\n## Database\n"


def test_apply_refuses_to_empty_a_nested_section() -> None:
    # s1 and s2 are the only descendants of the nested "Database" section.
    with pytest.raises(ValueError):
        apply(_nested_document(), (_delete("s1"), _delete("s2")))
