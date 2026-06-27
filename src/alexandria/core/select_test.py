from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.core.ir import Document, Section, Sentence
from alexandria.core.protocols import Candidate, Delete
from alexandria.core.select import apply

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _document(texts: list[str]) -> Document:
    vec: NDArray[np.float32] = np.ones(4, dtype=np.float32)
    sentences = tuple(
        Sentence(id=f"s{i}", text=t, token_count=1, embedding=vec) for i, t in enumerate(texts)
    )
    joined = "".join(texts)
    section = Section(header="", sentences=sentences, text=joined, token_count=len(texts), embedding=vec)
    return Document(
        embedding_model="hash-4", sections=(section,), text=joined, token_count=len(texts), embedding=vec
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
    vec: NDArray[np.float32] = np.ones(4, dtype=np.float32)
    section_a = Section(
        header="a",
        sentences=(
            Sentence(id="s0", text="a0\n", token_count=1, embedding=vec),
            Sentence(id="s1", text="a1\n", token_count=1, embedding=vec),
        ),
        text="a0\na1\n",
        token_count=2,
        embedding=vec,
    )
    section_b = Section(
        header="b",
        sentences=(Sentence(id="s2", text="b0\n", token_count=1, embedding=vec),),
        text="b0\n",
        token_count=1,
        embedding=vec,
    )
    return Document(
        embedding_model="hash-4",
        sections=(section_a, section_b),
        text="a0\na1\nb0\n",
        token_count=3,
        embedding=vec,
    )


def test_apply_refuses_to_empty_a_section() -> None:
    with pytest.raises(ValueError):
        apply(_two_section_document(), (_delete("s2"),))
