from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from alexandria.core.ir import Document, Section, SectionKind, Sentence

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec() -> NDArray[np.float32]:
    return np.ones(4, dtype=np.float32)


def _sentence(sid: str, text: str) -> Sentence:
    return Sentence(id=sid, text=text, token_count=1, embedding=_vec())


def test_sentences_flatten_in_document_order() -> None:
    a = _sentence("s0", "a\n")
    b = _sentence("s1", "b\n")
    section = Section(
        kind=SectionKind.PLAIN, header="", children=(a, b), text="a\nb\n", token_count=2, embedding=_vec()
    )
    document = Document(embedding_model="hash-4", sections=(section,), text="a\nb\n", token_count=2, embedding=_vec())
    assert [s.id for s in document.sentences] == ["s0", "s1"]


def test_nested_sections_flatten_in_document_order() -> None:
    inner = Section(
        kind=SectionKind.MARKDOWN,
        header="Database",
        children=(_sentence("s1", "## Database\n"), _sentence("s2", "Run migrations.\n")),
        text="## Database\nRun migrations.\n",
        token_count=2,
        embedding=_vec(),
    )
    outer = Section(
        kind=SectionKind.MARKDOWN,
        header="Setup",
        children=(_sentence("s0", "# Setup\n"), inner),
        text="# Setup\n## Database\nRun migrations.\n",
        token_count=3,
        embedding=_vec(),
    )
    document = Document(
        embedding_model="hash-4",
        sections=(outer,),
        text="# Setup\n## Database\nRun migrations.\n",
        token_count=3,
        embedding=_vec(),
    )
    assert [s.id for s in document.sentences] == ["s0", "s1", "s2"]


def test_rejects_text_not_matching_children() -> None:
    with pytest.raises(ValidationError):
        Section(
            kind=SectionKind.PLAIN,
            header="",
            children=(_sentence("s0", "x"),),
            text="WRONG",
            token_count=1,
            embedding=_vec(),
        )


def test_rejects_duplicate_sentence_ids() -> None:
    s = _sentence("s0", "x")
    section = Section(kind=SectionKind.PLAIN, header="", children=(s, s), text="xx", token_count=2, embedding=_vec())
    with pytest.raises(ValidationError):
        Document(embedding_model="hash-4", sections=(section,), text="xx", token_count=2, embedding=_vec())
