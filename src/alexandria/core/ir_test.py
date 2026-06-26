from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from alexandria.core.ir import Document, Section, Sentence

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec() -> NDArray[np.float32]:
    return np.ones(4, dtype=np.float32)


def test_sentences_flatten_in_document_order() -> None:
    a = Sentence(id="s0", text="a\n", token_count=1, embedding=_vec())
    b = Sentence(id="s1", text="b\n", token_count=1, embedding=_vec())
    section = Section(header="", sentences=(a, b), text="a\nb\n", token_count=2, embedding=_vec())
    document = Document(
        embedding_model="hash-4", sections=(section,), text="a\nb\n", token_count=2, embedding=_vec()
    )
    assert [s.id for s in document.sentences] == ["s0", "s1"]


def test_rejects_text_not_matching_children() -> None:
    s = Sentence(id="s0", text="x", token_count=1, embedding=_vec())
    with pytest.raises(ValidationError):
        Section(header="", sentences=(s,), text="WRONG", token_count=1, embedding=_vec())


def test_rejects_duplicate_sentence_ids() -> None:
    s = Sentence(id="s0", text="x", token_count=1, embedding=_vec())
    section = Section(header="", sentences=(s, s), text="xx", token_count=2, embedding=_vec())
    with pytest.raises(ValidationError):
        Document(
            embedding_model="hash-4", sections=(section,), text="xx", token_count=2, embedding=_vec()
        )
