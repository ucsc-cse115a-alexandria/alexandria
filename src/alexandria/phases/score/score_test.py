from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alexandria.core.registry import register_scorer
from alexandria.phases.represent import represent
from alexandria.phases.score import score
from alexandria.runtime.embedding import HashEmbedder

if TYPE_CHECKING:
    from alexandria.core.ir import Document


def _wrong_length(document: Document) -> list[float]:
    return [0.0] * (len(document.sentences) + 1)


register_scorer("wrong_length_probe")(_wrong_length)


def test_score_rejects_a_wrong_length_vector() -> None:
    document = represent("a\nb\n", HashEmbedder())
    with pytest.raises(ValueError):
        score(document, names=("wrong_length_probe",))
