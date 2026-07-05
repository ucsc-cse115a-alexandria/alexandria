from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from alexandria.core.envelope import DocumentEnvelope, PlanEnvelope, ScoredEnvelope
from alexandria.core.ir import Document, Section, SectionKind, Sentence, SentenceId
from alexandria.core.protocols import Candidate, Delete, Scores

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec(*values: float) -> NDArray[np.float32]:
    return np.array(values, dtype=np.float32)


def _document() -> Document:
    a = Sentence(id=SentenceId("s0"), text="a\n", token_count=1, embedding=_vec(0.1, 0.2, 0.3))
    b = Sentence(id=SentenceId("s1"), text="b\n", token_count=1, embedding=_vec(0.123456, -0.654321, 0.5))
    section = Section(
        kind=SectionKind.PLAIN, header="", children=(a, b), text="a\nb\n", token_count=2, embedding=_vec(0.4, 0.5, 0.6)
    )
    return Document(
        embedding_model="hash-4", sections=(section,), text="a\nb\n", token_count=2, embedding=_vec(0.7, 0.8, 0.9)
    )


def test_document_envelope_round_trips_embeddings_exactly() -> None:
    document = _document()
    envelope = DocumentEnvelope(document=document)

    parsed = DocumentEnvelope.model_validate_json(envelope.model_dump_json())

    for original, restored in zip(document.sentences, parsed.document.sentences, strict=True):
        assert restored.embedding.dtype == np.float32
        assert np.array_equal(restored.embedding, original.embedding)
    assert parsed.model_dump_json() == envelope.model_dump_json()


def test_scored_envelope_round_trips_the_score_bundle() -> None:
    document = _document()
    bundle: Scores = {"redundancy": {SentenceId("s0"): 0.99, SentenceId("s1"): 0.12}}
    envelope = ScoredEnvelope(document=document, scores=bundle)

    parsed = ScoredEnvelope.model_validate_json(envelope.model_dump_json())

    assert parsed.scores == bundle


def test_plan_envelope_round_trips_the_plan() -> None:
    document = _document()
    plan = (Candidate(edit=Delete(targets=(SentenceId("s0"),)), confidence=0.9, source="t", reason="redundant"),)
    envelope = PlanEnvelope(document=document, plan=plan)

    parsed = PlanEnvelope.model_validate_json(envelope.model_dump_json())

    assert parsed.plan == plan


def test_rejects_an_unknown_schema_version() -> None:
    payload = json.loads(DocumentEnvelope(document=_document()).model_dump_json())
    payload["schema_version"] = 2

    with pytest.raises(ValidationError):
        DocumentEnvelope.model_validate_json(json.dumps(payload))
