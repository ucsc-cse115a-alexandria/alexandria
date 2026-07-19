from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alexandria.ir.registry import get_scorer, register_scorer, scorer_peers
from alexandria.ir.similarity import similarity_matrix_for

if TYPE_CHECKING:
    from alexandria.ir.contracts import Scores
    from alexandria.ir.document import Document, SentenceId

DEFAULT_SCORER = "redundancy"


def most_similar(document: Document) -> list[tuple[SentenceId | None, float]]:
    """Each sentence's most-similar peer id and its cosine similarity (None, 0.0 if no peer)."""
    sentences = document.sentences
    if len(sentences) < 2:
        return [(None, 0.0) for _ in sentences]
    similarity = similarity_matrix_for(document).copy()  # copy: the shared matrix is read-only
    np.fill_diagonal(similarity, -np.inf)
    peer_indices = similarity.argmax(axis=1)  # first-max tie-break matches the per-row argmax it replaces
    return [(sentences[int(index)].id, float(row[index])) for row, index in zip(similarity, peer_indices, strict=True)]


@register_scorer(DEFAULT_SCORER, peers=most_similar)
def redundancy(document: Document) -> list[float]:
    """Score each sentence by its max cosine similarity to any other sentence."""
    return [similarity for _, similarity in most_similar(document)]


def score(document: Document, names: tuple[str, ...] = (DEFAULT_SCORER,)) -> Scores:
    """Run each named scorer, validate its length, and key its scores by sentence id."""
    bundle: Scores = {}
    sentences = document.sentences
    expected = len(sentences)
    for name in names:
        values = get_scorer(name)(document)
        if len(values) != expected:
            raise ValueError(f"scorer {name!r} returned {len(values)} scores for {expected} sentences")
        bundle[name] = {sentence.id: value for sentence, value in zip(sentences, values, strict=True)}
    return bundle


def score_rows(document: Document, bundle: Scores, scorers: tuple[str, ...]) -> list[dict[str, object]]:
    """Turn an already-scored Document into display rows: id, text, each scorer's value, and its peer."""
    sentences = document.sentences
    text_by_id = {s.id: s.text.strip() for s in sentences}
    peer_finders = [finder for name in scorers if (finder := scorer_peers(name)) is not None]
    peers = peer_finders[0](document) if peer_finders else None
    rows: list[dict[str, object]] = []
    for i, sentence in enumerate(sentences):
        row: dict[str, object] = {"id": sentence.id, "text": sentence.text.strip()}
        for name in scorers:
            row[name] = round(bundle[name][sentence.id], 4)
        if peers is not None:
            peer_id, _ = peers[i]
            row["most_similar_id"] = peer_id
            row["most_similar_text"] = text_by_id[peer_id] if peer_id is not None else None
        rows.append(row)
    return rows


__all__ = ["DEFAULT_SCORER", "most_similar", "redundancy", "score", "score_rows"]
