"""Library composition of the three phases — what the CLI calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.registry import required_scorers
from alexandria.core.select import apply
from alexandria.optimize import optimize
from alexandria.represent import represent
from alexandria.score import score
from alexandria.score.redundancy import most_similar

if TYPE_CHECKING:
    from alexandria.core.protocols import Embedder


def reduce(
    prompt: str,
    embedder: Embedder,
    *,
    optimizers: tuple[str, ...] = ("greedy_pairwise",),
    threshold: float = 0.85,
) -> str:
    """Run all three phases end to end and return the reduced prompt text."""
    document = represent(prompt, embedder)
    scores = score(document, names=_required_scorers(optimizers))
    plan = optimize(document, scores, embedder, names=optimizers, threshold=threshold)
    return apply(document, plan).text


def score_report(
    prompt: str, embedder: Embedder, *, scorers: tuple[str, ...] = ("redundancy",)
) -> list[dict[str, object]]:
    """Represent then score into display rows: id, text, each scorer's value, and redundancy's peer."""
    document = represent(prompt, embedder)
    bundle = score(document, names=scorers)
    sentences = document.sentences
    text_by_id = {s.id: s.text.strip() for s in sentences}
    peers = most_similar(document) if "redundancy" in scorers else None
    rows: list[dict[str, object]] = []
    for i, sentence in enumerate(sentences):
        row: dict[str, object] = {"id": sentence.id, "text": sentence.text.strip()}
        for name in scorers:
            row[name] = round(bundle[name][i], 4)
        if peers is not None:
            peer_id, _ = peers[i]
            row["most_similar_id"] = peer_id
            row["most_similar_text"] = text_by_id[peer_id] if peer_id is not None else None
        rows.append(row)
    return rows


def _required_scorers(optimizers: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for optimizer in optimizers:
        for scorer in required_scorers(optimizer):
            if scorer not in names:
                names.append(scorer)
    return tuple(names)
