"""Library composition of the three phases — what the CLI calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.registry import required_scorers, scorer_peers
from alexandria.core.select import apply
from alexandria.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.represent import represent
from alexandria.score import DEFAULT_SCORER, score

if TYPE_CHECKING:
    from alexandria.core.protocols import Embedder, OptimizerParams


def reduce(
    prompt: str,
    embedder: Embedder,
    *,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    params: OptimizerParams | None = None,
) -> str:
    """Run all three phases end to end and return the reduced prompt text."""
    document = represent(prompt, embedder)
    scores = score(document, names=_required_scorers(optimizers))
    plan = optimize(document, scores, embedder, names=optimizers, params=params)
    return apply(document, plan).text


def score_report(
    prompt: str, embedder: Embedder, *, scorers: tuple[str, ...] = (DEFAULT_SCORER,)
) -> list[dict[str, object]]:
    """Represent then score into display rows: id, text, each scorer's value, and its peer (if any)."""
    document = represent(prompt, embedder)
    bundle = score(document, names=scorers)
    sentences = document.sentences
    text_by_id = {s.id: s.text.strip() for s in sentences}
    peer_finders = [finder for name in scorers if (finder := scorer_peers(name)) is not None]
    peers = peer_finders[0](document) if peer_finders else None
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
    return tuple(dict.fromkeys(scorer for o in optimizers for scorer in required_scorers(o)))
