"""Library composition of the phases — what the CLI calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.registry import required_scorers, scorer_peers
from alexandria.phases.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.phases.represent import represent
from alexandria.phases.score import DEFAULT_SCORER, score
from alexandria.phases.select import DEFAULT_SELECTOR, select

if TYPE_CHECKING:
    from alexandria.core.protocols import Embedder, Params


def reduce(
    prompt: str,
    embedder: Embedder,
    *,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    selector: str = DEFAULT_SELECTOR,
    params: Params | None = None,
) -> str:
    """Run represent → score → optimize → select end to end and return the reduced prompt text."""
    document = represent(prompt, embedder)
    scores = score(document, names=_required_scorers(optimizers))
    plan = optimize(document, scores, names=optimizers, params=params)
    return select(document, plan, embedder, selector, params=params).text


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
