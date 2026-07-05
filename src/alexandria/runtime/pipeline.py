from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import Plan
from alexandria.ir.document import Document
from alexandria.ir.registry import required_scorers, scorer_peers
from alexandria.phases.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.phases.represent import represent
from alexandria.phases.score import DEFAULT_SCORER, score
from alexandria.phases.select import DEFAULT_SELECTOR, select

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder, Params, Scores


class ReduceResult(BaseModel):
    """The outcome of a reduction: the reduced Document, its source, and the applied candidates."""

    model_config = ConfigDict(frozen=True)
    document: Document
    source: Document
    applied: Plan

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def source_tokens(self) -> int:
        return self.source.token_count

    @property
    def reduced_tokens(self) -> int:
        return self.document.token_count


def reduce(
    prompt: str,
    embedder: Embedder,
    *,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    selector: str = DEFAULT_SELECTOR,
    params: Params | None = None,
) -> ReduceResult:
    """Run represent → score → optimize → select end to end and return the reduction."""
    document = represent(prompt, embedder)
    scores = score(document, names=_required_scorers(optimizers))
    plan = optimize(document, scores, names=optimizers, params=params)
    selection = select(document, plan, embedder, selector, params=params)
    return ReduceResult(document=selection.document, source=document, applied=selection.applied)


def score_report(
    prompt: str, embedder: Embedder, *, scorers: tuple[str, ...] = (DEFAULT_SCORER,)
) -> list[dict[str, object]]:
    """Represent then score into display rows: id, text, each scorer's value, and its peer (if any)."""
    document = represent(prompt, embedder)
    return score_rows(document, score(document, names=scorers), scorers)


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


def _required_scorers(optimizers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(scorer for o in optimizers for scorer in required_scorers(o)))
