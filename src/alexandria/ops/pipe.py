from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import Plan
from alexandria.ir.document import Document
from alexandria.ir.registry import required_scorers
from alexandria.ops.features.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import DEFAULT_SCORER, score, score_rows
from alexandria.ops.features.select import DEFAULT_SELECTOR, select

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder, Params


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


def _required_scorers(optimizers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(scorer for o in optimizers for scorer in required_scorers(o)))
