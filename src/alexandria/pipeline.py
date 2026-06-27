"""Library composition of the three phases — what the CLI calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.registry import required_scorers
from alexandria.core.select import apply
from alexandria.optimize import optimize
from alexandria.represent import represent
from alexandria.score import score

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder, Scores


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


def score_prompt(
    prompt: str, embedder: Embedder, *, scorers: tuple[str, ...] = ("redundancy",)
) -> tuple[Document, Scores]:
    """Represent then score; returns the Document and the score bundle for display."""
    document = represent(prompt, embedder)
    return document, score(document, names=scorers)


def _required_scorers(optimizers: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for optimizer in optimizers:
        for scorer in required_scorers(optimizer):
            if scorer not in names:
                names.append(scorer)
    return tuple(names)
