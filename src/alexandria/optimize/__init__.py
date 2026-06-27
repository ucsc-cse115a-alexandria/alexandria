"""Phase 3 — Optimize: run named optimizers and concatenate their Plan stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.registry import get_optimizer
from alexandria.optimize import greedy_pairwise

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Candidate, Embedder, Plan, Scores


def optimize(
    document: Document,
    scores: Scores,
    embedder: Embedder,
    names: tuple[str, ...] = ("greedy_pairwise",),
    *,
    threshold: float = 0.85,
    max_drift: float = 2.0,
) -> Plan:
    """Run each named optimizer and concatenate their candidate stacks into one ordered Plan."""
    plan: list[Candidate] = []
    for name in names:
        optimizer = get_optimizer(name)
        plan.extend(optimizer(document, scores, embedder, threshold=threshold, max_drift=max_drift))
    return tuple(plan)


__all__ = ["greedy_pairwise", "optimize"]
