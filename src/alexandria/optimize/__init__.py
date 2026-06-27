"""Phase 3 — Optimize: run named optimizers and concatenate their Plan stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.protocols import OptimizerParams
from alexandria.core.registry import get_optimizer
from alexandria.optimize import greedy_pairwise
from alexandria.optimize.greedy_pairwise import DEFAULT_OPTIMIZER

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Candidate, Embedder, Plan, Scores


def optimize(
    document: Document,
    scores: Scores,
    embedder: Embedder,
    names: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    *,
    params: OptimizerParams | None = None,
) -> Plan:
    """Run each named optimizer and concatenate their candidate stacks into one ordered Plan."""
    params = params or OptimizerParams()
    plan: list[Candidate] = []
    for name in names:
        optimizer = get_optimizer(name)
        plan.extend(optimizer(document, scores, embedder, params))
    return tuple(plan)


__all__ = ["DEFAULT_OPTIMIZER", "OptimizerParams", "greedy_pairwise", "optimize"]
