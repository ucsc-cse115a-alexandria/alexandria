"""Phase 3 — Optimize: run named optimizers and concatenate their Plan stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.protocols import Params
from alexandria.core.registry import get_optimizer
from alexandria.phases.optimize import greedy_pairwise
from alexandria.phases.optimize.greedy_pairwise import DEFAULT_OPTIMIZER

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Candidate, Plan, Scores


def optimize(
    document: Document,
    scores: Scores,
    names: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    *,
    params: Params | None = None,
) -> Plan:
    """Run each named optimizer and concatenate their candidate stacks into one ordered Plan."""
    params = params or Params()
    plan: list[Candidate] = []
    for name in names:
        optimizer = get_optimizer(name)
        plan.extend(optimizer(document, scores, params))
    return tuple(plan)


__all__ = ["DEFAULT_OPTIMIZER", "greedy_pairwise", "optimize"]
