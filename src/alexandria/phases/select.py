"""Phase 4 — Select: apply candidates under a drift budget, highest-confidence-first."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.apply import try_apply
from alexandria.core.protocols import Params
from alexandria.core.registry import get_selector, register_selector
from alexandria.core.similarity import cosine_distance

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder, Plan

DEFAULT_SELECTOR = "auto"


@register_selector(DEFAULT_SELECTOR)
def auto(document: Document, plan: Plan, embedder: Embedder, params: Params) -> Document:
    """Apply candidates highest-confidence-first, keeping each only while the reduced prompt stays
    within drift_budget cosine distance of the original Document embedding."""
    base = document.embedding
    current = document
    for candidate in sorted(plan, key=lambda candidate: candidate.confidence, reverse=True):
        trial = try_apply(current, candidate)
        if trial is None or trial is current:
            continue
        drift = cosine_distance(embedder.embed([trial.text])[0], base)
        if drift <= params.drift_budget:
            current = trial
    return current


def select(
    document: Document,
    plan: Plan,
    embedder: Embedder,
    name: str = DEFAULT_SELECTOR,
    *,
    params: Params | None = None,
) -> Document:
    """Run the named selector to fold the Plan into a reduced Document."""
    params = params or Params()
    return get_selector(name)(document, plan, embedder, params)


__all__ = ["DEFAULT_SELECTOR", "auto", "select"]
