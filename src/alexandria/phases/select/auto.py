"""auto selector: apply candidates by descending confidence while staying within the drift budget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.apply import try_apply
from alexandria.core.registry import register_selector
from alexandria.core.similarity import cosine_distance

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder, Params, Plan

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
