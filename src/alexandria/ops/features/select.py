from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.ir.contracts import Candidate, Params, Selection
from alexandria.ir.registry import get_selector, register_selector
from alexandria.ir.similarity import cosine_distance
from alexandria.utils.embedders import default_embedder

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder, Plan
    from alexandria.ir.document import Document

DEFAULT_SELECTOR = "auto"


@register_selector(DEFAULT_SELECTOR)
def auto(document: Document, plan: Plan, embedder: Embedder, params: Params) -> Selection:
    
    base = document.embedding
    current = document
    applied: list[Candidate] = []
    
    # Check if we are already under budget before doing any work
    if params.max_tokens is not None and current.token_count <= params.max_tokens:
        return Selection(document=current, applied=tuple(applied))
        
    for candidate in sorted(plan, key=lambda candidate: candidate.confidence, reverse=True):
        trial = current.apply(candidate)
        if trial is None or trial is current:
            continue
            
        drift = cosine_distance(embedder.embed([trial.text])[0], base)
        if drift <= params.drift_budget:
            current = trial
            applied.append(candidate)
            
            
            if params.max_tokens is not None and current.token_count <= params.max_tokens:
                break
                
    return Selection(document=current, applied=tuple(applied))


def select(
    document: Document,
    plan: Plan,
    embedder: Embedder | None = None,
    name: str = DEFAULT_SELECTOR,
    *,
    params: Params | None = None,
) -> Selection:
    """Run the named selector to fold the Plan into a reduced Document.

    When embedder is omitted, the default all-MiniLM-L6-v2 model is downloaded and built on first use.
    """
    embedder = embedder if embedder is not None else default_embedder()
    if embedder.model_id != document.embedding_model:
        raise ValueError(
            f"embedder {embedder.model_id!r} does not match document embedding model {document.embedding_model!r}"
        )
    params = params or Params()
    return get_selector(name)(document, plan, embedder, params)


__all__ = ["DEFAULT_SELECTOR", "auto", "select"]
