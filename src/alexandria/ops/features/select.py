from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.ir.contracts import Candidate, Params, Selection
from alexandria.ir.registry import get_selector, register_selector
from alexandria.ir.similarity import normalize
from alexandria.utils.embedders import default_embedder

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder, Plan
    from alexandria.ir.document import Document

DEFAULT_SELECTOR = "least_cos_sim_diff"


@register_selector(DEFAULT_SELECTOR)
def least_cos_sim_diff(document: Document, plan: Plan, embedder: Embedder, params: Params) -> Selection:
    """Apply candidates in ascending whole-document cos_sim_diff order, under the cumulative
    cos_sim_diff budget, stopping once params.max_tokens is met.

    Ranking embeds every candidate's trial text in one batched call; the cumulative re-check
    then embeds once per applied candidate.
    """
    if params.max_tokens is not None and document.token_count <= params.max_tokens:
        return Selection(document=document, applied=())

    base = normalize(document.embedding)  # normalize the fixed base once, not per cos_sim_diff measurement

    def cos_sim_diff_from_base(vector: NDArray[np.float32]) -> float:
        return 1.0 - float(normalize(vector) @ base)

    trials = [
        (candidate, trial)
        for candidate in plan
        if (trial := document.apply(candidate)) is not None and trial is not document
    ]
    vectors = embedder.embed([trial.text for _, trial in trials]) if trials else []
    ranked = sorted(zip(trials, vectors, strict=True), key=lambda pair: cos_sim_diff_from_base(pair[1]))

    current = document
    applied: list[Candidate] = []
    for (candidate, _), _ in ranked:
        trial = current.apply(candidate)
        if trial is None or trial is current:
            continue
        if cos_sim_diff_from_base(embedder.embed([trial.text])[0]) > params.cos_sim_diff_budget:
            continue
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

    When embedder is omitted, the default OpenAI text-embedding-3-small embedder is built on
    first use (requires an API key).
    """
    embedder = embedder if embedder is not None else default_embedder()
    if embedder.model_id != document.embedding_model:
        raise ValueError(
            f"embedder {embedder.model_id!r} does not match document embedding model {document.embedding_model!r}"
        )
    params = params or Params()
    return get_selector(name)(document, plan, embedder, params)


def apply_candidates(document: Document, candidates: tuple[Candidate, ...]) -> Document:
    """Fold Document.apply over the candidates; accept means accept — no cos_sim_diff budget re-filtering.

    A candidate whose edit would empty the document or a section (apply returns None) is skipped.
    """
    current = document
    for candidate in candidates:
        trial = current.apply(candidate)
        if trial is not None:
            current = trial
    return current


__all__ = ["DEFAULT_SELECTOR", "apply_candidates", "least_cos_sim_diff", "select"]
