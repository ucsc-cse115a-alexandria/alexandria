"""greedy_pairwise: drop the less load-bearing sentence of each redundant pair via A/B ablation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alexandria.core.protocols import Candidate, Delete
from alexandria.core.registry import register_optimizer
from alexandria.core.similarity import cosine_distance, cosine_similarity_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder, OptimizerParams, Plan, Scores

DEFAULT_OPTIMIZER = "greedy_pairwise"


@register_optimizer(DEFAULT_OPTIMIZER, requires=("redundancy",))
def greedy_pairwise(
    document: Document,
    scores: Scores,
    embedder: Embedder,
    params: OptimizerParams,
) -> Plan:
    """For each near-duplicate pair, drop the less load-bearing sentence — unless doing so would
    drift the prompt embedding more than max_drift (cosine distance) from the original prompt."""
    threshold = params.threshold
    max_drift = params.max_drift
    sentences = document.sentences
    redundancy = scores["redundancy"]
    embeddings = np.stack([s.embedding for s in sentences])
    similarity = cosine_similarity_matrix(embeddings)

    pairs = [
        (float(similarity[i, j]), i, j)
        for i in range(len(sentences))
        for j in range(i + 1, len(sentences))
        if similarity[i, j] >= threshold
    ]
    pairs.sort(key=lambda pair: pair[0], reverse=True)

    base = document.embedding
    order = [s.id for s in sentences]
    text_by_id = {s.id: s.text for s in sentences}
    present = set(order)
    candidates: list[Candidate] = []
    for sim, i, j in pairs:
        if redundancy[i] < threshold or redundancy[j] < threshold:
            continue
        a, b = sentences[i].id, sentences[j].id
        if a not in present or b not in present:
            continue
        drop, drift = _least_load_bearing(a, b, order, present, text_by_id, base, embedder)
        if drift > max_drift:
            continue
        present.discard(drop)
        candidates.append(
            Candidate(
                edit=Delete(targets=(drop,)),
                confidence=sim,
                source=DEFAULT_OPTIMIZER,
                reason=f"redundant (cosine {sim:.2f}); kept the more load-bearing instruction",
            )
        )
    return tuple(candidates)


def _least_load_bearing(
    a: str,
    b: str,
    order: list[str],
    present: set[str],
    text_by_id: dict[str, str],
    base: NDArray[np.float32],
    embedder: Embedder,
) -> tuple[str, float]:
    """Return the sentence whose removal drifts the prompt least, paired with that drift."""
    drift_a = _drift(a, order, present, text_by_id, base, embedder)
    drift_b = _drift(b, order, present, text_by_id, base, embedder)
    return (a, drift_a) if drift_a <= drift_b else (b, drift_b)


def _drift(
    drop_id: str,
    order: list[str],
    present: set[str],
    text_by_id: dict[str, str],
    base: NDArray[np.float32],
    embedder: Embedder,
) -> float:
    """Cosine distance from the original prompt embedding after also dropping drop_id."""
    kept_text = "".join(text_by_id[i] for i in order if i in present and i != drop_id)
    trial = embedder.embed([kept_text])[0]
    return cosine_distance(trial, base)
