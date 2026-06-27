"""greedy_pairwise: drop the less load-bearing sentence of each redundant pair via A/B ablation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alexandria.core.protocols import Candidate, Delete
from alexandria.core.registry import register_optimizer
from alexandria.core.similarity import cosine_similarity_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder, Plan, Scores


@register_optimizer("greedy_pairwise", requires=("redundancy",))
def greedy_pairwise(document: Document, scores: Scores, embedder: Embedder, *, threshold: float) -> Plan:
    """For each near-duplicate pair, keep the sentence whose removal changes the prompt meaning most."""
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
        drop = _least_load_bearing(a, b, order, present, text_by_id, base, embedder)
        present.discard(drop)
        candidates.append(
            Candidate(
                edit=Delete(targets=(drop,)),
                score=sim,
                source="greedy_pairwise",
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
) -> str:
    delta_a = _ablation_distance(a, order, present, text_by_id, base, embedder)
    delta_b = _ablation_distance(b, order, present, text_by_id, base, embedder)
    return a if delta_a <= delta_b else b


def _ablation_distance(
    drop_id: str,
    order: list[str],
    present: set[str],
    text_by_id: dict[str, str],
    base: NDArray[np.float32],
    embedder: Embedder,
) -> float:
    kept_text = "".join(text_by_id[i] for i in order if i in present and i != drop_id)
    trial = embedder.embed([kept_text])[0]
    return float(np.linalg.norm(trial - base))
