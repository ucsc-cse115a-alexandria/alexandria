from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.protocols import Candidate, Delete, Params
from alexandria.core.registry import get_optimizer, register_optimizer, required_scorers
from alexandria.core.similarity import similarity_matrix_for

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Plan, Scores

DEFAULT_OPTIMIZER = "greedy_pairwise"


@register_optimizer(DEFAULT_OPTIMIZER, requires=("redundancy",))
def greedy_pairwise(document: Document, scores: Scores, params: Params) -> Plan:
    """For each near-duplicate pair, propose deleting the more redundant sentence, ranked by similarity.

    Pure ranking only: the select phase decides how many of these proposals survive the drift budget.
    """
    threshold = params.threshold
    sentences = document.sentences
    redundancy = scores["redundancy"]
    similarity = similarity_matrix_for(document)

    pairs = [
        (float(similarity[i, j]), i, j)
        for i in range(len(sentences))
        for j in range(i + 1, len(sentences))
        if similarity[i, j] >= threshold
    ]
    pairs.sort(key=lambda pair: pair[0], reverse=True)

    present = {s.id for s in sentences}
    candidates: list[Candidate] = []
    for sim, i, j in pairs:
        a, b = sentences[i].id, sentences[j].id
        if a not in present or b not in present:
            continue
        drop = a if redundancy[a] >= redundancy[b] else b
        present.discard(drop)
        candidates.append(
            Candidate(
                edit=Delete(targets=(drop,)),
                confidence=sim,
                source=DEFAULT_OPTIMIZER,
                reason=f"redundant (cosine {sim:.2f}); dropped the more redundant instruction",
            )
        )
    return tuple(candidates)


def optimize(
    document: Document,
    scores: Scores,
    names: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    *,
    params: Params | None = None,
) -> Plan:
    """Run each named optimizer and concatenate their candidate stacks into one ordered Plan.

    Raises ValueError if any optimizer's required scorers are absent from `scores`, so a
    mispiped `score ... | optimize` fails at the boundary instead of raising a raw KeyError deeper in.
    """
    params = params or Params()
    for name in names:
        missing = [scorer for scorer in required_scorers(name) if scorer not in scores]
        if missing:
            raise ValueError(
                f"optimizer {name!r} requires scorer(s) {missing} absent from the input scores {sorted(scores)}"
            )
    plan: list[Candidate] = []
    for name in names:
        optimizer = get_optimizer(name)
        plan.extend(optimizer(document, scores, params))
    return tuple(plan)


__all__ = ["DEFAULT_OPTIMIZER", "greedy_pairwise", "optimize"]
