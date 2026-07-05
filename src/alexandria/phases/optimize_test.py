from __future__ import annotations

import pytest

from alexandria.core.protocols import Params
from alexandria.phases.optimize import greedy_pairwise, optimize
from alexandria.phases.represent import represent
from alexandria.phases.score import score
from alexandria.runtime.embedding import HashEmbedder


def test_emits_one_delete_for_a_duplicate_pair() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)
    scores = score(document, names=("redundancy",))

    plan = greedy_pairwise(document, scores, Params())

    ids = [s.id for s in document.sentences]
    assert len(plan) == 1
    assert plan[0].edit.targets in ((ids[0],), (ids[1],))
    assert plan[0].confidence > 0.99


def test_no_candidates_when_all_unique() -> None:
    embedder = HashEmbedder()
    document = represent("alpha\nbeta\ngamma\n", embedder)
    scores = score(document, names=("redundancy",))

    assert greedy_pairwise(document, scores, Params()) == ()


def test_keeps_one_of_three_identical_sentences() -> None:
    embedder = HashEmbedder()
    document = represent("dup\ndup\ndup\n", embedder)
    scores = score(document, names=("redundancy",))

    plan = greedy_pairwise(document, scores, Params())

    # Two deletes proposed; one copy must survive (no candidate targets all three).
    dropped = {target for candidate in plan for target in candidate.edit.targets}
    assert len(dropped) == 2


def test_optimize_rejects_a_missing_required_scorer() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)

    with pytest.raises(ValueError, match=r"greedy_pairwise.*redundancy"):
        optimize(document, {}, names=("greedy_pairwise",))
