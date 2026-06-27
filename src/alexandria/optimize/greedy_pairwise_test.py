from __future__ import annotations

from alexandria.embedding import HashEmbedder
from alexandria.optimize.greedy_pairwise import greedy_pairwise
from alexandria.represent import represent
from alexandria.score import score


def test_emits_one_delete_for_a_duplicate_pair() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)
    scores = score(document, names=("redundancy",))

    plan = greedy_pairwise(document, scores, embedder, threshold=0.85)

    assert len(plan) == 1
    assert plan[0].edit.targets in (("s0",), ("s1",))


def test_no_candidates_when_all_unique() -> None:
    embedder = HashEmbedder()
    document = represent("alpha\nbeta\ngamma\n", embedder)
    scores = score(document, names=("redundancy",))

    assert greedy_pairwise(document, scores, embedder, threshold=0.85) == ()


def test_skips_delete_when_drift_exceeds_max_drift() -> None:
    embedder = HashEmbedder()
    document = represent("repeat me\nrepeat me\nunique line\n", embedder)
    scores = score(document, names=("redundancy",))

    plan = greedy_pairwise(document, scores, embedder, threshold=0.85, max_drift=0.0)

    assert plan == ()
