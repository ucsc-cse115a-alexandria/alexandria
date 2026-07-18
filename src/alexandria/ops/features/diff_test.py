from __future__ import annotations

import pytest

from alexandria.ir.contracts import Candidate, Delete, Params, Replace
from alexandria.ir.document import Encoded, SentenceId
from alexandria.ops.features.diff import diffs
from alexandria.ops.features.optimize import optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import score
from alexandria.ops.features.select import select
from alexandria.utils.embedders import HashEmbedder, build_embedder

_REDUNDANT_SECTIONS = "# Alpha\nrepeat me\nrepeat me\n# Beta\necho twice\necho twice\n"


def test_one_diff_per_candidate_with_resolved_spans() -> None:
    document = represent(_REDUNDANT_SECTIONS, HashEmbedder())
    plan = optimize(document, score(document, names=("redundancy",)))

    result = diffs(document, plan)

    assert len(result) == len(plan) == 2
    assert [d.candidate.confidence for d in result] == sorted((c.confidence for c in plan), reverse=True)
    by_original = {d.spans[0].original: d for d in result}
    assert by_original["repeat me\n"].spans[0].section_path == ("Alpha",)
    assert by_original["echo twice\n"].spans[0].section_path == ("Beta",)
    for diff in result:
        assert [span.sentence_id for span in diff.spans] == list(diff.candidate.edit.targets)
        assert diff.replacement == ""


def test_accepting_every_diff_reproduces_the_automatic_run() -> None:
    embedder = HashEmbedder()
    document = represent(_REDUNDANT_SECTIONS, embedder)
    plan = optimize(document, score(document, names=("redundancy",)))
    params = Params(drift_budget=2.0)  # generous enough that the auto selector rejects nothing

    accepted = tuple(diff.candidate for diff in diffs(document, plan))
    interactive = select(document, accepted, embedder, params=params)
    automatic = select(document, plan, embedder, params=params)

    assert interactive.document.text == automatic.document.text
    assert interactive.applied == automatic.applied


def test_unknown_target_id_raises_at_the_boundary() -> None:
    document = represent("alpha\nbeta\n", HashEmbedder())
    ghost = Candidate(edit=Delete(targets=(SentenceId("s000000000000"),)), confidence=0.9, source="t", reason="r")

    with pytest.raises(ValueError, match="not in the document"):
        diffs(document, (ghost,))


def test_replace_diff_carries_the_replacement_text() -> None:
    document = represent("aaa\nbbb\nccc\n", build_embedder("deterministic"))
    ids = [s.id for s in document.sentences]
    replacement = Encoded(text="ab\n", token_count=1, embedding=document.sentences[0].embedding)
    candidate = Candidate(
        edit=Replace(targets=(ids[0], ids[1]), replacement=replacement), confidence=0.9, source="t", reason="r"
    )

    (diff,) = diffs(document, (candidate,))

    assert diff.replacement == "ab\n"
    assert [span.sentence_id for span in diff.spans] == [ids[0], ids[1]]
