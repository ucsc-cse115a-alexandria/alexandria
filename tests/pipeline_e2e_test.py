from __future__ import annotations

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import HashEmbedder


class _CannedMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del first, second, feedback
        return "repeat"


def test_reduce_merges_a_duplicate_pair_into_the_first_occurrence() -> None:
    embedder = HashEmbedder()
    # HashEmbedder only scores exact duplicates as redundant, and re-embeds edited text to an
    # unrelated vector, so a generous drift budget is needed to accept the merge.
    result = reduce("repeat me\nrepeat me\nunique line\n", embedder, _CannedMerger(), params=Params(drift_budget=2.0))

    assert result.text == "repeat\nunique line\n"
    assert result.reduced_tokens < result.source_tokens
    assert len(result.applied) == 1
