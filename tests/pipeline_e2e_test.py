from __future__ import annotations

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import HashEmbedder


class _CannedMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del first, second, feedback
        return "repeat"


def test_reduce_removes_an_exact_duplicate_without_calling_the_merger() -> None:
    embedder = HashEmbedder()
    # HashEmbedder only scores exact duplicates as redundant. The deterministic fast path keeps
    # the first verbatim and removes the second without spending a generation call.
    result = reduce(
        "repeat me\nrepeat me\nunique line\n", embedder, _CannedMerger(), params=Params(cos_sim_diff_budget=2.0)
    )

    assert result.text == "repeat me\nunique line\n"
    assert result.reduced_tokens < result.source_tokens
    assert len(result.applied) == 1
    assert result.merge_metrics.calls == 0
