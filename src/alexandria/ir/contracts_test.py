from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alexandria.ir.contracts import TargetedMerger, TrackedEmbedder, TrackedMerger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _CountingEmbedder:
    """Records every text the wrapped embedder is actually asked to embed."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    @property
    def model_id(self) -> str:
        return "counting-3"

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        self.embedded.extend(texts)
        return [np.array([text.count("a"), text.count("b"), text.count("c")], dtype=np.float32) for text in texts]


def test_tracked_embedder_caches_repeated_texts() -> None:
    inner = _CountingEmbedder()
    tracked = TrackedEmbedder(inner)

    first = tracked.embed(["alpha", "beta"])
    second = tracked.embed(["beta", "alpha", "gamma"])

    # "beta"/"alpha" are served from the cache; only "gamma" reaches the wrapped embedder again.
    assert inner.embedded == ["alpha", "beta", "gamma"]
    assert tracked.calls == 2
    assert tracked.texts == 3
    # Cached vectors are returned in the caller's input order and are the same arrays.
    assert second[0] is first[1]
    assert second[1] is first[0]


def test_tracked_embedder_deduplicates_within_one_batch() -> None:
    inner = _CountingEmbedder()
    tracked = TrackedEmbedder(inner)

    tracked.embed(["dup", "dup", "solo"])

    assert inner.embedded == ["dup", "solo"]
    assert tracked.calls == 1
    assert tracked.texts == 2


def test_tracked_embedder_skips_a_fully_cached_batch() -> None:
    inner = _CountingEmbedder()
    tracked = TrackedEmbedder(inner)

    tracked.embed(["one", "two"])
    tracked.embed(["two", "one"])

    assert inner.embedded == ["one", "two"]
    assert tracked.calls == 1
    assert tracked.texts == 2


class _FakeTargetedMerger:
    """Offline merger that echoes the prompt as a fixed number of candidates."""

    def __init__(self, candidate_count: int) -> None:
        self._candidate_count = candidate_count
        self.merge_calls = 0
        self.target_calls = 0

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        self.merge_calls += 1
        return first.strip()

    def merge_candidates_to_target(self, prompt: str, max_tokens: int) -> tuple[str, ...]:
        del max_tokens
        self.target_calls += 1
        return tuple(f"{prompt}#{index}" for index in range(self._candidate_count))


class _PairwiseOnlyMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first.strip()


def test_fake_targeted_merger_satisfies_the_protocol() -> None:
    assert isinstance(_FakeTargetedMerger(3), TargetedMerger)


def test_target_call_counts_as_one_call_and_one_job() -> None:
    tracked = TrackedMerger(_FakeTargetedMerger(3))

    candidates = tracked.merge_candidates_to_target("segment", 40)

    assert candidates == ("segment#0", "segment#1", "segment#2")
    assert tracked.calls == 1
    assert tracked.jobs_attempted == 1
    assert tracked.retries == 0
    assert tracked.candidates_generated == 3


def test_target_calls_accumulate_candidate_counts() -> None:
    tracked = TrackedMerger(_FakeTargetedMerger(2))

    tracked.merge_candidates_to_target("first", 40)
    tracked.merge_candidates_to_target("second", 40)

    assert tracked.calls == 2
    assert tracked.jobs_attempted == 2
    assert tracked.candidates_generated == 4


def test_target_requires_a_targeted_merger() -> None:
    tracked = TrackedMerger(_PairwiseOnlyMerger())

    with pytest.raises(TypeError):
        tracked.merge_candidates_to_target("segment", 40)


def test_pairwise_merge_still_counts_jobs_and_retries() -> None:
    tracked = TrackedMerger(_FakeTargetedMerger(3))

    tracked.merge("a", "b")
    tracked.merge("a", "b", feedback="try again")

    assert tracked.calls == 2
    assert tracked.jobs_attempted == 1
    assert tracked.retries == 1
