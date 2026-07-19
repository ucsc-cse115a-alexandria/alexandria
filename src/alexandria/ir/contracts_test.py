from __future__ import annotations

import pytest

from alexandria.ir.contracts import TargetedMerger, TrackedMerger


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
