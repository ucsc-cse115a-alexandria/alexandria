from __future__ import annotations

import pytest

from benchmarks.prompt_compression.statistics import paired_score_bootstrap


def test_paired_score_bootstrap_is_reproducible() -> None:
    original = [1.0, 1.0, 0.0, 0.5]
    compressed = [1.0, 0.5, 0.5, 0.5]
    first = paired_score_bootstrap(original, compressed, samples=2_000, seed=7)
    second = paired_score_bootstrap(original, compressed, samples=2_000, seed=7)
    assert first == second
    assert first.original_score == first.compressed_score
    assert first.retention == 1.0


def test_paired_score_bootstrap_handles_zero_original_score() -> None:
    estimate = paired_score_bootstrap([0.0, 0.0], [1.0, 0.0], samples=100)
    assert estimate.retention is None
    assert estimate.bootstrap_samples == 0
    assert not estimate.clears_release_threshold


def test_paired_score_bootstrap_rejects_unpaired_or_out_of_range_scores() -> None:
    with pytest.raises(ValueError, match="same number"):
        paired_score_bootstrap([1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="between 0 and 1"):
        paired_score_bootstrap([2.0], [1.0])
