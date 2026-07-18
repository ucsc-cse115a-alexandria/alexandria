from __future__ import annotations

import pytest

from benchmarks.babilong_8k.statistics import paired_retention_bootstrap


def test_identical_correct_outcomes_clear_the_retention_gate() -> None:
    estimate = paired_retention_bootstrap([True] * 10, [True] * 10, samples=1_000, seed=7)

    assert estimate.retention == 1.0
    assert estimate.confidence_low == 1.0
    assert estimate.confidence_high == 1.0
    assert estimate.clears_release_threshold


def test_bootstrap_is_paired_reproducible_and_fails_when_the_lower_bound_is_too_low() -> None:
    original = [True, True, True, True, False]
    compressed = [True, True, False, False, True]

    first = paired_retention_bootstrap(original, compressed, samples=2_000, seed=11)
    second = paired_retention_bootstrap(original, compressed, samples=2_000, seed=11)

    assert first == second
    assert first.retention == pytest.approx(0.75)
    assert first.accuracy_change_pp == pytest.approx(-20.0)
    assert first.confidence_low < first.release_threshold
    assert not first.clears_release_threshold


def test_bootstrap_requires_comparable_nonempty_outcomes() -> None:
    with pytest.raises(ValueError, match="same number"):
        paired_retention_bootstrap([True], [True, False])
    with pytest.raises(ValueError, match="at least one"):
        paired_retention_bootstrap([], [])
    with pytest.raises(ValueError, match="original accuracy is zero"):
        paired_retention_bootstrap([False], [True])
