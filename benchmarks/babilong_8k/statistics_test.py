from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.babilong_8k.statistics import paired_retention_bootstrap

_COMMITTED_RESULT = Path(__file__).parent / "results/2026-07-18-keep90-hard-target-n100-v1/summary.json"


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


def test_committed_retention_is_reproducible_from_paired_outcomes() -> None:
    payload = json.loads(_COMMITTED_RESULT.read_text(encoding="utf-8"))
    records = payload["records"]

    estimate = paired_retention_bootstrap(
        [record["original_correct"] for record in records],
        [record["compressed_correct"] for record in records],
        samples=payload["quality"]["bootstrap_samples"],
        seed=payload["quality"]["bootstrap_seed"],
        confidence_level=payload["quality"]["confidence_level"],
        release_threshold=payload["quality"]["release_threshold"],
    )

    expected = {field: payload["quality"][field] for field in type(estimate).model_fields}
    assert estimate.model_dump() == expected
    assert payload["target"]["successes"] == len(records)
    assert payload["target"]["failures"] == 0
