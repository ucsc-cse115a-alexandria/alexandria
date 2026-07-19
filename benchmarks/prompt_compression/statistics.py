from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence


class PairedEstimate(BaseModel):
    """Paired score retention and a reproducible percentile-bootstrap interval."""

    model_config = ConfigDict(frozen=True)
    n_cases: int = Field(ge=1)
    original_score: float = Field(ge=0.0, le=1.0)
    compressed_score: float = Field(ge=0.0, le=1.0)
    score_change_pp: float
    retention: float | None = Field(default=None, ge=0.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    confidence_low: float | None = Field(default=None, ge=0.0)
    confidence_high: float | None = Field(default=None, ge=0.0)
    bootstrap_samples: int = Field(ge=0)
    bootstrap_seed: int
    release_threshold: float = Field(ge=0.0)
    clears_release_threshold: bool


def paired_score_bootstrap(
    original_scores: Sequence[float | bool],
    compressed_scores: Sequence[float | bool],
    *,
    samples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
    release_threshold: float = 0.90,
) -> PairedEstimate:
    """Bootstrap paired case means and gate compressed/original score retention."""
    if len(original_scores) != len(compressed_scores):
        raise ValueError("original and compressed scores must contain the same number of paired cases")
    if not original_scores:
        raise ValueError("at least one paired case is required")
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    original = np.asarray(original_scores, dtype=np.float64)
    compressed = np.asarray(compressed_scores, dtype=np.float64)
    if np.any((original < 0.0) | (original > 1.0)) or np.any((compressed < 0.0) | (compressed > 1.0)):
        raise ValueError("scores must be between 0 and 1")
    original_score = float(np.mean(original))
    compressed_score = float(np.mean(compressed))
    if original_score == 0.0:
        return PairedEstimate(
            n_cases=len(original_scores),
            original_score=original_score,
            compressed_score=compressed_score,
            score_change_pp=(compressed_score - original_score) * 100.0,
            confidence_level=confidence_level,
            bootstrap_samples=0,
            bootstrap_seed=seed,
            release_threshold=release_threshold,
            clears_release_threshold=False,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(original), size=(samples, len(original)))
    original_means = np.mean(original[indices], axis=1)
    compressed_means = np.mean(compressed[indices], axis=1)
    valid = original_means > 0.0
    ratios = compressed_means[valid] / original_means[valid]
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(ratios, [tail, 1.0 - tail])
    return PairedEstimate(
        n_cases=len(original_scores),
        original_score=original_score,
        compressed_score=compressed_score,
        score_change_pp=(compressed_score - original_score) * 100.0,
        retention=compressed_score / original_score,
        confidence_level=confidence_level,
        confidence_low=float(low),
        confidence_high=float(high),
        bootstrap_samples=int(ratios.size),
        bootstrap_seed=seed,
        release_threshold=release_threshold,
        clears_release_threshold=float(low) >= release_threshold,
    )
