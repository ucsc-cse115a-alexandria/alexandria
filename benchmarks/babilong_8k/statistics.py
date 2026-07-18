from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

RELEASE_RETENTION_THRESHOLD = 0.90
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95


class RetentionEstimate(BaseModel):
    """Paired accuracy-retention estimate and its reproducible percentile bootstrap interval."""

    model_config = ConfigDict(frozen=True)
    n_cases: int = Field(ge=1)
    original_accuracy: float = Field(ge=0.0, le=1.0)
    compressed_accuracy: float = Field(ge=0.0, le=1.0)
    accuracy_change_pp: float
    retention: float = Field(ge=0.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    confidence_low: float = Field(ge=0.0)
    confidence_high: float = Field(ge=0.0)
    bootstrap_samples: int = Field(ge=1)
    bootstrap_seed: int
    release_threshold: float = Field(ge=0.0)
    clears_release_threshold: bool


def paired_retention_bootstrap(
    original_correct: Sequence[bool],
    compressed_correct: Sequence[bool],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    release_threshold: float = RELEASE_RETENTION_THRESHOLD,
) -> RetentionEstimate:
    """Bootstrap paired cases and gate on the lower bound of compressed/original accuracy retention."""
    if len(original_correct) != len(compressed_correct):
        raise ValueError("original and compressed outcomes must contain the same number of paired cases")
    if not original_correct:
        raise ValueError("at least one paired case is required")
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if release_threshold < 0.0:
        raise ValueError("release_threshold must be non-negative")

    original = np.asarray(original_correct, dtype=np.float64)
    compressed = np.asarray(compressed_correct, dtype=np.float64)
    original_accuracy = float(np.mean(original))
    compressed_accuracy = float(np.mean(compressed))
    if original_accuracy == 0.0:
        raise ValueError("accuracy retention is undefined when original accuracy is zero")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(original), size=(samples, len(original)))
    original_counts = np.sum(original[indices], axis=1)
    compressed_counts = np.sum(compressed[indices], axis=1)
    valid = original_counts > 0
    bootstrap_retention = compressed_counts[valid] / original_counts[valid]
    if not bootstrap_retention.size:  # pragma: no cover - guarded by non-zero observed accuracy
        raise ValueError("bootstrap produced no samples with non-zero original accuracy")

    tail = (1.0 - confidence_level) / 2.0
    confidence_low, confidence_high = np.quantile(bootstrap_retention, [tail, 1.0 - tail])
    retention = compressed_accuracy / original_accuracy
    return RetentionEstimate(
        n_cases=len(original_correct),
        original_accuracy=original_accuracy,
        compressed_accuracy=compressed_accuracy,
        accuracy_change_pp=(compressed_accuracy - original_accuracy) * 100.0,
        retention=retention,
        confidence_level=confidence_level,
        confidence_low=float(confidence_low),
        confidence_high=float(confidence_high),
        bootstrap_samples=int(bootstrap_retention.size),
        bootstrap_seed=seed,
        release_threshold=release_threshold,
        clears_release_threshold=float(confidence_low) >= release_threshold,
    )
