from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import Params, Plan
from alexandria.ir.document import Document
from alexandria.ir.registry import required_scorers
from alexandria.ir.similarity import normalize
from alexandria.ops.features.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import DEFAULT_SCORER, score, score_rows
from alexandria.ops.features.select import DEFAULT_SELECTOR, select
from alexandria.utils.embedders import default_embedder

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder


class ReduceResult(BaseModel):
    """The outcome of a reduction: the reduced Document, its source, and the applied candidates."""

    model_config = ConfigDict(frozen=True)
    document: Document
    source: Document
    applied: Plan

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def source_tokens(self) -> int:
        return self.source.token_count

    @property
    def reduced_tokens(self) -> int:
        return self.document.token_count


class ReportConfig(BaseModel):
    """The settings that produced an optimization report."""

    model_config = ConfigDict(frozen=True)
    model: str
    optimizers: tuple[str, ...]
    selector: str
    threshold: float
    drift_budget: float


class TokenMetrics(BaseModel):
    """Token counts and their derived reduction metrics."""

    model_config = ConfigDict(frozen=True)
    source: int
    reduced: int
    saved: int
    reduction_ratio: float


class QualityScores(BaseModel):
    """Instruction-preservation scores for the reduced prompt; larger is better."""

    model_config = ConfigDict(frozen=True)
    instruction_coverage: float
    minimum_instruction_similarity: float


class OptimizationReport(BaseModel):
    """Stable, machine-readable summary of one end-to-end optimization run."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = 1
    config: ReportConfig
    tokens: TokenMetrics
    quality: QualityScores
    applied_edits: int


class Regression(BaseModel):
    """One metric that moved beyond its allowed baseline tolerance."""

    model_config = ConfigDict(frozen=True)
    metric: str
    baseline: float
    current: float
    tolerance: float
    expected: Literal["at_least", "at_most"]


class ReportComparison(BaseModel):
    """Result of checking a current report against a compatible baseline."""

    model_config = ConfigDict(frozen=True)
    passed: bool
    regressions: tuple[Regression, ...]


def reduce(
    prompt: str,
    embedder: Embedder | None = None,
    *,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    selector: str = DEFAULT_SELECTOR,
    params: Params | None = None,
) -> ReduceResult:
    """Run represent → score → optimize → select end to end and return the reduction.

    When embedder is omitted, the default all-MiniLM-L6-v2 model is downloaded and built on first use.
    """
    embedder = embedder if embedder is not None else default_embedder()
    document = represent(prompt, embedder)
    scores = score(document, names=_required_scorers(optimizers))
    plan = optimize(document, scores, names=optimizers, params=params)
    selection = select(document, plan, embedder, selector, params=params)
    return ReduceResult(document=selection.document, source=document, applied=selection.applied)


def optimization_report(
    prompt: str,
    embedder: Embedder | None = None,
    *,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    selector: str = DEFAULT_SELECTOR,
    params: Params | None = None,
) -> OptimizationReport:
    """Run the optimizer and summarize compression plus instruction fidelity as JSON-ready data.

    The reduced prompt is represented again so its instruction embeddings are fresh; ``Document.apply``
    intentionally keeps pre-edit embeddings because the IR itself has no model access.
    """
    embedder = embedder if embedder is not None else default_embedder()
    params = params or Params()
    result = reduce(prompt, embedder, optimizers=optimizers, selector=selector, params=params)
    reduced = represent(result.text, embedder)

    saved = result.source_tokens - result.reduced_tokens
    ratio = saved / result.source_tokens if result.source_tokens else 0.0
    instruction_coverage, minimum_similarity = _instruction_quality(result.source, reduced)

    return OptimizationReport(
        config=ReportConfig(
            model=embedder.model_id,
            optimizers=optimizers,
            selector=selector,
            threshold=params.threshold,
            drift_budget=params.drift_budget,
        ),
        tokens=TokenMetrics(
            source=result.source_tokens,
            reduced=result.reduced_tokens,
            saved=saved,
            reduction_ratio=_round_score(ratio),
        ),
        quality=QualityScores(
            instruction_coverage=instruction_coverage,
            minimum_instruction_similarity=minimum_similarity,
        ),
        applied_edits=len(result.applied),
    )


def compare_reports(
    current: OptimizationReport,
    baseline: OptimizationReport,
    *,
    quality_tolerance: float = 0.0,
    token_tolerance: int = 0,
) -> ReportComparison:
    """Flag worse compression or fidelity relative to a compatible baseline report."""
    if quality_tolerance < 0.0:
        raise ValueError("quality_tolerance must be non-negative")
    if token_tolerance < 0:
        raise ValueError("token_tolerance must be non-negative")
    if current.config != baseline.config:
        raise ValueError("baseline config does not match the current report config")
    if current.tokens.source != baseline.tokens.source:
        raise ValueError(
            "baseline source token count does not match; regenerate the baseline for this fixture prompt"
        )

    regressions: list[Regression] = []
    _append_max_regression(
        regressions,
        "tokens.reduced",
        current.tokens.reduced,
        baseline.tokens.reduced,
        float(token_tolerance),
    )
    _append_min_regression(
        regressions,
        "quality.instruction_coverage",
        current.quality.instruction_coverage,
        baseline.quality.instruction_coverage,
        quality_tolerance,
    )
    _append_min_regression(
        regressions,
        "quality.minimum_instruction_similarity",
        current.quality.minimum_instruction_similarity,
        baseline.quality.minimum_instruction_similarity,
        quality_tolerance,
    )
    return ReportComparison(passed=not regressions, regressions=tuple(regressions))


def score_report(
    prompt: str, embedder: Embedder | None = None, *, scorers: tuple[str, ...] = (DEFAULT_SCORER,)
) -> list[dict[str, object]]:
    """Represent then score into display rows: id, text, each scorer's value, and its peer (if any).

    When embedder is omitted, the default all-MiniLM-L6-v2 model is downloaded and built on first use.
    """
    document = represent(prompt, embedder)
    return score_rows(document, score(document, names=scorers), scorers)


def _required_scorers(optimizers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(scorer for o in optimizers for scorer in required_scorers(o)))


def _instruction_quality(source: Document, reduced: Document) -> tuple[float, float]:
    """Return token-weighted mean and worst best-match similarity for source instructions."""
    source_vectors = normalize(np.stack([sentence.embedding for sentence in source.sentences]))
    reduced_vectors = normalize(np.stack([sentence.embedding for sentence in reduced.sentences]))
    best_matches = np.max(source_vectors @ reduced_vectors.T, axis=1)
    best_matches = np.clip(best_matches, -1.0, 1.0)
    weights = np.asarray([sentence.token_count for sentence in source.sentences], dtype=np.float64)
    mean = float(np.average(best_matches, weights=weights)) if weights.sum() else float(best_matches.mean())
    return _round_score(mean), _round_score(float(best_matches.min()))


def _round_score(value: float) -> float:
    return round(value, 6)


def _append_min_regression(
    regressions: list[Regression], metric: str, current: float, baseline: float, tolerance: float
) -> None:
    if current < baseline - tolerance:
        regressions.append(
            Regression(
                metric=metric,
                baseline=baseline,
                current=current,
                tolerance=tolerance,
                expected="at_least",
            )
        )


def _append_max_regression(
    regressions: list[Regression], metric: str, current: float, baseline: float, tolerance: float
) -> None:
    if current > baseline + tolerance:
        regressions.append(
            Regression(
                metric=metric,
                baseline=baseline,
                current=current,
                tolerance=tolerance,
                expected="at_most",
            )
        )
