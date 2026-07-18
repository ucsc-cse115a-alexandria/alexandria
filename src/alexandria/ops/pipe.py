from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import Candidate, Diff, MergeMetrics, Params, Plan, Replace, TrackedMerger
from alexandria.ir.document import Document, Encoded, Section, Sentence
from alexandria.ir.registry import required_scorers
from alexandria.ir.similarity import normalize
from alexandria.ops.features.diff import diffs
from alexandria.ops.features.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import DEFAULT_SCORER, score, score_rows
from alexandria.ops.features.select import DEFAULT_SELECTOR, select
from alexandria.utils.embedders import default_embedder
from alexandria.utils.merger import default_merger
from alexandria.utils.tokens import count_tokens

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder, SentenceMerger

MAX_TARGET_MERGE_ATTEMPTS = 5


class ReduceResult(BaseModel):
    """The outcome of a reduction: the reduced Document, its source, and the applied candidates."""

    model_config = ConfigDict(frozen=True)
    document: Document
    source: Document
    applied: Plan
    merge_metrics: MergeMetrics = MergeMetrics()

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def source_tokens(self) -> int:
        return self.source.token_count

    @property
    def reduced_tokens(self) -> int:
        return self.document.token_count


class TargetMergeError(ValueError):
    """A strict target failure with machine-readable merge usage and best observed quality."""

    def __init__(self, message: str, metrics: MergeMetrics, *, best_tokens: int, best_drift: float) -> None:
        super().__init__(message)
        self.metrics = metrics
        self.best_tokens = best_tokens
        self.best_drift = best_drift


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
    merger: SentenceMerger | None = None,
    *,
    api_key: str | None = None,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    selector: str = DEFAULT_SELECTOR,
    params: Params | None = None,
) -> ReduceResult:
    """Run represent → score → optimize → select end to end and return the reduction.

    When embedder or merger is omitted, the OpenAI defaults are built lazily (requires an API
    key: pass api_key, export OPENAI_API_KEY, or run `alexandria config set openai-api-key`).
    """
    embedder = embedder if embedder is not None else default_embedder(api_key)
    merger = merger if merger is not None else default_merger(api_key)
    tracked_merger = TrackedMerger(merger)
    document = represent(prompt, embedder)
    if params is not None and params.require_target and params.max_tokens is not None:
        reduced, applied_groups = _merge_to_target(document, embedder, tracked_merger, params)
        return ReduceResult(
            document=reduced,
            source=document,
            applied=(),
            merge_metrics=tracked_merger.metrics(
                proposed_edits=applied_groups,
                applied_edits=applied_groups,
            ),
        )
    scores = score(document, names=_required_scorers(optimizers))
    plan = optimize(document, scores, embedder, tracked_merger, names=optimizers, params=params)
    selection = select(document, plan, embedder, selector, params=params)
    if params is not None and params.max_tokens is not None and selection.document.token_count > params.max_tokens:
        # A target-sized proposal can still miss after cumulative drift checks. Resume exhaustively,
        # reusing every prior merger response from TrackedMerger instead of paying for the same call twice.
        exhaustive = params.model_copy(update={"max_tokens": None, "require_target": False})
        plan = optimize(document, scores, embedder, tracked_merger, names=optimizers, params=exhaustive)
        selection = select(document, plan, embedder, selector, params=params)
    return ReduceResult(
        document=selection.document,
        source=document,
        applied=selection.applied,
        merge_metrics=tracked_merger.metrics(proposed_edits=len(plan), applied_edits=len(selection.applied)),
    )


def _merge_to_target(
    document: Document, embedder: Embedder, merger: TrackedMerger, params: Params
) -> tuple[Document, int]:
    """Merge the largest content groups until token, structure, and embedding constraints pass."""
    if params.max_tokens is None:
        raise ValueError("target merge requires max_tokens")
    expected_structure = tuple(sentence.text.strip() for sentence in document.sentences if not sentence.optimizable)
    min_document_tokens = max(1, math.floor(params.max_tokens * 0.95))
    current = document
    applied_groups = 0
    best_tokens = document.token_count
    best_drift = float("inf")
    last_issues: list[str] = []
    while current.token_count > params.max_tokens:
        groups = _compressible_groups(current)
        if not groups:
            metrics = merger.metrics(proposed_edits=applied_groups, applied_edits=applied_groups)
            raise TargetMergeError(
                f"target merge cannot continue at {current.token_count} tokens: "
                "no multi-sentence content group remains",
                metrics,
                best_tokens=best_tokens,
                best_drift=best_drift,
            )
        group = max(groups, key=lambda sentences: sum(sentence.token_count for sentence in sentences))
        group_tokens = sum(sentence.token_count for sentence in group)
        fixed_tokens = current.token_count - group_tokens
        group_max_tokens = max(1, params.max_tokens - fixed_tokens)
        group_min_tokens = max(1, min_document_tokens - fixed_tokens)
        feedback: str | None = None
        accepted: Document | None = None
        for _ in range(MAX_TARGET_MERGE_ATTEMPTS):
            merged = merger.merge_to_target("".join(sentence.text for sentence in group), group_max_tokens, feedback)
            merged = merged.rstrip() + group[0].text[len(group[0].text.rstrip()) :]
            merged_tokens = count_tokens(merged)
            issues: list[str] = []
            if merged_tokens > group_max_tokens:
                issues.append(f"output used {merged_tokens} tokens; the hard limit is {group_max_tokens}")
            elif fixed_tokens + merged_tokens < min_document_tokens:
                issues.append(
                    f"output used only {merged_tokens} tokens; "
                    f"use the available budget of {group_min_tokens}-{group_max_tokens}"
                )
            replacement = Encoded(
                text=merged,
                token_count=merged_tokens,
                embedding=embedder.embed([merged])[0],
            )
            edit = Candidate(
                edit=Replace(targets=tuple(sentence.id for sentence in group), replacement=replacement),
                confidence=1.0,
                source="target_merge",
                reason=f"merged a content group to meet the {params.max_tokens}-token target",
            )
            trial = current.apply(edit)
            candidate = represent(trial.text, embedder) if trial is not None else None
            if candidate is None:
                issues.append("merge would empty a structural section")
            else:
                actual_structure = tuple(
                    sentence.text.strip() for sentence in candidate.sentences if not sentence.optimizable
                )
                if actual_structure != expected_structure:
                    issues.append("XML/Markdown boundary lines changed")
                drift = 1.0 - float(np.dot(normalize(candidate.embedding), normalize(document.embedding)))
                best_tokens = min(best_tokens, candidate.token_count)
                best_drift = min(best_drift, drift)
                if drift > params.drift_budget:
                    issues.append(
                        f"whole-prompt embedding drift was {drift:.4f}; maximum is {params.drift_budget:.4f}"
                    )
                if not issues:
                    accepted = candidate
                    break
            last_issues = issues
            feedback = "The previous compression was rejected: " + "; ".join(issues) + ". Correct every issue."
        if accepted is None:
            break
        current = accepted
        applied_groups += 1
    if min_document_tokens <= current.token_count <= params.max_tokens:
        return current, applied_groups
    details = "; ".join(last_issues) or "no acceptable output"
    metrics = merger.metrics(proposed_edits=applied_groups, applied_edits=applied_groups)
    raise TargetMergeError(
        f"target merge failed after {merger.calls} calls ({merger.retries} retries): {details}; "
        f"best output was {best_tokens} tokens with {best_drift:.4f} embedding drift",
        metrics,
        best_tokens=best_tokens,
        best_drift=best_drift,
    )


def _compressible_groups(document: Document) -> list[tuple[Sentence, ...]]:
    groups: list[tuple[Sentence, ...]] = []

    def walk(section: Section) -> None:
        direct = tuple(child for child in section.children if isinstance(child, Sentence) and child.optimizable)
        if len(direct) >= 2:
            groups.append(direct)
        for child in section.children:
            if isinstance(child, Section):
                walk(child)

    for section in document.sections:
        walk(section)
    return groups


class Proposal(BaseModel):
    """The reviewable form of a reduction: the source Document and its proposed edits as diffs."""

    model_config = ConfigDict(frozen=True)
    document: Document
    diffs: tuple[Diff, ...]


def propose(
    prompt: str,
    embedder: Embedder | None = None,
    merger: SentenceMerger | None = None,
    *,
    api_key: str | None = None,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    params: Params | None = None,
) -> Proposal:
    """Run represent → score → optimize → diffs, stopping before selection.

    When embedder or merger is omitted, the OpenAI defaults are built lazily (requires an API
    key: pass api_key, export OPENAI_API_KEY, or run `alexandria config set openai-api-key`).
    """
    embedder = embedder if embedder is not None else default_embedder(api_key)
    merger = merger if merger is not None else default_merger(api_key)
    document = represent(prompt, embedder)
    scores = score(document, names=_required_scorers(optimizers))
    plan = optimize(document, scores, embedder, merger, names=optimizers, params=params)
    return Proposal(document=document, diffs=diffs(document, plan))


def optimization_report(
    prompt: str,
    embedder: Embedder | None = None,
    merger: SentenceMerger | None = None,
    *,
    api_key: str | None = None,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    selector: str = DEFAULT_SELECTOR,
    params: Params | None = None,
) -> OptimizationReport:
    """Run the optimizer and summarize compression plus instruction fidelity as JSON-ready data.

    The reduced prompt is represented again so its instruction embeddings are fresh; ``Document.apply``
    intentionally keeps pre-edit embeddings because the IR itself has no model access.
    """
    embedder = embedder if embedder is not None else default_embedder(api_key)
    merger = merger if merger is not None else default_merger(api_key)
    params = params or Params()
    result = reduce(prompt, embedder, merger, optimizers=optimizers, selector=selector, params=params)
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
        raise ValueError("baseline source token count does not match; regenerate the baseline for this fixture prompt")

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

    When embedder is omitted, the OpenAI default is built lazily (requires an API key: export
    OPENAI_API_KEY or run `alexandria config set openai-api-key`).
    """
    embedder = embedder if embedder is not None else default_embedder()
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
