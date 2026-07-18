from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import (
    Candidate,
    Diff,
    MergeMetrics,
    Params,
    Plan,
    Replace,
    TargetMergeRoundMetrics,
    TrackedMerger,
)
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

MAX_TARGET_MERGE_ROUNDS = 5
TARGET_REFINEMENT_ROUNDS = 2
_GENERATED_MARKUP = re.compile(r"(?m)^\s*(?:#{1,6}\s+\S|</?[A-Za-z][\w.-]*>)\s*$")


@dataclass(frozen=True)
class _TargetCandidate:
    text: str
    document: Document
    drift: float
    minimum_coverage: float
    target_distance: int
    structure_valid: bool
    issues: tuple[str, ...]


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
    last_issues: tuple[str, ...] = ()
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
        full_group = max(groups, key=lambda sentences: sum(sentence.token_count for sentence in sentences))
        required_savings = current.token_count - params.max_tokens
        group = _target_merge_window(full_group, required_savings, document.embedding)
        group_tokens = sum(sentence.token_count for sentence in group)
        fixed_tokens = current.token_count - group_tokens
        group_max_tokens = max(1, params.max_tokens - fixed_tokens)
        group_min_tokens = max(1, min_document_tokens - fixed_tokens)
        source_segment = "".join(sentence.text for sentence in group)
        feedback: str | None = None
        accepted: Document | None = None
        base: _TargetCandidate | None = None
        target_reached_round: int | None = None
        for round_number in range(1, MAX_TARGET_MERGE_ROUNDS + 1):
            if base is not None and base.document.token_count > params.max_tokens:
                base_is_on_target = False
                overshoot = base.document.token_count - params.max_tokens
                round_max_tokens = max(1, group_max_tokens - overshoot - 8)
            elif base is not None and base.target_distance == 0:
                base_is_on_target = True
                round_max_tokens = min(group_max_tokens, count_tokens(base.text))
            else:
                base_is_on_target = False
                round_max_tokens = group_max_tokens
            generated = merger.merge_candidates_to_target(
                source_segment,
                round_max_tokens,
                feedback,
                base.text if base is not None else None,
            )
            evaluated = _evaluate_target_candidates(
                generated,
                current=current,
                source=document,
                group=group,
                embedder=embedder,
                min_document_tokens=min_document_tokens,
                max_document_tokens=params.max_tokens,
                group_min_tokens=group_min_tokens,
                group_max_tokens=group_max_tokens,
                drift_budget=params.drift_budget,
                expected_structure=expected_structure,
            )
            selectable = [candidate for candidate in evaluated if candidate.structure_valid]
            generated_best = min(selectable, key=_target_candidate_rank) if selectable else base
            if base_is_on_target and base is not None:
                selectable = [
                    candidate
                    for candidate in selectable
                    if candidate.target_distance == 0 and candidate.document.token_count <= base.document.token_count
                ]
            previous = base
            pool = [*([base] if base is not None else []), *selectable]
            if not pool:
                last_issues = ("the model returned no new non-empty candidates",)
                feedback = "Return exactly 10 distinct, non-empty candidates within the required token range."
                continue
            base = min(pool, key=_target_candidate_rank)
            improved = (
                previous is not None
                and base.document.token_count <= previous.document.token_count
                and base.drift < previous.drift
            )
            merger.record_target_round(
                TargetMergeRoundMetrics(
                    round=round_number,
                    base_tokens=current.token_count if previous is None else previous.document.token_count,
                    selected_tokens=base.document.token_count,
                    base_drift=0.0 if previous is None else previous.drift,
                    selected_drift=base.drift,
                    generated_best_tokens=(
                        generated_best.document.token_count
                        if generated_best is not None
                        else base.document.token_count
                    ),
                    generated_best_drift=generated_best.drift if generated_best is not None else base.drift,
                    improved=improved,
                )
            )
            best_tokens = base.document.token_count
            best_drift = base.drift
            last_issues = base.issues
            if base.target_distance == 0 and base.drift <= params.drift_budget:
                accepted = base.document
                break
            if base.target_distance == 0:
                target_reached_round = target_reached_round or round_number
                if round_number - target_reached_round >= TARGET_REFINEMENT_ROUNDS:
                    break
            feedback = _target_feedback(base, group, embedder)
        if accepted is None:
            break
        current = accepted
        applied_groups += 1
    if min_document_tokens <= current.token_count <= params.max_tokens:
        return current, applied_groups
    details = "; ".join(last_issues) or "no acceptable output"
    metrics = merger.metrics(proposed_edits=applied_groups, applied_edits=applied_groups)
    round_trace = ", ".join(
        f"r{round_metrics.round} base {round_metrics.base_tokens}/{round_metrics.base_drift:.4f}, "
        f"generated {round_metrics.generated_best_tokens}/{round_metrics.generated_best_drift:.4f}, "
        f"selected {round_metrics.selected_tokens}/{round_metrics.selected_drift:.4f}"
        for round_metrics in metrics.target_rounds
    )
    raise TargetMergeError(
        f"target merge failed after {merger.calls} calls ({merger.retries} retries): {details}; "
        f"best output was {best_tokens} tokens with {best_drift:.4f} embedding drift; rounds: {round_trace}",
        metrics,
        best_tokens=best_tokens,
        best_drift=best_drift,
    )


def _evaluate_target_candidates(
    generated: tuple[str, ...],
    *,
    current: Document,
    source: Document,
    group: tuple[Sentence, ...],
    embedder: Embedder,
    min_document_tokens: int,
    max_document_tokens: int,
    group_min_tokens: int,
    group_max_tokens: int,
    drift_budget: float,
    expected_structure: tuple[str, ...],
) -> tuple[_TargetCandidate, ...]:
    """Embed one generation round in batches and score every unique candidate."""
    suffix = group[0].text[len(group[0].text.rstrip()) :]
    texts = tuple(dict.fromkeys(candidate.rstrip() + suffix for candidate in generated if candidate.strip()))
    if not texts:
        return ()
    replacement_embeddings = embedder.embed(list(texts))
    trials: list[tuple[str, Document, int, bool, np.ndarray]] = []
    targets = tuple(sentence.id for sentence in group)
    for text, embedding in zip(texts, replacement_embeddings, strict=True):
        merged_tokens = count_tokens(text)
        replacement = Encoded(text=text, token_count=merged_tokens, embedding=embedding)
        edit = Candidate(
            edit=Replace(targets=targets, replacement=replacement),
            confidence=1.0,
            source="target_merge",
            reason=f"merged a content group to meet the {max_document_tokens}-token target",
        )
        trial = current.apply(edit)
        if trial is None:
            continue
        structure_valid = not _GENERATED_MARKUP.search(text)
        trials.append((text, trial, merged_tokens, structure_valid, embedding))
    if not trials:
        return ()
    whole_embeddings = embedder.embed([trial.text for _, trial, _, _, _ in trials])
    source_group_embeddings = np.stack([normalize(sentence.embedding) for sentence in group])
    evaluated: list[_TargetCandidate] = []
    for (text, trial, merged_tokens, structure_valid, replacement_embedding), whole_embedding in zip(
        trials, whole_embeddings, strict=True
    ):
        candidate = trial.model_copy(update={"embedding": whole_embedding})
        drift = max(0.0, 1.0 - float(np.dot(normalize(whole_embedding), normalize(source.embedding))))
        coverage = float(np.min(source_group_embeddings @ normalize(replacement_embedding)))
        if candidate.token_count < min_document_tokens:
            target_distance = min_document_tokens - candidate.token_count
        elif candidate.token_count > max_document_tokens:
            target_distance = candidate.token_count - max_document_tokens
        else:
            target_distance = 0
        issues: list[str] = []
        if merged_tokens > group_max_tokens:
            issues.append(f"output used {merged_tokens} tokens; the hard limit is {group_max_tokens}")
        elif candidate.token_count < min_document_tokens:
            issues.append(
                f"output used only {merged_tokens} tokens; "
                f"use the available budget of {group_min_tokens}-{group_max_tokens}"
            )
        if not structure_valid:
            issues.append("candidate introduced an XML or Markdown boundary line")
        actual_structure = tuple(sentence.text.strip() for sentence in candidate.sentences if not sentence.optimizable)
        if actual_structure != expected_structure:
            structure_valid = False
            issues.append("XML/Markdown boundary lines changed")
        if drift > drift_budget:
            issues.append(f"whole-prompt embedding drift was {drift:.4f}; maximum is {drift_budget:.4f}")
        evaluated.append(
            _TargetCandidate(
                text=text,
                document=candidate,
                drift=drift,
                minimum_coverage=coverage,
                target_distance=target_distance,
                structure_valid=structure_valid,
                issues=tuple(issues),
            )
        )
    return tuple(evaluated)


def _target_candidate_rank(candidate: _TargetCandidate) -> tuple[bool, int, float, float]:
    return (
        not candidate.structure_valid,
        candidate.target_distance,
        candidate.drift,
        -candidate.minimum_coverage,
    )


def _target_feedback(candidate: _TargetCandidate, group: tuple[Sentence, ...], embedder: Embedder) -> str:
    source_embeddings = np.stack([normalize(sentence.embedding) for sentence in group])
    replacement = normalize(
        next(sentence.embedding for sentence in candidate.document.sentences if sentence.text == candidate.text)
    )
    similarities = source_embeddings @ replacement
    least_covered = np.argsort(similarities)[:10]
    source_excerpts = [" ".join(group[int(index)].text.split())[:220] for index in least_covered]
    base_chunks = _feedback_chunks(candidate.text)
    base_embeddings = np.stack([normalize(vector) for vector in embedder.embed(list(base_chunks))])
    least_representative = np.argsort(base_embeddings @ replacement)[:10]
    base_excerpts = [" ".join(base_chunks[int(index)].split())[:220] for index in least_representative]
    problems = "; ".join(candidate.issues) or "improve semantic coverage while preserving the token range"
    rendered_swaps = "\n".join(
        f'{index}. Replace base passage "{base_excerpt}" with exact wording and meaning from source passage '
        f'"{source_excerpt}".'
        for index, (base_excerpt, source_excerpt) in enumerate(
            zip(base_excerpts, source_excerpts, strict=False), start=1
        )
        if base_excerpt and source_excerpt
    )
    return (
        f"The single current base still failed: {problems}. Its whole-prompt drift is {candidate.drift:.4f}, and its "
        f"minimum source-line similarity is {candidate.minimum_coverage:.4f}. Do not rewrite the whole base. Copy it "
        "nearly verbatim and make exactly one localized substitution per candidate using the correspondingly numbered "
        "swap below. Keep the substitution approximately token-neutral. Candidate N must perform only swap N.\n"
        f"{rendered_swaps}\nEvery revision must use no more than {count_tokens(candidate.text)} tokens."
    )


def _feedback_chunks(text: str) -> tuple[str, ...]:
    chunks = tuple(chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", text) if chunk.strip())
    return chunks if chunks else (text,)


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


def _target_merge_window(
    group: tuple[Sentence, ...], required_savings: int, document_embedding: np.ndarray
) -> tuple[Sentence, ...]:
    """Choose only enough contiguous content to make the requested saving practical.

    Rewriting an entire long context to save 10% makes each generated candidate unnecessarily huge. A window around
    twice the required saving lets the merger compress that window by roughly half. Large targets naturally saturate
    at the full group. Among similarly sized windows, prefer content most representative of the whole document so
    semantic outliers such as sparse task facts are less likely to be touched.
    """
    group_tokens = sum(sentence.token_count for sentence in group)
    desired_tokens = min(group_tokens, max(required_savings + 1, required_savings * 2))
    if desired_tokens >= group_tokens or len(group) == 2:
        return group
    normalized_document = normalize(document_embedding)
    best: tuple[Sentence, ...] | None = None
    best_similarity = float("-inf")
    end = 0
    window_tokens = 0
    for start in range(len(group)):
        while end < len(group) and window_tokens < desired_tokens:
            window_tokens += group[end].token_count
            end += 1
        if end - start >= 2 and window_tokens >= desired_tokens:
            window = group[start:end]
            similarity = sum(
                sentence.token_count * float(np.dot(normalize(sentence.embedding), normalized_document))
                for sentence in window
            ) / window_tokens
            if similarity > best_similarity:
                best = window
                best_similarity = similarity
        window_tokens -= group[start].token_count
        if end <= start + 1:
            end = start + 1
            window_tokens = 0
    return best if best is not None else group


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
