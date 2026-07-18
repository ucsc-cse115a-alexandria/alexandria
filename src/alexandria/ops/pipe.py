from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import (
    Candidate,
    Delete,
    Diff,
    MergeMetrics,
    Params,
    Plan,
    Replace,
    TargetMergeRoundMetrics,
    TrackedEmbedder,
    TrackedMerger,
)
from alexandria.ir.document import Document, Encoded, Section, Sentence
from alexandria.ir.registry import required_scorers
from alexandria.ir.similarity import cosine_distance, normalize, similarity_matrix_for
from alexandria.ops.features.diff import diffs
from alexandria.ops.features.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import DEFAULT_SCORER, score, score_rows
from alexandria.ops.features.select import DEFAULT_SELECTOR, select
from alexandria.utils.embedders import default_embedder
from alexandria.utils.merger import TARGET_CANDIDATES_PER_CALL, default_merger
from alexandria.utils.tokens import count_tokens, truncate_tokens

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder, SentenceMerger
    from alexandria.ir.document import SentenceId

MAX_TARGET_MERGE_ROUNDS = 2
TARGET_REFINEMENT_ROUNDS = 1
PRUNE_VERIFY_MAX_STEPS = 12  # drift-search probe cap
TARGET_GENERATION_HEADROOM = 0.0
MAX_REPAIR_VARIANTS = 6
_GENERATED_MARKUP = re.compile(r"(?m)^\s*(?:#{1,6}\s+\S|</?[A-Za-z][\w.-]*>)\s*$")
_REPAIR_BOUNDARY = re.compile(r"(?<=[.!?])(?:[ \t]+|\n+)|\n+")
_CONTENT_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "answer",
        "are",
        "at",
        "be",
        "been",
        "before",
        "did",
        "do",
        "does",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "please",
        "question",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
    }
)


@dataclass(frozen=True)
class _RepairVariant:
    text: str
    repaired_tokens: int


@dataclass(frozen=True)
class _TargetCandidate:
    text: str
    document: Document
    drift: float
    minimum_coverage: float
    target_distance: int
    target_undercut: int
    structure_valid: bool
    issues: tuple[str, ...]
    repaired_tokens: int
    coverage_gaps: tuple[int, ...]


@dataclass(frozen=True)
class _TargetMergeOutcome:
    """A finished target merge: the reduced document plus the work both phases performed."""

    document: Document
    applied_groups: int
    pruned_sentences: int
    pruned_tokens: int
    repaired_tokens: int
    final_drift: float
    drift_budget_met: bool


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


class InfeasibleTargetError(ValueError):
    """The protected prompt structure alone cannot fit within the requested token budget."""


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
    started = time.monotonic()
    embedder = TrackedEmbedder(embedder if embedder is not None else default_embedder(api_key))
    merger = merger if merger is not None else default_merger(api_key)
    tracked_merger = TrackedMerger(merger)
    document = represent(prompt, embedder)
    if params is not None and params.require_target and params.max_tokens is not None:
        try:
            outcome = _merge_to_target(document, embedder, tracked_merger, params)
        except TargetMergeError as error:
            error.metrics = _finalize_metrics(error.metrics, embedder, started)
            raise
        metrics = tracked_merger.metrics(
            proposed_edits=outcome.applied_groups, applied_edits=outcome.applied_groups
        ).model_copy(
            update={
                "pruned_sentences": outcome.pruned_sentences,
                "pruned_tokens": outcome.pruned_tokens,
                "repaired_tokens": outcome.repaired_tokens,
                "final_drift": outcome.final_drift,
                "drift_budget_met": outcome.drift_budget_met,
            }
        )
        return ReduceResult(
            document=outcome.document,
            source=document,
            applied=(),
            merge_metrics=_finalize_metrics(metrics, embedder, started),
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
        merge_metrics=_finalize_metrics(
            tracked_merger.metrics(proposed_edits=len(plan), applied_edits=len(selection.applied)),
            embedder,
            started,
        ),
    )


def _finalize_metrics(metrics: MergeMetrics, embedder: TrackedEmbedder, started: float) -> MergeMetrics:
    """Stamp the reduction-wide embedding counters and wall clock onto merger-built metrics."""
    return metrics.model_copy(
        update={
            "embed_calls": embedder.calls,
            "embed_texts": embedder.texts,
            "elapsed_seconds": time.monotonic() - started,
        }
    )


def _merge_to_target(
    document: Document, embedder: Embedder, merger: TrackedMerger, params: Params
) -> _TargetMergeOutcome:
    """Meet the hard token ceiling first, then minimize drift among feasible candidates."""
    if params.max_tokens is None:
        raise ValueError("target merge requires max_tokens")
    protected_tokens = sum(sentence.token_count for sentence in document.sentences if not sentence.optimizable)
    if protected_tokens > params.max_tokens:
        raise InfeasibleTargetError(
            f"protected structure uses {protected_tokens} tokens, exceeding the {params.max_tokens}-token target"
        )
    expected_structure = tuple(sentence.text.strip() for sentence in document.sentences if not sentence.optimizable)
    preferred_min_tokens = max(1, math.floor(params.max_tokens * 0.95))
    current = _prune_to_target(document, embedder, params, preferred_min_tokens, params.max_tokens)
    pruned_sentences = len(document.sentences) - len(current.sentences)
    pruned_tokens = document.token_count - current.token_count
    repaired_tokens = 0

    def merge_metrics(applied_groups: int, *, final_drift: float | None = None) -> MergeMetrics:
        return merger.metrics(proposed_edits=applied_groups, applied_edits=applied_groups).model_copy(
            update={
                "pruned_sentences": pruned_sentences,
                "pruned_tokens": pruned_tokens,
                "repaired_tokens": repaired_tokens,
                "final_drift": final_drift,
                "drift_budget_met": final_drift <= params.drift_budget if final_drift is not None else None,
            }
        )

    applied_groups = 0
    best_tokens = current.token_count
    best_drift = cosine_distance(current.embedding, document.embedding)
    last_issues: tuple[str, ...] = ()
    while current.token_count > params.max_tokens:
        groups = _compressible_groups(current)
        if not groups:
            raise InfeasibleTargetError(
                f"target cannot be reached from {current.token_count} tokens: no rewritable content remains"
            )
        full_group = max(groups, key=lambda sentences: sum(sentence.token_count for sentence in sentences))
        required_savings = current.token_count - params.max_tokens
        group = _target_merge_window(
            full_group,
            required_savings,
            document.embedding,
            protected_terms=_target_anchor_terms(current, full_group),
        )
        group_tokens = sum(sentence.token_count for sentence in group)
        fixed_tokens = current.token_count - group_tokens
        group_max_tokens = max(1, params.max_tokens - fixed_tokens)
        group_preferred_min_tokens = max(1, preferred_min_tokens - fixed_tokens)
        source_segment = "".join(sentence.text for sentence in group)
        feedback: str | None = None
        base: _TargetCandidate | None = None
        for round_number in range(1, MAX_TARGET_MERGE_ROUNDS + 1):
            if base is None:
                round_max_tokens = max(1, math.floor(group_max_tokens * (1.0 - TARGET_GENERATION_HEADROOM)))
            else:
                round_max_tokens = min(group_max_tokens, count_tokens(base.text))
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
                preferred_min_document_tokens=preferred_min_tokens,
                max_document_tokens=params.max_tokens,
                group_preferred_min_tokens=group_preferred_min_tokens,
                group_max_tokens=group_max_tokens,
                drift_budget=params.drift_budget,
                expected_structure=expected_structure,
            )
            selectable = [
                candidate
                for candidate in evaluated
                if candidate.structure_valid and candidate.document.token_count < current.token_count
            ]
            if not selectable:
                selectable = [
                    candidate
                    for candidate in _evaluate_target_candidates(
                        (source_segment,),
                        current=current,
                        source=document,
                        group=group,
                        embedder=embedder,
                        preferred_min_document_tokens=preferred_min_tokens,
                        max_document_tokens=params.max_tokens,
                        group_preferred_min_tokens=group_preferred_min_tokens,
                        group_max_tokens=group_max_tokens,
                        drift_budget=params.drift_budget,
                        expected_structure=expected_structure,
                    )
                    if candidate.structure_valid and candidate.document.token_count < current.token_count
                ]
            generated_best = min(selectable, key=_target_candidate_rank) if selectable else base
            previous = base
            pool = [*([base] if base is not None else []), *selectable]
            if not pool:
                last_issues = ("the model returned no new non-empty candidates",)
                feedback = (
                    f"Return exactly {TARGET_CANDIDATES_PER_CALL} distinct, non-empty candidates "
                    "that are shorter than the current source segment."
                )
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
            if base.document.token_count > params.max_tokens:
                break
            if base.drift <= params.drift_budget:
                break
            if round_number < MAX_TARGET_MERGE_ROUNDS:  # the final round's feedback would go unread
                feedback = _target_feedback(base, group, group_max_tokens)
        if base is None or base.document.token_count >= current.token_count:
            details = "; ".join(last_issues) or "no shorter structure-valid candidate"
            metrics = merge_metrics(applied_groups, final_drift=best_drift)
            raise TargetMergeError(
                f"target merge made no progress after {merger.calls} calls: {details}",
                metrics,
                best_tokens=best_tokens,
                best_drift=best_drift,
            )
        current = base.document
        repaired_tokens += base.repaired_tokens
        applied_groups += 1
    final_drift = cosine_distance(current.embedding, document.embedding)
    return _TargetMergeOutcome(
        document=current,
        applied_groups=applied_groups,
        pruned_sentences=pruned_sentences,
        pruned_tokens=pruned_tokens,
        repaired_tokens=repaired_tokens,
        final_drift=final_drift,
        drift_budget_met=final_drift <= params.drift_budget,
    )


def _prune_to_target(
    document: Document, embedder: Embedder, params: Params, min_tokens: int, max_tokens: int
) -> Document:
    """Delete redundant sentences toward the token window without any merge-model call.

    The plan orders deletions by descending redundancy; the whole plan is then drift-verified
    with one embedding call. When the full plan drifts over budget, bisect on the deletion count
    (a heuristic, since drift is not monotone in prefix length; every returned prefix is
    verified) and keep the longest passing prefix, falling back to no pruning at all.
    """
    deletions = _prune_plan(document, params.threshold, min_tokens, max_tokens)
    if not deletions:
        return document
    pruned = _apply_prune_prefix(document, deletions, len(deletions))
    pruned_embedding = embedder.embed([pruned.text])[0]
    drift = cosine_distance(pruned_embedding, document.embedding)
    if drift <= params.drift_budget:
        return pruned.model_copy(update={"embedding": pruned_embedding})
    passing, failing = 0, len(deletions)
    passing_document = document
    for _ in range(PRUNE_VERIFY_MAX_STEPS):
        middle = (passing + failing) // 2
        if middle == passing:
            break
        trial = _apply_prune_prefix(document, deletions, middle)
        trial_embedding = embedder.embed([trial.text])[0]
        drift = cosine_distance(trial_embedding, document.embedding)
        if drift <= params.drift_budget:
            passing = middle
            passing_document = trial.model_copy(update={"embedding": trial_embedding})
        else:
            failing = middle
    return passing_document


def _prune_plan(document: Document, threshold: float, min_tokens: int, max_tokens: int) -> tuple[SentenceId, ...]:
    """Deletions, most-redundant first, that approach max_tokens while never undershooting
    min_tokens or emptying a section."""
    sentences = document.sentences
    if len(sentences) < 2 or document.token_count <= max_tokens:
        return ()
    similarity = similarity_matrix_for(document).copy()  # copy: the shared matrix is read-only
    np.fill_diagonal(similarity, -np.inf)
    ancestors, surviving_counts = _sentence_ancestors(document)
    alive = np.ones(len(sentences), dtype=bool)
    remaining = document.token_count
    deletions: list[SentenceId] = []
    while remaining > max_tokens:
        redundancy = np.where(alive[None, :], similarity, -np.inf).max(axis=1)
        chosen: int | None = None
        for index in np.argsort(-redundancy):
            sentence = sentences[int(index)]
            if not alive[index] or not sentence.optimizable:
                continue
            if redundancy[index] < threshold:
                break  # descending order: everything after is below the eligibility floor
            if remaining - sentence.token_count < min_tokens:
                continue  # floor guard: this deletion would undershoot the window
            if any(surviving_counts[ancestor] <= 1 for ancestor in ancestors[sentence.id]):
                continue  # deleting the section's last sentence would empty it
            chosen = int(index)
            break
        if chosen is None:
            break
        alive[chosen] = False
        remaining -= sentences[chosen].token_count
        for ancestor in ancestors[sentences[chosen].id]:
            surviving_counts[ancestor] -= 1
        deletions.append(sentences[chosen].id)
    return tuple(deletions)


def _sentence_ancestors(document: Document) -> tuple[dict[SentenceId, tuple[int, ...]], list[int]]:
    """Each sentence's ancestor-section indices, plus every section's surviving-sentence count."""
    ancestors: dict[SentenceId, tuple[int, ...]] = {}
    counts: list[int] = []

    def walk(section: Section, lineage: tuple[int, ...]) -> None:
        index = len(counts)
        counts.append(len(section.sentences))
        for child in section.children:
            if isinstance(child, Sentence):
                ancestors[child.id] = (*lineage, index)
            else:
                walk(child, (*lineage, index))

    for section in document.sections:
        walk(section, ())
    return ancestors, counts


def _apply_prune_prefix(document: Document, deletions: tuple[SentenceId, ...], count: int) -> Document:
    candidate = Candidate(
        edit=Delete(targets=deletions[:count]),
        confidence=1.0,
        source="target_prune",
        reason="deleted redundant sentences to approach the token target",
    )
    applied = document.apply(candidate)
    if applied is None or applied is document:
        raise ValueError("prune plan produced an inapplicable deletion prefix")
    return applied


def _repair_chunks(text: str) -> tuple[str, ...]:
    chunks: list[str] = []
    start = 0
    for boundary in _REPAIR_BOUNDARY.finditer(text):
        end = boundary.end()
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        elif chunks:
            chunks[-1] += chunk
        start = end
    tail = text[start:]
    if tail.strip():
        chunks.append(tail)
    elif chunks:
        chunks[-1] += tail
    return tuple(chunks) if chunks else (text,)


def _repair_text_variants(text: str, max_tokens: int) -> tuple[_RepairVariant, ...]:
    """Build target-safe alternatives by removing bounded semantic spans, plus an exact token fallback."""
    original_tokens = count_tokens(text)
    if original_tokens <= max_tokens:
        return (_RepairVariant(text=text, repaired_tokens=0),)

    variants: dict[str, _RepairVariant] = {}
    chunks = _repair_chunks(text)
    required_savings = original_tokens - max_tokens
    chunk_tokens = [count_tokens(chunk) for chunk in chunks]
    if len(chunks) > 1:
        for start in range(len(chunks)):
            removed_tokens = 0
            for end in range(start, len(chunks)):
                removed_tokens += chunk_tokens[end]
                if removed_tokens < required_savings:
                    continue
                candidate = "".join((*chunks[:start], *chunks[end + 1 :]))
                if not candidate.strip():
                    break
                candidate_tokens = count_tokens(candidate)
                if candidate_tokens <= max_tokens:
                    variants.setdefault(
                        candidate,
                        _RepairVariant(text=candidate, repaired_tokens=original_tokens - candidate_tokens),
                    )
                    break

    truncated = truncate_tokens(text, max_tokens)
    variants.setdefault(
        truncated,
        _RepairVariant(text=truncated, repaired_tokens=original_tokens - count_tokens(truncated)),
    )
    ordered = sorted(variants.values(), key=lambda variant: (-count_tokens(variant.text), variant.text))
    if len(ordered) <= MAX_REPAIR_VARIANTS:
        return tuple(ordered)

    semantic_slots = MAX_REPAIR_VARIANTS - 1
    sampled = [ordered[index * (len(ordered) - 1) // max(semantic_slots - 1, 1)] for index in range(semantic_slots)]
    if truncated not in {variant.text for variant in sampled}:
        sampled.append(variants[truncated])
    return tuple(dict.fromkeys(sampled))


def _evaluate_target_candidates(
    generated: tuple[str, ...],
    *,
    current: Document,
    source: Document,
    group: tuple[Sentence, ...],
    embedder: Embedder,
    preferred_min_document_tokens: int,
    max_document_tokens: int,
    group_preferred_min_tokens: int,
    group_max_tokens: int,
    drift_budget: float,
    expected_structure: tuple[str, ...],
) -> tuple[_TargetCandidate, ...]:
    """Repair one generation round, then embed all segments, documents, and local chunks in one batch."""
    suffix = group[0].text[len(group[0].text.rstrip()) :]
    variants: dict[str, _RepairVariant] = {}
    for generated_text in generated:
        if not generated_text.strip():
            continue
        text = generated_text.rstrip() + suffix
        for variant in _repair_text_variants(text, group_max_tokens):
            existing = variants.get(variant.text)
            if existing is None or variant.repaired_tokens < existing.repaired_tokens:
                variants[variant.text] = variant
    if not variants:
        return ()

    pending: list[tuple[_RepairVariant, Document, int, bool, tuple[str, ...]]] = []
    targets = tuple(sentence.id for sentence in group)
    placeholder_embedding = group[0].embedding
    for variant in variants.values():
        merged_tokens = count_tokens(variant.text)
        replacement = Encoded(text=variant.text, token_count=merged_tokens, embedding=placeholder_embedding)
        edit = Candidate(
            edit=Replace(targets=targets, replacement=replacement),
            confidence=1.0,
            source="target_merge",
            reason=f"merged a content group to meet the {max_document_tokens}-token target",
        )
        trial = current.apply(edit)
        if trial is None:
            continue
        structure_valid = not _GENERATED_MARKUP.search(variant.text)
        pending.append((variant, trial, merged_tokens, structure_valid, _feedback_chunks(variant.text)))
    if not pending:
        return ()

    chunk_counts = [len(chunks) for _, _, _, _, chunks in pending]
    embedding_inputs = [variant.text for variant, _, _, _, _ in pending]
    embedding_inputs.extend(trial.text for _, trial, _, _, _ in pending)
    embedding_inputs.extend(chunk for _, _, _, _, chunks in pending for chunk in chunks)
    embeddings = embedder.embed(embedding_inputs)
    candidate_count = len(pending)
    replacement_embeddings = embeddings[:candidate_count]
    whole_embeddings = embeddings[candidate_count : candidate_count * 2]
    chunk_embeddings = embeddings[candidate_count * 2 :]

    source_group_embeddings = np.stack([normalize(sentence.embedding) for sentence in group])
    evaluated: list[_TargetCandidate] = []
    chunk_offset = 0
    for index, ((variant, _, merged_tokens, structure_valid, _), whole_embedding) in enumerate(
        zip(pending, whole_embeddings, strict=True)
    ):
        replacement = Encoded(
            text=variant.text,
            token_count=merged_tokens,
            embedding=replacement_embeddings[index],
        )
        edit = Candidate(
            edit=Replace(targets=targets, replacement=replacement),
            confidence=1.0,
            source="target_merge",
            reason=f"merged a content group to meet the {max_document_tokens}-token target",
        )
        applied = current.apply(edit)
        if applied is None:  # pragma: no cover - the placeholder application already proved feasibility
            continue
        candidate = applied.model_copy(update={"embedding": whole_embedding})
        drift = max(0.0, 1.0 - float(np.dot(normalize(whole_embedding), normalize(source.embedding))))
        output_chunk_embeddings = np.stack(
            [normalize(vector) for vector in chunk_embeddings[chunk_offset : chunk_offset + chunk_counts[index]]]
        )
        chunk_offset += chunk_counts[index]
        source_coverage = np.max(source_group_embeddings @ output_chunk_embeddings.T, axis=1)
        coverage = float(np.min(source_coverage))
        ordered_gaps = np.argsort(source_coverage)
        coverage_gaps = tuple(
            int(ordered_gaps[index % len(ordered_gaps)]) for index in range(TARGET_CANDIDATES_PER_CALL)
        )
        target_distance = max(0, candidate.token_count - max_document_tokens)
        target_undercut = max(0, preferred_min_document_tokens - candidate.token_count)
        issues: list[str] = []
        if variant.repaired_tokens:
            issues.append(f"deterministic repair removed {variant.repaired_tokens} tokens to enforce the hard limit")
        if candidate.token_count < preferred_min_document_tokens:
            issues.append(
                f"output used only {merged_tokens} tokens; prefer the available budget of "
                f"{group_preferred_min_tokens}-{group_max_tokens} when quality permits"
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
                text=variant.text,
                document=candidate,
                drift=drift,
                minimum_coverage=coverage,
                target_distance=target_distance,
                target_undercut=target_undercut,
                structure_valid=structure_valid,
                issues=tuple(issues),
                repaired_tokens=variant.repaired_tokens,
                coverage_gaps=coverage_gaps,
            )
        )
    return tuple(evaluated)


def _target_candidate_rank(candidate: _TargetCandidate) -> tuple[bool, bool, int, int, float, float]:
    return (
        not candidate.structure_valid,
        candidate.target_distance > 0,
        candidate.target_distance,
        candidate.target_undercut,
        -candidate.minimum_coverage,
        candidate.drift,
    )


def _target_feedback(candidate: _TargetCandidate, group: tuple[Sentence, ...], max_tokens: int) -> str:
    source_excerpts = [" ".join(group[index].text.split())[:220] for index in candidate.coverage_gaps]
    problems = "; ".join(candidate.issues) or "improve semantic coverage while preserving the token range"
    rendered_excerpts = "\n".join(
        f'{index}. Restore the exact meaning of source passage "{source_excerpt}".'
        for index, source_excerpt in enumerate(source_excerpts, start=1)
        if source_excerpt
    )
    return (
        f"The current base is within the hard token ceiling but needs a quality refinement: {problems}. Its "
        f"whole-prompt drift is {candidate.drift:.4f}, and its minimum source-line similarity is "
        f"{candidate.minimum_coverage:.4f}. Copy the base nearly verbatim and make one localized correction per "
        f"candidate using the correspondingly numbered excerpt below. Candidate N must perform only correction N.\n"
        f"{rendered_excerpts}\nEvery revision must use no more than {max_tokens} cl100k_base tokens. Remove redundant "
        "wording as needed; do not keep the edit token-neutral."
    )


def _feedback_chunks(text: str) -> tuple[str, ...]:
    chunks = tuple(chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", text) if chunk.strip())
    return chunks if chunks else (text,)


def _compressible_groups(document: Document) -> list[tuple[Sentence, ...]]:
    groups: list[tuple[Sentence, ...]] = []

    def walk(section: Section) -> None:
        direct = tuple(child for child in section.children if isinstance(child, Sentence) and child.optimizable)
        if direct:
            groups.append(direct)
        for child in section.children:
            if isinstance(child, Section):
                walk(child)

    for section in document.sections:
        walk(section)
    return groups


def _content_terms(text: str) -> frozenset[str]:
    return frozenset(
        word
        for match in _CONTENT_WORD.finditer(text.lower())
        if len(word := match.group()) >= 3 and word not in _QUERY_STOPWORDS
    )


def _target_anchor_terms(document: Document, group: tuple[Sentence, ...]) -> frozenset[str]:
    group_ids = {sentence.id for sentence in group}
    outside = [
        sentence
        for sentence in document.sentences
        if sentence.optimizable and sentence.id not in group_ids and sentence.text.strip()
    ]
    return _content_terms(outside[-1].text) if outside else frozenset()


def _target_merge_window(
    group: tuple[Sentence, ...],
    required_savings: int,
    document_embedding: np.ndarray,
    *,
    protected_terms: frozenset[str] = frozenset(),
) -> tuple[Sentence, ...]:
    """Choose only enough contiguous content to make the requested saving practical.

    Rewriting an entire long context to save 10% makes each generated candidate unnecessarily huge. A window around
    twice the required saving asks the merger to compress that window by roughly half. Large targets naturally
    saturate at the full group. Among similarly sized windows, prefer content most representative of the whole
    document so semantic outliers such as sparse task facts are less likely to be touched.
    """
    group_tokens = sum(sentence.token_count for sentence in group)
    desired_tokens = min(group_tokens, max(required_savings + 1, required_savings * 2))
    if desired_tokens >= group_tokens or len(group) <= 2:
        return group
    normalized_document = normalize(document_embedding)
    best: tuple[Sentence, ...] | None = None
    best_similarity = (False, float("-inf"), float("-inf"))
    end = 0
    window_tokens = 0
    for start in range(len(group)):
        while end < len(group) and window_tokens < desired_tokens:
            window_tokens += group[end].token_count
            end += 1
        if end - start >= 2 and window_tokens >= desired_tokens:
            window = group[start:end]
            sentence_similarities = [
                float(np.dot(normalize(sentence.embedding), normalized_document)) for sentence in window
            ]
            similarity = (
                not any(_content_terms(sentence.text) & protected_terms for sentence in window),
                min(sentence_similarities),
                sum(
                    sentence.token_count * sentence_similarity
                    for sentence, sentence_similarity in zip(window, sentence_similarities, strict=True)
                )
                / window_tokens,
            )
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
