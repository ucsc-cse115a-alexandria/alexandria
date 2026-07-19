from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from alexandria.ir.contracts import (
    Candidate,
    Delete,
    Replace,
    ReportedCandidate,
    TargetMergeRoundMetrics,
)
from alexandria.ir.document import Document, Encoded, Section, Sentence
from alexandria.ir.similarity import compute_cos_sim_diff, normalize, similarity_matrix_for
from alexandria.utils.tokens import count_tokens, truncate_tokens

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder, MergeMetrics, Params, ReductionReporter, TrackedMerger
    from alexandria.ir.document import SentenceId

PRUNE_VERIFY_MAX_STEPS = 12  # cos_sim_diff search probe cap
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
class _PendingCandidate:
    """A repaired variant whose Replace already proved feasible, awaiting its batched embeddings."""

    variant: _RepairVariant
    trial: Document
    merged_tokens: int
    structure_valid: bool
    coverage_chunks: tuple[str, ...]


@dataclass(frozen=True)
class _TargetCandidate:
    text: str
    document: Document
    cos_sim_diff: float
    minimum_coverage: float
    target_distance: int
    structure_valid: bool
    issues: tuple[str, ...]
    repaired_tokens: int


@dataclass(frozen=True)
class TargetMergeOutcome:
    """A finished target merge: the reduced document plus the work both phases performed."""

    document: Document
    applied_groups: int
    pruned_sentences: int
    pruned_tokens: int
    repaired_tokens: int
    final_cos_sim_diff: float
    cos_sim_diff_budget_met: bool


class InfeasibleTargetError(ValueError):
    """The protected prompt structure alone cannot fit within the requested token budget."""


class TargetMergeError(ValueError):
    """A strict target failure with machine-readable merge usage and best observed quality."""

    def __init__(self, message: str, metrics: MergeMetrics, *, best_tokens: int, best_cos_sim_diff: float) -> None:
        super().__init__(message)
        self.metrics = metrics
        self.best_tokens = best_tokens
        self.best_cos_sim_diff = best_cos_sim_diff


def merge_to_target(
    document: Document, embedder: Embedder, merger: TrackedMerger, params: Params, reporter: ReductionReporter
) -> TargetMergeOutcome:
    """Meet the hard token ceiling first, then minimize cos_sim_diff among feasible candidates."""
    if params.max_tokens is None:
        raise ValueError("target merge requires max_tokens")
    protected_tokens = sum(sentence.token_count for sentence in document.sentences if not sentence.optimizable)
    if protected_tokens > params.max_tokens:
        raise InfeasibleTargetError(
            f"protected structure uses {protected_tokens} tokens, exceeding the {params.max_tokens}-token target"
        )
    expected_structure = tuple(sentence.text.strip() for sentence in document.sentences if not sentence.optimizable)
    current = _prune_to_target(document, embedder, params, params.max_tokens)
    pruned_sentences = len(document.sentences) - len(current.sentences)
    pruned_tokens = document.token_count - current.token_count
    repaired_tokens = 0

    def merge_metrics(applied_groups: int, *, final_cos_sim_diff: float | None = None) -> MergeMetrics:
        return merger.metrics(proposed_edits=applied_groups, applied_edits=applied_groups).model_copy(
            update={
                "pruned_sentences": pruned_sentences,
                "pruned_tokens": pruned_tokens,
                "repaired_tokens": repaired_tokens,
                "final_cos_sim_diff": final_cos_sim_diff,
                "cos_sim_diff_budget_met": (
                    final_cos_sim_diff <= params.cos_sim_diff_budget if final_cos_sim_diff is not None else None
                ),
            }
        )

    applied_groups = 0
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
        source_segment = "".join(sentence.text for sentence in group)
        reporter.target_group(source_segment, group_tokens, required_savings)
        base_tokens = current.token_count
        base_cos_sim_diff = max(0.0, compute_cos_sim_diff(current.embedding, document.embedding))

        generated = merger.merge_candidates_to_target(source_segment, group_max_tokens)
        evaluated = _evaluate_target_candidates(
            generated,
            current=current,
            source=document,
            group=group,
            embedder=embedder,
            max_document_tokens=params.max_tokens,
            group_max_tokens=group_max_tokens,
            cos_sim_diff_budget=params.cos_sim_diff_budget,
            expected_structure=expected_structure,
        )
        generated_selectable = [
            candidate
            for candidate in evaluated
            if candidate.structure_valid and candidate.document.token_count < current.token_count
        ]
        generated_best = min(generated_selectable, key=_target_candidate_rank) if generated_selectable else None
        if generated_selectable:
            pool = generated_selectable
        else:
            # The source segment's repair variants are pure deletions, so they guarantee progress.
            evaluated = _evaluate_target_candidates(
                (source_segment,),
                current=current,
                source=document,
                group=group,
                embedder=embedder,
                max_document_tokens=params.max_tokens,
                group_max_tokens=group_max_tokens,
                cos_sim_diff_budget=params.cos_sim_diff_budget,
                expected_structure=expected_structure,
            )
            pool = [
                candidate
                for candidate in evaluated
                if candidate.structure_valid and candidate.document.token_count < current.token_count
            ]
        if not pool:
            reporter.target_group_done(applied=False, document_tokens=current.token_count)
            details = "; ".join(dict.fromkeys(issue for c in evaluated for issue in c.issues))
            metrics = merge_metrics(applied_groups, final_cos_sim_diff=base_cos_sim_diff)
            raise TargetMergeError(
                f"target merge made no progress after {merger.calls} calls: "
                f"{details or 'no shorter structure-valid candidate'}",
                metrics,
                best_tokens=current.token_count,
                best_cos_sim_diff=base_cos_sim_diff,
            )
        chosen = min(pool, key=_target_candidate_rank)
        improved = chosen.document.token_count < base_tokens and chosen.cos_sim_diff < base_cos_sim_diff
        merger.record_target_round(
            TargetMergeRoundMetrics(
                round=1,
                base_tokens=base_tokens,
                selected_tokens=chosen.document.token_count,
                base_cos_sim_diff=base_cos_sim_diff,
                selected_cos_sim_diff=chosen.cos_sim_diff,
                generated_best_tokens=(
                    generated_best.document.token_count if generated_best is not None else chosen.document.token_count
                ),
                generated_best_cos_sim_diff=(
                    generated_best.cos_sim_diff if generated_best is not None else chosen.cos_sim_diff
                ),
                improved=improved,
            )
        )
        candidate_list = tuple(
            ReportedCandidate(
                text=candidate.text,
                token_count=candidate.document.token_count,
                cos_sim_diff=candidate.cos_sim_diff,
                structure_valid=candidate.structure_valid,
            )
            for candidate in evaluated
        )
        selected = ReportedCandidate(
            text=chosen.text,
            token_count=chosen.document.token_count,
            cos_sim_diff=chosen.cos_sim_diff,
            structure_valid=chosen.structure_valid,
        )
        reporter.target_round(1, None, candidate_list, selected, generated_best is not None)
        current = chosen.document
        repaired_tokens += chosen.repaired_tokens
        reporter.target_group_done(applied=True, document_tokens=current.token_count)
        applied_groups += 1
    final_cos_sim_diff = compute_cos_sim_diff(current.embedding, document.embedding)
    return TargetMergeOutcome(
        document=current,
        applied_groups=applied_groups,
        pruned_sentences=pruned_sentences,
        pruned_tokens=pruned_tokens,
        repaired_tokens=repaired_tokens,
        final_cos_sim_diff=final_cos_sim_diff,
        cos_sim_diff_budget_met=final_cos_sim_diff <= params.cos_sim_diff_budget,
    )


def _prune_to_target(document: Document, embedder: Embedder, params: Params, max_tokens: int) -> Document:
    """Delete redundant sentences toward the token target without any merge-model call.

    The plan orders deletions by descending redundancy; the whole plan's cos_sim_diff is then verified
    with one embedding call. When the full plan exceeds the budget, bisect on the deletion count
    (a heuristic, since cos_sim_diff is not monotone in prefix length; every returned prefix is
    verified) and keep the longest passing prefix, falling back to no pruning at all.
    """
    deletions = _prune_plan(document, params.threshold, max_tokens)
    if not deletions:
        return document
    pruned = _apply_prune_prefix(document, deletions, len(deletions))
    pruned_embedding = embedder.embed([pruned.text])[0]
    cos_sim_diff = compute_cos_sim_diff(pruned_embedding, document.embedding)
    if cos_sim_diff <= params.cos_sim_diff_budget:
        return pruned.model_copy(update={"embedding": pruned_embedding})
    passing, failing = 0, len(deletions)
    passing_document = document
    for _ in range(PRUNE_VERIFY_MAX_STEPS):
        middle = (passing + failing) // 2
        if middle == passing:
            break
        trial = _apply_prune_prefix(document, deletions, middle)
        trial_embedding = embedder.embed([trial.text])[0]
        cos_sim_diff = compute_cos_sim_diff(trial_embedding, document.embedding)
        if cos_sim_diff <= params.cos_sim_diff_budget:
            passing = middle
            passing_document = trial.model_copy(update={"embedding": trial_embedding})
        else:
            failing = middle
    return passing_document


def _prune_plan(document: Document, threshold: float, max_tokens: int) -> tuple[SentenceId, ...]:
    """Deletions, most-redundant first, that approach max_tokens while never emptying a section."""
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
    max_document_tokens: int,
    group_max_tokens: int,
    cos_sim_diff_budget: float,
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

    pending: list[_PendingCandidate] = []
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
        pending.append(
            _PendingCandidate(
                variant=variant,
                trial=trial,
                merged_tokens=merged_tokens,
                structure_valid=structure_valid,
                coverage_chunks=_coverage_chunks(variant.text),
            )
        )
    if not pending:
        return ()

    embedding_inputs = [candidate.variant.text for candidate in pending]
    embedding_inputs.extend(candidate.trial.text for candidate in pending)
    embedding_inputs.extend(chunk for candidate in pending for chunk in candidate.coverage_chunks)
    embeddings = embedder.embed(embedding_inputs)
    count = len(pending)
    replacement_embeddings = embeddings[:count]
    whole_embeddings = embeddings[count : count * 2]
    chunk_embeddings = embeddings[count * 2 :]
    chunk_slices: list[list[NDArray[np.float32]]] = []
    offset = 0
    for candidate in pending:
        chunk_slices.append(chunk_embeddings[offset : offset + len(candidate.coverage_chunks)])
        offset += len(candidate.coverage_chunks)

    source_group_embeddings = np.stack([normalize(sentence.embedding) for sentence in group])
    evaluated: list[_TargetCandidate] = []
    for pending_candidate, replacement_embedding, whole_embedding, candidate_chunks in zip(
        pending, replacement_embeddings, whole_embeddings, chunk_slices, strict=True
    ):
        variant = pending_candidate.variant
        replacement = Encoded(
            text=variant.text,
            token_count=pending_candidate.merged_tokens,
            embedding=replacement_embedding,
        )
        edit = Candidate(
            edit=Replace(targets=targets, replacement=replacement),
            confidence=1.0,
            source="target_merge",
            reason=f"merged a content group to meet the {max_document_tokens}-token target",
        )
        # Re-apply with the real replacement embedding: later rounds read this merged sentence's
        # embedding (target-window selection and coverage), so the placeholder must not survive.
        applied = current.apply(edit)
        if applied is None:  # pragma: no cover - the placeholder application already proved feasibility
            continue
        candidate = applied.model_copy(update={"embedding": whole_embedding})
        cos_sim_diff = max(0.0, 1.0 - float(np.dot(normalize(whole_embedding), normalize(source.embedding))))
        output_chunk_embeddings = np.stack([normalize(vector) for vector in candidate_chunks])
        source_coverage = np.max(source_group_embeddings @ output_chunk_embeddings.T, axis=1)
        coverage = float(np.min(source_coverage))
        target_distance = max(0, candidate.token_count - max_document_tokens)
        issues: list[str] = []
        structure_valid = pending_candidate.structure_valid
        if variant.repaired_tokens:
            issues.append(f"deterministic repair removed {variant.repaired_tokens} tokens to enforce the hard limit")
        if not structure_valid:
            issues.append("candidate introduced an XML or Markdown boundary line")
        actual_structure = tuple(sentence.text.strip() for sentence in candidate.sentences if not sentence.optimizable)
        if actual_structure != expected_structure:
            structure_valid = False
            issues.append("XML/Markdown boundary lines changed")
        if cos_sim_diff > cos_sim_diff_budget:
            issues.append(f"whole-prompt cos_sim_diff was {cos_sim_diff:.4f}; maximum is {cos_sim_diff_budget:.4f}")
        evaluated.append(
            _TargetCandidate(
                text=variant.text,
                document=candidate,
                cos_sim_diff=cos_sim_diff,
                minimum_coverage=coverage,
                target_distance=target_distance,
                structure_valid=structure_valid,
                issues=tuple(issues),
                repaired_tokens=variant.repaired_tokens,
            )
        )
    return tuple(evaluated)


def _target_candidate_rank(candidate: _TargetCandidate) -> tuple[bool, bool, int, float, float]:
    return (
        not candidate.structure_valid,
        candidate.target_distance > 0,
        candidate.target_distance,
        candidate.cos_sim_diff,
        -candidate.minimum_coverage,
    )


def _coverage_chunks(text: str) -> tuple[str, ...]:
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
    group_similarities = [float(np.dot(normalize(sentence.embedding), normalized_document)) for sentence in group]
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
            sentence_similarities = group_similarities[start:end]
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
