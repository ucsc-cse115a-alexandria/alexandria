from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.ir.contracts import SILENT_REPORTER, Candidate, Delete, Params, Replace
from alexandria.ir.document import Encoded
from alexandria.ir.registry import get_optimizer, register_optimizer, required_scorers
from alexandria.ir.similarity import compute_cos_sim_diff, similarity_matrix_for
from alexandria.utils.tokens import count_tokens

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from alexandria.ir.contracts import Embedder, Plan, ReductionReporter, Scores, SentenceMerger
    from alexandria.ir.document import Document, Sentence

DEFAULT_OPTIMIZER = "merge_rewrite"
MAX_MERGE_ATTEMPTS = 3  # LLM attempts per pair; each retry feeds back why the last rewrite was rejected
_KEEP_FIRST_SIMILARITY = 0.99  # merged ≈ first sentence -> keep the original text, just delete the second


@register_optimizer(DEFAULT_OPTIMIZER)
def merge_rewrite(
    document: Document,
    scores: Scores,
    embedder: Embedder,
    merger: SentenceMerger,
    params: Params,
    reporter: ReductionReporter = SILENT_REPORTER,
) -> Plan:
    """For each near-duplicate pair, rewrite both sentences into one via the merger, kept at the
    first occurrence; ranked by pair similarity.

    Each pair runs a generate -> measure -> feedback loop (at most MAX_MERGE_ATTEMPTS): a rewrite
    whose whole-document cos_sim_diff would exceed params.cos_sim_diff_budget is fed back to
    the merger for another try, and a pair that never fits is skipped. Emits Delete when the
    merged sentence is effectively the first one (it already covers both). A sentence is
    rewritten at most once; after a Delete the unchanged first sentence stays pairable, so a
    triple duplicate collapses fully.
    """
    del scores
    if embedder.model_id != document.embedding_model:
        raise ValueError(
            f"embedder {embedder.model_id!r} does not match document embedding model {document.embedding_model!r}"
        )
    sentences = document.sentences
    similarity = similarity_matrix_for(document)
    optimizable = [index for index, sentence in enumerate(sentences) if sentence.optimizable]
    pairs = [
        (float(similarity[i, j]), i, j)
        for position, i in enumerate(optimizable)
        for j in optimizable[position + 1 :]
        if similarity[i, j] >= params.threshold
    ]
    pairs.sort(key=lambda pair: pair[0], reverse=True)

    present = {s.id for s in sentences}
    candidates: list[Candidate] = []
    projected = document
    for sim, i, j in pairs:
        first, second = sentences[i], sentences[j]
        if first.id not in present or second.id not in present:
            continue
        reporter.redundant_pair(first.text, second.text, sim)
        candidate = _merge_pair(document, first, second, sim, embedder, merger, params)
        if candidate is None:
            reporter.pair_merged(None, "skipped")
        elif candidate.edit.op == "delete":
            reporter.pair_merged(first.text, "delete")
        else:
            reporter.pair_merged(candidate.edit.replacement.text, "replace")
        if candidate is not None and candidate.edit.op == "delete":
            present.discard(second.id)  # first is unchanged, so it may pair again
            candidates.append(candidate)
        else:
            present -= {first.id, second.id}  # rewritten or judged unmergeable: both are consumed
            if candidate is not None:
                candidates.append(candidate)
        if candidate is not None:
            next_projected = projected.apply(candidate)
            if next_projected is not None and next_projected is not projected:
                projected = next_projected
                if params.max_tokens is not None and projected.token_count <= params.max_tokens:
                    break
    return tuple(candidates)


def _merge_pair(
    document: Document,
    first: Sentence,
    second: Sentence,
    sim: float,
    embedder: Embedder,
    merger: SentenceMerger,
    params: Params,
) -> Candidate | None:
    """One pair's generate -> measure -> feedback loop; None when no attempt fits the budget."""
    if first.text == second.text:
        candidate = Candidate(
            edit=Delete(targets=(second.id,)),
            confidence=sim,
            source=DEFAULT_OPTIMIZER,
            reason="exact duplicate; removed without a merge model call",
        )
        trial = document.apply(candidate)
        if trial is None:
            return None
        cos_sim_diff = compute_cos_sim_diff(embedder.embed([trial.text])[0], document.embedding)
        return candidate if cos_sim_diff <= params.cos_sim_diff_budget else None

    feedback: str | None = None
    for _ in range(MAX_MERGE_ATTEMPTS):
        merged = _with_tail(merger.merge(first.text, second.text, feedback), second.text)
        embedding = embedder.embed([merged])[0]
        candidate = _candidate_for(first, second, merged, embedding, sim)
        if candidate is None:
            feedback = (
                f'Your rewrite "{merged.strip()}" is not shorter than the two instructions combined; make it shorter.'
            )
            continue
        trial = document.apply(candidate)
        if trial is None:
            return None  # the edit would empty a section; no rewrite can fix that
        cos_sim_diff = compute_cos_sim_diff(embedder.embed([trial.text])[0], document.embedding)
        if cos_sim_diff <= params.cos_sim_diff_budget:
            return candidate
        feedback = (
            f'Your rewrite "{merged.strip()}" changed the whole document\'s embedding by '
            f"{cos_sim_diff:.3f} against a budget of {params.cos_sim_diff_budget:.3f}. Preserve more of the "
            "original meaning, even at the cost of a few more tokens."
        )
    return None


def _candidate_for(
    first: Sentence, second: Sentence, merged: str, embedding: NDArray[np.float32], sim: float
) -> Candidate | None:
    """The edit one merge attempt proposes: Delete when the first sentence already covers both,
    Replace when the merge saves tokens, None otherwise."""
    if 1.0 - compute_cos_sim_diff(embedding, first.embedding) >= _KEEP_FIRST_SIMILARITY:
        return Candidate(
            edit=Delete(targets=(second.id,)),
            confidence=sim,
            source=DEFAULT_OPTIMIZER,
            reason=f"redundant (cosine {sim:.2f}); the first instruction already covers both",
        )
    token_count = count_tokens(merged)
    if token_count >= first.token_count + second.token_count:
        return None
    return Candidate(
        edit=Replace(
            targets=(first.id, second.id),
            replacement=Encoded(text=merged, token_count=token_count, embedding=embedding),
        ),
        confidence=sim,
        source=DEFAULT_OPTIMIZER,
        reason=f"redundant (cosine {sim:.2f}); merged both instructions at the first occurrence",
    )


def _with_tail(merged: str, original: str) -> str:
    """Give a merged span the final target's trailing whitespace so its next boundary stays intact."""
    tail = original[len(original.rstrip()) :]
    return merged.rstrip() + tail


def optimize(
    document: Document,
    scores: Scores,
    embedder: Embedder,
    merger: SentenceMerger,
    names: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    *,
    params: Params | None = None,
    reporter: ReductionReporter = SILENT_REPORTER,
) -> Plan:
    """Run each named optimizer and concatenate their candidate stacks into one ordered Plan.

    Raises ValueError if any optimizer's required scorers are absent from `scores`, so a
    mispiped `score ... | optimize` fails at the boundary instead of raising a raw KeyError deeper in.
    """
    params = params or Params()
    for name in names:
        missing = [scorer for scorer in required_scorers(name) if scorer not in scores]
        if missing:
            raise ValueError(
                f"optimizer {name!r} requires scorer(s) {missing} absent from the input scores {sorted(scores)}"
            )
    plan: list[Candidate] = []
    for name in names:
        plan.extend(get_optimizer(name)(document, scores, embedder, merger, params, reporter))
    return tuple(plan)


__all__ = ["DEFAULT_OPTIMIZER", "MAX_MERGE_ATTEMPTS", "merge_rewrite", "optimize"]
