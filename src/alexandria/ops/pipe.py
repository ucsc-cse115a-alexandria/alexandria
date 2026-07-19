from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import (
    SILENT_REPORTER,
    Diff,
    MergeMetrics,
    Params,
    Plan,
    Selection,
    TrackedEmbedder,
    TrackedMerger,
)
from alexandria.ir.document import Document
from alexandria.ir.registry import required_scorers
from alexandria.ops.features.diff import diffs
from alexandria.ops.features.optimize import DEFAULT_OPTIMIZER, optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import DEFAULT_SCORER, score, score_rows
from alexandria.ops.features.select import DEFAULT_SELECTOR, select
from alexandria.ops.features.target import TargetMergeError, merge_to_target
from alexandria.utils.embedders import default_embedder
from alexandria.utils.merger import default_merger

if TYPE_CHECKING:
    from alexandria.ir.contracts import Embedder, ReductionReporter, Scores, SentenceMerger
    from alexandria.ops.features.target import TargetMergeOutcome


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


class Proposal(BaseModel):
    """The reviewable form of a reduction: the source Document and its proposed edits as diffs."""

    model_config = ConfigDict(frozen=True)
    document: Document
    diffs: tuple[Diff, ...]


def reduce(
    prompt: str,
    embedder: Embedder | None = None,
    merger: SentenceMerger | None = None,
    *,
    api_key: str | None = None,
    optimizers: tuple[str, ...] = (DEFAULT_OPTIMIZER,),
    selector: str = DEFAULT_SELECTOR,
    params: Params | None = None,
    reporter: ReductionReporter = SILENT_REPORTER,
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
            outcome = merge_to_target(document, embedder, tracked_merger, params, reporter)
        except TargetMergeError as error:
            error.metrics = _finalize_metrics(error.metrics, embedder, started)
            raise
        selection = Selection(document=outcome.document, applied=())
        base_metrics = _target_merge_metrics(tracked_merger, outcome)
    else:
        scores = score(document, names=_required_scorers(optimizers))
        plan = optimize(document, scores, embedder, tracked_merger, names=optimizers, params=params, reporter=reporter)
        selection = select(document, plan, embedder, selector, params=params)
        if params is not None and params.max_tokens is not None and selection.document.token_count > params.max_tokens:
            plan, selection = _retry_exhaustively(
                document, scores, embedder, tracked_merger, optimizers, selector, params, reporter
            )
        base_metrics = tracked_merger.metrics(proposed_edits=len(plan), applied_edits=len(selection.applied))

    return ReduceResult(
        document=selection.document,
        source=document,
        applied=selection.applied,
        merge_metrics=_finalize_metrics(base_metrics, embedder, started),
    )


def _retry_exhaustively(
    document: Document,
    scores: Scores,
    embedder: Embedder,
    tracked_merger: TrackedMerger,
    optimizers: tuple[str, ...],
    selector: str,
    params: Params,
    reporter: ReductionReporter,
) -> tuple[Plan, Selection]:
    """Re-run optimize + select without a ceiling when a target-sized proposal overshoots after
    cumulative cos_sim_diff checks, reusing every cached merger response instead of paying for a call twice."""
    exhaustive = params.model_copy(update={"max_tokens": None, "require_target": False})
    plan = optimize(document, scores, embedder, tracked_merger, names=optimizers, params=exhaustive, reporter=reporter)
    selection = select(document, plan, embedder, selector, params=params)
    return plan, selection


def _target_merge_metrics(tracked_merger: TrackedMerger, outcome: TargetMergeOutcome) -> MergeMetrics:
    """Merger counters plus the prune, repair, and cos_sim_diff figures a target run produced."""
    return tracked_merger.metrics(
        proposed_edits=outcome.applied_groups, applied_edits=outcome.applied_groups
    ).model_copy(
        update={
            "pruned_sentences": outcome.pruned_sentences,
            "pruned_tokens": outcome.pruned_tokens,
            "repaired_tokens": outcome.repaired_tokens,
            "final_cos_sim_diff": outcome.final_cos_sim_diff,
            "cos_sim_diff_budget_met": outcome.cos_sim_diff_budget_met,
        }
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
