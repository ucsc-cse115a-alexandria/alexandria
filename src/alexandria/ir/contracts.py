from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alexandria.ir.document import Document, Encoded, SentenceId

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Scores = dict[str, dict[SentenceId, float]]  # scorer name -> sentence id -> score

# A scorer may also expose, per sentence, its most-similar peer id and that similarity.
Peers = Callable[[Document], list[tuple[SentenceId | None, float]]]

Threshold = Annotated[float, Field(ge=0.0, le=1.0)]
Drift = Annotated[float, Field(ge=0.0)]  # cosine distance from the original prompt; 0.01 == 1%


class Params(BaseModel):
    """Tuning knobs shared by the optimize and select phases; the single source of truth for defaults."""

    model_config = ConfigDict(frozen=True)
    threshold: Threshold = 0.85
    drift_budget: Drift = 0.01
    max_tokens: int | None = Field(default=None, ge=1)
    require_target: bool = False


class MergeMetrics(BaseModel):
    """Generation work performed by a sentence merger during one reduction."""

    model_config = ConfigDict(frozen=True)
    calls: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    jobs_attempted: int = Field(default=0, ge=0)
    proposed_edits: int = Field(default=0, ge=0)
    applied_edits: int = Field(default=0, ge=0)


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]: ...


class SentenceMerger(Protocol):
    """Rewrites two overlapping instructions as one that preserves both meanings in fewer tokens.

    feedback carries the caller's rejection of a previous attempt (the rejected rewrite and why
    it failed), so a retry can correct course; None on the first attempt.
    """

    def merge(self, first: str, second: str, feedback: str | None = None) -> str: ...


@runtime_checkable
class TargetedMerger(Protocol):
    """Compress a prompt content segment to a hard token budget, using feedback on retries."""

    def merge_to_target(self, prompt: str, max_tokens: int, feedback: str | None = None) -> str: ...


class TrackedMerger:
    """Decorate any sentence merger with request and retry counters."""

    def __init__(self, merger: SentenceMerger) -> None:
        self._merger = merger
        self._cache: dict[tuple[str, str, str | None], str] = {}
        self.calls = 0
        self.retries = 0
        self.jobs_attempted = 0

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        key = (first, second, feedback)
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        if feedback is None:
            self.jobs_attempted += 1
        else:
            self.retries += 1
        merged = self._merger.merge(first, second, feedback)
        self._cache[key] = merged
        return merged

    def merge_to_target(self, prompt: str, max_tokens: int, feedback: str | None = None) -> str:
        if not isinstance(self._merger, TargetedMerger):
            raise TypeError("strict token targets require a merger with merge_to_target support")
        self.calls += 1
        if feedback is None:
            self.jobs_attempted += 1
        else:
            self.retries += 1
        return self._merger.merge_to_target(prompt, max_tokens, feedback)

    def metrics(self, *, proposed_edits: int, applied_edits: int) -> MergeMetrics:
        return MergeMetrics(
            calls=self.calls,
            retries=self.retries,
            jobs_attempted=self.jobs_attempted,
            proposed_edits=proposed_edits,
            applied_edits=applied_edits,
        )


def _reject_duplicate_targets(targets: tuple[SentenceId, ...]) -> tuple[SentenceId, ...]:
    """One edit must never name the same sentence twice — that would delete-then-miss or double-count."""
    if len(set(targets)) != len(targets):
        raise ValueError(f"edit targets must be unique, got {list(targets)}")
    return targets


class Delete(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["delete"] = "delete"
    targets: tuple[SentenceId, ...] = Field(min_length=1)

    _no_duplicate_targets = field_validator("targets")(_reject_duplicate_targets)


class Replace(BaseModel):
    """Swap the first target's text for the merged replacement and remove the remaining targets."""

    model_config = ConfigDict(frozen=True)
    op: Literal["replace"] = "replace"
    targets: tuple[SentenceId, ...] = Field(min_length=2)
    replacement: Encoded  # merged text with its token count and embedding, precomputed at plan time

    _no_duplicate_targets = field_validator("targets")(_reject_duplicate_targets)


Edit = Annotated[Delete | Replace, Field(discriminator="op")]


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)
    edit: Edit
    confidence: float
    source: str
    reason: str


Plan = tuple[Candidate, ...]


class DiffSpan(BaseModel):
    """One sentence an edit removes or rewrites: its id, section location, and original text."""

    model_config = ConfigDict(frozen=True)
    sentence_id: SentenceId
    section_path: tuple[str, ...]  # Section.header values from the root section down to the sentence's parent
    original: str


class Diff(BaseModel):
    """A displayable rendering of one Candidate: where it applies, what it removes, what replaces it."""

    model_config = ConfigDict(frozen=True)
    candidate: Candidate
    spans: tuple[DiffSpan, ...] = Field(min_length=1)
    replacement: str  # "" for Delete; the merged text for Replace


class Scorer(Protocol):
    def __call__(self, document: Document) -> list[float]: ...


class Optimizer(Protocol):
    def __call__(
        self, document: Document, scores: Scores, embedder: Embedder, merger: SentenceMerger, params: Params
    ) -> Plan: ...


class Selection(BaseModel):
    """A selector's result: the reduced Document and the candidates it actually applied."""

    model_config = ConfigDict(frozen=True)
    document: Document
    applied: Plan


class Selector(Protocol):
    def __call__(self, document: Document, plan: Plan, embedder: Embedder, params: Params) -> Selection: ...
