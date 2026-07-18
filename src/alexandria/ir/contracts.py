from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

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


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]: ...


class Delete(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["delete"] = "delete"
    targets: tuple[SentenceId, ...] = Field(min_length=1)


class Replace(BaseModel):
    """Swap the first target's text for the merged replacement and remove the remaining targets."""

    model_config = ConfigDict(frozen=True)
    op: Literal["replace"] = "replace"
    targets: tuple[SentenceId, ...] = Field(min_length=2)
    replacement: Encoded  # merged text with its token count and embedding, precomputed at plan time


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
    def __call__(self, document: Document, scores: Scores, params: Params) -> Plan: ...


class Selection(BaseModel):
    """A selector's result: the reduced Document and the candidates it actually applied."""

    model_config = ConfigDict(frozen=True)
    document: Document
    applied: Plan


class Selector(Protocol):
    def __call__(self, document: Document, plan: Plan, embedder: Embedder, params: Params) -> Selection: ...
