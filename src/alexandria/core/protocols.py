from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from alexandria.core.ir import Document, SentenceId

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

ScoreVector = tuple[float, ...]
Scores = dict[str, ScoreVector]

# A scorer may also expose, per sentence, its most-similar peer id and that similarity.
Peers = Callable[[Document], list[tuple[SentenceId | None, float]]]

Threshold = Annotated[float, Field(ge=0.0, le=1.0)]
Drift = Annotated[float, Field(ge=0.0)]  # cosine distance from the original prompt; 0.01 == 1%


class Params(BaseModel):
    """Tuning knobs shared by the optimize and select phases; the single source of truth for defaults."""

    model_config = ConfigDict(frozen=True)
    threshold: Threshold = 0.85
    drift_budget: Drift = 0.01


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]: ...


class Delete(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["delete"] = "delete"
    targets: tuple[SentenceId, ...] = Field(min_length=1)


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)
    edit: Delete
    confidence: float
    source: str
    reason: str


Plan = tuple[Candidate, ...]


class Scorer(Protocol):
    def __call__(self, document: Document) -> list[float]: ...


class Optimizer(Protocol):
    def __call__(self, document: Document, scores: Scores, params: Params) -> Plan: ...


class Selector(Protocol):
    def __call__(self, document: Document, plan: Plan, embedder: Embedder, params: Params) -> Document: ...
