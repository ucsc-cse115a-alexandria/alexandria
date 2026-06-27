"""Phase contracts (Embedder/Scorer/Optimizer) and the edit data they exchange."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from alexandria.core.ir import Document, SentenceId

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

ScoreVector = tuple[float, ...]
Scores = dict[str, ScoreVector]


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]: ...


class Delete(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["delete"] = "delete"
    targets: tuple[SentenceId, ...] = Field(min_length=1)


Edit = Delete


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)
    edit: Edit
    score: float
    source: str
    reason: str


Plan = tuple[Candidate, ...]


class Scorer(Protocol):
    def __call__(self, document: Document) -> list[float]: ...


class Optimizer(Protocol):
    def __call__(
        self,
        document: Document,
        scores: Scores,
        embedder: Embedder,
        *,
        threshold: float,
        max_drift: float = 2.0,
    ) -> Plan: ...
