"""Self-contained envelopes that carry one phase's output to the next over stdin/stdout as JSON."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import MergeMetrics, Plan, Scores
from alexandria.ir.document import Document

# The wire format version. A parse of a file stamped with any other value fails as a ValidationError.
SCHEMA_VERSION = 1


class DocumentEnvelope(BaseModel):
    """Represent's output: the Document, ready for the score phase."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = SCHEMA_VERSION
    document: Document


class ScoredEnvelope(BaseModel):
    """Score's output: the Document plus the name-keyed score bundle the optimize phase consumes."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = SCHEMA_VERSION
    document: Document
    scores: Scores


class PlanEnvelope(BaseModel):
    """Optimize's output: the Document plus the ranked Plan the select phase folds."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = SCHEMA_VERSION
    document: Document
    plan: Plan
    merge_metrics: MergeMetrics = MergeMetrics()


__all__ = ["SCHEMA_VERSION", "DocumentEnvelope", "PlanEnvelope", "ScoredEnvelope"]
