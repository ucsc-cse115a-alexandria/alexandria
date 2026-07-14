from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from alexandria.ir.contracts import Embedder, Params
from alexandria.ir.similarity import cosine_distance
from alexandria.ops.pipe import reduce


class OptimizationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    original_text: str
    compressed_text: str
    source_tokens: int
    reduced_tokens: int
    tokens_saved: int
    token_reduction: float
    quality_score: float
    quality_drift: float
    quality_gate: float
    quality_passed: bool


def report(
    prompt: str,
    embedder: Embedder,
    *,
    params: Params | None = None,
) -> OptimizationReport:
    params = params or Params()

    result = reduce(prompt, embedder, params=params)

    original_vector, compressed_vector = embedder.embed([prompt, result.text])

    drift = cosine_distance(
        original_vector,
        compressed_vector,
    )
    quality_score = 1.0 - drift
    quality_gate = 1.0 - params.drift_budget

    tokens_saved = result.source_tokens - result.reduced_tokens
    token_reduction = tokens_saved / result.source_tokens if result.source_tokens else 0.0

    return OptimizationReport(
        original_text=prompt,
        compressed_text=result.text,
        source_tokens=result.source_tokens,
        reduced_tokens=result.reduced_tokens,
        tokens_saved=tokens_saved,
        token_reduction=round(token_reduction, 6),
        quality_score=round(quality_score, 6),
        quality_drift=round(drift, 6),
        quality_gate=round(quality_gate, 6),
        quality_passed=drift <= params.drift_budget,
    )
