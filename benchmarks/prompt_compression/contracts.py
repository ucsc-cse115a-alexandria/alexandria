from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field

from alexandria.ir.contracts import MergeMetrics

if TYPE_CHECKING:
    from pathlib import Path


class PromptParts(BaseModel):
    """A lossless prompt split whose long context may be replaced independently."""

    model_config = ConfigDict(frozen=True)
    prefix: str = ""
    context: str = Field(min_length=1)
    suffix: str = ""

    @computed_field
    @property
    def prompt(self) -> str:
        return self.prefix + self.context + self.suffix

    def replace_context(self, context: str) -> PromptParts:
        return self.model_copy(update={"context": context})


class BenchmarkCase(BaseModel):
    """One static, automatically scored model input."""

    model_config = ConfigDict(frozen=True)
    key: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    task: str = Field(min_length=1)
    prompt_parts: PromptParts
    expected_answers: tuple[str, ...] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @computed_field
    @property
    def prompt(self) -> str:
        return self.prompt_parts.prompt


class BenchmarkVerdict(BaseModel):
    """Case-level official or official-equivalent score."""

    model_config = ConfigDict(frozen=True)
    score: float = Field(ge=0.0, le=1.0)
    correct: bool
    parsed_answer: str | None = None


class UsageRecord(BaseModel):
    """One metered API call made while compressing or answering."""

    model_config = ConfigDict(frozen=True)
    category: str
    model: str
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)


class GenerationResult(BaseModel):
    """Text returned by the answer model and stable response metadata."""

    model_config = ConfigDict(frozen=True)
    text: str
    model: str
    response_id: str | None = None


class ConditionRecord(BaseModel):
    """One case evaluated under original or a named compression condition."""

    model_config = ConfigDict(frozen=True)
    case_key: str
    benchmark: str
    task: str
    condition: str
    reduction_percent: float = Field(ge=0.0, lt=100.0)
    source_tokens: int = Field(ge=1)
    target_tokens: int = Field(ge=1)
    sent_tokens: int = Field(ge=1)
    prompt_sha256: str
    response: str
    response_model: str
    response_id: str | None = None
    verdict: BenchmarkVerdict
    configured_cos_sim_diff_budget: float | None = Field(default=None, ge=0.0)
    context_embedding_cosine_difference: float = Field(default=0.0, ge=0.0)
    context_cos_sim_diff_budget_met: bool | None = None
    prompt_embedding_cosine_difference: float = Field(default=0.0, ge=0.0)
    compression_elapsed_seconds: float = Field(default=0.0, ge=0.0)
    answer_elapsed_seconds: float = Field(default=0.0, ge=0.0)
    merge_metrics: MergeMetrics = MergeMetrics()
    usage: tuple[UsageRecord, ...] = ()
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @computed_field
    @property
    def token_reduction(self) -> float:
        return 1.0 - self.sent_tokens / self.source_tokens


class BenchmarkAdapter(Protocol):
    """Dataset-specific loading and response verification behind the common runner."""

    name: str
    default_data_dir: Path
    provenance: ClassVar[dict[str, str]]

    def load_cases(
        self,
        n: int | None,
        *,
        seed: int,
        data_dir: Path,
        min_source_tokens: int,
        max_source_tokens: int | None,
    ) -> tuple[BenchmarkCase, ...]: ...

    def verify(self, case: BenchmarkCase, response: str) -> BenchmarkVerdict: ...
