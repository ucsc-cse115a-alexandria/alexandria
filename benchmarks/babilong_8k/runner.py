from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, computed_field

from alexandria.ir.contracts import MergeMetrics
from benchmarks.babilong_8k.cases import CaseVerdict, TaskName

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from benchmarks.babilong_8k.cases import BABILongCase

_ENCODING = tiktoken.get_encoding("cl100k_base")


class CaseRecord(BaseModel):
    """One BABILong case run under one experiment condition."""

    model_config = ConfigDict(frozen=True)
    key: str
    task: TaskName
    prompt: str
    source_tokens: int = Field(ge=1)
    sent_tokens: int = Field(ge=1)
    response: str
    verdict: CaseVerdict
    merge_metrics: MergeMetrics = MergeMetrics()

    @computed_field
    @property
    def ratio(self) -> float:
        return self.sent_tokens / self.source_tokens


class ExperimentResult(BaseModel):
    """All BABILong records for one condition plus task accuracy and token totals."""

    model_config = ConfigDict(frozen=True)
    label: str
    model: str
    records: tuple[CaseRecord, ...] = Field(min_length=1)

    @computed_field
    @property
    def accuracy(self) -> float:
        return sum(record.verdict.correct for record in self.records) / len(self.records)

    @computed_field
    @property
    def mean_source_tokens(self) -> float:
        return sum(record.source_tokens for record in self.records) / len(self.records)

    @computed_field
    @property
    def mean_sent_tokens(self) -> float:
        return sum(record.sent_tokens for record in self.records) / len(self.records)

    @computed_field
    @property
    def token_reduction(self) -> float:
        return 1 - sum(record.sent_tokens for record in self.records) / sum(
            record.source_tokens for record in self.records
        )

    @computed_field
    @property
    def merge_calls(self) -> int:
        return sum(record.merge_metrics.calls for record in self.records)

    @computed_field
    @property
    def merge_retries(self) -> int:
        return sum(record.merge_metrics.retries for record in self.records)


class CompressionResult(BaseModel):
    """A compressed prompt plus generation work used to produce it."""

    model_config = ConfigDict(frozen=True)
    prompt: str = Field(min_length=1)
    merge_metrics: MergeMetrics


def run_experiment(
    cases: Sequence[BABILongCase],
    generate: Callable[[str], str],
    *,
    label: str,
    model: str,
    transform: Callable[[str], str | CompressionResult] | None = None,
) -> ExperimentResult:
    """Optionally compress every full input, answer it, and apply the official-equivalent verifier."""
    records: list[CaseRecord] = []
    for case in cases:
        transformed = case.prompt if transform is None else transform(case.prompt)
        if isinstance(transformed, CompressionResult):
            sent = transformed.prompt
            merge_metrics = transformed.merge_metrics
        else:
            sent = transformed
            merge_metrics = MergeMetrics()
        response = generate(sent)
        records.append(
            CaseRecord(
                key=case.key,
                task=case.task,
                prompt=sent,
                source_tokens=len(_ENCODING.encode(case.prompt)),
                sent_tokens=len(_ENCODING.encode(sent)),
                response=response,
                verdict=case.verify(response),
                merge_metrics=merge_metrics,
            )
        )
    return ExperimentResult(label=label, model=model, records=tuple(records))


def compare(*results: ExperimentResult) -> str:
    """Render token reduction and downstream task accuracy against the first condition."""
    if not results:
        raise ValueError("at least one result is required")
    baseline = results[0]
    baseline_keys = tuple(record.key for record in baseline.records)
    if any(tuple(record.key for record in result.records) != baseline_keys for result in results[1:]):
        raise ValueError("all results must contain the same cases in the same order")
    header = (
        "| Condition | Mean input tokens | Token reduction | Merge calls | Retries | Task accuracy | Accuracy change |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|"
    rows: list[str] = []
    for index, result in enumerate(results):
        passed = sum(record.verdict.correct for record in result.records)
        if index == 0:
            delta = "—"
        else:
            delta_pp = 100 * (result.accuracy - baseline.accuracy)
            delta = "±0.0 pp" if delta_pp == 0 else f"{delta_pp:+.1f} pp"
        rows.append(
            f"| {result.label} | {result.mean_sent_tokens:,.1f} | {result.token_reduction * 100:.1f}% | "
            f"{result.merge_calls} | {result.merge_retries} | "
            f"{result.accuracy * 100:.1f}% ({passed}/{len(result.records)}) | {delta} |"
        )
    return "\n".join([header, divider, *rows])
