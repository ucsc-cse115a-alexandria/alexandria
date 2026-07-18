from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, computed_field

from benchmarks.ifeval.cases import CaseVerdict

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from benchmarks.ifeval.cases import IFEvalCase

_ENCODING = tiktoken.get_encoding("cl100k_base")


class CaseRecord(BaseModel):
    """One case run under one experiment condition."""

    model_config = ConfigDict(frozen=True)
    key: int
    prompt: str
    source_tokens: int = Field(ge=1)
    sent_tokens: int = Field(ge=1)
    response: str
    verdict: CaseVerdict

    @computed_field
    @property
    def ratio(self) -> float:
        """Sent tokens as a fraction of source tokens (1.0 = no compression)."""
        return self.sent_tokens / self.source_tokens


class ExperimentResult(BaseModel):
    """All records for one condition plus the four official IFEval accuracies."""

    model_config = ConfigDict(frozen=True)
    label: str
    model: str
    records: tuple[CaseRecord, ...] = Field(min_length=1)

    @computed_field
    @property
    def prompt_strict(self) -> float:
        """Fraction of cases whose response follows every instruction, strict evaluation."""
        return sum(all(record.verdict.strict) for record in self.records) / len(self.records)

    @computed_field
    @property
    def prompt_loose(self) -> float:
        """Fraction of cases whose response follows every instruction, loose evaluation."""
        return sum(all(record.verdict.loose) for record in self.records) / len(self.records)

    @computed_field
    @property
    def inst_strict(self) -> float:
        """Fraction of individual instructions followed, strict evaluation."""
        flags = [flag for record in self.records for flag in record.verdict.strict]
        return sum(flags) / len(flags)

    @computed_field
    @property
    def inst_loose(self) -> float:
        """Fraction of individual instructions followed, loose evaluation."""
        flags = [flag for record in self.records for flag in record.verdict.loose]
        return sum(flags) / len(flags)

    @computed_field
    @property
    def mean_ratio(self) -> float:
        """Mean of the per-case compression ratios."""
        return sum(record.ratio for record in self.records) / len(self.records)


def run_experiment(
    cases: Sequence[IFEvalCase],
    generate: Callable[[str], str],
    *,
    label: str,
    model: str,
    transform: Callable[[str], str] | None = None,
) -> ExperimentResult:
    """Send each case's prompt (optionally transformed) through generate and verify the response."""
    records: list[CaseRecord] = []
    for case in cases:
        sent = case.prompt if transform is None else transform(case.prompt)
        response = generate(sent)
        records.append(
            CaseRecord(
                key=case.key,
                prompt=sent,
                source_tokens=len(_ENCODING.encode(case.prompt)),
                sent_tokens=len(_ENCODING.encode(sent)),
                response=response,
                verdict=case.verify(response),
            )
        )
    return ExperimentResult(label=label, model=model, records=tuple(records))


_COLUMNS = ("label", "n", "prompt_strict", "inst_strict", "prompt_loose", "inst_loose", "mean_ratio")


def compare(*results: ExperimentResult) -> str:
    """A markdown table of the four accuracies and mean compression ratio per experiment."""
    header = "| " + " | ".join(_COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    rows = [
        f"| {result.label} | {len(result.records)} | {result.prompt_strict:.3f} | {result.inst_strict:.3f}"
        f" | {result.prompt_loose:.3f} | {result.inst_loose:.3f} | {result.mean_ratio:.3f} |"
        for result in results
    ]
    return "\n".join([header, divider, *rows])
