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


_METRICS = (
    ("prompt_strict", "Prompt-level strict accuracy"),
    ("inst_strict", "Instruction-level strict accuracy"),
    ("prompt_loose", "Prompt-level loose accuracy"),
    ("inst_loose", "Instruction-level loose accuracy"),
)


def _counts(result: ExperimentResult) -> dict[str, tuple[int, int]]:
    strict_flags = [flag for record in result.records for flag in record.verdict.strict]
    loose_flags = [flag for record in result.records for flag in record.verdict.loose]
    return {
        "prompt_strict": (sum(all(record.verdict.strict) for record in result.records), len(result.records)),
        "inst_strict": (sum(strict_flags), len(strict_flags)),
        "prompt_loose": (sum(all(record.verdict.loose) for record in result.records), len(result.records)),
        "inst_loose": (sum(loose_flags), len(loose_flags)),
    }


def compare(*results: ExperimentResult) -> str:
    """A markdown table of mean compression and official IFEval accuracies.

    Each accuracy is an IFEval mean: prompt-level means every instruction in a prompt must
    pass; instruction-level means each instruction is counted separately. Later rows show
    their change against the first result in percentage points.
    """
    baseline = _counts(results[0])
    header = "| Condition | Mean compression | " + " | ".join(label for _, label in _METRICS) + " |"
    divider = "|" + "|".join("---" for _ in range(len(_METRICS) + 2)) + "|"
    rows: list[str] = []
    for index, result in enumerate(results):
        cells = [result.label, f"{(1 - result.mean_ratio) * 100:.1f}%"]
        counts = _counts(result)
        for metric, _ in _METRICS:
            passed, total = counts[metric]
            percent = 100 * passed / total
            if index == 0:
                cells.append(f"{percent:.1f}% ({passed}/{total})")
            else:
                base_passed, base_total = baseline[metric]
                delta = percent - 100 * base_passed / base_total
                delta_text = "±0.0 pp" if delta == 0 else f"{delta:+.1f} pp"
                cells.append(f"{percent:.1f}% ({delta_text}; {passed}/{total})")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])
