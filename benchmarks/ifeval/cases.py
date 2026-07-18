from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.ifeval.vendor import evaluation_lib

DATA_PATH = Path("data/ifeval/input_data.jsonl")

type KwargValue = str | int | float | list[str]


class InstructionCheck(BaseModel):
    """One official instruction id with the parameters its checker is built from."""

    model_config = ConfigDict(frozen=True)
    instruction_id: str
    kwargs: dict[str, KwargValue]


class CaseVerdict(BaseModel):
    """Official strict and loose pass/fail, one entry per instruction."""

    model_config = ConfigDict(frozen=True)
    strict: tuple[bool, ...]
    loose: tuple[bool, ...]


class IFEvalCase(BaseModel):
    """One official IFEval prompt bundled with its instructions and verifier."""

    model_config = ConfigDict(frozen=True)
    key: int
    prompt: str
    checks: tuple[InstructionCheck, ...] = Field(min_length=1)

    def verify(self, response: str) -> CaseVerdict:
        """Judge a response with the official strict and loose evaluations."""
        example = evaluation_lib.InputExample(
            key=self.key,
            instruction_id_list=[check.instruction_id for check in self.checks],
            prompt=self.prompt,
            kwargs=[dict(check.kwargs) for check in self.checks],
        )
        responses = {self.prompt: response}
        strict = evaluation_lib.test_instruction_following_strict(example, responses)
        loose = evaluation_lib.test_instruction_following_loose(example, responses)
        return CaseVerdict(
            strict=tuple(strict.follow_instruction_list),
            loose=tuple(loose.follow_instruction_list),
        )


class _RawCase(BaseModel):
    """Schema of one line of the official input_data.jsonl."""

    key: int
    prompt: str
    instruction_id_list: list[str] = Field(min_length=1)
    kwargs: list[dict[str, KwargValue]]

    @model_validator(mode="after")
    def _lengths_match(self) -> Self:
        if len(self.kwargs) != len(self.instruction_id_list):
            raise ValueError("kwargs and instruction_id_list must have the same length")
        return self

    def to_case(self) -> IFEvalCase:
        checks = tuple(
            InstructionCheck(instruction_id=instruction_id, kwargs=kwargs)
            for instruction_id, kwargs in zip(self.instruction_id_list, self.kwargs, strict=True)
        )
        return IFEvalCase(key=self.key, prompt=self.prompt, checks=checks)


def load_cases(n: int | None = None, data_path: Path = DATA_PATH) -> tuple[IFEvalCase, ...]:
    """The official IFEval cases; with n, the n longest prompts by character count (ties by key)."""
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found; run `uv run python -m scripts.download_ifeval_data` first")
    with data_path.open() as f:
        cases = [_RawCase.model_validate(json.loads(line)).to_case() for line in f]
    if n is not None:
        cases = sorted(cases, key=lambda case: (-len(case.prompt), case.key))[:n]
    return tuple(cases)
