from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.babilong_8k.prompts import TaskName, format_prompt

DATA_DIR = Path("data/babilong/8k")
TASKS: tuple[TaskName, ...] = ("qa1", "qa2", "qa3", "qa4", "qa5")

TASK_LABELS: dict[TaskName, frozenset[str]] = {
    "qa1": frozenset({"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"}),
    "qa2": frozenset({"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"}),
    "qa3": frozenset({"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"}),
    "qa4": frozenset({"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"}),
    "qa5": frozenset({"bill", "fred", "jeff", "mary", "apple", "football", "milk"}),
}


class CaseVerdict(BaseModel):
    """Official-equivalent BABILong label correctness for one response."""

    model_config = ConfigDict(frozen=True)
    correct: bool


class BABILongCase(BaseModel):
    """One BABILong 8k input bundled with its deterministic label verifier."""

    model_config = ConfigDict(frozen=True)
    key: str
    task: TaskName
    source_index: int = Field(ge=0)
    prompt: str = Field(min_length=1)
    question: str = Field(min_length=1)
    target: str = Field(min_length=1)

    def verify(self, response: str) -> CaseVerdict:
        """Apply BABILong's qa1-qa5 single-label comparison."""
        output = response.lower().split(".", 1)[0]
        output = output.split("<context>", 1)[0].split("<example>", 1)[0]
        labels = TASK_LABELS[self.task]
        labels_in_question = {label for label in labels if label in self.question.lower()}
        labels_in_output = {label for label in labels if label in output} - labels_in_question
        return CaseVerdict(correct=self.target.lower() in labels_in_output and len(labels_in_output) == 1)


class _RawCase(BaseModel):
    input: str = Field(min_length=1)
    question: str = Field(min_length=1)
    target: str = Field(min_length=1)

    def to_case(self, task: TaskName, source_index: int) -> BABILongCase:
        return BABILongCase(
            key=f"{task}:{source_index}",
            task=task,
            source_index=source_index,
            prompt=format_prompt(task, self.input, self.question),
            question=self.question,
            target=self.target,
        )


def _load_task(task: TaskName, data_dir: Path) -> list[BABILongCase]:
    path = data_dir / f"{task}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run python -m scripts.download_babilong_8k_data` first")
    raw_cases = [_RawCase.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    return [raw.to_case(task, index) for index, raw in enumerate(raw_cases)]


def load_cases(
    n: int | None = None,
    *,
    seed: int = 42,
    data_dir: Path = DATA_DIR,
) -> tuple[BABILongCase, ...]:
    """Load qa1-qa5, optionally taking a reproducible task-balanced sample."""
    by_task = {task: _load_task(task, data_dir) for task in TASKS}
    if n is None:
        return tuple(case for task in TASKS for case in by_task[task])
    if n < 1:
        raise ValueError("n must be at least 1")
    total = sum(len(cases) for cases in by_task.values())
    if n > total:
        raise ValueError(f"requested {n} cases, but only {total} are available")

    per_task, extra = divmod(n, len(TASKS))
    selected: list[BABILongCase] = []
    for task_index, task in enumerate(TASKS):
        quota = per_task + (task_index < extra)
        cases = list(by_task[task])
        random.Random(seed + task_index).shuffle(cases)
        if quota > len(cases):
            raise ValueError(f"task {task} has only {len(cases)} cases, but balanced sampling needs {quota}")
        selected.extend(cases[:quota])
    return tuple(selected)
