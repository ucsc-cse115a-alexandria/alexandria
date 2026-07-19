from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel, Field

from alexandria.utils.tokens import count_tokens
from benchmarks.prompt_compression.adapters import balanced_sample
from benchmarks.prompt_compression.contracts import BenchmarkCase, BenchmarkVerdict, PromptParts


class _RawRULERCase(BaseModel):
    index: int | str | None = None
    question: str = Field(min_length=1)
    expected_answer: object
    length: int | None = None


def _answer_strings(value: object) -> tuple[str, ...]:
    answers = tuple(str(item) for item in cast("list[object]", value)) if isinstance(value, list) else (str(value),)
    if not answers or any(not answer for answer in answers):
        raise ValueError("RULERv2 expected_answer must contain at least one non-empty value")
    return answers


_SUFFIX_MARKERS: dict[str, tuple[str, ...]] = {
    "mk_niah_basic": ("\nWhat is the special magic",),
    "mv_niah_basic": ("\nWhat are all the special magic",),
    "mk_niah_easy": ("\n\nPlease copy the ",),
    "mk_niah_medium": ("\n\nHere are some examples",),
    "mk_niah_hard": ("\n\nHere are some examples",),
    "mv_niah_easy": ("\n\nPlease find and copy",),
    "mv_niah_medium": ("\n\nPlease first copy all",),
    "mv_niah_hard": ("\n\nPlease copy the ",),
    "qa_basic": ("\n\nPlease copy the Document",),
    "qa_easy": ("\n\nQuestion: ",),
    "qa_medium": ("\n\nPlease first find and copy paste",),
    "qa_hard": ("\n\nPlease answer the following",),
}


def _prompt_parts(task: str, prompt: str) -> PromptParts:
    markers = _SUFFIX_MARKERS.get(task, ())
    suffix_start = max((prompt.rfind(marker) for marker in markers), default=-1)
    if suffix_start < 0:
        context, separator, final_problem = prompt.rpartition("\n\n")
        return (
            PromptParts(context=context, suffix=separator + final_problem)
            if separator
            else PromptParts(context=prompt)
        )

    prefix_end = prompt.find("\n\n") + 2
    if task in {"mk_niah_basic", "mv_niah_basic"}:
        prefix_end = prompt.find("\n") + 1
    if prefix_end <= 1 or prefix_end >= suffix_start:
        raise ValueError(f"cannot identify the official RULERv2 context boundaries for {task}")
    return PromptParts(
        prefix=prompt[:prefix_end], context=prompt[prefix_end:suffix_start], suffix=prompt[suffix_start:]
    )


def _word_error_similarity(prediction: str, reference: str) -> float:
    predicted_words = prediction.lower().split()
    reference_words = reference.lower().split()
    if not reference_words:
        return float(not predicted_words)
    previous = list(range(len(reference_words) + 1))
    for predicted_index, predicted_word in enumerate(predicted_words, start=1):
        current = [predicted_index]
        for reference_index, reference_word in enumerate(reference_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[reference_index] + 1,
                    previous[reference_index - 1] + (predicted_word != reference_word),
                )
            )
        previous = current
    return max(0.0, 1.0 - previous[-1] / len(reference_words))


def _reference_score(prediction: str, reference: str) -> float:
    if reference.lower() in prediction.lower():
        return 1.0
    return _word_error_similarity(prediction, reference)


_BOXED_ANSWER = re.compile(r"\\boxed\{([^{}]+)\}", re.IGNORECASE)
_LABEL_ANSWER = re.compile(r"(?:final\s+answer\s+is|answer\s*:)\s*[\(<{*_]*([A-D])\b", re.IGNORECASE)


def _multiple_choice_answer(response: str) -> str | None:
    boxed = _BOXED_ANSWER.findall(response)
    if boxed:
        matches = re.findall(r"\b([A-D])\b", boxed[-1].upper())
        if matches:
            return matches[-1]
    labeled = _LABEL_ANSWER.findall(response)
    if labeled:
        return labeled[-1].upper()
    stripped = response.strip().upper()
    return stripped if stripped in {"A", "B", "C", "D"} else None


def _paths(data_dir: Path) -> tuple[Path, ...]:
    if data_dir.is_file():
        return (data_dir,)
    nested = sorted(data_dir.glob("*/test.jsonl"))
    return tuple(nested or sorted(data_dir.glob("*.jsonl")))


class RULERv2Adapter:
    name = "ruler_v2"
    default_data_dir = Path("data/ruler_v2")
    provenance: ClassVar[dict[str, str]] = {
        "entrypoint": "NVIDIA/RULER@4809570a2a40e803bfe341773e561524224c2e7c",
        "dataset_and_metric": "NVIDIA-NeMo/Skills@74b8649734a6ecc2d3beca89311e1a5e02da48fa",
    }

    def load_cases(
        self,
        n: int | None,
        *,
        seed: int,
        data_dir: Path,
        min_source_tokens: int,
        max_source_tokens: int | None,
    ) -> tuple[BenchmarkCase, ...]:
        paths = _paths(data_dir)
        if not paths:
            raise FileNotFoundError(f"no RULERv2 JSONL files found under {data_dir}")
        cases: list[BenchmarkCase] = []
        for path in paths:
            task = path.parent.name if path.name == "test.jsonl" else path.stem
            for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                raw = _RawRULERCase.model_validate_json(line)
                tokens = count_tokens(raw.question)
                if tokens < min_source_tokens or (max_source_tokens is not None and tokens > max_source_tokens):
                    continue
                index = raw.index if raw.index is not None else row_index
                cases.append(
                    BenchmarkCase(
                        key=f"{task}:{index}",
                        benchmark=self.name,
                        task=task,
                        prompt_parts=_prompt_parts(task, raw.question),
                        expected_answers=_answer_strings(raw.expected_answer),
                        metadata={"official_length": str(raw.length or tokens)},
                    )
                )
        return balanced_sample(cases, n, seed=seed)

    def verify(self, case: BenchmarkCase, response: str) -> BenchmarkVerdict:
        if case.task in {"mk_niah_medium", "mk_niah_hard"}:
            parsed = _multiple_choice_answer(response)
            correct = parsed == case.expected_answers[0]
            return BenchmarkVerdict(score=float(correct), correct=correct, parsed_answer=parsed)

        prediction = response.split("\n\n")[-1] if case.task == "mv_niah_medium" else response
        reference_scores = [_reference_score(prediction, answer) for answer in case.expected_answers]
        score = max(reference_scores) if case.task.startswith("qa_") else sum(reference_scores) / len(reference_scores)
        return BenchmarkVerdict(score=score, correct=score == 1.0, parsed_answer=response.strip())
