from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from alexandria.utils.tokens import count_tokens
from benchmarks.prompt_compression.adapters import balanced_sample
from benchmarks.prompt_compression.contracts import BenchmarkCase, BenchmarkVerdict, PromptParts

PROMPT_PREFIX = "Please read the following text and answer the question below.\n\n<text>\n"
PROMPT_SUFFIX = (
    "\n</text>\n\nWhat is the correct answer to this question: {question}\n"
    "Choices:\n(A) {choice_a}\n(B) {choice_b}\n(C) {choice_c}\n(D) {choice_d}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)
_ANSWER = re.compile(r"The correct answer is \(?([A-D])\)?")


class _RawLongBenchCase(BaseModel):
    id: str = Field(alias="_id")
    domain: str
    sub_domain: str
    difficulty: str
    length: str
    question: str
    choice_A: str
    choice_B: str
    choice_C: str
    choice_D: str
    answer: str
    context: str = Field(min_length=1)

    def to_case(self) -> BenchmarkCase:
        suffix = PROMPT_SUFFIX.format(
            question=self.question.strip(),
            choice_a=self.choice_A.strip(),
            choice_b=self.choice_B.strip(),
            choice_c=self.choice_C.strip(),
            choice_d=self.choice_D.strip(),
        )
        return BenchmarkCase(
            key=self.id,
            benchmark="longbench_v2",
            task=self.domain,
            prompt_parts=PromptParts(prefix=PROMPT_PREFIX, context=self.context.strip(), suffix=suffix),
            expected_answers=(self.answer,),
            metadata={
                "sub_domain": self.sub_domain,
                "difficulty": self.difficulty,
                "length": self.length,
            },
        )


class LongBenchV2Adapter:
    name = "longbench_v2"
    default_data_dir = Path("data/longbench_v2")
    provenance: ClassVar[dict[str, str]] = {
        "dataset": "zai-org/LongBench-v2@2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        "prompt_and_metric": "THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e",
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
        path = data_dir if data_dir.is_file() else data_dir / "data.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `uv run python -m scripts.download_longbench_v2_data`")
        rows = json.loads(path.read_text(encoding="utf-8"))
        cases = [_RawLongBenchCase.model_validate(row).to_case() for row in rows]
        eligible = [
            case
            for case in cases
            if count_tokens(case.prompt) >= min_source_tokens
            and (max_source_tokens is None or count_tokens(case.prompt) <= max_source_tokens)
        ]
        return balanced_sample(eligible, n, seed=seed)

    def verify(self, case: BenchmarkCase, response: str) -> BenchmarkVerdict:
        cleaned = response.replace("*", "")
        match = _ANSWER.search(cleaned)
        parsed = match.group(1) if match is not None else None
        correct = parsed == case.expected_answers[0]
        return BenchmarkVerdict(score=float(correct), correct=correct, parsed_answer=parsed)
