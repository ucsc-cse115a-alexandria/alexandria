from __future__ import annotations

from pathlib import Path
from typing import ClassVar, cast

from alexandria.utils.tokens import count_tokens
from benchmarks.babilong_8k import TaskName, load_cases
from benchmarks.prompt_compression.adapters import balanced_sample
from benchmarks.prompt_compression.contracts import BenchmarkCase, BenchmarkVerdict, PromptParts


def _parts(prompt: str) -> PromptParts:
    opening = "<context>\n"
    closing = "\n</context>"
    prefix, separator, remainder = prompt.partition(opening)
    context, closing_separator, suffix = remainder.rpartition(closing)
    if not separator or not closing_separator:
        raise ValueError("BABILong prompt does not contain the expected context boundaries")
    return PromptParts(prefix=prefix + opening, context=context, suffix=closing + suffix)


class BABILongAdapter:
    name = "babilong_8k"
    default_data_dir = Path("data/babilong/8k")
    provenance: ClassVar[dict[str, str]] = {
        "dataset": "RMT-team/babilong@ee0d588794c7ac098062ee0d247c733d62e94fe2",
        "prompt_and_metric": "booydar/babilong@38da79d79519ef87aa46ae804f838e1eab7f86d7",
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
        converted = [
            BenchmarkCase(
                key=case.key,
                benchmark=self.name,
                task=case.task,
                prompt_parts=_parts(case.prompt),
                expected_answers=(case.target,),
                metadata={"question": case.question, "source_index": str(case.source_index)},
            )
            for case in load_cases(data_dir=data_dir)
            if count_tokens(case.prompt) >= min_source_tokens
            and (max_source_tokens is None or count_tokens(case.prompt) <= max_source_tokens)
        ]
        return balanced_sample(converted, n, seed=seed)

    def verify(self, case: BenchmarkCase, response: str) -> BenchmarkVerdict:
        from benchmarks.babilong_8k.cases import BABILongCase

        original = BABILongCase(
            key=case.key,
            task=cast("TaskName", case.task),
            source_index=int(case.metadata["source_index"]),
            prompt=case.prompt,
            question=case.metadata["question"],
            target=case.expected_answers[0],
        )
        correct = original.verify(response).correct
        return BenchmarkVerdict(score=float(correct), correct=correct, parsed_answer=response.strip())
