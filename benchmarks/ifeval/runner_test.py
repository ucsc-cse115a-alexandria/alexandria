from __future__ import annotations

from benchmarks.ifeval.cases import IFEvalCase, InstructionCheck
from benchmarks.ifeval.runner import compare, run_experiment

NO_COMMA = InstructionCheck(instruction_id="punctuation:no_comma", kwargs={})
CASES = (
    IFEvalCase(key=1, prompt="Write a haiku about tea without using commas.", checks=(NO_COMMA,)),
    IFEvalCase(key=2, prompt="Write a couplet about rain without using commas.", checks=(NO_COMMA,)),
)

RESPONSES = {
    "Write a haiku about tea without using commas.": "Steam rises softly",
    "Write a couplet about rain without using commas.": "Rain, rain",
    "Write a haiku about tea": "Tea, tea",
    "Write a couplet about rain": "Rain falls",
}


def generate(prompt: str) -> str:
    return RESPONSES[prompt]


def truncate(prompt: str) -> str:
    return prompt.rsplit(" without", 1)[0]


def test_run_experiment_aggregates_verdicts() -> None:
    result = run_experiment(CASES, generate, label="original", model="stub")
    assert result.prompt_strict == 0.5
    assert result.inst_strict == 0.5
    assert result.prompt_loose == 0.5
    assert result.inst_loose == 0.5
    assert result.mean_ratio == 1.0
    assert [record.key for record in result.records] == [1, 2]


def test_run_experiment_records_transformed_prompts_and_ratio() -> None:
    result = run_experiment(CASES, generate, label="c95", model="stub", transform=truncate)
    first = result.records[0]
    assert first.prompt == "Write a haiku about tea"
    assert first.sent_tokens < first.source_tokens
    assert result.mean_ratio < 1.0
    assert result.records[0].verdict.strict == (False,)
    assert result.records[1].verdict.strict == (True,)


def test_result_serializes_summary_fields() -> None:
    result = run_experiment(CASES, generate, label="original", model="stub")
    dumped = result.model_dump()
    assert dumped["prompt_strict"] == 0.5
    assert dumped["records"][0]["ratio"] == 1.0


def test_compare_renders_one_row_per_result() -> None:
    result = run_experiment(CASES, generate, label="original", model="stub")
    table = compare(result)
    assert table.splitlines()[0] == "| label | n | prompt_strict | inst_strict | prompt_loose | inst_loose | mean_ratio |"
    assert "| original | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 |" in table
