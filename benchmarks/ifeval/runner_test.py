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


def test_compare_baseline_row_shows_fractions_and_percent() -> None:
    result = run_experiment(CASES, generate, label="original", model="stub")
    table = compare(result)
    assert (
        table.splitlines()[0]
        == "| condition | tokens saved | prompt_strict | inst_strict | prompt_loose | inst_loose |"
    )
    assert "| original | 0.0% | 1/2 (50.0%) | 1/2 (50.0%) | 1/2 (50.0%) | 1/2 (50.0%) |" in table


def test_compare_later_rows_show_delta_vs_baseline() -> None:
    original = run_experiment(CASES, generate, label="original", model="stub")
    compressed = run_experiment(CASES, generate, label="c95", model="stub", transform=truncate)
    row = compare(original, compressed).splitlines()[3]
    # same 1/2 strict outcome as baseline -> ±0.0pt; tokens were actually saved
    assert row.startswith("| c95 | ")
    assert "| 1/2 (±0.0pt) |" in row
    assert "| 0.0% |" not in row


def test_compare_delta_is_signed_percentage_points() -> None:
    passing = run_experiment(CASES[:1], generate, label="original", model="stub")
    failing = run_experiment(CASES[1:], generate, label="worse", model="stub")
    row = compare(passing, failing).splitlines()[3]
    assert "| 0/1 (-100.0pt) |" in row
