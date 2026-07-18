from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.ifeval.cases import InstructionCheck, load_cases

NO_COMMA_LINE = (
    '{"key": 3615, "prompt": "Write a riddle about Camilla that doesn\'t use commas.",'
    ' "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}'
)
NUMBER_WORDS_LINE = (
    '{"key": 1092, "prompt": "Write a short blog post about a trip to Japan using less than 300 words.",'
    ' "instruction_id_list": ["length_constraints:number_words"],'
    ' "kwargs": [{"relation": "less than", "num_words": 300}]}'
)


def write_data(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "input_data.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_cases_parses_official_lines(tmp_path: Path) -> None:
    cases = load_cases(data_path=write_data(tmp_path, [NO_COMMA_LINE, NUMBER_WORDS_LINE]))
    assert [case.key for case in cases] == [3615, 1092]
    assert cases[0].checks == (InstructionCheck(instruction_id="punctuation:no_comma", kwargs={}),)
    assert cases[1].checks[0].kwargs == {"relation": "less than", "num_words": 300}


def test_load_cases_rejects_mismatched_kwargs(tmp_path: Path) -> None:
    bad = json.dumps({"key": 1, "prompt": "p", "instruction_id_list": ["punctuation:no_comma"], "kwargs": []})
    with pytest.raises(ValidationError):
        load_cases(data_path=write_data(tmp_path, [bad]))


def test_load_cases_selects_longest_prompts(tmp_path: Path) -> None:
    cases = load_cases(n=1, data_path=write_data(tmp_path, [NO_COMMA_LINE, NUMBER_WORDS_LINE]))
    assert [case.key for case in cases] == [1092]


def test_load_cases_missing_file_names_the_download_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_ifeval_data"):
        load_cases(data_path=tmp_path / "missing.jsonl")


def test_verify_strict_and_loose_agree_on_clean_pass(tmp_path: Path) -> None:
    (case,) = load_cases(data_path=write_data(tmp_path, [NO_COMMA_LINE]))
    verdict = case.verify("A riddle without any comma at all")
    assert verdict.strict == (True,)
    assert verdict.loose == (True,)


def test_verify_loose_forgives_a_comma_in_the_first_line(tmp_path: Path) -> None:
    (case,) = load_cases(data_path=write_data(tmp_path, [NO_COMMA_LINE]))
    verdict = case.verify("Sure, here is your riddle:\nNo commas in this line")
    assert verdict.strict == (False,)
    assert verdict.loose == (True,)


def test_verify_counts_words(tmp_path: Path) -> None:
    (case,) = load_cases(data_path=write_data(tmp_path, [NUMBER_WORDS_LINE]))
    assert case.verify("A short trip note.").strict == (True,)
    assert case.verify(" ".join(["word"] * 301)).strict == (False,)
