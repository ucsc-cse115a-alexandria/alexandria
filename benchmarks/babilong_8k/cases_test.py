from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmarks.babilong_8k.cases import TASKS, BABILongCase, load_cases

if TYPE_CHECKING:
    from pathlib import Path


def write_data(data_dir: Path, cases_per_task: int = 3) -> Path:
    data_dir.mkdir()
    for task in TASKS:
        rows = [
            {
                "input": f"Irrelevant prose. Mary journeyed to the bathroom. Row {index}.",
                "question": "Where is Mary?",
                "target": "bathroom",
            }
            for index in range(cases_per_task)
        ]
        (data_dir / f"{task}.json").write_text(json.dumps(rows))
    return data_dir


def test_load_cases_builds_full_official_prompt(tmp_path: Path) -> None:
    cases = load_cases(data_dir=write_data(tmp_path / "data"))
    assert len(cases) == 15
    prompt = cases[0].prompt
    assert "<example>" in prompt
    assert "<context>\nIrrelevant prose." in prompt
    assert "Question: Where is Mary?" in prompt


def test_load_cases_balances_reproducible_sample(tmp_path: Path) -> None:
    data_dir = write_data(tmp_path / "data")
    first = load_cases(n=10, seed=7, data_dir=data_dir)
    second = load_cases(n=10, seed=7, data_dir=data_dir)
    assert [case.key for case in first] == [case.key for case in second]
    assert {task: sum(case.task == task for case in first) for task in TASKS} == dict.fromkeys(TASKS, 2)


def test_load_cases_missing_data_names_downloader(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_babilong_8k_data"):
        load_cases(data_dir=tmp_path / "missing")


def test_verify_accepts_only_the_target_label() -> None:
    case = BABILongCase(
        key="qa1:0",
        task="qa1",
        source_index=0,
        prompt="prompt",
        question="Where is Mary?",
        target="bathroom",
    )
    assert case.verify("The most recent location of Mary is bathroom.").correct
    assert case.verify("bathroom").correct
    assert not case.verify("bedroom").correct
    assert not case.verify("bathroom or bedroom").correct


def test_verify_ignores_labels_already_in_question() -> None:
    case = BABILongCase(
        key="qa5:0",
        task="qa5",
        source_index=0,
        prompt="prompt",
        question="Who did Mary give the apple to?",
        target="Fred",
    )
    assert case.verify("Mary gave the apple to Fred.").correct
