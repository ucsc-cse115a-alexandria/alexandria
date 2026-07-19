from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.ruler_v2.cases import RULERv2Adapter

if TYPE_CHECKING:
    from pathlib import Path


def test_loads_official_jsonl_shape_and_protects_final_problem(tmp_path: Path) -> None:
    task_dir = tmp_path / "mk_niah_basic"
    task_dir.mkdir()
    row = {
        "index": 7,
        "question": "Long haystack with key 123.\n\nWhat is the key?",
        "expected_answer": ["123"],
        "length": 32,
    }
    (task_dir / "test.jsonl").write_text(json.dumps(row) + "\n")
    adapter = RULERv2Adapter()
    case = adapter.load_cases(1, seed=42, data_dir=tmp_path, min_source_tokens=0, max_source_tokens=None)[0]
    assert case.key == "mk_niah_basic:7"
    assert case.prompt_parts.context == "Long haystack with key 123."
    assert case.prompt_parts.suffix == "\n\nWhat is the key?"
    assert adapter.verify(case, "The answer is 123").correct


def test_adapter_separates_official_basic_instruction_haystack_and_query(tmp_path: Path) -> None:
    path = tmp_path / "mk_niah_basic.jsonl"
    question = (
        "A special magic number is hidden within the following text. Make sure to memorize it. "
        "I will quiz you about the number afterwards.\n"
        "One of the special magic numbers for blue-key is: 123.\n"
        "What is the special magic number for blue-key mentioned in the provided text? "
        "The special magic number for blue-key mentioned in the provided text is"
    )
    path.write_text(json.dumps({"question": question, "expected_answer": ["123"]}) + "\n")
    case = RULERv2Adapter().load_cases(1, seed=42, data_dir=path, min_source_tokens=0, max_source_tokens=None)[0]
    assert case.prompt == question
    assert case.prompt_parts.prefix.endswith("afterwards.\n")
    assert case.prompt_parts.context == "One of the special magic numbers for blue-key is: 123."
    assert case.prompt_parts.suffix.startswith("\nWhat is the special magic")


def test_multivalue_verifier_awards_partial_credit(tmp_path: Path) -> None:
    path = tmp_path / "mv_niah_basic.jsonl"
    path.write_text(json.dumps({"question": "context", "expected_answer": ["one", "two"]}) + "\n")
    adapter = RULERv2Adapter()
    case = adapter.load_cases(1, seed=42, data_dir=path, min_source_tokens=0, max_source_tokens=None)[0]
    verdict = adapter.verify(case, "one")
    assert verdict.score == 0.5
    assert not verdict.correct


def test_verifier_reproduces_task_specific_ruler2_matching(tmp_path: Path) -> None:
    rows = {
        "mv_niah_medium": ("target answer", "wrong draft\n\ntarget answer"),
        "qa_hard": (["accepted one", "accepted two"], "accepted two"),
        "mk_niah_hard": ("C", "The final answer is \\boxed{C}"),
    }
    adapter = RULERv2Adapter()
    for task, (expected, response) in rows.items():
        path = tmp_path / f"{task}.jsonl"
        path.write_text(json.dumps({"question": "context", "expected_answer": expected}) + "\n")
        case = adapter.load_cases(1, seed=42, data_dir=path, min_source_tokens=0, max_source_tokens=None)[0]
        assert adapter.verify(case, response).correct


def test_verifier_uses_official_word_error_similarity(tmp_path: Path) -> None:
    path = tmp_path / "mk_niah_basic.jsonl"
    path.write_text(json.dumps({"question": "context", "expected_answer": ["one two"]}) + "\n")
    adapter = RULERv2Adapter()
    case = adapter.load_cases(1, seed=42, data_dir=path, min_source_tokens=0, max_source_tokens=None)[0]
    assert adapter.verify(case, "one three").score == 0.5
