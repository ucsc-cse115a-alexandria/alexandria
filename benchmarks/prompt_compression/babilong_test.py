from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.babilong_8k.cases import TASKS
from benchmarks.prompt_compression.babilong import BABILongAdapter

if TYPE_CHECKING:
    from pathlib import Path


def _write_data(data_dir: Path) -> Path:
    data_dir.mkdir()
    row = {
        "input": "Irrelevant prose. Mary journeyed to the bathroom.",
        "question": "Where is Mary?",
        "target": "bathroom",
    }
    for task in TASKS:
        (data_dir / f"{task}.json").write_text(json.dumps([row]))
    return data_dir


def test_common_adapter_preserves_fixed_babilong_prompt_and_verifier(tmp_path: Path) -> None:
    adapter = BABILongAdapter()
    cases = adapter.load_cases(
        1,
        seed=42,
        data_dir=_write_data(tmp_path / "data"),
        min_source_tokens=0,
        max_source_tokens=None,
    )
    case = cases[0]
    assert case.prompt_parts.prefix.endswith("<context>\n")
    assert case.prompt_parts.context == "Irrelevant prose. Mary journeyed to the bathroom."
    assert case.prompt_parts.suffix.startswith("\n</context>\n\nQuestion: Where is Mary?")
    assert adapter.verify(case, "bathroom").correct
    assert not adapter.verify(case, "bedroom").correct
