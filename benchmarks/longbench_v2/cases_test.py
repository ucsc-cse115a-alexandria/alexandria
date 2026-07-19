from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.longbench_v2.cases import PROMPT_PREFIX, LongBenchV2Adapter

if TYPE_CHECKING:
    from pathlib import Path


def _row() -> dict[str, str]:
    return {
        "_id": "lb-1",
        "domain": "Code Repository Understanding",
        "sub_domain": "python",
        "difficulty": "easy",
        "length": "short",
        "question": "Which option is correct?",
        "choice_A": "alpha",
        "choice_B": "beta",
        "choice_C": "gamma",
        "choice_D": "delta",
        "answer": "B",
        "context": "A long repository context.",
    }


def test_loads_official_shape_and_reproduces_prompt(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text(json.dumps([_row()]))
    adapter = LongBenchV2Adapter()
    case = adapter.load_cases(1, seed=42, data_dir=tmp_path, min_source_tokens=0, max_source_tokens=None)[0]
    assert case.prompt.startswith(PROMPT_PREFIX)
    assert case.prompt_parts.context == "A long repository context."
    assert "(B) beta" in case.prompt_parts.suffix
    assert adapter.verify(case, "The correct answer is (B)").correct
    assert not adapter.verify(case, "B").correct
