from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from alexandria.ir.contracts import MergeMetrics
from benchmarks.babilong_8k.cases import TASKS
from benchmarks.prompt_compression.contracts import PromptParts
from scripts.prompt_compression_benchmark import compress_parts, condition_name, main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from _pytest.capture import CaptureFixture


def test_condition_name_uses_retained_percentage() -> None:
    assert condition_name(50) == "keep50"
    assert condition_name(25) == "keep75"
    assert condition_name(10) == "keep90"
    assert condition_name(5) == "keep95"


def test_compress_parts_uses_compare_cos_sim_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(text="short context", merge_metrics=MergeMetrics())
    comparison = SimpleNamespace(cos_sim_diff=0.125)
    monkeypatch.setattr("scripts.prompt_compression_benchmark.reduce", lambda *_, **__: result)
    monkeypatch.setattr("scripts.prompt_compression_benchmark.compare", lambda *_, **__: comparison)
    monkeypatch.setattr("scripts.prompt_compression_benchmark.default_embedder", lambda: object())
    monkeypatch.setattr("scripts.prompt_compression_benchmark.default_merger", lambda **__: object())

    compressed = compress_parts(PromptParts(context="one two three four five six seven eight"), 50)

    assert compressed[-1] == 0.125


def test_dry_run_prints_reproducible_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = {
        "input": "Irrelevant prose. Mary journeyed to the bathroom.",
        "question": "Where is Mary?",
        "target": "bathroom",
    }
    for task in TASKS:
        (data_dir / f"{task}.json").write_text(json.dumps([row]))
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prompt_compression_benchmark",
            "--benchmark",
            "babilong_8k",
            "--n",
            "1",
            "--data-dir",
            str(data_dir),
            "--out",
            str(output_dir),
            "--dry-run",
        ],
    )
    main()
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["case_keys"] == ["qa1:0"]
    assert manifest["command"].endswith(f"--out {output_dir} --dry-run")
    assert isinstance(manifest["implementation_dirty"], bool)
    assert manifest["compression"]["compressible_prompt_part"] == "context"
    assert not (output_dir / "manifest.json").exists()
