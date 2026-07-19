from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from benchmarks.babilong_8k.cases import TASKS
from scripts.prompt_compression_benchmark import condition_name, main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from _pytest.capture import CaptureFixture


def test_condition_name_uses_retained_percentage() -> None:
    assert condition_name(50) == "keep50"
    assert condition_name(25) == "keep75"
    assert condition_name(10) == "keep90"
    assert condition_name(5) == "keep95"


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
    assert manifest["compression"]["compressible_prompt_part"] == "context"
    assert not (output_dir / "manifest.json").exists()
