from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from alexandria.ir.contracts import MergeMetrics, Params
from benchmarks.babilong_8k.cases import TASKS
from benchmarks.prompt_compression.contracts import PromptParts
from scripts.prompt_compression_benchmark import (
    compress_parts,
    compress_parts_to_cos_sim_budget,
    condition_name,
    cos_sim_budget_condition_name,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from _pytest.capture import CaptureFixture


def test_condition_name_uses_retained_percentage() -> None:
    assert condition_name(50) == "keep50"
    assert condition_name(25) == "keep75"
    assert condition_name(10) == "keep90"
    assert condition_name(5) == "keep95"
    assert cos_sim_budget_condition_name(0.0025) == "budget0p0025"


def test_compress_parts_uses_compare_cos_sim_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(text="short context", merge_metrics=MergeMetrics())
    comparison = SimpleNamespace(cos_sim_diff=0.125)

    def fake_reduce(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return result

    def fake_compare(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return comparison

    def fake_merger(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr("scripts.prompt_compression_benchmark.reduce", fake_reduce)
    monkeypatch.setattr("scripts.prompt_compression_benchmark.compare", fake_compare)
    monkeypatch.setattr("scripts.prompt_compression_benchmark.default_embedder", lambda: object())
    monkeypatch.setattr("scripts.prompt_compression_benchmark.default_merger", fake_merger)

    compressed = compress_parts(PromptParts(context="one two three four five six seven eight"), 50)

    assert compressed[-1] == 0.125


def test_cos_budget_compression_is_best_effort_and_records_context_and_prompt_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_params: list[Params] = []
    result = SimpleNamespace(text="short context", merge_metrics=MergeMetrics())
    comparisons = iter((SimpleNamespace(cos_sim_diff=0.01), SimpleNamespace(cos_sim_diff=0.008)))

    def fake_reduce(*_args: object, **kwargs: object) -> SimpleNamespace:
        params = kwargs["params"]
        assert isinstance(params, Params)
        seen_params.append(params)
        return result

    def fake_compare(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return next(comparisons)

    def fake_merger(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr("scripts.prompt_compression_benchmark.reduce", fake_reduce)
    monkeypatch.setattr("scripts.prompt_compression_benchmark.compare", fake_compare)
    monkeypatch.setattr("scripts.prompt_compression_benchmark.default_embedder", lambda: object())
    monkeypatch.setattr("scripts.prompt_compression_benchmark.default_merger", fake_merger)

    compressed = compress_parts_to_cos_sim_budget(PromptParts(context="one two three four five six seven eight"), 0.02)

    params = seen_params[0]
    assert params.cos_sim_diff_budget == 0.02
    assert params.require_target is False
    assert compressed[-3:] == (0.008, 0.01, True)


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


def test_budget_dry_run_prints_safety_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = {"input": "Context.", "question": "Where?", "target": "there"}
    for task in TASKS:
        (data_dir / f"{task}.json").write_text(json.dumps([row]))
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
            str(tmp_path / "run"),
            "--cos-sim-diff-budgets",
            "0.005",
            "0.02",
            "--max-generation-calls-per-condition",
            "12",
            "--max-condition-seconds",
            "60",
            "--max-estimated-cost-usd",
            "2",
            "--dry-run",
        ],
    )

    main()

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["experiment_mode"] == "cos_sim_diff_budget"
    assert manifest["cos_sim_diff_budgets"] == [0.005, 0.02]
    assert manifest["compression"]["require_target"] is False
    assert manifest["compression"]["max_generation_calls_per_condition"] == 12
    assert manifest["max_estimated_cost_usd"] == 2.0
