from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmarks.prompt_compression.sweep_runner import (
    SWEEP_POINTS,
    ExecutionKind,
    PointStatus,
    SweepPoint,
    build_index_entry,
    build_sweep_index,
    classify_point_status,
    collect_artifacts,
    extract_summary_metrics,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_sweep_point_table_covers_p0_through_p9_with_runbook_out_dirs() -> None:
    ids = [point.sweep_point for point in SWEEP_POINTS]
    assert ids == [f"P{index}" for index in range(10)]
    assert SWEEP_POINTS[0].out_dir.name == "2026-07-20-p0-original-n50-v1"
    assert SWEEP_POINTS[1].execution_kind.value == "inline"
    assert SWEEP_POINTS[2].out_dir.name == "2026-07-20-p2-budget-0.0025-n50-v1"
    assert SWEEP_POINTS[7].out_dir.name == "2026-07-20-p7-reduction-50-n50-v1"


def test_extract_summary_metrics_from_fixture_summary() -> None:
    summary = {
        "conditions": {
            "original": {"accuracy": 0.72},
            "budget0p0025": {
                "accuracy": 0.75,
                "token_reduction": 0.0065,
                "mean_prompt_embedding_cosine_difference": 0.0022,
                "compression_seconds": 100.0,
                "answer_seconds": 25.5,
                "estimated_cost_usd": 0.42,
            },
        },
        "comparisons": {
            "budget0p0025": {
                "accuracy_retention": {
                    "retention": 1.0416666666666667,
                },
            },
        },
        "expected_conditions": ["budget0p0025"],
    }
    metrics = extract_summary_metrics(summary, "P2")
    assert metrics == {
        "original_accuracy": 0.72,
        "compressed_conditions": ["budget0p0025"],
        "compressed_condition": "budget0p0025",
        "compressed_accuracy": 0.75,
        "mean_token_reduction": 0.0065,
        "mean_prompt_cosine_difference": 0.0022,
        "estimated_cost_usd": 0.42,
        "wall_clock_seconds": 125.5,
        "accuracy_retention": 1.0416666666666667,
    }


def test_collect_artifacts_tracks_evidence_contract_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (out_dir / "summary.json").write_text("{}", encoding="utf-8")
    (out_dir / "prompts.jsonl.gz").write_bytes(b"\x1f\x8b")
    (out_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (out_dir / "api_events.jsonl").write_text("{}\n", encoding="utf-8")
    (out_dir / "records.jsonl").write_text("", encoding="utf-8")

    artifacts = collect_artifacts(out_dir)
    assert artifacts == {
        "manifest": True,
        "summary": True,
        "prompts": True,
        "report": True,
        "api_events": True,
        "records_count": 0,
        "errors_count": 0,
    }


def test_p0_exit_code_one_counts_as_success_when_manifest_exists(tmp_path: Path) -> None:
    out_dir = tmp_path / "p0"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (out_dir / "summary.json").write_text("{}", encoding="utf-8")

    point = SweepPoint(
        sweep_point="P0",
        type="Baseline",
        setting="original",
        mechanism="gate",
        out_dir=out_dir,
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=1,
    )
    assert classify_point_status(point, exit_code=1, out_dir=out_dir) is PointStatus.SUCCESS
    assert classify_point_status(point, exit_code=0, out_dir=out_dir) is PointStatus.FAILED


def test_failed_point_is_recorded_without_blocking_other_index_entries(tmp_path: Path) -> None:
    success_dir = tmp_path / "ok"
    failed_dir = tmp_path / "bad"
    success_dir.mkdir()
    failed_dir.mkdir()
    summary = {
        "conditions": {
            "original": {"accuracy": 0.7},
            "keep50": {
                "accuracy": 0.5,
                "token_reduction": 0.5,
                "mean_prompt_embedding_cosine_difference": 0.07,
                "compression_seconds": 10.0,
                "answer_seconds": 5.0,
                "estimated_cost_usd": 1.25,
            },
        },
        "comparisons": {
            "keep50": {
                "accuracy_retention": {
                    "retention": 0.7142857142857143,
                },
            },
        },
    }
    (success_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (success_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    ok_point = SweepPoint(
        sweep_point="P7",
        type="Stronger, hard-target",
        setting="50% forced reduction",
        mechanism="--reductions 50",
        out_dir=success_dir,
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
    )
    bad_point = SweepPoint(
        sweep_point="P8",
        type="Stronger, hard-target",
        setting="30% forced reduction",
        mechanism="--reductions 30",
        out_dir=failed_dir,
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
    )

    index = build_sweep_index(
        (ok_point, bad_point),
        run_results={
            "P7": (0, PointStatus.SUCCESS, None),
            "P8": (2, PointStatus.FAILED, "exit code 2, expected 0"),
        },
    )
    by_point = {entry["sweep_point"]: entry for entry in index["points"]}  # type: ignore[index]
    assert by_point["P7"]["status"] == "success"
    assert by_point["P8"]["status"] == "failed"
    assert by_point["P7"]["summary_metrics"]["compressed_accuracy"] == 0.5  # type: ignore[index]
    assert by_point["P7"]["summary_metrics"]["accuracy_retention"] == pytest.approx(0.7142857142857143)  # type: ignore[index]
    assert by_point["P7"]["summary_metrics"]["wall_clock_seconds"] == 15.0  # type: ignore[index]
    assert by_point["P7"]["summary_metrics"]["estimated_cost_usd"] == 1.25  # type: ignore[index]
    assert by_point["P8"]["summary_metrics"] is None


def test_build_index_entry_marks_p0_original_only(tmp_path: Path) -> None:
    out_dir = tmp_path / "p0"
    out_dir.mkdir()
    summary = {"conditions": {"original": {"accuracy": 0.68}}}
    (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    point = SweepPoint(
        sweep_point="P0",
        type="Baseline",
        setting="Uncompressed original prompt",
        mechanism="gate",
        out_dir=out_dir,
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=1,
    )
    entry = build_index_entry(point, exit_code=1, status=PointStatus.SUCCESS)
    assert entry["summary_metrics"]["compressed_conditions"] == []  # type: ignore[index]
    assert entry["summary_metrics"]["original_accuracy"] == 0.68  # type: ignore[index]


def test_main_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    from benchmarks.prompt_compression.sweep_runner import main

    assert main(["--dry-run", "--points", "P0,P1"]) == 0
    output = capsys.readouterr().out
    assert "P0:" in output
    assert "P1:" in output
    assert "inline:library_default" in output
