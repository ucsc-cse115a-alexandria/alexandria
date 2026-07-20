"""Execute the compression-strength sweep matrix and build a combined index for analysis."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from alexandria.ir.contracts import MergeMetrics, Params
from alexandria.ops.features.compare import compare
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import OPENAI_EMBEDDING_MODEL, default_embedder
from alexandria.utils.merger import MERGE_MODEL, default_merger
from alexandria.utils.tokens import count_tokens
from benchmarks.prompt_compression.adapters import get_adapter
from benchmarks.prompt_compression.runner import benchmark_report, summarize_records
from benchmarks.prompt_compression.store import RunStore
from scripts.prompt_compression_benchmark import DEFAULT_MODEL, _generate, _git_dirty, _git_sha, _record

if TYPE_CHECKING:
    from collections.abc import Sequence

RUNBOOK_PATH = Path("benchmarks/prompt_compression/sweep-matrix-runbook.md")
SWEEP_INDEX_PATH = Path("benchmarks/babilong_8k/results/2026-07-20-sweep-index-v1.json")
BABILONG_DATA_DIR = Path("data/babilong/8k")
N_CASES = 50
SEED = 42
MODEL = "gpt-5.6-luna"


class ExecutionKind(StrEnum):
    SUBPROCESS = "subprocess"
    INLINE = "inline"


class PointStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


def _budget_argv(budget: str, out_suffix: str) -> tuple[str, ...]:
    out = f"benchmarks/babilong_8k/results/{out_suffix}"
    return (
        "scripts.prompt_compression_benchmark",
        "--benchmark",
        "babilong_8k",
        "--n",
        str(N_CASES),
        "--seed",
        str(SEED),
        "--cos-sim-diff-budgets",
        budget,
        "--model",
        MODEL,
        "--merge-model",
        MODEL,
        "--min-original-accuracy",
        "0.50",
        "--out",
        out,
    )


def _reduction_argv(reduction: str, out_suffix: str) -> tuple[str, ...]:
    out = f"benchmarks/babilong_8k/results/{out_suffix}"
    return (
        "scripts.prompt_compression_benchmark",
        "--benchmark",
        "babilong_8k",
        "--n",
        str(N_CASES),
        "--seed",
        str(SEED),
        "--reductions",
        reduction,
        "--model",
        MODEL,
        "--merge-model",
        MODEL,
        "--min-original-accuracy",
        "0.50",
        "--out",
        out,
    )


@dataclass(frozen=True, slots=True)
class SweepPoint:
    sweep_point: str
    type: str
    setting: str
    mechanism: str
    out_dir: Path
    execution_kind: ExecutionKind
    expect_exit_code: int
    argv: tuple[str, ...] | None = None

    @property
    def command(self) -> str:
        if self.execution_kind is ExecutionKind.INLINE:
            return "inline:library_default"
        if self.argv is None:
            for point in SWEEP_POINTS:
                if point.sweep_point == self.sweep_point:
                    return point.command
            raise ValueError(f"no argv configured for sweep point {self.sweep_point}")
        return "uv run " + shlex.join(("python", "-m", *self.argv))


SWEEP_POINTS: tuple[SweepPoint, ...] = (
    SweepPoint(
        sweep_point="P0",
        type="Baseline",
        setting="Uncompressed original prompt",
        mechanism="--min-original-accuracy 1.0 gate (original only)",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p0-original-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=1,
        argv=(
            "scripts.prompt_compression_benchmark",
            "--benchmark",
            "babilong_8k",
            "--n",
            str(N_CASES),
            "--seed",
            str(SEED),
            "--reductions",
            "50",
            "--min-original-accuracy",
            "1.0",
            "--model",
            MODEL,
            "--out",
            "benchmarks/babilong_8k/results/2026-07-20-p0-original-n50-v1",
        ),
    ),
    SweepPoint(
        sweep_point="P1",
        type="Baseline",
        setting="Current default compression (`Params()` defaults)",
        mechanism="Inline runner → `reduce(..., params=Params())`",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p1-library-default-n50-v1"),
        execution_kind=ExecutionKind.INLINE,
        expect_exit_code=0,
    ),
    SweepPoint(
        sweep_point="P2",
        type="Weaker",
        setting="Tight semantic budget",
        mechanism="--cos-sim-diff-budgets 0.0025",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p2-budget-0.0025-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_budget_argv("0.0025", "2026-07-20-p2-budget-0.0025-n50-v1"),
    ),
    SweepPoint(
        sweep_point="P3",
        type="Weaker",
        setting="Slightly tight semantic budget",
        mechanism="--cos-sim-diff-budgets 0.005",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p3-budget-0.005-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_budget_argv("0.005", "2026-07-20-p3-budget-0.005-n50-v1"),
    ),
    SweepPoint(
        sweep_point="P4",
        type="Reference",
        setting="Mid semantic budget",
        mechanism="--cos-sim-diff-budgets 0.01",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p4-budget-0.01-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_budget_argv("0.01", "2026-07-20-p4-budget-0.01-n50-v1"),
    ),
    SweepPoint(
        sweep_point="P5",
        type="Stronger",
        setting="Loose semantic budget",
        mechanism="--cos-sim-diff-budgets 0.015",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p5-budget-0.015-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_budget_argv("0.015", "2026-07-20-p5-budget-0.015-n50-v1"),
    ),
    SweepPoint(
        sweep_point="P6",
        type="Stronger",
        setting="Loosest allowed semantic budget",
        mechanism="--cos-sim-diff-budgets 0.02",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p6-budget-0.02-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_budget_argv("0.02", "2026-07-20-p6-budget-0.02-n50-v1"),
    ),
    SweepPoint(
        sweep_point="P7",
        type="Stronger, hard-target",
        setting="50% forced reduction",
        mechanism="--reductions 50",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p7-reduction-50-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_reduction_argv("50", "2026-07-20-p7-reduction-50-n50-v1"),
    ),
    SweepPoint(
        sweep_point="P8",
        type="Stronger, hard-target",
        setting="30% forced reduction",
        mechanism="--reductions 30",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p8-reduction-30-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_reduction_argv("30", "2026-07-20-p8-reduction-30-n50-v1"),
    ),
    SweepPoint(
        sweep_point="P9",
        type="Stronger, hard-target",
        setting="10% forced reduction",
        mechanism="--reductions 10",
        out_dir=Path("benchmarks/babilong_8k/results/2026-07-20-p9-reduction-10-n50-v1"),
        execution_kind=ExecutionKind.SUBPROCESS,
        expect_exit_code=0,
        argv=_reduction_argv("10", "2026-07-20-p9-reduction-10-n50-v1"),
    ),
)


class _Tee:
    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)  # type: ignore[union-attr]
            stream.flush()  # type: ignore[union-attr]
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()  # type: ignore[union-attr]


def ensure_babilong_data() -> None:
    if BABILONG_DATA_DIR.is_dir() and any(BABILONG_DATA_DIR.glob("*.json")):
        return
    print(f"Downloading BABILong 8k data into {BABILONG_DATA_DIR}...", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.download_babilong_8k_data"], check=True)


def run_library_default_point(out_dir: Path, *, log_path: Path) -> int:
    """Port of the P1 inline runbook body using library-default Params()."""
    condition = "default"
    params = Params()
    merge_model_name = MERGE_MODEL
    adapter = get_adapter("babilong_8k")
    cases = adapter.load_cases(
        N_CASES,
        seed=SEED,
        data_dir=adapter.default_data_dir,
        min_source_tokens=0,
        max_source_tokens=None,
    )
    store = RunStore(out_dir)
    store.write_manifest(
        {
            "schema_version": 1,
            "implementation_commit": _git_sha(),
            "implementation_dirty": _git_dirty(),
            "command": "inline:library_default",
            "benchmark": adapter.name,
            "provenance": adapter.provenance,
            "data_dir": str(adapter.default_data_dir),
            "n_cases": len(cases),
            "case_keys": [case.key for case in cases],
            "seed": SEED,
            "experiment_mode": "library_default_inline",
            "library_default_params": {
                "threshold": params.threshold,
                "cos_sim_diff_budget": params.cos_sim_diff_budget,
                "max_tokens": params.max_tokens,
                "require_target": params.require_target,
            },
            "model": DEFAULT_MODEL,
            "compression": {"merge_model": merge_model_name, "embedding_model": OPENAI_EMBEDDING_MODEL},
        }
    )
    client = OpenAI(timeout=120.0)
    completed = store.completed_keys()
    with log_path.open("a", encoding="utf-8") as log_file:
        tee = _Tee(sys.__stdout__, log_file)
        previous_stdout = sys.stdout
        sys.stdout = tee  # type: ignore[assignment]
        try:
            for case in cases:
                source_tokens = count_tokens(case.prompt)
                if (case.key, "original") not in completed:
                    started = time.monotonic()
                    generation = _generate(client, DEFAULT_MODEL, "none", case.prompt)
                    record = _record(
                        case=case,
                        condition="original",
                        reduction_percent=0.0,
                        prompt=case.prompt,
                        target_tokens=source_tokens,
                        sent_tokens=source_tokens,
                        generation=generation,
                        adapter=adapter,
                        prompt_cosine_difference=0.0,
                        compression_elapsed=0.0,
                        answer_elapsed=time.monotonic() - started,
                        merge_metrics=MergeMetrics(),
                        usage=(),
                    )
                    store.append(record, case.prompt)
                if (case.key, condition) in completed:
                    continue
                parts = case.prompt_parts
                embedder = default_embedder()
                merger = default_merger(model=merge_model_name)
                compression_started = time.monotonic()
                result = reduce(parts.context, embedder, merger, params=params)
                compressed = parts.replace_context(result.text)
                prompt_difference = float(compare(case.prompt, compressed.prompt, embedder).cos_sim_diff)
                compression_elapsed = time.monotonic() - compression_started
                sent_tokens = count_tokens(compressed.prompt)
                answer_started = time.monotonic()
                generation = _generate(client, DEFAULT_MODEL, "none", compressed.prompt)
                answer_elapsed = time.monotonic() - answer_started
                actual_reduction = max(0.0, (1.0 - sent_tokens / source_tokens) * 100.0)
                record = _record(
                    case=case,
                    condition=condition,
                    reduction_percent=actual_reduction,
                    prompt=compressed.prompt,
                    target_tokens=source_tokens,
                    sent_tokens=sent_tokens,
                    generation=generation,
                    adapter=adapter,
                    prompt_cosine_difference=prompt_difference,
                    compression_elapsed=compression_elapsed,
                    answer_elapsed=answer_elapsed,
                    merge_metrics=result.merge_metrics,
                    usage=(),
                    configured_cos_sim_diff_budget=params.cos_sim_diff_budget,
                )
                store.append(record, compressed.prompt)
            records = store.load_records()
            summary = summarize_records(records, bootstrap_seed=SEED)
            report = benchmark_report(summary)
            store.write_summary(summary, report)
            print(report)
        finally:
            sys.stdout = previous_stdout
    return 0


def run_subprocess_point(point: SweepPoint, *, log_path: Path) -> int:
    if point.argv is None:
        raise ValueError(f"subprocess point {point.sweep_point} has no argv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- sweep_runner invoking {point.sweep_point} at {datetime.now(UTC).isoformat()} ---\n")
        log_file.flush()
        process = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", *point.argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ,
        )
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("subprocess stdout was not captured")
        for line in stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return process.wait()


def classify_point_status(
    point: SweepPoint,
    *,
    exit_code: int | None,
    out_dir: Path,
) -> PointStatus:
    manifest_path = out_dir / "manifest.json"
    summary_path = out_dir / "summary.json"
    records_path = out_dir / "records.jsonl"
    has_manifest = manifest_path.is_file()
    has_summary = summary_path.is_file()
    has_records = records_path.is_file() and records_path.stat().st_size > 0

    if not has_manifest and not has_records:
        return PointStatus.FAILED

    if has_summary and has_manifest:
        if exit_code is None or exit_code == point.expect_exit_code:
            return PointStatus.SUCCESS
        return PointStatus.FAILED

    if has_manifest or has_records:
        return PointStatus.PARTIAL

    return PointStatus.FAILED


def _compressed_conditions(summary: dict[str, object]) -> list[str]:
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict):
        return []
    return [name for name in conditions if name != "original"]


def _primary_compressed_condition(summary: dict[str, object], sweep_point: str) -> str | None:
    compressed = _compressed_conditions(summary)
    if not compressed:
        return None
    if len(compressed) == 1:
        return compressed[0]
    expected = summary.get("expected_conditions")
    if isinstance(expected, list) and expected:
        first = expected[0]
        return str(first) if first in compressed else compressed[0]
    if sweep_point == "P1":
        return "default" if "default" in compressed else compressed[0]
    return compressed[0]


def extract_summary_metrics(summary: dict[str, object], sweep_point: str) -> dict[str, object] | None:
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict):
        return None
    original = conditions.get("original")
    if not isinstance(original, dict):
        return None
    compressed_name = _primary_compressed_condition(summary, sweep_point)
    metrics: dict[str, object] = {
        "original_accuracy": original.get("accuracy"),
        "compressed_conditions": _compressed_conditions(summary),
    }
    if compressed_name is None:
        return metrics
    compressed = conditions.get(compressed_name)
    if not isinstance(compressed, dict):
        return metrics
    metrics["compressed_condition"] = compressed_name
    metrics["compressed_accuracy"] = compressed.get("accuracy")
    metrics["mean_token_reduction"] = compressed.get("token_reduction")
    metrics["mean_prompt_cosine_difference"] = compressed.get("mean_prompt_embedding_cosine_difference")
    return metrics


def collect_artifacts(out_dir: Path) -> dict[str, object]:
    store = RunStore(out_dir)
    records = store.load_records()
    errors = store.load_errors()
    return {
        "manifest": (out_dir / "manifest.json").is_file(),
        "summary": (out_dir / "summary.json").is_file(),
        "records_count": len(records),
        "errors_count": len(errors),
    }


def build_index_entry(
    point: SweepPoint,
    *,
    exit_code: int | None,
    status: PointStatus,
    error: str | None = None,
) -> dict[str, object]:
    out_dir = point.out_dir
    artifacts = collect_artifacts(out_dir)
    summary_metrics: dict[str, object] | None = None
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_metrics = extract_summary_metrics(summary, point.sweep_point)

    entry: dict[str, object] = {
        "sweep_point": point.sweep_point,
        "type": point.type,
        "setting": point.setting,
        "mechanism": point.mechanism,
        "command": point.command,
        "out_dir": str(out_dir),
        "execution_kind": point.execution_kind.value,
        "status": status.value,
        "exit_code": exit_code,
        "expected_exit_code": point.expect_exit_code,
        "artifacts": artifacts,
        "summary_metrics": summary_metrics,
    }
    if error is not None:
        entry["error"] = error
    return entry


def build_sweep_index(
    points: Sequence[SweepPoint],
    *,
    run_results: dict[str, tuple[int | None, PointStatus, str | None]] | None = None,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for point in points:
        if run_results and point.sweep_point in run_results:
            exit_code, status, error = run_results[point.sweep_point]
        else:
            exit_code = None
            status = classify_point_status(point, exit_code=exit_code, out_dir=point.out_dir)
            error = None
        entries.append(
            build_index_entry(point, exit_code=exit_code, status=status, error=error)
        )
    return {
        "schema_version": 1,
        "runbook_path": str(RUNBOOK_PATH),
        "implementation_commit": _git_sha(),
        "generated_at": datetime.now(UTC).isoformat(),
        "points": entries,
    }


def write_sweep_index(index: dict[str, object], path: Path = SWEEP_INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_point(point: SweepPoint) -> tuple[int | None, PointStatus, str | None]:
    out_dir = point.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    try:
        if point.execution_kind is ExecutionKind.INLINE:
            exit_code = run_library_default_point(out_dir, log_path=log_path)
        else:
            exit_code = run_subprocess_point(point, log_path=log_path)
    except Exception as error:
        log_path.open("a", encoding="utf-8").write(f"\nSWEEP RUNNER ERROR: {error}\n")
        return None, PointStatus.FAILED, f"{type(error).__name__}: {error}"

    status = classify_point_status(point, exit_code=exit_code, out_dir=out_dir)
    error_message = None
    if status is PointStatus.FAILED and exit_code != point.expect_exit_code:
        error_message = f"exit code {exit_code}, expected {point.expect_exit_code}"
    return exit_code, status, error_message


def parse_points(raw: str | None) -> tuple[SweepPoint, ...]:
    if raw is None:
        return SWEEP_POINTS
    requested = {part.strip().upper() for part in raw.split(",") if part.strip()}
    selected = tuple(point for point in SWEEP_POINTS if point.sweep_point in requested)
    if len(selected) != len(requested):
        unknown = requested - {point.sweep_point for point in SWEEP_POINTS}
        raise ValueError(f"unknown sweep points: {sorted(unknown)}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", help="Comma-separated subset, e.g. P0,P2")
    parser.add_argument("--index-only", action="store_true", help="Rebuild sweep index without executing points")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and exit")
    parser.add_argument(
        "--index-out",
        type=Path,
        default=SWEEP_INDEX_PATH,
        help="Path for combined sweep index JSON",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    load_dotenv()
    points = parse_points(args.points)

    if args.dry_run:
        for point in points:
            print(f"{point.sweep_point}: {point.command}")
        return 0

    if not args.index_only and not os.getenv("OPENAI_API_KEY"):
        print(
            "Warning: OPENAI_API_KEY is not set; sweep points will fail until credentials are configured.",
            file=sys.stderr,
            flush=True,
        )

    run_results: dict[str, tuple[int | None, PointStatus, str | None]] = {}
    if not args.index_only:
        ensure_babilong_data()
        for point in points:
            print(f"Running {point.sweep_point} → {point.out_dir}", flush=True)
            run_results[point.sweep_point] = run_point(point)
            exit_code, status, error = run_results[point.sweep_point]
            print(
                f"Finished {point.sweep_point}: status={status.value} exit_code={exit_code}",
                flush=True,
            )
            if error:
                print(f"  note: {error}", flush=True)

    index = build_sweep_index(SWEEP_POINTS, run_results=run_results or None)
    write_sweep_index(index, args.index_out)
    print(f"Wrote sweep index to {args.index_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
