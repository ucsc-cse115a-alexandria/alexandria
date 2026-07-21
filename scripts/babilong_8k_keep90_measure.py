#!/usr/bin/env python3
"""Measure BABILong 8k keep-90 compression, answers, usage, time, and paired retention."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import numpy as np
from openai import OpenAI
from openai.resources.embeddings import Embeddings
from openai.resources.responses.responses import Responses
from pydantic import BaseModel, ConfigDict, Field

from alexandria.ir.contracts import MergeMetrics, Params
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import default_embedder
from alexandria.utils.merger import default_merger
from alexandria.utils.tokens import count_tokens
from benchmarks.babilong_8k import ExperimentResult, load_cases, paired_retention_bootstrap

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarks.babilong_8k import BABILongCase

MODEL = "gpt-5.6-luna"
MERGE_REASONING = "low"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_KEEP_PERCENT = 90.0
GPT_INPUT_PER_MILLION = 1.00
GPT_CACHED_INPUT_PER_MILLION = 0.10
GPT_OUTPUT_PER_MILLION = 6.00
EMBEDDING_INPUT_PER_MILLION = 0.02


class UsageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    category: str
    model: str
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)


class MeasuredCase(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: str
    task: str
    target: str
    source_tokens: int = Field(ge=1)
    target_max_tokens: int = Field(ge=1)
    reduced_tokens: int = Field(ge=1)
    compressed_prompt: str = Field(min_length=1)
    response: str
    correct: bool
    compression_elapsed_seconds: float = Field(ge=0.0)
    answer_elapsed_seconds: float = Field(ge=0.0)
    merge_metrics: MergeMetrics


class RawRun(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[2] = 2
    implementation_commit: str
    model: str = MODEL
    answer_reasoning: str = "none"
    merge_reasoning: str = MERGE_REASONING
    keep_percent: float
    seed: int
    requested_cases: int
    records: tuple[MeasuredCase, ...] = ()
    usage: tuple[UsageRecord, ...] = ()


def _git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to record the benchmark implementation commit")
    result = subprocess.run(
        [git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _response_usage(category: str, response: Any, elapsed_seconds: float) -> UsageRecord | None:
    usage = response.usage
    if usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    return UsageRecord(
        category=category,
        model=str(response.model),
        input_tokens=int(usage.input_tokens),
        cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        output_tokens=int(usage.output_tokens),
        total_tokens=int(usage.total_tokens),
        elapsed_seconds=elapsed_seconds,
    )


def _embedding_usage(response: Any, elapsed_seconds: float) -> UsageRecord:
    return UsageRecord(
        category="embedding",
        model=EMBEDDING_MODEL,
        input_tokens=int(response.usage.prompt_tokens),
        cached_input_tokens=0,
        output_tokens=0,
        total_tokens=int(response.usage.total_tokens),
        elapsed_seconds=elapsed_seconds,
    )


def _meter_openai(ledger: list[UsageRecord]) -> ExitStack:
    original_create = Responses.create
    original_parse = Responses.parse
    original_embedding = Embeddings.create

    def create(resource: Responses, *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        response = original_create(resource, *args, **kwargs)
        record = _response_usage("answer", response, time.monotonic() - started)
        if record is not None:
            ledger.append(record)
        return response

    def parse(resource: Responses, *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        response = original_parse(resource, *args, **kwargs)
        record = _response_usage("target_merge", response, time.monotonic() - started)
        if record is not None:
            ledger.append(record)
        return response

    def embedding(resource: Embeddings, *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        response = original_embedding(resource, *args, **kwargs)
        ledger.append(_embedding_usage(response, time.monotonic() - started))
        return response

    stack = ExitStack()
    stack.enter_context(patch.object(Responses, "create", create))
    stack.enter_context(patch.object(Responses, "parse", parse))
    stack.enter_context(patch.object(Embeddings, "create", embedding))
    return stack


def _write_raw(path: Path, run: RawRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _load_or_start(path: Path, *, keep: float, seed: int, n: int) -> RawRun:
    if not path.exists():
        return RawRun(
            implementation_commit=_git_sha(),
            keep_percent=keep,
            seed=seed,
            requested_cases=n,
        )
    run = RawRun.model_validate_json(path.read_text(encoding="utf-8"))
    expected = (keep, seed, n)
    actual = (run.keep_percent, run.seed, run.requested_cases)
    if actual != expected:
        raise ValueError(f"existing raw run uses keep/seed/n={actual}, expected {expected}")
    return run


def _measure_case(
    case: BABILongCase,
    *,
    keep: float,
    generate: Callable[[str], str],
) -> MeasuredCase:
    source_tokens = count_tokens(case.prompt)
    target_max_tokens = max(1, math.floor(source_tokens * keep / 100.0))
    compression_started = time.monotonic()
    result = reduce(
        case.prompt,
        default_embedder(),
        default_merger(),
        params=Params(max_tokens=target_max_tokens, require_target=True),
    )
    compression_elapsed = time.monotonic() - compression_started
    if result.reduced_tokens > target_max_tokens:
        raise RuntimeError(
            f"hard-target invariant failed for {case.key}: {result.reduced_tokens} > {target_max_tokens}"
        )

    answer_started = time.monotonic()
    response = generate(result.text)
    answer_elapsed = time.monotonic() - answer_started
    return MeasuredCase(
        key=case.key,
        task=case.task,
        target=case.target,
        source_tokens=source_tokens,
        target_max_tokens=target_max_tokens,
        reduced_tokens=result.reduced_tokens,
        compressed_prompt=result.text,
        response=response,
        correct=case.verify(response).correct,
        compression_elapsed_seconds=compression_elapsed,
        answer_elapsed_seconds=answer_elapsed,
        merge_metrics=result.merge_metrics,
    )


def _aggregate_usage(records: tuple[UsageRecord, ...]) -> dict[str, dict[str, float | int]]:
    grouped: defaultdict[str, list[UsageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record)
    aggregated: dict[str, dict[str, float | int]] = {}
    for category, category_records in sorted(grouped.items()):
        aggregated[category] = {
            "requests": len(category_records),
            "input_tokens": sum(record.input_tokens for record in category_records),
            "cached_input_tokens": sum(record.cached_input_tokens for record in category_records),
            "output_tokens": sum(record.output_tokens for record in category_records),
            "total_tokens": sum(record.total_tokens for record in category_records),
            "elapsed_seconds": sum(record.elapsed_seconds for record in category_records),
        }
    return aggregated


def _estimated_cost(records: tuple[UsageRecord, ...]) -> dict[str, float]:
    costs = {"answer": 0.0, "target_merge": 0.0, "embedding": 0.0}
    for record in records:
        if record.category == "embedding":
            costs["embedding"] += record.input_tokens * EMBEDDING_INPUT_PER_MILLION / 1_000_000
            continue
        uncached = record.input_tokens - record.cached_input_tokens
        costs[record.category] += (
            uncached * GPT_INPUT_PER_MILLION
            + record.cached_input_tokens * GPT_CACHED_INPUT_PER_MILLION
            + record.output_tokens * GPT_OUTPUT_PER_MILLION
        ) / 1_000_000
    costs["total"] = sum(costs.values())
    return costs


def _write_baseline(path: Path, baseline: ExperimentResult, measurement: dict[str, Any] | None) -> None:
    payload = {
        "schema_version": 1,
        "label": baseline.label,
        "model": baseline.model,
        "accuracy": baseline.accuracy,
        "mean_input_prompt_tokens": baseline.mean_sent_tokens,
        "measurement": measurement,
        "records": [
            {
                "key": record.key,
                "task": record.task,
                "source_tokens": record.source_tokens,
                "sent_tokens": record.sent_tokens,
                "response": record.response,
                "correct": record.verdict.correct,
            }
            for record in baseline.records
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    run: RawRun,
    baseline: ExperimentResult,
    baseline_measurement: dict[str, Any] | None,
) -> None:
    baseline_by_key = {record.key: record for record in baseline.records}
    if tuple(record.key for record in run.records) != tuple(baseline_by_key):
        raise ValueError("baseline and measured run must contain the same cases in the same order")
    original_correct = [baseline_by_key[record.key].verdict.correct for record in run.records]
    compressed_correct = [record.correct for record in run.records]
    retention = paired_retention_bootstrap(original_correct, compressed_correct)
    source_tokens = sum(record.source_tokens for record in run.records)
    reduced_tokens = sum(record.reduced_tokens for record in run.records)
    compression_times = [record.compression_elapsed_seconds for record in run.records]
    answer_times = [record.answer_elapsed_seconds for record in run.records]
    cos_sim_diffs = [record.merge_metrics.final_cos_sim_diff or 0.0 for record in run.records]
    run_cost = _estimated_cost(run.usage)
    original_cost = (
        float(baseline_measurement["api_usage_and_cost"]["total_estimated_usd"])
        if baseline_measurement is not None
        else None
    )
    original_wall_clock = (
        float(baseline_measurement["timing_seconds"]["total_wall_clock"]) if baseline_measurement is not None else None
    )
    payload = {
        "schema_version": 2,
        "method": {
            "benchmark": "BABILong 8k qa1-qa5 task-balanced sample",
            "n_cases": len(run.records),
            "seed": run.seed,
            "keep_percent": run.keep_percent,
            "model": run.model,
            "answer_reasoning": run.answer_reasoning,
            "merge_reasoning": run.merge_reasoning,
            "tokenizer": "cl100k_base",
            "bootstrap": "paired percentile bootstrap over case indices",
            "pricing_usd_per_million": {
                "gpt_input": GPT_INPUT_PER_MILLION,
                "gpt_cached_input": GPT_CACHED_INPUT_PER_MILLION,
                "gpt_output": GPT_OUTPUT_PER_MILLION,
                "embedding_input": EMBEDDING_INPUT_PER_MILLION,
            },
            "pricing_source": "https://developers.openai.com/api/docs/pricing",
        },
        "implementation_commit": run.implementation_commit,
        "target": {
            "successes": sum(record.reduced_tokens <= record.target_max_tokens for record in run.records),
            "failures": sum(record.reduced_tokens > record.target_max_tokens for record in run.records),
            "source_tokens": source_tokens,
            "reduced_tokens": reduced_tokens,
            "token_reduction": 1.0 - reduced_tokens / source_tokens,
            "mean_input_prompt_tokens": reduced_tokens / len(run.records),
            "original_mean_input_prompt_tokens": source_tokens / len(run.records),
        },
        "quality": {
            **retention.model_dump(),
            "cos_sim_diff_budget_met": sum(
                record.merge_metrics.cos_sim_diff_budget_met is True for record in run.records
            ),
            "mean_cos_sim_diff": float(np.mean(cos_sim_diffs)),
            "p95_cos_sim_diff": float(np.quantile(cos_sim_diffs, 0.95)),
        },
        "performance": {
            "merge_calls": sum(record.merge_metrics.calls for record in run.records),
            "merge_retries": sum(record.merge_metrics.retries for record in run.records),
            "merge_candidates": sum(record.merge_metrics.candidates_generated for record in run.records),
            "repaired_tokens": sum(record.merge_metrics.repaired_tokens for record in run.records),
            "compression_total_seconds": sum(compression_times),
            "compression_mean_seconds": float(np.mean(compression_times)),
            "compression_p95_seconds": float(np.quantile(compression_times, 0.95)),
            "answer_total_seconds": sum(answer_times),
            "answer_mean_seconds": float(np.mean(answer_times)),
        },
        "api_usage": _aggregate_usage(run.usage),
        "estimated_api_cost_usd": {
            **run_cost,
            "original_baseline": original_cost,
            "combined_with_original": run_cost["total"] + original_cost if original_cost is not None else None,
        },
        "wall_clock_seconds": {
            "original_baseline": original_wall_clock,
            "compression_and_compressed_answers": sum(compression_times) + sum(answer_times),
            "combined_sequential": (
                original_wall_clock + sum(compression_times) + sum(answer_times)
                if original_wall_clock is not None
                else None
            ),
        },
        "records": [
            {
                "key": record.key,
                "original_correct": baseline_by_key[record.key].verdict.correct,
                "compressed_correct": record.correct,
                "source_tokens": record.source_tokens,
                "reduced_tokens": record.reduced_tokens,
                "response": record.response,
            }
            for record in run.records
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", type=float, default=DEFAULT_KEEP_PERCENT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_path = args.out / "raw.json"
    summary_path = args.out / "summary.json"
    run = _load_or_start(raw_path, keep=args.keep, seed=args.seed, n=args.n)
    cases = load_cases(n=args.n, seed=args.seed, data_dir=args.data_dir)
    completed = {record.key for record in run.records}
    records = list(run.records)
    ledger = list(run.usage)
    client = OpenAI(timeout=120.0)

    def generate(prompt: str) -> str:
        return client.responses.create(model=MODEL, reasoning={"effort": "none"}, input=prompt).output_text

    with _meter_openai(ledger):
        for case in cases:
            if case.key in completed:
                continue
            print(f"measuring {case.key} ({len(records) + 1}/{len(cases)})", flush=True)
            try:
                records.append(_measure_case(case, keep=args.keep, generate=generate))
            finally:
                run = run.model_copy(update={"records": tuple(records), "usage": tuple(ledger)})
                _write_raw(raw_path, run)

    baseline = ExperimentResult.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    baseline_measurement = (
        json.loads(args.baseline_summary.read_text(encoding="utf-8")) if args.baseline_summary is not None else None
    )
    _write_baseline(args.out / "original.json", baseline, baseline_measurement)
    _write_summary(summary_path, run, baseline, baseline_measurement)
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
