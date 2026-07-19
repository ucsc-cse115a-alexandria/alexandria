#!/usr/bin/env python3
"""Run paired original-vs-compressed benchmark conditions with resumable raw logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from openai import OpenAI

from alexandria.ir.contracts import MergeMetrics, Params
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import OPENAI_EMBEDDING_MODEL, default_embedder
from alexandria.utils.merger import MERGE_MODEL, default_merger
from alexandria.utils.tokens import count_tokens, truncate_tokens
from benchmarks.prompt_compression.adapters import get_adapter
from benchmarks.prompt_compression.contracts import ConditionRecord, GenerationResult, PromptParts
from benchmarks.prompt_compression.metering import OpenAIUsageMeter, estimate_cost
from benchmarks.prompt_compression.runner import benchmark_report, summarize_records
from benchmarks.prompt_compression.store import RunStore

if TYPE_CHECKING:
    from benchmarks.prompt_compression.contracts import BenchmarkAdapter, BenchmarkCase, UsageRecord

DEFAULT_MODEL = "gpt-5.6-luna"


def condition_name(reduction_percent: float) -> str:
    keep = 100.0 - reduction_percent
    rendered = f"{keep:g}".replace(".", "p")
    return f"keep{rendered}"


def compress_parts(parts: PromptParts, reduction_percent: float) -> tuple[PromptParts, int, int, MergeMetrics, float]:
    """Compress only context while enforcing a hard ceiling on the reconstructed full prompt."""
    source_tokens = count_tokens(parts.prompt)
    target_tokens = max(1, math.floor(source_tokens * (1.0 - reduction_percent / 100.0)))
    context_tokens = count_tokens(parts.context)
    fixed_effect = source_tokens - context_tokens
    context_budget = target_tokens - fixed_effect
    if context_budget < 1:
        raise ValueError(
            f"fixed prompt structure leaves no context budget: source={source_tokens}, target={target_tokens}"
        )
    started = time.monotonic()
    result = reduce(
        parts.context,
        default_embedder(),
        default_merger(),
        params=Params(max_tokens=context_budget, require_target=True),
    )
    compressed_context = result.text
    compressed = parts.replace_context(compressed_context)
    sent_tokens = count_tokens(compressed.prompt)
    repaired = 0
    while sent_tokens > target_tokens:
        overshoot = sent_tokens - target_tokens
        before = count_tokens(compressed_context)
        compressed_context = truncate_tokens(compressed_context, max(1, before - overshoot - 1))
        repaired += before - count_tokens(compressed_context)
        compressed = parts.replace_context(compressed_context)
        sent_tokens = count_tokens(compressed.prompt)
    metrics = result.merge_metrics.model_copy(
        update={"repaired_tokens": result.merge_metrics.repaired_tokens + repaired}
    )
    return compressed, target_tokens, sent_tokens, metrics, time.monotonic() - started


def _git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found")
    return subprocess.run(
        [git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _generate(client: OpenAI, model: str, reasoning: str, prompt: str) -> GenerationResult:
    response = client.responses.create(model=model, reasoning={"effort": reasoning}, input=prompt)
    return GenerationResult(text=response.output_text, model=str(response.model), response_id=response.id)


def _record(
    *,
    case: BenchmarkCase,
    condition: str,
    reduction_percent: float,
    prompt: str,
    target_tokens: int,
    sent_tokens: int,
    generation: GenerationResult,
    adapter: BenchmarkAdapter,
    compression_elapsed: float,
    answer_elapsed: float,
    merge_metrics: MergeMetrics,
    usage: tuple[UsageRecord, ...],
) -> ConditionRecord:
    verdict = adapter.verify(case, generation.text)
    return ConditionRecord(
        case_key=case.key,
        benchmark=case.benchmark,
        task=case.task,
        condition=condition,
        reduction_percent=reduction_percent,
        source_tokens=count_tokens(case.prompt),
        target_tokens=target_tokens,
        sent_tokens=sent_tokens,
        prompt_sha256=_prompt_hash(prompt),
        response=generation.text,
        response_model=generation.model,
        response_id=generation.response_id,
        verdict=verdict,
        compression_elapsed_seconds=compression_elapsed,
        answer_elapsed_seconds=answer_elapsed,
        merge_metrics=merge_metrics,
        usage=usage,
        estimated_cost_usd=estimate_cost(usage),
        metadata=case.metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("babilong_8k", "ruler_v2", "longbench_v2"), required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reductions", type=float, nargs="+", default=(50.0, 25.0, 10.0, 5.0))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--min-source-tokens", type=int, default=0)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--release-threshold", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if any(reduction <= 0.0 or reduction >= 100.0 for reduction in args.reductions):
        raise ValueError("every reduction must be greater than 0 and less than 100")
    adapter = get_adapter(args.benchmark)
    data_dir = args.data_dir or adapter.default_data_dir
    cases = adapter.load_cases(
        args.n,
        seed=args.seed,
        data_dir=data_dir,
        min_source_tokens=args.min_source_tokens,
        max_source_tokens=args.max_source_tokens,
    )
    token_counts = [count_tokens(case.prompt) for case in cases]
    store = RunStore(args.out)
    manifest = {
        "schema_version": 1,
        "implementation_commit": _git_sha(),
        "command": shlex.join(sys.argv),
        "benchmark": args.benchmark,
        "provenance": adapter.provenance,
        "data_dir": str(data_dir),
        "n_cases": len(cases),
        "case_keys": [case.key for case in cases],
        "seed": args.seed,
        "reductions_percent": args.reductions,
        "model": args.model,
        "reasoning": args.reasoning,
        "tokenizer": "cl100k_base",
        "eligibility": {
            "min_source_tokens": args.min_source_tokens,
            "max_source_tokens": args.max_source_tokens,
        },
        "compression": {
            "compressible_prompt_part": "context",
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "merge_model": MERGE_MODEL,
            "require_target": True,
        },
        "source_token_distribution": {
            "min": min(token_counts),
            "mean": float(np.mean(token_counts)),
            "p50": float(np.quantile(token_counts, 0.50)),
            "p95": float(np.quantile(token_counts, 0.95)),
            "max": max(token_counts),
        },
        "pricing_usd_per_million": {
            "answer_or_merge_input": 1.00,
            "answer_or_merge_cached_input": 0.10,
            "answer_or_merge_output": 6.00,
            "embedding_input": 0.02,
        },
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    store.write_manifest(manifest)

    completed = store.completed_keys()
    client = OpenAI(timeout=120.0)
    with OpenAIUsageMeter() as meter:
        for case in cases:
            source_tokens = count_tokens(case.prompt)
            if (case.key, "original") not in completed:
                usage_start = len(meter.records)
                started = time.monotonic()
                with meter.category("answer"):
                    generation = _generate(client, args.model, args.reasoning, case.prompt)
                answer_elapsed = time.monotonic() - started
                usage = tuple(meter.records[usage_start:])
                record = _record(
                    case=case,
                    condition="original",
                    reduction_percent=0.0,
                    prompt=case.prompt,
                    target_tokens=source_tokens,
                    sent_tokens=source_tokens,
                    generation=generation,
                    adapter=adapter,
                    compression_elapsed=0.0,
                    answer_elapsed=answer_elapsed,
                    merge_metrics=MergeMetrics(),
                    usage=usage,
                )
                store.append(record, case.prompt)
                print(f"completed {case.key} original", flush=True)

            for reduction in args.reductions:
                condition = condition_name(reduction)
                if (case.key, condition) in completed:
                    continue
                usage_start = len(meter.records)
                with meter.category("compression"):
                    parts, target_tokens, sent_tokens, metrics, compression_elapsed = compress_parts(
                        case.prompt_parts, reduction
                    )
                answer_started = time.monotonic()
                with meter.category("answer"):
                    generation = _generate(client, args.model, args.reasoning, parts.prompt)
                answer_elapsed = time.monotonic() - answer_started
                usage = tuple(meter.records[usage_start:])
                record = _record(
                    case=case,
                    condition=condition,
                    reduction_percent=reduction,
                    prompt=parts.prompt,
                    target_tokens=target_tokens,
                    sent_tokens=sent_tokens,
                    generation=generation,
                    adapter=adapter,
                    compression_elapsed=compression_elapsed,
                    answer_elapsed=answer_elapsed,
                    merge_metrics=metrics,
                    usage=usage,
                )
                store.append(record, parts.prompt)
                print(f"completed {case.key} {condition}", flush=True)

    records = store.load_records()
    summary = summarize_records(
        records,
        release_threshold=args.release_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    report = benchmark_report(summary)
    store.write_summary(summary, report)
    print(report)


if __name__ == "__main__":
    main()
