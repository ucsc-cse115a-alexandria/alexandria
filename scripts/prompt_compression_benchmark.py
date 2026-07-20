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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from openai import OpenAI, OpenAIError

from alexandria.ir.contracts import MergeMetrics, Params
from alexandria.ops.features.compare import compare
from alexandria.ops.pipe import reduce
from alexandria.utils.embedders import OPENAI_EMBEDDING_MODEL, default_embedder
from alexandria.utils.merger import MERGE_MODEL, default_merger
from alexandria.utils.tokens import count_tokens, truncate_tokens
from benchmarks.prompt_compression.adapters import get_adapter
from benchmarks.prompt_compression.budget_runner import budget_benchmark_report, summarize_budget_records
from benchmarks.prompt_compression.contracts import ConditionRecord, GenerationResult, PromptParts
from benchmarks.prompt_compression.metering import (
    EMBEDDING_INPUT_USD_PER_MILLION,
    OpenAIUsageMeter,
    UsageLimitExceeded,
    estimate_cost,
    pricing_for_model,
)
from benchmarks.prompt_compression.runner import benchmark_report, summarize_records
from benchmarks.prompt_compression.store import RunStore

if TYPE_CHECKING:
    from openai.types.shared_params.reasoning import Reasoning

    from benchmarks.prompt_compression.contracts import BenchmarkAdapter, BenchmarkCase, UsageRecord

DEFAULT_MODEL = "gpt-5.6-luna"


def condition_name(reduction_percent: float) -> str:
    keep = 100.0 - reduction_percent
    rendered = f"{keep:g}".replace(".", "p")
    return f"keep{rendered}"


def cos_sim_budget_condition_name(budget: float) -> str:
    """Return a filesystem- and table-safe semantic-budget condition name."""
    return f"budget{budget:g}".replace(".", "p")


def compress_parts(
    parts: PromptParts, reduction_percent: float, *, merge_model: str = MERGE_MODEL
) -> tuple[PromptParts, int, int, MergeMetrics, float, float]:
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
        default_merger(model=merge_model),
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
    prompt_cosine_difference = float(compare(parts.prompt, compressed.prompt, default_embedder()).cos_sim_diff)
    return compressed, target_tokens, sent_tokens, metrics, time.monotonic() - started, prompt_cosine_difference


def compress_parts_to_cos_sim_budget(
    parts: PromptParts,
    budget: float,
    *,
    minimum_retained_percent: float = 50.0,
    merge_model: str = MERGE_MODEL,
) -> tuple[PromptParts, int, int, MergeMetrics, float, float, float, bool]:
    """Best-effort context compression bounded by semantic change, without forcing a token target."""
    source_tokens = count_tokens(parts.prompt)
    target_tokens = max(1, math.floor(source_tokens * minimum_retained_percent / 100.0))
    context_tokens = count_tokens(parts.context)
    fixed_effect = source_tokens - context_tokens
    context_ceiling = target_tokens - fixed_effect
    if context_ceiling < 1:
        raise ValueError(
            f"fixed prompt structure leaves no context budget: source={source_tokens}, target={target_tokens}"
        )
    embedder = default_embedder()
    started = time.monotonic()
    result = reduce(
        parts.context,
        embedder,
        default_merger(model=merge_model),
        params=Params(
            cos_sim_diff_budget=budget,
            max_tokens=context_ceiling,
            require_target=False,
        ),
    )
    compressed = parts.replace_context(result.text)
    context_difference = float(compare(parts.context, result.text, embedder).cos_sim_diff)
    prompt_difference = float(compare(parts.prompt, compressed.prompt, embedder).cos_sim_diff)
    return (
        compressed,
        target_tokens,
        count_tokens(compressed.prompt),
        result.merge_metrics,
        time.monotonic() - started,
        prompt_difference,
        context_difference,
        context_difference <= budget + 1e-12,
    )


def _git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found")
    return subprocess.run([git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _git_dirty() -> bool:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found")
    status = subprocess.run([git, "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    return bool(status.strip())


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _generate(client: OpenAI, model: str, reasoning: str, prompt: str) -> GenerationResult:
    response = client.responses.create(
        model=model,
        reasoning=cast("Reasoning", {"effort": reasoning}),
        input=prompt,
    )
    return GenerationResult(text=response.output_text, model=str(response.model))


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
    prompt_cosine_difference: float,
    compression_elapsed: float,
    answer_elapsed: float,
    merge_metrics: MergeMetrics,
    usage: tuple[UsageRecord, ...],
    configured_cos_sim_diff_budget: float | None = None,
    context_cosine_difference: float = 0.0,
    context_budget_met: bool | None = None,
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
        verdict=verdict,
        configured_cos_sim_diff_budget=configured_cos_sim_diff_budget,
        context_embedding_cosine_difference=context_cosine_difference,
        context_cos_sim_diff_budget_met=context_budget_met,
        prompt_embedding_cosine_difference=prompt_cosine_difference,
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
    parser.add_argument("--reductions", type=float, nargs="+")
    parser.add_argument("--cos-sim-diff-budgets", type=float, nargs="+")
    parser.add_argument("--minimum-retained-percent", type=float, default=50.0)
    parser.add_argument("--max-generation-calls-per-condition", type=int, default=30)
    parser.add_argument("--max-condition-seconds", type=float, default=300.0)
    parser.add_argument("--max-estimated-cost-usd", type=float)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--merge-model", default=MERGE_MODEL)
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--min-source-tokens", type=int, default=0)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--release-threshold", type=float, default=0.90)
    parser.add_argument("--min-original-accuracy", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.reductions is not None and args.cos_sim_diff_budgets is not None:
        raise ValueError("choose either --reductions or --cos-sim-diff-budgets, not both")
    reductions = args.reductions if args.reductions is not None else (50.0, 25.0, 10.0, 5.0)
    budget_mode = args.cos_sim_diff_budgets is not None
    budgets = tuple(args.cos_sim_diff_budgets or ())
    if any(reduction <= 0.0 or reduction >= 100.0 for reduction in reductions):
        raise ValueError("every reduction must be greater than 0 and less than 100")
    if any(budget < 0.0 or budget > 0.02 for budget in budgets):
        raise ValueError("every cos_sim_diff budget must be between 0 and 0.02")
    if not 0.0 < args.minimum_retained_percent < 100.0:
        raise ValueError("minimum retained percent must be greater than 0 and less than 100")
    if args.max_generation_calls_per_condition < 1:
        raise ValueError("maximum generation calls per condition must be at least 1")
    if args.max_condition_seconds <= 0.0:
        raise ValueError("maximum condition seconds must be greater than 0")
    if args.max_estimated_cost_usd is not None and args.max_estimated_cost_usd <= 0.0:
        raise ValueError("maximum estimated cost must be greater than 0")
    if not 0.0 <= args.min_original_accuracy <= 1.0:
        raise ValueError("minimum original accuracy must be between 0 and 1")
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
    implementation_dirty = _git_dirty()
    store = RunStore(args.out)
    manifest = {
        "schema_version": 1,
        "implementation_commit": _git_sha(),
        "implementation_dirty": implementation_dirty,
        "command": shlex.join(sys.argv),
        "benchmark": args.benchmark,
        "provenance": adapter.provenance,
        "data_dir": str(data_dir),
        "n_cases": len(cases),
        "case_keys": [case.key for case in cases],
        "seed": args.seed,
        "experiment_mode": "cos_sim_diff_budget" if budget_mode else "hard_token_target",
        "reductions_percent": None if budget_mode else reductions,
        "cos_sim_diff_budgets": budgets if budget_mode else None,
        "model": args.model,
        "reasoning": args.reasoning,
        "minimum_original_accuracy": args.min_original_accuracy,
        "tokenizer": "cl100k_base",
        "eligibility": {
            "min_source_tokens": args.min_source_tokens,
            "max_source_tokens": args.max_source_tokens,
        },
        "compression": {
            "compressible_prompt_part": "context",
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "merge_model": args.merge_model,
            "require_target": not budget_mode,
            "minimum_retained_percent": args.minimum_retained_percent if budget_mode else None,
            "max_generation_calls_per_condition": (args.max_generation_calls_per_condition if budget_mode else None),
            "max_condition_seconds": args.max_condition_seconds if budget_mode else None,
        },
        "source_token_distribution": {
            "min": min(token_counts),
            "mean": float(np.mean(token_counts)),
            "p50": float(np.quantile(token_counts, 0.50)),
            "p95": float(np.quantile(token_counts, 0.95)),
            "max": max(token_counts),
        },
        "pricing_usd_per_million": {
            "answer_model": pricing_for_model(args.model),
            "merge_model": pricing_for_model(args.merge_model),
            "embedding_input": EMBEDDING_INPUT_USD_PER_MILLION,
            "context_tier": "standard short context",
        },
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
        "max_estimated_cost_usd": args.max_estimated_cost_usd,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    store.write_manifest(manifest)

    completed = store.completed_keys()
    terminal_failures = frozenset(
        (str(error["case_key"]), str(error["condition"]))
        for error in store.load_errors()
        if bool(error.get("terminal")) and "case_key" in error and "condition" in error
    )
    client = OpenAI(timeout=120.0)
    with OpenAIUsageMeter(
        store.append_api_event,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        initial_estimated_cost_usd=store.api_event_cost(),
    ) as meter:
        # Establish that the answer model can solve this subset before spending on compression.
        for case in cases:
            source_tokens = count_tokens(case.prompt)
            if (case.key, "original") not in completed:
                usage_start = len(meter.records)
                started = time.monotonic()
                with (
                    meter.scope(
                        case_key=case.key,
                        condition="original",
                        max_generation_calls=1,
                        max_elapsed_seconds=args.max_condition_seconds,
                    ),
                    meter.category("answer"),
                ):
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
                    prompt_cosine_difference=0.0,
                    compression_elapsed=0.0,
                    answer_elapsed=answer_elapsed,
                    merge_metrics=MergeMetrics(),
                    usage=usage,
                )
                store.append(record, case.prompt)
                print(f"completed {case.key} original", flush=True)

        original_records = tuple(record for record in store.load_records() if record.condition == "original")
        original_accuracy = sum(record.verdict.correct for record in original_records) / len(original_records)
        print(
            f"original baseline accuracy: {original_accuracy * 100:.1f}% "
            f"(minimum {args.min_original_accuracy * 100:.1f}%)",
            flush=True,
        )
        if original_accuracy < args.min_original_accuracy:
            summary = summarize_records(
                original_records,
                release_threshold=args.release_threshold,
                minimum_original_accuracy=args.min_original_accuracy,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.seed,
            )
            report = benchmark_report(summary)
            store.write_summary(summary, report)
            print(report)
            raise SystemExit("original baseline is too weak; compressed conditions were not run")

        for case in cases:
            for reduction in () if budget_mode else reductions:
                condition = condition_name(reduction)
                if (case.key, condition) in completed:
                    continue
                usage_start = len(meter.records)
                with meter.category("compression"):
                    (
                        parts,
                        target_tokens,
                        sent_tokens,
                        metrics,
                        compression_elapsed,
                        prompt_cosine_difference,
                    ) = compress_parts(case.prompt_parts, reduction, merge_model=args.merge_model)
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
                    prompt_cosine_difference=prompt_cosine_difference,
                    compression_elapsed=compression_elapsed,
                    answer_elapsed=answer_elapsed,
                    merge_metrics=metrics,
                    usage=usage,
                )
                store.append(record, parts.prompt)
                print(f"completed {case.key} {condition}", flush=True)

        if budget_mode:
            for case in cases:
                source_tokens = count_tokens(case.prompt)
                for budget in budgets:
                    condition = cos_sim_budget_condition_name(budget)
                    key = (case.key, condition)
                    if key in completed or key in terminal_failures:
                        continue
                    usage_start = len(meter.records)
                    condition_started = time.monotonic()
                    try:
                        with meter.scope(
                            case_key=case.key,
                            condition=condition,
                            max_generation_calls=args.max_generation_calls_per_condition,
                            max_elapsed_seconds=args.max_condition_seconds,
                        ):
                            with meter.category("compression"):
                                (
                                    parts,
                                    target_tokens,
                                    sent_tokens,
                                    metrics,
                                    compression_elapsed,
                                    prompt_cosine_difference,
                                    context_cosine_difference,
                                    context_budget_met,
                                ) = compress_parts_to_cos_sim_budget(
                                    case.prompt_parts,
                                    budget,
                                    minimum_retained_percent=args.minimum_retained_percent,
                                    merge_model=args.merge_model,
                                )
                            answer_started = time.monotonic()
                            with meter.category("answer"):
                                generation = _generate(client, args.model, args.reasoning, parts.prompt)
                            answer_elapsed = time.monotonic() - answer_started
                    except (OpenAIError, UsageLimitExceeded, ValueError) as error:
                        terminal = isinstance(error, UsageLimitExceeded)
                        store.append_errors(
                            (
                                {
                                    "recorded_at": datetime.now(UTC).isoformat(),
                                    "case_key": case.key,
                                    "condition": condition,
                                    "configured_cos_sim_diff_budget": budget,
                                    "error_type": type(error).__name__,
                                    "error_message": str(error),
                                    "elapsed_seconds": time.monotonic() - condition_started,
                                    "terminal": terminal,
                                    "api_events_persisted": len(meter.records) - usage_start,
                                    "estimated_cost_usd": estimate_cost(tuple(meter.records[usage_start:])),
                                },
                            )
                        )
                        print(f"incomplete {case.key} {condition}: {error}", flush=True)
                        if terminal and "estimated API cost limit" in str(error):
                            raise SystemExit(str(error)) from error
                        continue
                    usage = tuple(meter.records[usage_start:])
                    actual_reduction = max(0.0, (1.0 - sent_tokens / source_tokens) * 100.0)
                    record = _record(
                        case=case,
                        condition=condition,
                        reduction_percent=actual_reduction,
                        prompt=parts.prompt,
                        target_tokens=target_tokens,
                        sent_tokens=sent_tokens,
                        generation=generation,
                        adapter=adapter,
                        prompt_cosine_difference=prompt_cosine_difference,
                        compression_elapsed=compression_elapsed,
                        answer_elapsed=answer_elapsed,
                        merge_metrics=metrics,
                        usage=usage,
                        configured_cos_sim_diff_budget=budget,
                        context_cosine_difference=context_cosine_difference,
                        context_budget_met=context_budget_met,
                    )
                    store.append(record, parts.prompt)
                    print(
                        f"completed {case.key} {condition}: retained={sent_tokens / source_tokens:.3f} "
                        f"context_diff={context_cosine_difference:.6f} "
                        f"full_diff={prompt_cosine_difference:.6f} correct={record.verdict.correct}",
                        flush=True,
                    )

    records = store.load_records()
    if budget_mode:
        expected_conditions = tuple(cos_sim_budget_condition_name(budget) for budget in budgets)
        summary = summarize_budget_records(
            records,
            expected_conditions=expected_conditions,
            expected_case_keys=[case.key for case in cases],
            errors=store.load_errors(),
            release_threshold=args.release_threshold,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        )
        report = budget_benchmark_report(summary)
    else:
        summary = summarize_records(
            records,
            release_threshold=args.release_threshold,
            minimum_original_accuracy=args.min_original_accuracy,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        )
        report = benchmark_report(summary)
    store.write_summary(summary, report)
    print(report)


if __name__ == "__main__":
    main()
