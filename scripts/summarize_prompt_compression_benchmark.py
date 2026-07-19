#!/usr/bin/env python3
"""Recompute benchmark confidence intervals and release decisions from saved raw records."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.prompt_compression.runner import benchmark_report, summarize_records
from benchmarks.prompt_compression.store import RunStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--release-threshold", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    store = RunStore(args.run_dir)
    summary = summarize_records(
        store.load_records(),
        release_threshold=args.release_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report = benchmark_report(summary)
    store.write_summary(summary, report)
    print(report)


if __name__ == "__main__":
    main()
