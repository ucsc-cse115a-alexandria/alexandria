#!/usr/bin/env python3
"""Plot a single-benchmark retained-context accuracy curve from one saved run summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from scripts.plot_prompt_compression_benchmarks import BENCHMARK_LABELS, retained_percent

ACCENT = "#10A37F"


def build_points(summary: dict[str, Any]) -> dict[str, Any]:
    """Build the retention-curve points from one saved run summary."""
    qualification = summary.get("baseline_qualification", {})
    if not bool(qualification.get("qualifies", True)):
        raise ValueError("the original baseline does not qualify, so the curve has no reference")
    points = [
        {
            "condition": condition,
            "retained_percent": retained_percent(condition),
            "accuracy": float(raw["accuracy"]),
            "token_reduction": float(raw["token_reduction"]),
        }
        for condition, raw in sorted(summary["conditions"].items(), key=lambda item: retained_percent(item[0]))
    ]
    return {
        "schema_version": 1,
        "benchmark": str(summary["benchmark"]),
        "n_cases": int(summary["conditions"]["original"]["n_cases"]),
        "original_accuracy": float(summary["conditions"]["original"]["accuracy"]),
        "points": points,
    }


def plot_accuracy_curve(aggregate: dict[str, Any], path: Path) -> None:
    """Render task accuracy against retained prompt share."""
    points = aggregate["points"]
    x = [point["retained_percent"] for point in points]
    y = [point["accuracy"] for point in points]
    original_accuracy = aggregate["original_accuracy"]
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.axhline(original_accuracy, color="#9CA3AF", linewidth=1.2, linestyle="--", zorder=1)
    axis.annotate(
        f"Original {original_accuracy * 100:.0f}%",
        (x[0], original_accuracy),
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=9,
        color="#6B7280",
    )
    axis.plot(x, y, marker="o", linewidth=2.4, markersize=6, color=ACCENT, zorder=5)
    for point in points:
        axis.annotate(
            f"{point['accuracy'] * 100:.0f}%",
            (point["retained_percent"], point["accuracy"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=9,
            color=ACCENT,
        )
    axis.set_xlabel("Prompt retained (%)")
    axis.set_ylabel("Task accuracy")
    axis.set_xticks(x)
    axis.set_ylim(-0.03, 1.03)
    axis.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    label = BENCHMARK_LABELS.get(aggregate["benchmark"], aggregate["benchmark"])
    n_cases = aggregate["n_cases"]
    axis.set_title(
        f"{label} accuracy vs. retained prompt context (n={n_cases})",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_outputs(summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write the machine-readable aggregate and the committed figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = build_points(summary)
    (out_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    plot_accuracy_curve(aggregate, out_dir / "accuracy_vs_retained.png")
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    write_outputs(summary, args.out_dir)


if __name__ == "__main__":
    main()
