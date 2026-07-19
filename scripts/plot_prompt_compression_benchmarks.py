#!/usr/bin/env python3
"""Plot comparable retained-context benchmark curves from saved run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

BENCHMARK_LABELS = {
    "babilong_8k": "BABILong 8k",
    "ruler_v2": "RULERv2",
    "longbench_v2": "LongBench v2",
}
COLORS = {
    "babilong_8k": "#10A37F",
    "ruler_v2": "#7C3AED",
    "longbench_v2": "#2563EB",
    "macro_average": "#EA580C",
}


def retained_percent(condition: str) -> float:
    """Return the complete-prompt retention target encoded in a condition name."""
    if condition == "original":
        return 100.0
    if not condition.startswith("keep"):
        raise ValueError(f"unsupported benchmark condition {condition!r}")
    return float(condition.removeprefix("keep").replace("p", "."))


def load_summaries(run_dirs: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Load exactly one summary for each supported benchmark."""
    summaries: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        path = run_dir / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark = str(payload["benchmark"])
        if benchmark not in BENCHMARK_LABELS:
            raise ValueError(f"unsupported benchmark {benchmark!r} in {path}")
        if benchmark in summaries:
            raise ValueError(f"duplicate summary for {benchmark}")
        summaries[benchmark] = payload
    missing = set(BENCHMARK_LABELS) - summaries.keys()
    if missing:
        raise ValueError(f"missing summaries for: {', '.join(sorted(missing))}")
    return summaries


def aggregate_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the cross-benchmark points used by plots and user-facing tables."""
    condition_sets = [set(summary["conditions"]) for summary in summaries.values()]
    if any(conditions != condition_sets[0] for conditions in condition_sets[1:]):
        raise ValueError("all benchmark summaries must contain the same conditions")
    conditions = sorted(condition_sets[0], key=retained_percent)
    original_accuracies = {
        benchmark: float(summary["conditions"]["original"]["accuracy"]) for benchmark, summary in summaries.items()
    }
    if any(accuracy == 0.0 for accuracy in original_accuracies.values()):
        raise ValueError("accuracy retention is undefined when an original accuracy is zero")
    points: list[dict[str, Any]] = []
    for condition in conditions:
        benchmark_points: dict[str, dict[str, float]] = {}
        for benchmark in BENCHMARK_LABELS:
            raw = summaries[benchmark]["conditions"][condition]
            benchmark_points[benchmark] = {
                "accuracy": float(raw["accuracy"]),
                "accuracy_retention": float(raw["accuracy"]) / original_accuracies[benchmark],
                "official_score": float(raw["official_score"]),
                "cos_sim_diff": float(raw["mean_prompt_embedding_cosine_difference"]),
                "token_reduction": float(raw["token_reduction"]),
                "wall_clock_seconds": float(raw["reduction_seconds"]) + float(raw["execution_seconds"]),
                "estimated_cost_usd": float(raw["estimated_cost_usd"]),
            }
        points.append(
            {
                "condition": condition,
                "retained_percent": retained_percent(condition),
                "benchmarks": benchmark_points,
                "macro_average": {
                    metric: float(np.mean([point[metric] for point in benchmark_points.values()]))
                    for metric in (
                        "accuracy",
                        "accuracy_retention",
                        "official_score",
                        "cos_sim_diff",
                        "token_reduction",
                    )
                },
                "total_wall_clock_seconds": sum(point["wall_clock_seconds"] for point in benchmark_points.values()),
                "total_estimated_cost_usd": sum(point["estimated_cost_usd"] for point in benchmark_points.values()),
            }
        )
    return {"schema_version": 1, "benchmarks": list(BENCHMARK_LABELS), "points": points}


def _style_axis(axis: Any, *, ylabel: str) -> None:
    axis.set_xlabel("Prompt retained (%)")
    axis.set_ylabel(ylabel)
    axis.set_xticks([50, 60, 70, 80, 90, 100])
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def _legend_average_first(axis: Any, *, ncols: int = 2) -> None:
    handles, labels = axis.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda index: labels[index] != "Average")
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        ncols=ncols,
        loc="best",
    )


def _plot_metric(aggregate: dict[str, Any], metric: str, ylabel: str, title: str, path: Path) -> None:
    points = aggregate["points"]
    x = [point["retained_percent"] for point in points]
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for benchmark, label in BENCHMARK_LABELS.items():
        y = [point["benchmarks"][benchmark][metric] for point in points]
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=1.1,
            markersize=3.5,
            color=COLORS[benchmark],
            alpha=0.35,
            label=label,
            zorder=2,
        )
    average = [point["macro_average"][metric] for point in points]
    axis.plot(
        x,
        average,
        marker="o",
        linewidth=2.4,
        markersize=6,
        color=COLORS["macro_average"],
        label="Average",
        zorder=5,
    )
    _style_axis(axis, ylabel=ylabel)
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=14)
    if metric in {"accuracy", "accuracy_retention", "official_score"}:
        if metric != "accuracy_retention":
            axis.set_ylim(-0.03, 1.03)
        axis.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    _legend_average_first(axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_efficiency(aggregate: dict[str, Any], path: Path) -> None:
    points = aggregate["points"]
    x = [point["retained_percent"] for point in points]
    figure, (cost_axis, time_axis) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    cost_axis.plot(
        x,
        [point["total_estimated_cost_usd"] for point in points],
        marker="o",
        linewidth=2.5,
        color="#D97706",
    )
    time_axis.plot(
        x,
        [point["total_wall_clock_seconds"] for point in points],
        marker="o",
        linewidth=2.5,
        color="#DC2626",
    )
    _style_axis(cost_axis, ylabel="Estimated API cost (USD)")
    _style_axis(time_axis, ylabel="Measured wall-clock time (seconds)")
    cost_axis.set_title("Cost across all three benchmarks", loc="left", fontsize=14, fontweight="bold")
    time_axis.set_title("Time across all three benchmarks", loc="left", fontsize=14, fontweight="bold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_semantic_tradeoff(aggregate: dict[str, Any], path: Path) -> None:
    """Plot downstream quality directly against whole-prompt semantic change."""
    points = aggregate["points"]
    figure, (accuracy_axis, retention_axis) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    series = (*BENCHMARK_LABELS.items(), ("macro_average", "Average"))
    for key, label in series:
        x = [
            point["macro_average"]["cos_sim_diff"]
            if key == "macro_average"
            else point["benchmarks"][key]["cos_sim_diff"]
            for point in points
        ]
        for axis, metric in ((accuracy_axis, "accuracy"), (retention_axis, "accuracy_retention")):
            y = [
                point["macro_average"][metric] if key == "macro_average" else point["benchmarks"][key][metric]
                for point in points
            ]
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2.4 if key == "macro_average" else 1.1,
                markersize=6 if key == "macro_average" else 3.5,
                color=COLORS[key],
                alpha=1.0 if key == "macro_average" else 0.35,
                label=label,
                zorder=5 if key == "macro_average" else 2,
            )
            if key == "macro_average":
                for semantic_change, quality, point in zip(x, y, points, strict=True):
                    axis.annotate(
                        f"{point['retained_percent']:.0f}%",
                        (semantic_change, quality),
                        xytext=(5, 6),
                        textcoords="offset points",
                        fontsize=8,
                        color=COLORS[key],
                    )
    for axis in (accuracy_axis, retention_axis):
        axis.set_xlabel("Mean cos_sim_diff (lower is better)")
        axis.grid(color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    accuracy_axis.set_ylabel("Task accuracy")
    accuracy_axis.set_ylim(-0.03, 1.05)
    accuracy_axis.set_title("Absolute accuracy", loc="left", fontsize=14, fontweight="bold")
    retention_axis.set_ylabel("Accuracy retention (original = 100%)")
    retention_axis.set_title("Original-relative retention", loc="left", fontsize=14, fontweight="bold")
    _legend_average_first(retention_axis, ncols=1)
    figure.suptitle("Downstream quality vs. semantic change", x=0.01, ha="left", fontsize=17, fontweight="bold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_outputs(summaries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Write machine-readable aggregates and the three committed figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_summaries(summaries)
    (out_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    _plot_metric(
        aggregate,
        "accuracy",
        "Task accuracy",
        "Accuracy vs. retained prompt context",
        out_dir / "accuracy_vs_retained.png",
    )
    _plot_metric(
        aggregate,
        "accuracy_retention",
        "Accuracy retention (original = 100%)",
        "Accuracy retention vs. retained prompt context",
        out_dir / "accuracy_retention_vs_retained.png",
    )
    _plot_metric(
        aggregate,
        "cos_sim_diff",
        "Mean cos_sim_diff",
        "Semantic change vs. retained prompt context",
        out_dir / "cos_sim_diff_vs_retained.png",
    )
    _plot_efficiency(aggregate, out_dir / "cost_and_time_vs_retained.png")
    _plot_semantic_tradeoff(aggregate, out_dir / "semantic_change_vs_accuracy.png")
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_outputs(load_summaries(args.run_dirs), args.out_dir)


if __name__ == "__main__":
    main()
