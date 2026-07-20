#!/usr/bin/env python3
"""Plot semantic-budget benchmark curves from saved n-case summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BENCHMARK_LABELS = {
    "babilong_8k": "BABILong 8k",
    "ruler_v2": "RULERv2",
}
COLORS = {
    "babilong_8k": "#10A37F",
    "ruler_v2": "#7C3AED",
    "average": "#EA580C",
}


def configured_budget(condition: str) -> float:
    """Return the configured cosine-difference ceiling encoded in a condition."""
    if condition == "original":
        return 0.0
    if not condition.startswith("budget"):
        raise ValueError(f"unsupported benchmark condition {condition!r}")
    return float(condition.removeprefix("budget").replace("p", "."))


def load_summaries(run_dirs: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Load one budget-mode summary for each included benchmark."""
    summaries: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        path = run_dir / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark = str(payload["benchmark"])
        if benchmark not in BENCHMARK_LABELS:
            raise ValueError(f"unsupported benchmark {benchmark!r} in {path}")
        if payload.get("experiment_mode") != "cos_sim_diff_budget":
            raise ValueError(f"summary is not a cos_sim_diff budget run: {path}")
        if benchmark in summaries:
            raise ValueError(f"duplicate summary for {benchmark}")
        summaries[benchmark] = payload
    missing = set(BENCHMARK_LABELS) - summaries.keys()
    if missing:
        raise ValueError(f"missing summaries for: {', '.join(sorted(missing))}")
    return summaries


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload[key]
    if not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def aggregate_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build equal-weight cross-benchmark points for figures and report tables."""
    if set(summaries) != set(BENCHMARK_LABELS):
        raise ValueError("summaries must contain BABILong 8k and RULERv2")
    condition_sets = [set(summary["conditions"]) for summary in summaries.values()]
    if any(conditions != condition_sets[0] for conditions in condition_sets[1:]):
        raise ValueError("all benchmark summaries must contain the same conditions")
    conditions = sorted(condition_sets[0], key=configured_budget)
    points: list[dict[str, Any]] = []
    for condition in conditions:
        benchmark_points: dict[str, dict[str, float]] = {}
        for benchmark in BENCHMARK_LABELS:
            raw = summaries[benchmark]["conditions"][condition]
            if not isinstance(raw, dict):
                raise TypeError(f"condition {condition!r} has no completed records for {benchmark}")
            if condition == "original":
                completion_rate = 1.0
                budget_compliance = 1.0
                completed_cases = int(_number(raw, "n_cases"))
                expected_cases = completed_cases
            else:
                comparison = summaries[benchmark]["comparisons"][condition]
                completion_rate = _number(comparison, "completion_rate")
                budget_compliance = _number(comparison, "context_budget_compliance")
                completed_cases = int(_number(comparison, "completed_cases"))
                expected_cases = int(_number(comparison, "expected_cases"))
            benchmark_points[benchmark] = {
                "completed_cases": completed_cases,
                "expected_cases": expected_cases,
                "accuracy": _number(raw, "accuracy"),
                "official_score": _number(raw, "official_score"),
                "token_reduction": _number(raw, "token_reduction"),
                "realized_cos_sim_diff": _number(raw, "mean_prompt_embedding_cosine_difference"),
                "completion_rate": completion_rate,
                "budget_compliance": budget_compliance,
                "wall_clock_seconds": _number(raw, "reduction_seconds") + _number(raw, "execution_seconds"),
                "estimated_cost_usd": _number(raw, "estimated_cost_usd"),
            }
        points.append(
            {
                "condition": condition,
                "configured_budget": configured_budget(condition),
                "benchmarks": benchmark_points,
                "average": {
                    metric: float(np.mean([point[metric] for point in benchmark_points.values()]))
                    for metric in (
                        "accuracy",
                        "official_score",
                        "token_reduction",
                        "realized_cos_sim_diff",
                        "completion_rate",
                        "budget_compliance",
                    )
                },
                "total_wall_clock_seconds": sum(point["wall_clock_seconds"] for point in benchmark_points.values()),
                "total_estimated_cost_usd": sum(point["estimated_cost_usd"] for point in benchmark_points.values()),
            }
        )
    return {
        "schema_version": 1,
        "experiment_mode": "cos_sim_diff_budget",
        "average_definition": "equal-weight mean of BABILong 8k and RULERv2",
        "benchmarks": list(BENCHMARK_LABELS),
        "points": points,
    }


def _series() -> tuple[tuple[str, str], ...]:
    return (
        ("average", "Average"),
        *((benchmark, label) for benchmark, label in BENCHMARK_LABELS.items()),
    )


def _values(aggregate: dict[str, Any], key: str, metric: str) -> list[float]:
    return [
        point["average"][metric] if key == "average" else point["benchmarks"][key][metric]
        for point in aggregate["points"]
    ]


def _plot_line(axis: Any, x: Sequence[float], y: Sequence[float], key: str, label: str) -> None:
    is_average = key == "average"
    axis.plot(
        x,
        y,
        marker="o",
        linewidth=2.8 if is_average else 1.2,
        markersize=6 if is_average else 3.8,
        color=COLORS[key],
        alpha=1.0 if is_average else 0.35,
        label=label,
        zorder=5 if is_average else 2,
    )


def _style_axis(axis: Any, *, xlabel: str, ylabel: str, percent_y: bool = False) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    if percent_y:
        axis.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")


def _legend_average_first(axis: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda index: labels[index] != "Average")
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        ncols=3,
        loc="best",
    )


def _plot_quality_and_reduction(aggregate: dict[str, Any], path: Path) -> None:
    x = [point["configured_budget"] for point in aggregate["points"]]
    figure, (accuracy_axis, reduction_axis) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    for key, label in _series():
        _plot_line(accuracy_axis, x, _values(aggregate, key, "accuracy"), key, label)
        _plot_line(reduction_axis, x, _values(aggregate, key, "token_reduction"), key, label)
    for axis in (accuracy_axis, reduction_axis):
        axis.xaxis.set_major_formatter(lambda value, _position: f"{value:.4f}".rstrip("0").rstrip("."))
    _style_axis(
        accuracy_axis,
        xlabel="Configured cos_sim_diff budget",
        ylabel="Task accuracy",
        percent_y=True,
    )
    _style_axis(
        reduction_axis,
        xlabel="Configured cos_sim_diff budget",
        ylabel="Mean prompt token reduction",
        percent_y=True,
    )
    accuracy_axis.set_ylim(-0.03, 1.03)
    reduction_axis.set_ylim(bottom=0.0)
    reduction_axis.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.1f}%")
    accuracy_axis.set_title("Downstream quality", loc="left", fontsize=14, fontweight="bold")
    reduction_axis.set_title("Prompt reduction", loc="left", fontsize=14, fontweight="bold")
    _legend_average_first(accuracy_axis)
    figure.suptitle(
        "Quality and reduction under semantic-change budgets",
        x=0.01,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_pareto(aggregate: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for key, label in _series():
        x = _values(aggregate, key, "token_reduction")
        y = _values(aggregate, key, "accuracy")
        _plot_line(axis, x, y, key, label)
        if key == "average":
            for reduction, accuracy, point in zip(x, y, aggregate["points"], strict=True):
                budget = point["configured_budget"]
                annotation = "original" if budget == 0 else f"≤ {budget:g}"
                axis.annotate(annotation, (reduction, accuracy), xytext=(6, 6), textcoords="offset points", fontsize=8)
    _style_axis(axis, xlabel="Mean prompt token reduction", ylabel="Task accuracy", percent_y=True)
    axis.xaxis.set_major_formatter(lambda value, _position: f"{value * 100:.1f}%")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Accuracy vs. prompt reduction", loc="left", fontsize=17, fontweight="bold", pad=14)
    _legend_average_first(axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_semantic_tradeoff(aggregate: dict[str, Any], path: Path) -> None:
    figure, (accuracy_axis, reduction_axis) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    for key, label in _series():
        x = _values(aggregate, key, "realized_cos_sim_diff")
        _plot_line(accuracy_axis, x, _values(aggregate, key, "accuracy"), key, label)
        _plot_line(reduction_axis, x, _values(aggregate, key, "token_reduction"), key, label)
    _style_axis(accuracy_axis, xlabel="Mean realized full-prompt cos_sim_diff", ylabel="Task accuracy", percent_y=True)
    _style_axis(
        reduction_axis,
        xlabel="Mean realized full-prompt cos_sim_diff",
        ylabel="Mean prompt token reduction",
        percent_y=True,
    )
    accuracy_axis.set_ylim(-0.03, 1.03)
    reduction_axis.set_ylim(bottom=0.0)
    reduction_axis.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.1f}%")
    accuracy_axis.set_title("Semantic change vs. quality", loc="left", fontsize=14, fontweight="bold")
    reduction_axis.set_title("Semantic change vs. reduction", loc="left", fontsize=14, fontweight="bold")
    _legend_average_first(accuracy_axis)
    figure.suptitle("Observed semantic-change trade-off", x=0.01, ha="left", fontsize=17, fontweight="bold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_reliability(aggregate: dict[str, Any], path: Path) -> None:
    points = [point for point in aggregate["points"] if point["condition"] != "original"]
    x = [point["configured_budget"] for point in points]
    figure, (compliance_axis, completion_axis) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for key, label in _series():
        compliance = [
            point["average"]["budget_compliance"]
            if key == "average"
            else point["benchmarks"][key]["budget_compliance"]
            for point in points
        ]
        completion = [
            point["average"]["completion_rate"] if key == "average" else point["benchmarks"][key]["completion_rate"]
            for point in points
        ]
        _plot_line(compliance_axis, x, compliance, key, label)
        _plot_line(completion_axis, x, completion, key, label)
    _style_axis(
        compliance_axis,
        xlabel="Configured cos_sim_diff budget",
        ylabel="Context-budget compliance",
        percent_y=True,
    )
    _style_axis(
        completion_axis,
        xlabel="Configured cos_sim_diff budget",
        ylabel="Completed case-condition pairs",
        percent_y=True,
    )
    for axis in (compliance_axis, completion_axis):
        axis.set_ylim(-0.03, 1.03)
    compliance_axis.set_title("Budget reliability", loc="left", fontsize=14, fontweight="bold")
    completion_axis.set_title("Operational completion", loc="left", fontsize=14, fontweight="bold")
    _legend_average_first(compliance_axis)
    figure.suptitle("Measurement reliability", x=0.01, ha="left", fontsize=17, fontweight="bold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_outputs(summaries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Write the machine-readable aggregate and publication figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_summaries(summaries)
    (out_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    _plot_quality_and_reduction(aggregate, out_dir / "quality_and_reduction_vs_budget.png")
    _plot_pareto(aggregate, out_dir / "accuracy_vs_token_reduction.png")
    _plot_semantic_tradeoff(aggregate, out_dir / "semantic_change_tradeoff.png")
    _plot_reliability(aggregate, out_dir / "reliability_vs_budget.png")
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_outputs(load_summaries(args.run_dirs), args.out_dir)


if __name__ == "__main__":
    main()
