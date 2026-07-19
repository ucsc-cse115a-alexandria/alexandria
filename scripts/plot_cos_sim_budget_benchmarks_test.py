from __future__ import annotations

from typing import Any

import pytest

from scripts.plot_cos_sim_budget_benchmarks import aggregate_summaries, configured_budget, write_outputs


def _condition(accuracy: float, reduction: float, difference: float) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "official_score": accuracy,
        "token_reduction": reduction,
        "mean_prompt_embedding_cosine_difference": difference,
        "reduction_seconds": 2.0,
        "execution_seconds": 1.0,
        "estimated_cost_usd": 0.1,
    }


def _summary(benchmark: str, accuracy: float) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "conditions": {
            "original": _condition(accuracy, 0.0, 0.0),
            "budget0p0025": _condition(accuracy - 0.1, 0.2, 0.002),
            "budget0p02": _condition(accuracy, 0.4, 0.015),
        },
        "comparisons": {
            "budget0p0025": {"completion_rate": 1.0, "context_budget_compliance": 0.9},
            "budget0p02": {"completion_rate": 0.98, "context_budget_compliance": 1.0},
        },
    }


def test_configured_budget_parses_condition_names() -> None:
    assert configured_budget("original") == 0.0
    assert configured_budget("budget0p0025") == 0.0025
    assert configured_budget("budget0p02") == 0.02
    with pytest.raises(ValueError, match="unsupported"):
        configured_budget("keep50")


def test_aggregate_summaries_equal_weights_benchmarks() -> None:
    aggregate = aggregate_summaries(
        {
            "babilong_8k": _summary("babilong_8k", 0.8),
            "ruler_v2": _summary("ruler_v2", 0.6),
        }
    )

    assert [point["configured_budget"] for point in aggregate["points"]] == [0.0, 0.0025, 0.02]
    assert aggregate["points"][0]["average"]["accuracy"] == pytest.approx(0.7)
    assert aggregate["points"][1]["average"]["token_reduction"] == pytest.approx(0.2)
    assert aggregate["points"][1]["average"]["budget_compliance"] == pytest.approx(0.9)
    assert aggregate["points"][2]["average"]["completion_rate"] == pytest.approx(0.98)


def test_aggregate_summaries_requires_matching_conditions() -> None:
    left = _summary("babilong_8k", 0.8)
    right = _summary("ruler_v2", 0.6)
    del right["conditions"]["budget0p02"]

    with pytest.raises(ValueError, match="same conditions"):
        aggregate_summaries({"babilong_8k": left, "ruler_v2": right})


def test_write_outputs_creates_aggregate_and_four_figures(tmp_path) -> None:
    aggregate = write_outputs(
        {
            "babilong_8k": _summary("babilong_8k", 0.8),
            "ruler_v2": _summary("ruler_v2", 0.6),
        },
        tmp_path,
    )

    assert aggregate["average_definition"].startswith("equal-weight")
    assert {path.name for path in tmp_path.iterdir()} == {
        "aggregate_summary.json",
        "quality_and_reduction_vs_budget.png",
        "accuracy_vs_token_reduction.png",
        "semantic_change_tradeoff.png",
        "reliability_vs_budget.png",
    }
    assert all(path.stat().st_size > 1_000 for path in tmp_path.glob("*.png"))
