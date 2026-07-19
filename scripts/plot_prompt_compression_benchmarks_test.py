from __future__ import annotations

from typing import Any

import pytest

from scripts.plot_prompt_compression_benchmarks import aggregate_summaries, retained_percent


def _condition(accuracy: float, cos_sim_diff: float, cost: float) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "official_score": accuracy,
        "mean_prompt_embedding_cosine_difference": cos_sim_diff,
        "token_reduction": 0.5,
        "reduction_seconds": 2.0,
        "execution_seconds": 1.0,
        "estimated_cost_usd": cost,
    }


def test_retained_percent_parses_original_and_decimal_conditions() -> None:
    assert retained_percent("keep50") == 50.0
    assert retained_percent("keep87p5") == 87.5
    assert retained_percent("original") == 100.0
    with pytest.raises(ValueError, match="unsupported"):
        retained_percent("compressed")


def test_aggregate_summaries_adds_macro_accuracy_cost_and_time() -> None:
    summaries: dict[str, dict[str, Any]] = {
        benchmark: {
            "conditions": {
                "original": _condition(1.0, 0.0, 0.1),
                "keep50": _condition(accuracy, 0.2, 0.2),
            }
        }
        for benchmark, accuracy in {
            "babilong_8k": 1.0,
            "ruler_v2": 0.5,
            "longbench_v2": 0.0,
        }.items()
    }

    aggregate = aggregate_summaries(summaries)

    assert [point["retained_percent"] for point in aggregate["points"]] == [50.0, 100.0]
    keep50 = aggregate["points"][0]
    assert keep50["macro_average"]["accuracy"] == 0.5
    assert keep50["macro_average"]["accuracy_retention"] == 0.5
    assert keep50["total_estimated_cost_usd"] == pytest.approx(0.6)
    assert keep50["total_wall_clock_seconds"] == 9.0


def test_aggregate_summaries_rejects_mismatched_conditions() -> None:
    summaries: dict[str, dict[str, Any]] = {
        "babilong_8k": {"conditions": {"original": _condition(1.0, 0.0, 0.1)}},
        "ruler_v2": {"conditions": {"original": _condition(1.0, 0.0, 0.1)}},
        "longbench_v2": {
            "conditions": {
                "original": _condition(1.0, 0.0, 0.1),
                "keep50": _condition(1.0, 0.1, 0.2),
            }
        },
    }

    with pytest.raises(ValueError, match="same conditions"):
        aggregate_summaries(summaries)


def test_aggregate_summaries_excludes_failed_baseline_from_curves() -> None:
    summaries: dict[str, dict[str, Any]] = {
        "babilong_8k": {
            "baseline_qualification": {"qualifies": True},
            "conditions": {
                "original": _condition(0.8, 0.0, 0.1),
                "keep50": _condition(0.4, 0.2, 0.2),
            },
        },
        "ruler_v2": {
            "baseline_qualification": {"qualifies": True},
            "conditions": {
                "original": _condition(0.6, 0.0, 0.1),
                "keep50": _condition(0.3, 0.1, 0.2),
            },
        },
        "longbench_v2": {
            "baseline_qualification": {
                "qualifies": False,
                "decision": "FAIL: original accuracy does not clear the minimum baseline",
            },
            "conditions": {"original": _condition(0.48, 0.0, 1.8)},
        },
    }

    aggregate = aggregate_summaries(summaries)

    assert aggregate["qualified_benchmarks"] == ["babilong_8k", "ruler_v2"]
    assert aggregate["excluded_benchmarks"] == {
        "longbench_v2": "FAIL: original accuracy does not clear the minimum baseline"
    }
    assert aggregate["baseline_only"]["longbench_v2"]["accuracy"] == 0.48
    assert aggregate["points"][0]["macro_average"]["accuracy"] == pytest.approx(0.35)
    assert set(aggregate["points"][0]["benchmarks"]) == {"babilong_8k", "ruler_v2"}
