from __future__ import annotations

from typing import Any

import pytest

from scripts.plot_babilong_retention import build_points


def _summary(*, qualifies: bool = True) -> dict[str, Any]:
    return {
        "benchmark": "babilong_8k",
        "baseline_qualification": {"qualifies": qualifies},
        "conditions": {
            "original": {"accuracy": 0.8, "token_reduction": 0.0, "n_cases": 50},
            "keep90": {"accuracy": 0.6, "token_reduction": 0.11},
        },
    }


def test_build_points_orders_conditions_by_retained_percent() -> None:
    aggregate = build_points(_summary())

    assert aggregate["benchmark"] == "babilong_8k"
    assert aggregate["n_cases"] == 50
    assert aggregate["original_accuracy"] == 0.8
    assert [point["condition"] for point in aggregate["points"]] == ["keep90", "original"]
    keep90 = aggregate["points"][0]
    assert keep90["retained_percent"] == 90.0
    assert keep90["accuracy"] == 0.6
    assert keep90["token_reduction"] == 0.11


def test_build_points_rejects_unqualified_baseline() -> None:
    with pytest.raises(ValueError, match="does not qualify"):
        build_points(_summary(qualifies=False))
