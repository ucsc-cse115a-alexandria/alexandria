from __future__ import annotations

import pytest

from scripts.babilong_8k_phase1 import require_target_reduction


def test_require_target_reduction_accepts_met_target() -> None:
    require_target_reduction(1_000, 100, 90)


def test_require_target_reduction_rejects_unmet_target() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"target reduction 90% was not met: achieved 0.3% \(73743 -> 73520 tokens\)",
    ):
        require_target_reduction(73_743, 73_520, 90)
