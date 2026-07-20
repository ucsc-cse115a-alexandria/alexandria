from __future__ import annotations

import pytest
from pydantic import ValidationError

from alexandria.ir.contracts import Params
from alexandria.ops.report import OptimizationReport, compare_reports, optimization_report
from alexandria.utils.embedders import HashEmbedder


class _FirstWinsMerger:
    """Offline merger that returns the first sentence, so every merge lands as a Delete of the second."""

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first.strip()

    def merge_candidates_to_target(self, prompt: str, max_tokens: int) -> tuple[str, ...]:
        del max_tokens
        return (prompt,)


class _OtherMerger(_FirstWinsMerger):
    pass


def test_optimization_report_includes_compression_and_quality_metrics() -> None:
    report = optimization_report(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(cos_sim_diff_budget=2.0),
    )

    assert report.tokens.reduced < report.tokens.source
    assert report.tokens.saved == report.tokens.source - report.tokens.reduced
    assert report.quality.instruction_coverage == 1.0
    assert report.quality.minimum_instruction_similarity == 1.0
    assert report.schema_version == 3
    assert report.config.merger.endswith("._FirstWinsMerger")


def test_report_config_captures_target_settings() -> None:
    report = optimization_report(
        "one line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(max_tokens=100, require_target=True),
    )

    assert report.config.max_tokens == 100
    assert report.config.require_target is True


def test_report_comparison_rejects_a_different_merger_identity() -> None:
    prompt = "repeat me\nrepeat me\nunique line\n"
    baseline = optimization_report(prompt, HashEmbedder(), _FirstWinsMerger(), params=Params(cos_sim_diff_budget=2.0))
    current = optimization_report(prompt, HashEmbedder(), _OtherMerger(), params=Params(cos_sim_diff_budget=2.0))

    with pytest.raises(ValueError, match="config does not match"):
        compare_reports(current, baseline)


def test_report_schema_v3_rejects_a_v2_baseline() -> None:
    report = optimization_report(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(cos_sim_diff_budget=2.0),
    )
    payload = report.model_dump(mode="json")
    payload["schema_version"] = 2

    with pytest.raises(ValidationError):
        OptimizationReport.model_validate(payload)


def test_compare_reports_flags_token_and_quality_regressions() -> None:
    baseline = optimization_report(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        _FirstWinsMerger(),
        params=Params(cos_sim_diff_budget=2.0),
    )
    current = baseline.model_copy(
        update={
            "tokens": baseline.tokens.model_copy(update={"reduced": baseline.tokens.reduced + 1}),
            "quality": baseline.quality.model_copy(
                update={"instruction_coverage": baseline.quality.instruction_coverage - 0.1}
            ),
        }
    )

    comparison = compare_reports(current, baseline)

    assert not comparison.passed
    assert {regression.metric for regression in comparison.regressions} == {
        "tokens.reduced",
        "quality.instruction_coverage",
    }
