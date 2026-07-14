from __future__ import annotations

from alexandria.ir.contracts import Params
from alexandria.ops.pipe import compare_reports, optimization_report, score_report
from alexandria.utils.embedders import HashEmbedder


def test_score_report_adds_the_redundant_peer() -> None:
    rows = score_report("repeat me\nrepeat me\nunique line\n", HashEmbedder())
    assert rows[0]["most_similar_id"] == rows[1]["id"]
    assert rows[0]["most_similar_text"] == "repeat me"
    redundancy = rows[0]["redundancy"]
    assert isinstance(redundancy, float)
    assert redundancy > 0.99


def test_score_report_single_sentence_has_no_peer() -> None:
    rows = score_report("only one\n", HashEmbedder())
    assert rows[0]["most_similar_id"] is None
    assert rows[0]["most_similar_text"] is None


def test_optimization_report_includes_compression_and_quality_metrics() -> None:
    report = optimization_report(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        params=Params(drift_budget=2.0),
    )

    assert report.tokens.reduced < report.tokens.source
    assert report.tokens.saved == report.tokens.source - report.tokens.reduced
    assert report.quality.instruction_coverage == 1.0
    assert report.quality.minimum_instruction_similarity == 1.0


def test_compare_reports_flags_token_and_quality_regressions() -> None:
    baseline = optimization_report(
        "repeat me\nrepeat me\nunique line\n",
        HashEmbedder(),
        params=Params(drift_budget=2.0),
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
