from __future__ import annotations

from alexandria.cli.verbose import VerboseReporter
from alexandria.ir.contracts import ReportedCandidate


def test_redundant_pair_shows_both_sentences_and_similarity() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)

    reporter.redundant_pair("first sentence", "second sentence", 0.987)

    joined = "\n".join(lines)
    assert "0.99" in joined
    assert "first sentence" in joined
    assert "second sentence" in joined


def test_pair_merged_replace_shows_the_merged_text() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)

    reporter.pair_merged("the merged sentence", "replace")

    assert any("the merged sentence" in line for line in lines)


def test_pair_merged_delete_mentions_keeping_the_first() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)

    reporter.pair_merged(None, "delete")

    assert any("kept the first" in line for line in lines)


def test_pair_merged_skipped_mentions_the_drift_budget() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)

    reporter.pair_merged(None, "skipped")

    assert any("skipped" in line for line in lines)


def test_target_group_shows_token_and_savings_counts() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)

    reporter.target_group("some source text", group_tokens=42, required_savings=10)

    joined = "\n".join(lines)
    assert "42" in joined
    assert "10" in joined
    assert "some source text" in joined


def test_target_round_reports_candidates_and_selection() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)
    candidates = (
        ReportedCandidate(text="candidate one", token_count=10, drift=0.05, structure_valid=True),
        ReportedCandidate(text="candidate two", token_count=8, drift=0.9, structure_valid=False),
    )
    selected = candidates[0]

    reporter.target_round(1, "base text", candidates, selected, True)

    joined = "\n".join(lines)
    assert "round 1" in joined
    assert "base text" in joined
    assert "candidate one" in joined
    assert "candidate two" in joined
    assert "[invalid structure]" in joined
    assert "selected" in joined
    assert "generated" in joined


def test_target_round_with_no_base_omits_the_search_base_line() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)
    selected = ReportedCandidate(text="only candidate", token_count=5, drift=0.1, structure_valid=True)

    reporter.target_round(1, None, (selected,), selected, False)

    joined = "\n".join(lines)
    assert "search base" not in joined
    assert "kept base" in joined


def test_target_group_done_applied_shows_the_new_token_count() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)

    reporter.target_group_done(applied=True, document_tokens=123)

    assert any("123" in line and "applied" in line for line in lines)


def test_target_group_done_not_applied_reports_failure() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)

    reporter.target_group_done(applied=False, document_tokens=99)

    assert any("not applied" in line for line in lines)


def test_long_text_is_clipped_with_an_ellipsis() -> None:
    lines: list[str] = []
    reporter = VerboseReporter(lines.append)
    long_text = " ".join(f"word{i}" for i in range(100))

    reporter.redundant_pair(long_text, "short", 0.9)

    a_line = next(line for line in lines if line.strip().startswith("A:"))
    assert a_line.endswith("…")
    assert len(a_line) < len(long_text)
