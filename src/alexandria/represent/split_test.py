from __future__ import annotations

from alexandria.represent.split import split


def test_split_is_lossless() -> None:
    prompt = "a\n\nb\nc\n"
    assert "".join(split(prompt)) == prompt


def test_split_groups_one_line_per_segment() -> None:
    assert split("first\nsecond\n") == ["first\n", "second\n"]


def test_blank_prompt_yields_no_segments() -> None:
    assert split("   \n\n") == []


def test_leading_newlines_are_lossless() -> None:
    prompt = "\n\nfirst\nsecond\n"
    assert "".join(split(prompt)) == prompt
