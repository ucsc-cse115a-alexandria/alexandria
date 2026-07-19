from __future__ import annotations

from alexandria.utils.merger import trim_to_last_sentence


def test_trim_keeps_text_already_ending_on_final_punctuation() -> None:
    text = "  Keep every rule. Ship it.  "

    assert trim_to_last_sentence(text) == "Keep every rule. Ship it."


def test_trim_keeps_text_ending_on_quote() -> None:
    assert trim_to_last_sentence('She said "go"') == 'She said "go"'


def test_trim_cuts_dangling_fragment_at_last_sentence_boundary() -> None:
    text = "First fact. Second fact. And then an incomplete thought that ran out of budg"

    assert trim_to_last_sentence(text) == "First fact. Second fact."


def test_trim_cuts_on_newline_boundary_keeping_the_period() -> None:
    text = "First fact.\nA half-written follow"

    assert trim_to_last_sentence(text) == "First fact."


def test_trim_keeps_closing_paren_after_period() -> None:
    text = "A rule (see note.) then unfinished tex"

    assert trim_to_last_sentence(text) == "A rule (see note.)"


def test_trim_returns_stripped_text_when_no_boundary_found() -> None:
    text = "  one unbroken clause with no terminal punctuation  "

    assert trim_to_last_sentence(text) == "one unbroken clause with no terminal punctuation"
