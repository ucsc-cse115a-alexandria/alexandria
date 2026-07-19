from __future__ import annotations

import tiktoken

from alexandria.utils.tokens import count_tokens, truncate_tokens


def test_count_tokens_matches_cl100k_base() -> None:
    text = "Always write tests for new code.\n"
    assert count_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))


def test_literal_special_token_text_is_counted_as_ordinary_input() -> None:
    text = "repository code contains <|endoftext|> as a literal"
    expected = tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=())
    assert count_tokens(text) == len(expected)
    assert count_tokens(truncate_tokens(text, 4)) <= 4


def test_truncate_tokens_enforces_the_limit_and_preserves_trailing_newline() -> None:
    text = "Always write focused tests for every new behavior.\n"

    truncated = truncate_tokens(text, 5)

    assert truncated.endswith("\n")
    assert truncated.strip()
    assert count_tokens(truncated) <= 5


def test_truncate_tokens_leaves_text_within_the_limit_unchanged() -> None:
    text = "Already short.\n"

    assert truncate_tokens(text, count_tokens(text)) == text
