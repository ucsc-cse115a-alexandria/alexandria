from __future__ import annotations

import tiktoken

from alexandria.utils.tokens import count_tokens


def test_count_tokens_matches_cl100k_base() -> None:
    text = "Always write tests for new code.\n"
    assert count_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))
