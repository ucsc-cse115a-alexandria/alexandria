"""Token counting shared by the represent phase and the optimizers."""

from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count under cl100k_base — the one encoding every Alexandria token number uses."""
    return len(_ENCODING.encode(text))
