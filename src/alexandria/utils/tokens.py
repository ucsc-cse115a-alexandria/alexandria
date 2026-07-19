"""Token counting shared by the represent phase and the optimizers."""

from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count under cl100k_base — the one encoding every Alexandria token number uses."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text under the shared tokenizer while retaining trailing whitespace when possible."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    encoded = _ENCODING.encode(text, disallowed_special=())
    if len(encoded) <= max_tokens:
        return text

    body = text.rstrip()
    suffix = text[len(body) :]
    body_tokens = _ENCODING.encode(body, disallowed_special=())
    if suffix:
        for body_limit in range(min(len(body_tokens), max_tokens), 0, -1):
            candidate = _ENCODING.decode(body_tokens[:body_limit]).rstrip() + suffix
            if candidate.strip() and count_tokens(candidate) <= max_tokens:
                return candidate

    return _ENCODING.decode(encoded[:max_tokens])
