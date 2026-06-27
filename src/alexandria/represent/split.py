"""Split a prompt into individual instruction segments."""

from __future__ import annotations

import re

_SEPARATOR = re.compile(r"\n+")


def split(prompt: str) -> list[str]:
    """Split a prompt into segments such that ``"".join(split(prompt)) == prompt``.

    Each segment is one line of content plus the newline(s) that follow it, so the original
    separators survive and the pieces re-concatenate into the exact input. A blank prompt yields [].
    """
    if not prompt.strip():
        return []
    segments: list[str] = []
    pos = 0
    for match in _SEPARATOR.finditer(prompt):
        content = prompt[pos : match.start()]
        separator = match.group()
        if content:
            segments.append(content + separator)
        elif segments:
            segments[-1] += separator
        else:
            segments.append(separator)
        pos = match.end()
    tail = prompt[pos:]
    if tail:
        segments.append(tail)
    return segments
