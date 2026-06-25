"""Split prompts into individual instruction segments."""

import re

_LIST_MARKER = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


def segment_instructions(prompt: str) -> list[str]:
    """Split a prompt into individual instructions, one per non-empty line.

    Leading list markers (``-``, ``*``, ``•``, ``1.``, ``1)``) and surrounding
    whitespace are stripped so each returned segment is the bare instruction
    text, ready for embedding and clustering downstream.
    """
    segments: list[str] = []
    for line in prompt.splitlines():
        instruction = _LIST_MARKER.sub("", line.strip())
        if instruction:
            segments.append(instruction)
    return segments
