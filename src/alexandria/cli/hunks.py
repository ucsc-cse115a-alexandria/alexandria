"""Shared diff-hunk computation for the terminal and browser review UIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from alexandria.ir.document import Document, SentenceId

HunkKind = Literal["removed", "added", "context", "gap", "empty"]
"""How one hunk line should read: a removed sentence, a replacement (`+`) line, kept context,
a `···` gap between distant hunks, or the empty-selection message."""

_EMPTY_MESSAGE = "(no edits accepted — output equals the original)"


def hunk_lines(
    document: Document,
    removed: frozenset[SentenceId],
    added: Mapping[SentenceId, str] | None = None,
) -> list[tuple[HunkKind, str]]:
    """Diff hunk for the sentences ``removed`` takes out, each with one sentence of context.

    The removals are known exactly, so the hunk never guesses with a text diff — on large
    documents difflib's junk heuristics turn a pure one-line deletion into huge phantom "+"
    blocks. Distant hunks are separated by a ``gap`` marker. When ``added`` maps a removed
    sentence id to replacement text, that text's lines follow the sentence as ``added`` lines.

    Returns neutral ``(kind, text)`` tuples so terminal (ANSI) and HTML callers style them.
    """
    sentences = document.sentences
    shown: set[int] = set()
    for position, sentence in enumerate(sentences):
        if sentence.id in removed:
            shown.update(index for index in (position - 1, position, position + 1) if 0 <= index < len(sentences))
    if not shown:
        return [("empty", _EMPTY_MESSAGE)]
    body: list[tuple[HunkKind, str]] = []
    previous: int | None = None
    for index in sorted(shown):
        if previous is not None and index > previous + 1:
            body.append(("gap", "···"))
        sentence = sentences[index]
        kind: HunkKind = "removed" if sentence.id in removed else "context"
        body.extend((kind, line) for line in sentence.text.splitlines() or [""])
        if added is not None:
            body.extend(("added", line) for line in added.get(sentence.id, "").splitlines())
        previous = index
    return body
