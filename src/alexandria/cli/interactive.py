from __future__ import annotations

import difflib
import shutil
from typing import TYPE_CHECKING, Self

import click
from pydantic import BaseModel, ConfigDict, model_validator

from alexandria.ir.contracts import Candidate, Diff

if TYPE_CHECKING:
    from collections.abc import Callable

    from alexandria.ir.document import Document

_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_HOME_AND_CLEAR = "\x1b[H\x1b[J"
_UP_KEYS = ("\x1b[A", "k")
_DOWN_KEYS = ("\x1b[B", "j")
_TOGGLE_KEYS = ("\r", "\n", " ")


class ReviewState(BaseModel):
    """The selection state machine behind the interactive review; every transition returns a new state."""

    model_config = ConfigDict(frozen=True)
    diffs: tuple[Diff, ...]
    cursor: int
    accepted: frozenset[int]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not 0 <= self.cursor < len(self.diffs):
            raise ValueError(f"cursor {self.cursor} out of range for {len(self.diffs)} diffs")
        if not all(0 <= index < len(self.diffs) for index in self.accepted):
            raise ValueError("accepted indices must point into diffs")
        return self

    def move(self, delta: int) -> ReviewState:
        cursor = min(max(self.cursor + delta, 0), len(self.diffs) - 1)
        return self.model_copy(update={"cursor": cursor})

    def toggle(self) -> ReviewState:
        return self.model_copy(update={"accepted": self.accepted ^ {self.cursor}})

    def toggle_all(self) -> ReviewState:
        everything = frozenset(range(len(self.diffs)))
        return self.model_copy(update={"accepted": frozenset() if self.accepted == everything else everything})

    def accepted_candidates(self) -> tuple[Candidate, ...]:
        """The accepted diffs' candidates in list (confidence) order — the order they are applied."""
        return tuple(diff.candidate for index, diff in enumerate(self.diffs) if index in self.accepted)


def render(state: ReviewState, document: Document, size: tuple[int, int]) -> str:
    """One complete frame: header, list viewport, live diff preview, and key help."""
    columns, lines = size
    reduced = apply_candidates(document, state.accepted_candidates())
    header = (
        f"Alexandria — {len(state.accepted)}/{len(state.diffs)} accepted"
        f" · {document.token_count} → {reduced.token_count} tokens"
    )
    footer = "↑/↓ move · enter toggle · a all · d done · q quit"
    body_rows = max(lines - 3, 6)  # minus header, separator, footer
    list_rows = min(2 * len(state.diffs) + 2, max(body_rows // 2, 4))
    return "\n".join(
        [
            click.style(header, bold=True),
            *_list_lines(state, list_rows, columns),
            click.style("─ diff (original → selection) " + "─" * 20, dim=True),
            *_preview_lines(document.text, reduced.text, body_rows - list_rows),
            click.style(footer, dim=True),
        ]
    )


def _list_lines(state: ReviewState, rows: int, columns: int) -> list[str]:
    """The scrolling checkbox viewport: two lines per edit, hidden rows marked with ↑/↓ more."""
    capacity = max((rows - 2) // 2, 1)  # two rows per item, two reserved for the more markers
    total = len(state.diffs)
    start = min(max(state.cursor - capacity // 2, 0), max(total - capacity, 0))
    end = min(start + capacity, total)
    lines: list[str] = []
    if start > 0:
        lines.append(click.style(f"  ↑ {start} more", dim=True))
    for index in range(start, end):
        diff = state.diffs[index]
        marker = "▸" if index == state.cursor else " "
        box = "[x]" if index in state.accepted else "[ ]"
        span = diff.spans[0]
        location = " > ".join(part for part in span.section_path if part) or "(top level)"
        row = f"{marker} {box} {diff.candidate.confidence:.2f}  {location}"[:columns]
        lines.append(click.style(row, bold=True) if index == state.cursor else row)
        lines.append(click.style(f"      - {span.original.strip()}"[:columns], fg="red"))
    if end < total:
        lines.append(click.style(f"  ↓ {total - end} more", dim=True))
    return lines


def _preview_lines(original: str, reduced: str, rows: int) -> list[str]:
    """A unified diff of original vs. the current selection, truncated to the preview rows."""
    body = [
        line
        for line in difflib.unified_diff(original.splitlines(), reduced.splitlines(), lineterm="", n=1)
        if not line.startswith(("---", "+++", "@@"))
    ]
    if not body:
        return [click.style("(no edits accepted — output equals the original)", dim=True)]
    if len(body) > rows:
        hidden = len(body) - max(rows - 1, 1)
        body = [*body[: max(rows - 1, 1)], click.style(f"… {hidden} more lines", dim=True)]
    palette = {"-": "red", "+": "green"}
    return [click.style(line, fg=palette[line[0]]) if line[0] in palette else line for line in body]


def _write_stderr(text: str) -> None:
    click.echo(text, err=True, nl=False)


def review(
    document: Document,
    diffs: tuple[Diff, ...],
    read_key: Callable[[], str] | None = None,
    write: Callable[[str], None] = _write_stderr,
) -> tuple[Candidate, ...] | None:
    """Run the review loop: render a frame, read one key, transition.

    Returns the accepted candidates on 'd' (done) and None on 'q' (quit). Both hooks are
    injected so tests can script the whole loop without a terminal; read_key defaults to
    click.getchar, resolved per call so tests can also monkeypatch it.
    """
    read_key = read_key if read_key is not None else click.getchar
    state = ReviewState(diffs=diffs, cursor=0, accepted=frozenset())
    write(_HIDE_CURSOR)
    try:
        while True:
            columns, lines = shutil.get_terminal_size()
            write(_HOME_AND_CLEAR + render(state, document, (columns, lines)))
            key = read_key()
            if key in _UP_KEYS:
                state = state.move(-1)
            elif key in _DOWN_KEYS:
                state = state.move(1)
            elif key in _TOGGLE_KEYS:
                state = state.toggle()
            elif key == "a":
                state = state.toggle_all()
            elif key == "d":
                return state.accepted_candidates()
            elif key == "q":
                return None
    finally:
        write(_SHOW_CURSOR)


def apply_candidates(document: Document, candidates: tuple[Candidate, ...]) -> Document:
    """Fold Document.apply over the candidates; accept means accept — no drift-budget re-filtering.

    A candidate whose edit would empty the document or a section (apply returns None) is skipped.
    """
    current = document
    for candidate in candidates:
        trial = current.apply(candidate)
        if trial is not None:
            current = trial
    return current
