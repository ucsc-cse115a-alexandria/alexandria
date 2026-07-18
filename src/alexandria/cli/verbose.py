from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from alexandria.ir.contracts import ReportedCandidate


def _clip(text: str, limit: int = 200) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


class VerboseReporter:
    """Renders reduction progress as plain text to an injected writer."""

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write

    def redundant_pair(self, first: str, second: str, similarity: float) -> None:
        self._write(f"● redundant pair (similarity {similarity:.2f})")
        self._write(f"    A: {_clip(first)}")
        self._write(f"    B: {_clip(second)}")

    def pair_merged(self, merged: str | None, decision: str) -> None:
        if decision == "replace":
            self._write(f"    → merged: {_clip(merged or '')}")
        elif decision == "delete":
            self._write("    → kept the first; the second is redundant")
        else:
            self._write("    → skipped: no rewrite fit the drift budget")

    def target_group(self, source_segment: str, group_tokens: int, required_savings: int) -> None:
        self._write(f"▶ compressing a content group of {group_tokens} tokens (need to save {required_savings})")
        self._write(f"    source: {_clip(source_segment)}")

    def target_round(
        self,
        round_number: int,
        base: str | None,
        candidates: tuple[ReportedCandidate, ...],
        selected: ReportedCandidate,
        selected_from_generation: bool,
    ) -> None:
        self._write(f"  round {round_number}:")
        if base is not None:
            self._write(f"    search base: {_clip(base)}")
        for index, candidate in enumerate(candidates, start=1):
            structure_note = "" if candidate.structure_valid else " [invalid structure]"
            self._write(
                f"    candidate {index}: {candidate.token_count} tok, drift {candidate.drift:.4f}"
                f"{structure_note}: {_clip(candidate.text)}"
            )
        origin = "generated" if selected_from_generation else "kept base"
        self._write(f"    → selected ({origin}): {selected.token_count} tok, drift {selected.drift:.4f}")

    def target_group_done(self, applied: bool, document_tokens: int) -> None:
        if applied:
            self._write(f"  ✓ applied — document now {document_tokens} tokens")
        else:
            self._write("  ✗ group not applied")
