"""Phase 2 — Score: run named scorers into a length-validated Scores bundle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.registry import get_scorer
from alexandria.score import redundancy

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Scores


def score(document: Document, names: tuple[str, ...] = ("redundancy",)) -> Scores:
    """Run each named scorer and validate its vector length against the Document."""
    bundle: dict[str, tuple[float, ...]] = {}
    expected = len(document.sentences)
    for name in names:
        values = get_scorer(name)(document)
        if len(values) != expected:
            raise ValueError(f"scorer {name!r} returned {len(values)} scores for {expected} sentences")
        bundle[name] = tuple(values)
    return bundle


__all__ = ["redundancy", "score"]
