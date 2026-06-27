"""Phase 4 — Select: choose and apply candidates under a drift budget, highest-confidence-first."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.core.protocols import Params
from alexandria.core.registry import get_selector
from alexandria.phases.select import auto
from alexandria.phases.select.auto import DEFAULT_SELECTOR

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder, Plan


def select(
    document: Document,
    plan: Plan,
    embedder: Embedder,
    name: str = DEFAULT_SELECTOR,
    *,
    params: Params | None = None,
) -> Document:
    """Run the named selector to fold the Plan into a reduced Document."""
    params = params or Params()
    return get_selector(name)(document, plan, embedder, params)


__all__ = ["DEFAULT_SELECTOR", "auto", "select"]
