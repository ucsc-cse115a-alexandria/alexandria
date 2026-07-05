from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alexandria.core.protocols import Selection
from alexandria.core.registry import (
    get_optimizer,
    get_scorer,
    get_selector,
    register_optimizer,
    register_scorer,
    register_selector,
    required_scorers,
)

if TYPE_CHECKING:
    from alexandria.core.ir import Document
    from alexandria.core.protocols import Embedder, Params, Plan, Scores


def test_lookup_round_trip() -> None:
    @register_scorer("reg_test_scorer")
    def scorer(document: Document) -> list[float]:
        return [0.0 for _ in document.sentences]

    assert get_scorer("reg_test_scorer") is scorer


def test_duplicate_name_raises() -> None:
    @register_scorer("reg_test_dup")
    def first(document: Document) -> list[float]:
        return [0.0 for _ in document.sentences]

    assert get_scorer("reg_test_dup") is first

    def second(document: Document) -> list[float]:
        return [0.0 for _ in document.sentences]

    with pytest.raises(ValueError, match="duplicate scorer name"):
        register_scorer("reg_test_dup")(second)


def test_optimizer_with_unregistered_requires_raises() -> None:
    @register_optimizer("reg_test_opt", requires=("does_not_exist",))
    def opt(document: Document, scores: Scores, params: Params) -> Plan:
        del document, scores, params
        return ()

    assert callable(opt)

    with pytest.raises(ValueError, match="unregistered scorers"):
        required_scorers("reg_test_opt")


def test_selector_round_trip() -> None:
    @register_selector("reg_test_selector")
    def selector(document: Document, plan: Plan, embedder: Embedder, params: Params) -> Selection:
        del embedder, params
        return Selection(document=document, applied=plan)

    assert get_selector("reg_test_selector") is selector


def test_unknown_lookups_raise() -> None:
    with pytest.raises(ValueError, match="unknown scorer"):
        get_scorer("nope")
    with pytest.raises(ValueError, match="unknown optimizer"):
        get_optimizer("nope")
    with pytest.raises(ValueError, match="unknown selector"):
        get_selector("nope")
