"""Name-keyed registries for scorers, optimizers, and selectors; reject dup names, validate requires."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from alexandria.core.protocols import Optimizer, Peers, Scorer, Selector

_scorers: dict[str, Scorer] = {}
_peers: dict[str, Peers | None] = {}
_optimizers: dict[str, Optimizer] = {}
_requires: dict[str, tuple[str, ...]] = {}
_selectors: dict[str, Selector] = {}


def register_scorer(name: str, *, peers: Peers | None = None) -> Callable[[Scorer], Scorer]:
    def decorator(fn: Scorer) -> Scorer:
        if name in _scorers:
            raise ValueError(f"duplicate scorer name: {name!r}")
        _scorers[name] = fn
        _peers[name] = peers
        return fn

    return decorator


def register_optimizer(name: str, *, requires: tuple[str, ...] = ()) -> Callable[[Optimizer], Optimizer]:
    def decorator(fn: Optimizer) -> Optimizer:
        if name in _optimizers:
            raise ValueError(f"duplicate optimizer name: {name!r}")
        _optimizers[name] = fn
        _requires[name] = requires
        return fn

    return decorator


def get_scorer(name: str) -> Scorer:
    try:
        return _scorers[name]
    except KeyError:
        raise ValueError(f"unknown scorer: {name!r}") from None


def scorer_peers(name: str) -> Peers | None:
    """The peer-finder a scorer registered, or None if it produces no peers."""
    get_scorer(name)  # validates the name exists
    return _peers[name]


def get_optimizer(name: str) -> Optimizer:
    try:
        optimizer = _optimizers[name]
    except KeyError:
        raise ValueError(f"unknown optimizer: {name!r}") from None
    missing = [r for r in _requires[name] if r not in _scorers]
    if missing:
        raise ValueError(f"optimizer {name!r} requires unregistered scorers: {missing}")
    return optimizer


def required_scorers(name: str) -> tuple[str, ...]:
    get_optimizer(name)
    return _requires[name]


def register_selector(name: str) -> Callable[[Selector], Selector]:
    def decorator(fn: Selector) -> Selector:
        if name in _selectors:
            raise ValueError(f"duplicate selector name: {name!r}")
        _selectors[name] = fn
        return fn

    return decorator


def get_selector(name: str) -> Selector:
    try:
        return _selectors[name]
    except KeyError:
        raise ValueError(f"unknown selector: {name!r}") from None
