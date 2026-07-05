from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from alexandria.ir.contracts import Optimizer, Peers, Scorer, Selector


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(fn: T) -> T:
            if name in self._entries:
                raise ValueError(f"duplicate {self._kind} name: {name!r}")
            self._entries[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"unknown {self._kind}: {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._entries


_scorers: Registry[Scorer] = Registry("scorer")
_optimizers: Registry[Optimizer] = Registry("optimizer")
_selectors: Registry[Selector] = Registry("selector")
_peers: dict[str, Peers | None] = {}
_requires: dict[str, tuple[str, ...]] = {}


def register_scorer(name: str, *, peers: Peers | None = None) -> Callable[[Scorer], Scorer]:
    base = _scorers.register(name)

    def decorator(fn: Scorer) -> Scorer:
        registered = base(fn)
        _peers[name] = peers
        return registered

    return decorator


def register_optimizer(name: str, *, requires: tuple[str, ...] = ()) -> Callable[[Optimizer], Optimizer]:
    base = _optimizers.register(name)

    def decorator(fn: Optimizer) -> Optimizer:
        registered = base(fn)
        _requires[name] = requires
        return registered

    return decorator


register_selector = _selectors.register
get_scorer = _scorers.get
get_selector = _selectors.get


def scorer_peers(name: str) -> Peers | None:
    _scorers.get(name)  # validates the name exists
    return _peers[name]


def get_optimizer(name: str) -> Optimizer:
    optimizer = _optimizers.get(name)
    missing = [r for r in _requires[name] if r not in _scorers]
    if missing:
        raise ValueError(f"optimizer {name!r} requires unregistered scorers: {missing}")
    return optimizer


def required_scorers(name: str) -> tuple[str, ...]:
    get_optimizer(name)
    return _requires[name]
