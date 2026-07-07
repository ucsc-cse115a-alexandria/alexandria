from typing import TYPE_CHECKING

import numpy as np
import pytest
import tiktoken
from inflate_redundancy import inflate

if TYPE_CHECKING:
    from collections.abc import Callable

_ENCODING = tiktoken.get_encoding("cl100k_base")
_ORIGINAL = "Reply in JSON. Use only lowercase keys."


class ConstantEmbedder:
    """Every text maps to the same vector, so compare() reports similarity 1.0."""

    model_id = "constant"

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [np.ones(8, dtype=np.float32) for _ in texts]


_RESTATEMENT = " Remember: reply in JSON and keep every key lowercase."


def make_growing_generate() -> tuple[list[str], Callable[[str], str]]:
    """Fake LLM: the original plus one more canned restatement per call, recording each prompt."""
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return _ORIGINAL + _RESTATEMENT * len(calls)

    return calls, generate


class OrthogonalEmbedder:
    """Every text maps to a fresh orthogonal vector, so compare() reports similarity 0.0."""

    model_id = "orthogonal"

    def __init__(self) -> None:
        self._calls = 0

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for _ in texts:
            vector = np.zeros(64, dtype=np.float32)
            vector[self._calls % 64] = 1.0
            self._calls += 1
            vectors.append(vector)
        return vectors


def test_inflate_expands_until_token_ratio_is_met() -> None:
    factor = 4.0
    calls, generate = make_growing_generate()
    inflated = inflate(_ORIGINAL, factor, generate, ConstantEmbedder(), _ENCODING)
    assert len(_ENCODING.encode(inflated)) >= factor * len(_ENCODING.encode(_ORIGINAL))
    assert len(calls) > 1  # the append-style continuation loop ran


def test_inflate_raises_after_max_attempts_below_gate() -> None:
    attempts = []

    def long_generate(prompt: str) -> str:
        attempts.append(prompt)
        return _ORIGINAL + _RESTATEMENT * 10  # long enough on the first call of each attempt

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        inflate(_ORIGINAL, 1.2, long_generate, OrthogonalEmbedder(), _ENCODING, max_attempts=2)
    assert len(attempts) == 2
