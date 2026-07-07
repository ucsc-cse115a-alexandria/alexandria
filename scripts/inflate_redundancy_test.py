from typing import TYPE_CHECKING

import numpy as np
import pytest
import tiktoken
from inflate_redundancy import inflate, section_feedback

if TYPE_CHECKING:
    from collections.abc import Callable

_ENCODING = tiktoken.get_encoding("cl100k_base")
_ORIGINAL = "Reply in JSON. Use only lowercase keys."


class ConstantEmbedder:
    model_id = "constant"

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [np.ones(8, dtype=np.float32) for _ in texts]


_RESTATEMENT = " Remember: reply in JSON and keep every key lowercase."


def make_growing_generate() -> tuple[list[str], Callable[[str], str]]:
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return _ORIGINAL + _RESTATEMENT * len(calls)

    return calls, generate


class OrthogonalEmbedder:
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
    assert len(calls) > 1


def test_section_feedback_flags_structure_change() -> None:
    notes = section_feedback("# alpha\nstay calm.\n", "# renamed\nstay calm.\n", ConstantEmbedder())
    assert len(notes) == 1
    assert "structure changed" in notes[0]


class KeywordEmbedder:
    """Texts containing "quack" embed orthogonally to all others."""

    model_id = "keyword"

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for text in texts:
            vector = np.zeros(2, dtype=np.float32)
            vector[1 if "quack" in text else 0] = 1.0
            vectors.append(vector)
        return vectors


def test_retry_feedback_names_drifted_section() -> None:
    original = "# alpha\nstay calm and answer briefly.\n# beta\nalways cite your sources.\n"
    drifted = "# alpha\nstay calm and answer briefly. stay calm.\n# beta\nalways cite your sources. quack quack.\n"
    good = "# alpha\nstay calm and answer briefly. stay calm.\n# beta\nalways cite your sources. cite them.\n"
    prompts = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return drifted if len(prompts) == 1 else good

    assert inflate(original, 1.2, generate, KeywordEmbedder(), _ENCODING) == good
    assert 'section "beta" drifted' in prompts[1]
    assert 'section "alpha"' not in prompts[1]


def test_inflate_raises_after_max_attempts_below_gate() -> None:
    attempts = []

    def long_generate(prompt: str) -> str:
        attempts.append(prompt)
        return _ORIGINAL + _RESTATEMENT * 10  # skips the expand loop

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        inflate(_ORIGINAL, 1.2, long_generate, OrthogonalEmbedder(), _ENCODING, max_attempts=2)
    assert len(attempts) == 2
