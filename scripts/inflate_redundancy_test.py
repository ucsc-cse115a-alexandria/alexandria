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


_RESTATEMENT_PIECE = "Reply in JSON. Keep every key lowercase."


def make_restating_generate() -> tuple[list[str], Callable[[str], str]]:
    """Fake generate for the append-only loop: every call returns one restatement piece."""
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return _RESTATEMENT_PIECE

    return calls, generate


class OrthogonalEmbedder:
    model_id = "orthogonal"

    def __init__(self) -> None:
        self._calls = 0

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for _ in texts:
            vector = np.zeros(64, dtype=np.float32)
            vector[self._calls % 64] = 1.0
            self._calls += 1
            vectors.append(vector)
        return vectors


def test_inflate_expands_then_truncates_to_target() -> None:
    factor = 4.0
    target = round(factor * len(_ENCODING.encode(_ORIGINAL)))
    calls, generate = make_restating_generate()
    inflated = inflate(_ORIGINAL, factor, generate, ConstantEmbedder(), _ENCODING)
    assert len(_ENCODING.encode(inflated)) == target  # truncated to exactly factor x
    assert len(calls) > 1  # append-only needed several restatement pieces to reach the target
    assert inflated.startswith(_RESTATEMENT_PIECE)  # first piece kept intact


def test_section_feedback_flags_structure_change() -> None:
    notes = section_feedback("# alpha\nstay calm.\n", "# renamed\nstay calm.\n", ConstantEmbedder())
    assert len(notes) == 1
    assert "structure changed" in notes[0]


class KeywordEmbedder:
    """Texts containing "quack" embed orthogonally to all others."""

    model_id = "keyword"

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for text in texts:
            vector = np.zeros(2, dtype=np.float32)
            vector[1 if "quack" in text else 0] = 1.0
            vectors.append(vector)
        return vectors


def test_retry_feedback_names_changed_section() -> None:
    original = "# alpha\nalways cite the listed sources carefully.\n"
    changed = "# alpha\nquack quack quack quack quack quack quack.\n"
    prompts: list[str] = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        # Attempt 2's prompt carries meaning-change feedback; the first attempt and expansions change it.
        return original if "changed the meaning" in prompt else changed

    assert inflate(original, 1.0, generate, KeywordEmbedder(), _ENCODING) == original
    assert 'section "alpha" changed meaning' in prompts[-1]


def test_inflate_raises_after_max_attempts_below_gate() -> None:
    attempts: list[str] = []

    def long_generate(prompt: str) -> str:
        attempts.append(prompt)
        return _RESTATEMENT_PIECE * 10  # long enough to skip the expand loop

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        inflate(_ORIGINAL, 1.2, long_generate, OrthogonalEmbedder(), _ENCODING, max_attempts=2)
    assert len(attempts) == 2
