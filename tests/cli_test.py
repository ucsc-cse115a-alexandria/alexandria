from __future__ import annotations

from typing import TYPE_CHECKING

from alexandria.__main__ import main

if TYPE_CHECKING:
    import pytest


def test_main_prints_each_segmented_instruction(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "You are a helpful assistant.",
        "Always answer in English.",
        "Keep responses concise.",
        "Always answer in English.",
    ]
