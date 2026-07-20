"""Shared pytest configuration: register the `ai` marker and skip it without an OpenAI key.

Tests marked `ai` run the pipeline against the real OpenAI defaults, so they cost money and need
network access. They are skipped automatically unless a key is resolvable (env var or config file),
which keeps the default suite and CI offline while letting a developer opt in by exporting a key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alexandria.utils.config import ENV_VAR, resolve_api_key

if TYPE_CHECKING:
    from collections.abc import Iterable

AI_MARKER = "ai"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{AI_MARKER}: live end-to-end test that calls the OpenAI API; skipped unless a key is set "
        f"({ENV_VAR} or the alexandria config file).",
    )


def pytest_collection_modifyitems(items: Iterable[pytest.Item]) -> None:
    if resolve_api_key() is not None:
        return
    skip = pytest.mark.skip(reason=f"requires a resolvable OpenAI key ({ENV_VAR} or config file) for a live run")
    for item in items:
        if AI_MARKER in item.keywords:
            item.add_marker(skip)
