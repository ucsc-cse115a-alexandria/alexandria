from __future__ import annotations

import stat
from typing import TYPE_CHECKING

import pytest

from alexandria.utils.config import (
    MISSING_KEY_MESSAGE,
    config_path,
    require_api_key,
    resolve_api_key,
    write_config_key,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


@pytest.mark.usefixtures("isolated_config")
def test_write_then_resolve_round_trips() -> None:
    path = write_config_key("sk-test-123")

    assert path == config_path()
    assert path.read_text() == 'openai-api-key = "sk-test-123"\n'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert resolve_api_key() == "sk-test-123"


@pytest.mark.usefixtures("isolated_config")
def test_resolution_order_is_explicit_then_env_then_file(monkeypatch: pytest.MonkeyPatch) -> None:
    write_config_key("sk-file")
    assert resolve_api_key() == "sk-file"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert resolve_api_key() == "sk-env"
    assert resolve_api_key("sk-explicit") == "sk-explicit"


@pytest.mark.usefixtures("isolated_config")
def test_require_api_key_raises_the_setup_message_when_unset() -> None:
    with pytest.raises(ValueError, match="alexandria config set openai-api-key"):
        require_api_key()
    assert "OPENAI_API_KEY" in MISSING_KEY_MESSAGE


@pytest.mark.usefixtures("isolated_config")
def test_require_api_key_returns_the_resolved_key() -> None:
    write_config_key("sk-file")
    assert require_api_key() == "sk-file"


@pytest.mark.usefixtures("isolated_config")
def test_write_rejects_keys_that_would_break_the_toml() -> None:
    with pytest.raises(ValueError, match="invalid OpenAI API key"):
        write_config_key('sk-"quoted"')
