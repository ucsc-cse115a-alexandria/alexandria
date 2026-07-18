"""OpenAI API key resolution and the on-disk config file (the imperative shell for credentials)."""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path

_KEY_FIELD = "openai-api-key"
ENV_VAR = "OPENAI_API_KEY"
MISSING_KEY_MESSAGE = (
    f"OpenAI API key not found. Set it with `alexandria config set {_KEY_FIELD}` or export {ENV_VAR}."
)


def config_path() -> Path:
    """~/.config/alexandria/config.toml, honoring XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "alexandria" / "config.toml"


def read_config_key() -> str | None:
    """The stored key, or None when the config file or field is absent."""
    path = config_path()
    if not path.is_file():
        return None
    value = tomllib.loads(path.read_text()).get(_KEY_FIELD)
    return value if isinstance(value, str) and value else None


def write_config_key(key: str) -> Path:
    """Save the key with owner-only permissions; returns the path written."""
    if not key or any(character in key for character in '"\\\n'):
        raise ValueError("invalid OpenAI API key")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as file:
        file.write(f'{_KEY_FIELD} = "{key}"\n')
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # tighten a pre-existing looser file that O_CREAT left untouched
    return path


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Explicit argument > OPENAI_API_KEY env var > config file; None when nowhere set."""
    return explicit or os.environ.get(ENV_VAR) or read_config_key()


def require_api_key(explicit: str | None = None) -> str:
    """resolve_api_key, but a missing key raises with the CLI setup instructions."""
    key = resolve_api_key(explicit)
    if key is None:
        raise ValueError(MISSING_KEY_MESSAGE)
    return key
