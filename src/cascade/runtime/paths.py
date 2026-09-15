"""Platform-native locations for writable CASCADE runtime state.

Installed package directories are immutable inputs.  All application-managed
configuration, cache, state, and diagnostic files resolve through this module
so CLI and Studio use the same Windows and Linux conventions.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

_DIRECTORIES = PlatformDirs("CASCADE", appauthor=False, roaming=False)


def _location(variable: str, default: str | Path) -> Path:
    configured = os.environ.get(variable)
    value = configured if configured else default
    return Path(value).expanduser().resolve()


def config_directory() -> Path:
    """Return the user-specific directory for persistent configuration."""
    return _location("CASCADE_CONFIG_DIR", _DIRECTORIES.user_config_path)


def cache_directory(*parts: str) -> Path:
    """Return the user-specific cache directory, optionally with children."""
    root = _location("CASCADE_CACHE_DIR", _DIRECTORIES.user_cache_path)
    return root.joinpath(*parts)


def state_directory(*parts: str) -> Path:
    """Return the user-specific persistent state directory."""
    root = _location("CASCADE_STATE_DIR", _DIRECTORIES.user_state_path)
    return root.joinpath(*parts)


def log_directory() -> Path:
    """Return the user-specific diagnostic log directory."""
    return _location("CASCADE_LOG_DIR", _DIRECTORIES.user_log_path)


def project_directory() -> Path:
    """Return Studio's default project directory."""
    configured = os.environ.get("CASCADE_PROJECT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "CASCADE Projects" / "Untitled").resolve()


__all__ = [
    "cache_directory",
    "config_directory",
    "log_directory",
    "project_directory",
    "state_directory",
]
