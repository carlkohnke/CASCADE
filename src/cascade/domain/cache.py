"""Content-addressed cache paths for constructed tissue domains."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

from cascade.utils.hashing import file_sha256


def default_domain_cache_dir() -> Path:
    """Return CASCADE's platform-neutral user cache directory for domains."""
    override = os.environ.get("CASCADE_DOMAIN_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (root / "cascade" / "domains").resolve()


def domain_cache_path(source: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Key a constructed domain by source bytes and the installed SVV version."""
    source_path = Path(source).expanduser().resolve()
    root = (
        default_domain_cache_dir()
        if cache_dir is None
        else Path(cache_dir).expanduser().resolve()
    )
    try:
        svv_version = importlib.metadata.version("svv")
    except importlib.metadata.PackageNotFoundError:
        svv_version = "unknown"
    safe_version = "".join(
        char if char.isalnum() or char in ".-_" else "_" for char in svv_version
    )
    return root / f"surface-v1-{file_sha256(source_path)}-svv-{safe_version}.dmn"


__all__ = ["default_domain_cache_dir", "domain_cache_path"]
