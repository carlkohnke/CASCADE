"""Filesystem locations for generated CASCADE output.

Package directories are read-only in many installations, so runtime output
must never be located relative to ``__file__``. Relative paths are resolved
from the current working directory, or from ``CASCADE_OUTPUT_DIR`` when set.
Diagnostic logs may be redirected separately with ``CASCADE_LOG_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

from cascade.runtime.paths import log_directory


def output_directory() -> Path:
    """Return the root directory used for relative generated-output paths."""
    configured = os.environ.get("CASCADE_OUTPUT_DIR")
    return (
        Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
    )


def resolve_output_path(value: str | Path) -> Path:
    """Resolve an output filename without referring to the installed package."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = output_directory() / path
    return path.resolve()


def diagnostic_log_path(filename: str) -> Path:
    """Return a writable location for a diagnostic log and create its directory."""
    directory = log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


__all__ = ["diagnostic_log_path", "output_directory", "resolve_output_path"]
