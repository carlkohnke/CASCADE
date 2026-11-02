"""Conservative console setup for Windows and redirected command output."""

from __future__ import annotations

import os
import sys

# CASCADE prints non-ASCII text (for example "…" and "µ"). Child Python
# processes write UTF-8 under any locale, and parents decode it as UTF-8.
UTF8_CHILD_ENVIRONMENT = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def utf8_child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default: this environment) with UTF-8 child I/O."""
    environment = dict(os.environ if base is None else base)
    environment.update(UTF8_CHILD_ENVIRONMENT)
    return environment


def configure_console_error_handling() -> None:
    """Keep unencodable paths from crashing a CLI without changing encoding."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass


__all__ = [
    "UTF8_CHILD_ENVIRONMENT",
    "configure_console_error_handling",
    "utf8_child_environment",
]
