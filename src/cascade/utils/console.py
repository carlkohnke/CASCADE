"""Conservative console setup for Windows and redirected command output."""

from __future__ import annotations

import sys


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


__all__ = ["configure_console_error_handling"]
