"""Persist Studio exceptions for console-free Windows launches."""

from __future__ import annotations

import traceback
from pathlib import Path
from types import TracebackType

from cascade.exporting.paths import diagnostic_log_path


def studio_error_log_path() -> Path:
    """Return the persistent Studio exception log."""
    return diagnostic_log_path("cascade-studio-errors.log")


def write_studio_exception(
    exception_type: type[BaseException],
    exception: BaseException,
    trace: TracebackType | None,
) -> Path | None:
    """Append one complete exception record without masking the original error."""
    try:
        path = studio_error_log_path()
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n--- CASCADE Studio exception ---\n")
            handle.write("".join(traceback.format_exception(exception_type, exception, trace)))
        return path
    except OSError:
        return None


__all__ = ["studio_error_log_path", "write_studio_exception"]
