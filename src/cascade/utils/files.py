"""Durable file publication helpers shared by CLI and Studio."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import TextIO

_PUBLICATION_LOCKS = tuple(RLock() for _index in range(64))


def _publication_lock(path: Path) -> RLock:
    normalized = os.path.normcase(str(path))
    return _PUBLICATION_LOCKS[hash(normalized) % len(_PUBLICATION_LOCKS)]


def _replace_file(temporary: Path, target: Path) -> None:
    delays = (
        (0.0,)
        if os.name != "nt"
        else (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
    )
    last_error: PermissionError | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            os.replace(temporary, target)
            return
        except PermissionError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


@contextmanager
def atomic_text_writer(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Iterator[TextIO]:
    """Yield a same-directory temporary file and atomically publish it on exit."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        # Windows rejects overlapping replacements of one destination with
        # WinError 5. Studio writes from one process, so a bounded lock stripe
        # serializes publication without fixed ``.tmp`` names or lock files.
        # Windows filesystem filters can retain the replaced destination for a
        # few milliseconds, so bounded retries handle that documented sharing
        # violation while preserving the original exception if it persists.
        with _publication_lock(target):
            _replace_file(temporary, target)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    """Write text through a unique, flushed, same-directory temporary file."""
    with atomic_text_writer(path, encoding=encoding, newline=newline) as handle:
        handle.write(text)


__all__ = ["atomic_text_writer", "atomic_write_text"]
