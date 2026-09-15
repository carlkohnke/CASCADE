"""Single-instance ownership for CASCADE Studio."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLockFile

from cascade.runtime.paths import state_directory


def studio_lock_path() -> Path:
    """Return the platform-native Studio instance-lock path."""
    return state_directory("studio.lock")


def acquire_studio_lock() -> QLockFile | None:
    """Acquire Studio's cross-platform process lock without blocking."""
    path = studio_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(path))
    if not lock.tryLock(0):
        return None
    return lock


__all__ = ["acquire_studio_lock", "studio_lock_path"]
