"""Native Windows integration for the frameless Studio shell."""

from __future__ import annotations

import ctypes
import sys
from typing import Any

from PySide6.QtGui import QGuiApplication


_GWL_STYLE = -16
_WS_MAXIMIZEBOX = 0x00010000
_WS_THICKFRAME = 0x00040000
WINDOWS_SNAP_STYLE = _WS_MAXIMIZEBOX | _WS_THICKFRAME


def snap_eligible_style(style: int) -> int:
    """Return ``style`` with the Win32 bits required for Windows Snap."""

    return int(style) | WINDOWS_SNAP_STYLE


def enable_windows_snap(window: Any) -> bool:
    """Restore Windows Snap eligibility without adding native decorations.

    Qt's ``FramelessWindowHint`` removes the sizing/maximize style bits that
    Windows uses to decide whether a top-level window can participate in Snap.
    Restoring those bits after the native handle exists keeps CASCADE's custom
    chrome while allowing ``startSystemMove()`` to hand edge tiling to Windows.
    """

    if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
        return False

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_window_long = user32.GetWindowLongPtrW
    get_window_long.argtypes = (ctypes.c_void_p, ctypes.c_int)
    get_window_long.restype = ctypes.c_ssize_t
    set_window_long = user32.SetWindowLongPtrW
    set_window_long.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t)
    set_window_long.restype = ctypes.c_ssize_t

    handle = ctypes.c_void_p(int(window.winId()))
    ctypes.set_last_error(0)
    style = int(get_window_long(handle, _GWL_STYLE))
    error = ctypes.get_last_error()
    if style == 0 and error:
        raise ctypes.WinError(error)

    updated = snap_eligible_style(style)
    if updated != style:
        ctypes.set_last_error(0)
        previous = int(set_window_long(handle, _GWL_STYLE, updated))
        error = ctypes.get_last_error()
        if previous == 0 and error:
            raise ctypes.WinError(error)
    return True


__all__ = ("WINDOWS_SNAP_STYLE", "enable_windows_snap", "snap_eligible_style")
