"""Platform cursor integration for the CASCADE desktop window."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QPixmap
from PySide6.QtWidgets import QApplication, QWidget


_DEFAULT_WINDOWS_ARROW = Path("/mnt/c/Windows/Cursors/aero_arrow.cur")


def apply_wsl_windows_pointer(window: QWidget) -> bool:
    """Use the host Windows arrow instead of XWayland's oversized pointer.

    Qt's XCB transport selects an X cursor even though WSLg ultimately presents
    the application in Windows. Loading the cursor from the host installation
    restores the familiar Windows pointer without shipping a cursor asset or
    affecting native Linux, macOS, or Windows sessions.
    """
    if not os.environ.get("WSL_INTEROP"):
        return False
    if QApplication.platformName().lower() != "xcb":
        return False

    cursor_path = Path(
        os.environ.get("CASCADE_WINDOWS_CURSOR", str(_DEFAULT_WINDOWS_ARROW))
    ).expanduser()
    if not cursor_path.is_file():
        return False

    pixmap = QPixmap(str(cursor_path))
    if pixmap.isNull():
        return False

    try:
        size = max(16, min(int(os.environ.get("CASCADE_WINDOWS_CURSOR_SIZE", "32")), 64))
    except ValueError:
        size = 32
    pixmap = pixmap.scaled(
        size,
        size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    # The arrow tip in the Windows asset is at its upper-left corner.
    window.setCursor(QCursor(pixmap, 1, 1))
    return True
