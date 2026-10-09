"""Verify Studio construction without a display or persistent user state.

Run with ``python -m cascade.gui.smoke_test`` after installing the GUI extra.
Installation and release checks use this to catch errors inside page builders,
which importing ``cascade.gui`` alone cannot detect.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    # These settings apply only to this disposable verification process.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["CASCADE_RENDER_BACKEND"] = "software"
    with TemporaryDirectory(prefix="cascade-gui-check-") as directory:
        for name in ("PROJECT", "CONFIG", "STATE", "CACHE", "LOG", "OUTPUT"):
            os.environ[f"CASCADE_{name}_DIR"] = str(Path(directory) / name.lower())

        from PySide6.QtWidgets import QApplication
        from cascade.gui.ui_helpers import APP_STYLE
        from cascade.gui.window import MainWindow

        app = QApplication([])
        app.setApplicationName("CASCADE Studio startup check")
        app.setStyle("Fusion")
        app.setStyleSheet(APP_STYLE)
        window = MainWindow()
        try:
            # Exercise both automatic and explicit root assignment, including
            # the zip(strict=...) calls involved in the startup regression.
            vessels = window.pages[2]
            vessels.topology.setCurrentIndex(vessels.topology.findData("forest"))
            vessels.inlet_count.setValue(3)
            vessels._set_root_values(vessels._roots_from_fields())
            assert len(vessels.root_rows) == 3
            for index in range(len(window.pages)):
                window.nav.setCurrentRow(index)
                app.processEvents()
            print(f"CASCADE Studio GUI ready: {len(window.pages)} pages constructed")
        finally:
            window.runner.shutdown()
            window.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
