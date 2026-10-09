from __future__ import annotations

import subprocess
import sys
import textwrap


def test_studio_smoke_constructs_every_page():
    result = subprocess.run(
        [sys.executable, "-m", "cascade.gui.smoke_test"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "8 pages constructed" in result.stdout


def test_studio_smoke_fails_on_qt_signal_callback_errors():
    script = textwrap.dedent("""
        import cascade.gui.window as module
        from cascade.gui.smoke_test import main

        def fail(*args):
            raise RuntimeError("injected GUI callback failure")

        class BrokenWindow(module.MainWindow):
            def __init__(self):
                super().__init__()
                self.pages[2].topology.currentIndexChanged.connect(fail)

        module.MainWindow = BrokenWindow
        main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "injected GUI callback failure" in result.stderr
    assert "GUI ready" not in result.stdout
