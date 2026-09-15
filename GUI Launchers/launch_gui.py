"""Start CASCADE Studio with the Python environment running this file.

This small launcher is useful for desktop shortcuts or file-manager launches.
Install the GUI dependencies first, then run it with the configured CASCADE
environment (normally ``.venv/bin/python`` on Linux/WSL).
"""

from cascade.gui.main import main


if __name__ == "__main__":
    raise SystemExit(main())
