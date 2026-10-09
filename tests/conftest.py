from __future__ import annotations

import os
from pathlib import Path


if os.environ.get("CASCADE_TEST_INSTALLED") != "1":
    _SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
    _existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(_SOURCE_ROOT), *([_existing_pythonpath] if _existing_pythonpath else [])]
    )
