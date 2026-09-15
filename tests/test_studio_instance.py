from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cascade.gui.instance import acquire_studio_lock, studio_lock_path


def test_studio_instance_lock_rejects_second_owner(
    monkeypatch, tmp_path: Path
) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("CASCADE_STATE_DIR", str(state))

    first = acquire_studio_lock()
    assert first is not None
    try:
        assert studio_lock_path() == state.resolve() / "studio.lock"
        assert acquire_studio_lock() is None
    finally:
        first.unlock()

    replacement = acquire_studio_lock()
    assert replacement is not None
    replacement.unlock()


def test_studio_instance_lock_is_process_wide(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "process lock"
    monkeypatch.setenv("CASCADE_STATE_DIR", str(state))
    environment = os.environ.copy()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from cascade.gui.instance import acquire_studio_lock; "
                "lock=acquire_studio_lock(); assert lock is not None; "
                "print('LOCKED', flush=True); input()"
            ),
        ],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        assert acquire_studio_lock() is None
    finally:
        stdout, stderr = child.communicate("\n", timeout=5)
        assert child.returncode == 0, stdout + stderr

    replacement = acquire_studio_lock()
    assert replacement is not None
    replacement.unlock()
