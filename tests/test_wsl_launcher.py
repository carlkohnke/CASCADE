from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Linux/WSL bash launcher")


def _python(path: Path, label: str, *, supported: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '#!/bin/bash\nif [[ "$1" == "-c" ]]; then\n'
        f'  exit {0 if supported else 1}\nfi\necho "launched {label}"\n'
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout with spaces"
    launcher = repo / "scripts" / "wsl" / "launch-studio.sh"
    launcher.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts" / "wsl" / "launch-studio.sh", launcher)
    config = tmp_path / "shared config"
    config.mkdir()
    state = tmp_path / "state"
    env = {key: value for key, value in os.environ.items() if not key.startswith("CASCADE_")}
    env.update(CASCADE_CONFIG_DIR=str(config), CASCADE_STATE_DIR=str(state))
    return repo, launcher, config, state, env


def _launch(checkout):
    return subprocess.run(
        ["bash", str(checkout[1])], env=checkout[4],
        capture_output=True, text=True, timeout=10,
    )


def test_local_venv_wins_over_stale_shared_python39(checkout):
    repo, _, config, state, _ = checkout
    _python(repo / ".venv" / "bin" / "python", "local")
    old = _python(config / "old-python", "old", supported=False)
    (config / ".cascade_python").write_text(str(old))
    result = _launch(checkout)
    assert result.returncode == 0, result.stderr
    assert (state / "gui.log").read_text().strip() == "launched local"


def test_explicit_python_override_wins_over_both_records(checkout):
    repo, _, config, state, env = checkout
    old = _python(config / "old-python", "old", supported=False)
    for directory in (repo, config):
        (directory / ".cascade_python").write_text(str(old))
    override = _python(repo / "custom environment" / "python", "override")
    env["CASCADE_PYTHON"] = str(override)
    result = _launch(checkout)
    assert result.returncode == 0, result.stderr
    assert (state / "gui.log").read_text().strip() == "launched override"


def test_checkout_record_preserves_custom_environment(checkout):
    repo, _, _, state, _ = checkout
    _python(repo / ".venv" / "bin" / "python", "default")
    custom = _python(repo / "custom environment" / "python", "custom")
    (repo / ".cascade_python").write_text(str(custom))
    result = _launch(checkout)
    assert result.returncode == 0, result.stderr
    assert (state / "gui.log").read_text().strip() == "launched custom"


@pytest.mark.parametrize("supported", [True, False])
def test_shared_fallback_checks_python_before_gui(checkout, supported):
    _, _, config, state, _ = checkout
    shared = _python(config / "python", "shared", supported=supported)
    (config / ".cascade_python").write_text(str(shared))
    result = _launch(checkout)
    if supported:
        assert result.returncode == 0, result.stderr
        assert (state / "gui.log").read_text().strip() == "launched shared"
    else:
        assert result.returncode == 1
        assert "requires 64-bit CPython 3.12" in result.stderr
        assert "python3.12 setup_linux.py" in result.stderr
        assert not (state / "gui.log").exists()
