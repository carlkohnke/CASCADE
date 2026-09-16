from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "windows" / "install-cascade.ps1"
DOUBLE_CLICK = ROOT / "Install CASCADE for Windows.cmd"


def test_double_click_launcher_uses_repository_relative_installer() -> None:
    text = DOUBLE_CLICK.read_text(encoding="utf-8")
    assert "%~dp0scripts\\windows\\install-cascade.ps1" in text
    assert "ExecutionPolicy Bypass" in text
    assert "%*" in text
    assert "CASCADE_INSTALLER_NO_PAUSE" in text


def test_installer_has_native_isolated_release_guards() -> None:
    text = INSTALLER.read_text(encoding="utf-8")
    folded = text.casefold()
    assert "--only-binary=:all:" in text
    assert "gpu-cu13" in text
    assert "--require-gpu" in text
    assert "--no-gpu-probe" in text
    assert "cascade studio.lnk" in folded
    assert '"path", "user"' in folded
    assert "print(sys.executable)" not in text
    assert "wsl.exe" not in folded
    assert "/mnt/" not in folded


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell syntax check")
def test_installer_parses_in_windows_powershell() -> None:
    powershell = shutil.which("powershell.exe")
    assert powershell is not None
    command = (
        "$errors=$null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{INSTALLER}',"
        "[ref]$null,[ref]$errors); "
        "if($errors.Count){$errors | ForEach-Object {Write-Error $_}; exit 1}"
    )
    subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )
