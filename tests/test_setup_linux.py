from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_setup_linux() -> ModuleType:
    path = ROOT / "setup_linux.py"
    module = ModuleType("cascade_test_setup_linux")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), path, "exec"), module.__dict__)
    return module


def test_windows_cuda_root_accepts_nested_nvrtc_dll(
    monkeypatch, tmp_path: Path
) -> None:
    setup_linux = _load_setup_linux()
    toolkit = tmp_path / "CUDA Toolkit"
    include = toolkit / "include"
    binaries = toolkit / "bin" / "x86_64"
    include.mkdir(parents=True)
    binaries.mkdir(parents=True)
    (include / "cuda_fp16.h").write_bytes(b"")
    (binaries / "nvrtc64_134_0.dll").write_bytes(b"")
    monkeypatch.setattr(setup_linux.sys, "platform", "win32")

    assert setup_linux._is_cuda_root(toolkit)


def test_cuda_toolkit_versions_are_sorted_numerically() -> None:
    setup_linux = _load_setup_linux()
    versions = [Path("v9.2"), Path("v13.1"), Path("v12.8")]

    assert sorted(versions, key=setup_linux._cuda_version_key, reverse=True) == [
        Path("v13.1"),
        Path("v12.8"),
        Path("v9.2"),
    ]


def test_recreate_refuses_repository_and_non_venv_directories(tmp_path: Path) -> None:
    setup_linux = _load_setup_linux()

    with pytest.raises(ValueError, match="protected path"):
        setup_linux._validate_recreate_target(ROOT)

    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    with pytest.raises(ValueError, match="missing pyvenv.cfg"):
        setup_linux._validate_recreate_target(ordinary)


def test_recreate_accepts_only_marked_virtual_environment(tmp_path: Path) -> None:
    setup_linux = _load_setup_linux()
    environment = tmp_path / "safe-venv"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    assert setup_linux._validate_recreate_target(environment) == environment.resolve()


def test_recreate_refuses_symlinked_environment(tmp_path: Path) -> None:
    setup_linux = _load_setup_linux()
    environment = tmp_path / "real-venv"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    link = tmp_path / "linked-venv"
    try:
        link.symlink_to(environment, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) in {1, 50, 1314}:
            pytest.skip(f"Windows filesystem/account cannot create symlinks: {error}")
        raise

    with pytest.raises(ValueError, match="symlinked"):
        setup_linux._validate_recreate_target(link)


def test_launcher_path_is_published_after_verification_dry_run(
    tmp_path: Path, capsys
) -> None:
    setup_linux = _load_setup_linux()

    assert setup_linux.main(["--venv", str(tmp_path / "new-venv"), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert output.index("cascade.commands.main --help") < output.index(
        ".cascade_python"
    )


def test_platform_setup_entrypoints_have_unambiguous_locations() -> None:
    assert (ROOT / "Install CASCADE for Windows.cmd").is_file()
    assert (ROOT / "setup_linux.py").is_file()
    assert not (ROOT / "setup_env.py").exists()
    assert not (ROOT / "GUI Launchers").exists()


def test_wsl_launchers_delegate_to_the_relocated_linux_launcher() -> None:
    launcher = (ROOT / "scripts" / "wsl" / "launch-studio.sh").read_text()
    silent = (ROOT / "scripts" / "wsl" / "launch-from-windows.vbs").read_text()
    diagnostic = (
        ROOT / "scripts" / "wsl" / "launch-diagnostic.cmd"
    ).read_text()

    assert 'LAUNCHER_DIR/../..' in launcher
    assert "setup_linux.py" in launcher
    assert "./launch-studio.sh" in silent
    assert "./launch-studio.sh" in diagnostic


def test_source_distribution_manifest_includes_platform_entrypoints() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text()

    assert "include pyproject.toml setup_linux.py" in manifest
    assert "recursive-include scripts/windows *.ps1" in manifest
    assert "recursive-include scripts/wsl *.cmd *.sh *.vbs" in manifest


@pytest.mark.parametrize("version", ["3.9.20", "3.13.1", "3.12.3 (32-bit)"])
def test_setup_rejects_unsupported_python_before_mutating_environment(
    monkeypatch, tmp_path, version
):
    setup = _load_setup_linux()
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, version, ""),
    )
    monkeypatch.setattr(setup, "_run", lambda *args, **kwargs: pytest.fail("mutation"))
    with pytest.raises(ValueError, match="requires 64-bit CPython 3.12"):
        setup.main(["--venv", str(tmp_path / "environment")])
    assert not (tmp_path / "environment").exists()


def test_setup_validates_existing_venv_not_just_creator(monkeypatch, tmp_path):
    setup = _load_setup_linux()
    environment = tmp_path / "environment"
    python = setup._venv_python(environment)
    python.parent.mkdir(parents=True)
    python.touch()
    checked = []

    def validate(executable, dry_run):
        checked.append(executable)
        if executable == str(python):
            raise ValueError("existing environment has Python 3.9")

    monkeypatch.setattr(setup, "_validate_python312", validate)
    monkeypatch.setattr(setup, "_run", lambda *args, **kwargs: pytest.fail("mutation"))
    with pytest.raises(ValueError, match="existing environment"):
        setup.main(["--venv", str(environment)])
    assert checked == [sys.executable, str(python)]


def test_gui_verification_failure_does_not_publish_launcher(monkeypatch, tmp_path):
    setup = _load_setup_linux()
    monkeypatch.setattr(setup, "_validate_python312", lambda *args: None)

    def run(command, *args, **kwargs):
        if any("cascade.gui.smoke_test" in arg for arg in command):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup, "_run", run)
    monkeypatch.setattr(
        setup, "_write_python_path", lambda *args: pytest.fail("published broken GUI")
    )
    with pytest.raises(subprocess.CalledProcessError):
        setup.main(["--venv", str(tmp_path / "environment"), "--gui"])


def test_setup_records_checkout_and_shared_interpreters(monkeypatch, tmp_path):
    setup = _load_setup_linux()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    config = tmp_path / "config"
    monkeypatch.setattr(setup, "ROOT", checkout)
    monkeypatch.setattr(setup, "_config_dir", lambda: config)
    python = checkout / ".venv" / "bin" / "python"
    setup._write_python_path(python, False)
    assert (checkout / ".cascade_python").read_text().strip() == str(python)
    assert (config / ".cascade_python").read_text().strip() == str(python)


@pytest.mark.parametrize(
    "requested, driver, expected, message",
    [
        ("auto", 13020, "cu13", "Selected --gpu cu13."),
        ("auto", 12040, "cu12", "Selected --gpu cu12."),
        ("auto", 11080, "cu11", "Selected --gpu cu11."),
        ("auto", 10020, "none", "Installing CPU-only dependencies"),
        ("auto", None, "none", "No NVIDIA driver was detected."),
        ("cu13", 12040, "cu13", "WARNING: --gpu cu13 needs a newer NVIDIA driver"),
        ("cu12", 13020, "cu12", "matching GPU family: cu13."),
        ("cu12", None, "cu12", "No NVIDIA driver was detected."),
        ("none", 12040, "none", ""),
    ],
)
def test_gpu_family_selection_follows_driver(
    monkeypatch, capsys, requested, driver, expected, message
) -> None:
    setup_linux = _load_setup_linux()
    probe = setup_linux._cuda_driver_module()
    monkeypatch.setattr(probe, "cuda_driver_version", lambda: driver)
    monkeypatch.setattr(probe, "cuda_devices", lambda: ())
    monkeypatch.setattr(setup_linux, "_cuda_driver_module", lambda: probe)

    assert setup_linux._select_gpu_family(requested) == expected
    output = capsys.readouterr().out
    assert message in output
    if requested == "cu13" and driver is not None and driver < 13000:
        assert "580 or later" in output
    if requested in {"auto", "cu12"} or requested == "none":
        assert "WARNING" not in output


def test_dry_run_installs_the_auto_selected_requirements(monkeypatch, capsys) -> None:
    setup_linux = _load_setup_linux()
    probe = setup_linux._cuda_driver_module()
    monkeypatch.setattr(probe, "cuda_driver_version", lambda: 12040)
    monkeypatch.setattr(probe, "cuda_devices", lambda: ())
    monkeypatch.setattr(setup_linux, "_cuda_driver_module", lambda: probe)

    assert setup_linux.main(["--gpu", "auto", "--dry-run", "--skip-verify"]) == 0
    output = capsys.readouterr().out
    assert "requirements/gpu-cu12.txt" in output.replace("\\", "/")
    assert "gpu-cu13.txt" not in output


@pytest.mark.parametrize(
    "requested, compute, expected, message",
    [
        ("auto", (7, 0), "cu12", "Selected --gpu cu12."),
        ("auto", (3, 7), "cu11", "Selected --gpu cu11."),
        ("cu13", (7, 0), "cu13", "WARNING: --gpu cu13 cannot compile for this GPU (sm_70)"),
        ("cu12", (7, 0), "cu12", "(sm_86, sm_70); matching GPU family: cu12."),
    ],
)
def test_gpu_family_selection_follows_oldest_gpu(
    monkeypatch, capsys, requested, compute, expected, message
) -> None:
    setup_linux = _load_setup_linux()
    probe = setup_linux._cuda_driver_module()
    monkeypatch.setattr(probe, "cuda_driver_version", lambda: 13020)
    monkeypatch.setattr(
        probe,
        "cuda_devices",
        lambda: (probe.CudaDevice(0, (8, 6), 0), probe.CudaDevice(1, compute, 3)),
    )
    monkeypatch.setattr(setup_linux, "_cuda_driver_module", lambda: probe)

    assert setup_linux._select_gpu_family(requested) == expected
    assert message in capsys.readouterr().out
