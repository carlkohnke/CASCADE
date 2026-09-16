from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_setup_env() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "setup_env.py"
    spec = importlib.util.spec_from_file_location("cascade_test_setup_env", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_cuda_root_accepts_nested_nvrtc_dll(
    monkeypatch, tmp_path: Path
) -> None:
    setup_env = _load_setup_env()
    toolkit = tmp_path / "CUDA Toolkit"
    include = toolkit / "include"
    binaries = toolkit / "bin" / "x86_64"
    include.mkdir(parents=True)
    binaries.mkdir(parents=True)
    (include / "cuda_fp16.h").write_bytes(b"")
    (binaries / "nvrtc64_134_0.dll").write_bytes(b"")
    monkeypatch.setattr(setup_env.sys, "platform", "win32")

    assert setup_env._is_cuda_root(toolkit)


def test_cuda_toolkit_versions_are_sorted_numerically() -> None:
    setup_env = _load_setup_env()
    versions = [Path("v9.2"), Path("v13.1"), Path("v12.8")]

    assert sorted(versions, key=setup_env._cuda_version_key, reverse=True) == [
        Path("v13.1"),
        Path("v12.8"),
        Path("v9.2"),
    ]
