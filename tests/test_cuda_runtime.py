from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.accelerators import backend, cuda_runtime


class _FakeDistribution:
    def __init__(self, root: Path, files: list[Path]) -> None:
        self._root = root
        self.files = files

    def locate_file(self, relative: Path) -> Path:
        return self._root / relative


def _fake_component_distribution(tmp_path: Path) -> tuple[_FakeDistribution, Path]:
    library_relative = (
        Path("nvidia/cu13/bin/x86_64/cudart64_13.dll")
        if os.name == "nt"
        else Path("nvidia/cu13/lib/libcudart.so.13")
    )
    header_relative = Path("nvidia/cu13/include/cuda_fp16.h")
    for relative in (library_relative, header_relative):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return _FakeDistribution(tmp_path, [library_relative, header_relative]), library_relative


def test_component_wheel_nested_library_directory_is_discovered(
    monkeypatch, tmp_path: Path
) -> None:
    distribution, library_relative = _fake_component_distribution(tmp_path)
    monkeypatch.setattr(cuda_runtime, "_COMPONENT_DISTRIBUTIONS", ("fake-cuda",))
    monkeypatch.setattr(
        cuda_runtime.metadata,
        "distribution",
        lambda _name: distribution,
    )

    assert cuda_runtime.cuda_component_library_dirs() == (
        (tmp_path / library_relative).resolve().parent,
    )


def test_component_wheel_headers_are_a_cuda_root(monkeypatch, tmp_path: Path) -> None:
    distribution, _library_relative = _fake_component_distribution(tmp_path)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_COMPONENT_DISTRIBUTIONS", ("fake-cuda",))
    monkeypatch.setattr(
        cuda_runtime.metadata,
        "distribution",
        lambda _name: distribution,
    )
    monkeypatch.setattr(cuda_runtime, "_configured_file_root", lambda: None)
    monkeypatch.setattr(cuda_runtime, "_system_cuda_roots", lambda: ())

    assert cuda_runtime.discover_cuda_path() == (tmp_path / "nvidia/cu13").resolve()


def test_per_user_cuda_path_file_is_honored(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "configuration"
    toolkit = tmp_path / "CUDA Toolkit"
    include = toolkit / "include"
    include.mkdir(parents=True)
    (include / "cuda_fp16.h").write_bytes(b"")
    config.mkdir()
    (config / ".cascade_cuda_path").write_text(str(toolkit), encoding="utf-8")

    monkeypatch.setenv("CASCADE_CONFIG_DIR", str(config))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_component_cuda_roots", lambda: ())
    monkeypatch.setattr(cuda_runtime, "_system_cuda_roots", lambda: ())

    assert cuda_runtime.cuda_path_file() == config.resolve() / ".cascade_cuda_path"
    assert cuda_runtime.discover_cuda_path() == toolkit.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows DLL search semantics")
def test_windows_dll_directories_are_retained_and_idempotent(
    monkeypatch, tmp_path: Path
) -> None:
    distribution, library_relative = _fake_component_distribution(tmp_path)
    opened: list[str] = []
    handles: list[object] = []

    def add_dll_directory(value: str) -> object:
        opened.append(value)
        handle = object()
        handles.append(handle)
        return handle

    monkeypatch.setattr(cuda_runtime, "_COMPONENT_DISTRIBUTIONS", ("fake-cuda",))
    monkeypatch.setattr(
        cuda_runtime.metadata,
        "distribution",
        lambda _name: distribution,
    )
    monkeypatch.setattr(cuda_runtime.os, "add_dll_directory", add_dll_directory)
    monkeypatch.setattr(cuda_runtime, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(cuda_runtime, "_DLL_DIRECTORY_PATHS", set())

    expected = str((tmp_path / library_relative).resolve().parent)
    assert cuda_runtime.preload_cuda_component_libraries() == (expected,)
    assert cuda_runtime.preload_cuda_component_libraries() == (expected,)
    assert opened == [expected]
    assert cuda_runtime._DLL_DIRECTORY_HANDLES == handles


def test_isolated_probe_bootstraps_before_importing_cupy() -> None:
    assert backend._PROBE_CODE.index("configure_cuda_runtime()") < (
        backend._PROBE_CODE.index("import cupy")
    )
    assert backend._PROBE_CODE.index("import cupy") < (
        backend._PROBE_CODE.index("configure_cupy_compiler_paths(cp)")
    )


def test_windows_cupy_compiler_uses_short_paths_for_unicode_includes(
    monkeypatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, object]]] = []

    def compile_module(source, options=(), **kwargs):
        calls.append((source, options, kwargs))
        return "compiled"

    fake_compiler = SimpleNamespace(_compile_module_with_cache=compile_module)
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(compiler=fake_compiler))
    monkeypatch.setattr(cuda_runtime.os, "name", "nt")
    monkeypatch.setattr(
        cuda_runtime,
        "_windows_short_path",
        lambda value: str(value).replace("Unicode Ω", "UNICOD~1"),
    )

    assert cuda_runtime.configure_cupy_compiler_paths(fake_cupy)
    assert cuda_runtime.configure_cupy_compiler_paths(fake_cupy)
    result = fake_compiler._compile_module_with_cache(
        "kernel", (r"-IC:\Unicode Ω\cupy\include", "--std=c++17"), arch="86"
    )

    assert result == "compiled"
    assert calls == [
        (
            "kernel",
            (r"-IC:\UNICOD~1\cupy\include", "--std=c++17"),
            {"arch": "86"},
        )
    ]


def _run_config(*, cext_mode: str, tissue_mode: str, solver: str = "network_ext"):
    return SimpleNamespace(
        simulation=SimpleNamespace(
            geometry_only=False,
            concentration_solver=solver,
            skip_tissue_oxygen=False,
            tissue_accel=tissue_mode,
            cext={},
        ),
        runtime_settings={
            "cext": {"accel_mode": cext_mode},
            "tissue": {"accel_mode": tissue_mode},
        },
    )


def test_auto_acceleration_falls_back_to_cpu_when_gpu_probe_fails(
    monkeypatch, capsys
) -> None:
    config = _run_config(cext_mode="auto", tissue_mode="auto")
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    monkeypatch.setattr(
        backend,
        "probe_gpu_runtime",
        lambda: backend.GpuProbeResult(False, "GPU unavailable", "No CuPy"),
    )

    assert backend.require_gpu_runtime(config) is None
    assert config.runtime_settings["cext"]["accel_mode"] == "cpu"
    assert config.runtime_settings["tissue"]["accel_mode"] == "cpu"
    assert config.simulation.tissue_accel == "cpu"
    assert "using CPU-compatible solver paths" in capsys.readouterr().out


def test_explicit_gpu_acceleration_still_fails_when_gpu_probe_fails(
    monkeypatch,
) -> None:
    config = _run_config(cext_mode="gpu", tissue_mode="cpu")
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    monkeypatch.setattr(
        backend,
        "probe_gpu_runtime",
        lambda: backend.GpuProbeResult(False, "GPU unavailable", "No CuPy"),
    )

    with pytest.raises(RuntimeError, match="GPU preflight failed"):
        backend.require_gpu_runtime(config)


def test_gpu_only_solver_still_fails_when_auto_probe_fails(monkeypatch) -> None:
    config = _run_config(
        cext_mode="auto", tissue_mode="cpu", solver="network_ext_hybrid_bg"
    )
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    monkeypatch.setattr(
        backend,
        "probe_gpu_runtime",
        lambda: backend.GpuProbeResult(False, "GPU unavailable", "No CuPy"),
    )

    with pytest.raises(RuntimeError, match="GPU preflight failed"):
        backend.require_gpu_runtime(config)
