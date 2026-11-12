from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cascade.accelerators import backend, cuda_driver, cuda_runtime
from cascade.diagnostics import environment


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
    return _FakeDistribution(
        tmp_path, [library_relative, header_relative]
    ), library_relative


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
    monkeypatch.setattr(
        cuda_runtime,
        "installed_cupy_distributions",
        lambda: (("cupy-cuda13x", "14.2.0"),),
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
    monkeypatch.setattr(
        cuda_runtime,
        "installed_cupy_distributions",
        lambda: (("cupy-cuda13x", "14.2.0"),),
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

    def component_distribution(name: str) -> _FakeDistribution:
        # Other CUDA families must remain absent; reporting them as installed
        # triggers real DLL preloading of the empty fixture file.
        if name == "fake-cuda":
            return distribution
        raise cuda_runtime.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cuda_runtime, "_COMPONENT_DISTRIBUTIONS", ("fake-cuda",))
    monkeypatch.setattr(
        cuda_runtime.metadata,
        "distribution",
        component_distribution,
    )
    monkeypatch.setattr(
        cuda_runtime,
        "installed_cupy_distributions",
        lambda: (("cupy-cuda13x", "14.2.0"),),
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
        lambda **_: backend.GpuProbeResult(False, "GPU unavailable", "No CuPy"),
    )

    assert backend.require_gpu_runtime(config) is None
    assert config.runtime_settings["cext"]["accel_mode"] == "cpu"
    assert config.runtime_settings["tissue"]["accel_mode"] == "cpu"
    assert config.simulation.tissue_accel == "cpu"
    assert config.runtime_settings["oxygen"]["network_transport_accel"] == "cpu"
    assert config.runtime_settings["hemodynamics"]["kirchhoff_solver"] == "spsolve"
    assert "using CPU-compatible solver paths" in capsys.readouterr().out


def test_explicit_gpu_acceleration_still_fails_when_gpu_probe_fails(
    monkeypatch,
) -> None:
    config = _run_config(cext_mode="gpu", tissue_mode="cpu")
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    monkeypatch.setattr(
        backend,
        "probe_gpu_runtime",
        lambda **_: backend.GpuProbeResult(False, "GPU unavailable", "No CuPy"),
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
        lambda **_: backend.GpuProbeResult(False, "GPU unavailable", "No CuPy"),
    )

    with pytest.raises(RuntimeError, match="GPU preflight failed"):
        backend.require_gpu_runtime(config)


@pytest.mark.parametrize("stage", ["flow", "transport"])
def test_explicit_graph_gpu_stage_requires_preflight(monkeypatch, stage):
    config = _run_config(cext_mode="cpu", tissue_mode="cpu", solver="network")
    config.simulation.skip_tissue_oxygen = True
    config.runtime_settings["hemodynamics"] = {"kirchhoff_solver": "spsolve"}
    config.runtime_settings["oxygen"] = {"network_transport_accel": "cpu"}
    if stage == "flow":
        config.runtime_settings["hemodynamics"]["kirchhoff_solver"] = "gpu_amg"
    else:
        config.runtime_settings["oxygen"]["network_transport_accel"] = "gpu"
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    monkeypatch.setattr(
        backend,
        "probe_gpu_runtime",
        lambda **_: backend.GpuProbeResult(False, "Unavailable", "No CUDA"),
    )
    assert backend.gpu_requested(config)
    with pytest.raises(RuntimeError, match="GPU preflight failed"):
        backend.require_gpu_runtime(config)


@pytest.mark.parametrize(
    "driver, expected",
    [
        (None, "gpu-cu13"),
        (13020, "gpu-cu13"),
        (12040, "gpu-cu12"),
        (11080, "gpu-cu11"),
    ],
)
def test_gpu_preflight_hint_matches_driver(monkeypatch, driver, expected) -> None:
    config = _run_config(cext_mode="gpu", tissue_mode="cpu")
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    monkeypatch.setattr(backend, "cuda_driver_version", lambda: driver)
    monkeypatch.setattr(backend, "cuda_devices", lambda: ())
    monkeypatch.setattr(
        backend,
        "probe_gpu_runtime",
        lambda **_: backend.GpuProbeResult(False, "GPU unavailable", "No CuPy"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        backend.require_gpu_runtime(config)
    message = str(excinfo.value)
    assert f"cascade-vascular[{expected}]" in message
    if driver is not None and driver < 13000:
        assert f"supports up to CUDA {driver // 1000}." in message
        assert "580 or later" in message


# Library and header paths from the published Linux and Windows component wheels.
_WHEEL_LAYOUTS = {
    13: {
        "nvidia-cuda-runtime": (
            "nvidia/cu13/lib/libcudart.so.13",
            "nvidia/cu13/bin/x86_64/cudart64_13.dll",
            "nvidia/cu13/include/cuda_fp16.h",
        ),
        "nvidia-cufft": (
            "nvidia/cu13/lib/libcufft.so.12",
            "nvidia/cu13/bin/x86_64/cufft64_12.dll",
        ),
        "nvidia-nvjitlink": (
            "nvidia/cu13/lib/libnvJitLink.so.13",
            "nvidia/cu13/bin/x86_64/nvJitLink_130_0.dll",
        ),
    },
    12: {
        "nvidia-cuda-runtime-cu12": (
            "nvidia/cuda_runtime/lib/libcudart.so.12",
            "nvidia/cuda_runtime/bin/cudart64_12.dll",
            "nvidia/cuda_runtime/include/cuda_fp16.h",
        ),
        "nvidia-cuda-nvrtc-cu12": (
            "nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
            "nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.4",
            "nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll",
        ),
        "nvidia-cufft-cu12": (
            "nvidia/cufft/lib/libcufft.so.11",
            "nvidia/cufft/lib/libcufftw.so.11",
            "nvidia/cufft/bin/cufft64_11.dll",
        ),
        "nvidia-nvjitlink-cu12": (
            "nvidia/nvjitlink/lib/libnvJitLink.so.12",
            "nvidia/nvjitlink/bin/nvJitLink_120_0.dll",
        ),
    },
    11: {
        "nvidia-cuda-runtime-cu11": (
            "nvidia/cuda_runtime/lib/libcudart.so.11.0",
            "nvidia/cuda_runtime/bin/cudart64_110.dll",
            "nvidia/cuda_runtime/include/cuda_fp16.h",
        ),
        "nvidia-cuda-nvrtc-cu11": (
            "nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2",
            "nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.11.8",
            "nvidia/cuda_nvrtc/bin/nvrtc64_112_0.dll",
        ),
        "nvidia-cublas-cu11": (
            "nvidia/cublas/lib/libcublas.so.11",
            "nvidia/cublas/lib/libcublasLt.so.11",
            "nvidia/cublas/bin/cublas64_11.dll",
        ),
        "nvidia-cufft-cu11": (
            "nvidia/cufft/lib/libcufft.so.10",
            "nvidia/cufft/bin/cufft64_10.dll",
        ),
        "nvidia-cusolver-cu11": (
            "nvidia/cusolver/lib/libcusolver.so.11",
            "nvidia/cusolver/lib/libcusolverMg.so.11",
            "nvidia/cusolver/bin/cusolver64_11.dll",
        ),
    },
}
_CUPY_FOR_FAMILY = {13: "cupy-cuda13x", 12: "cupy-cuda12x", 11: "cupy-cuda11x"}


def _install_fake_wheels(
    monkeypatch, tmp_path: Path, families: tuple[int, ...], cupy: tuple[str, str]
) -> Path:
    """Expose fake component wheels for ``families`` and one CuPy distribution."""
    site = tmp_path / "site-packages"
    distributions = {}
    for family in families:
        for name, files in _WHEEL_LAYOUTS[family].items():
            relatives = [Path(value) for value in files]
            for relative in relatives:
                (site / relative).parent.mkdir(parents=True, exist_ok=True)
                (site / relative).write_bytes(b"")
            distributions[name] = _FakeDistribution(site, relatives)

    def distribution(name: str):
        try:
            return distributions[name]
        except KeyError:
            raise cuda_runtime.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(cuda_runtime.metadata, "distribution", distribution)
    monkeypatch.setattr(cuda_runtime, "installed_cupy_distributions", lambda: (cupy,))
    return site


@pytest.mark.parametrize(
    "family, expected",
    [
        (13, {"nvidia/cu13/lib"}),
        (12, {"nvidia/cuda_runtime/lib", "nvidia/cuda_nvrtc/lib", "nvidia/cufft/lib",
              "nvidia/nvjitlink/lib"}),
        (11, {"nvidia/cuda_runtime/lib", "nvidia/cuda_nvrtc/lib", "nvidia/cublas/lib",
              "nvidia/cufft/lib", "nvidia/cusolver/lib"}),
    ],
)
def test_component_library_dirs_follow_each_cuda_wheel_layout(
    monkeypatch, tmp_path: Path, family: int, expected: set[str]
) -> None:
    # Stale wheels from every other family must not be picked up.
    site = _install_fake_wheels(
        monkeypatch, tmp_path, (13, 12, 11), (_CUPY_FOR_FAMILY[family], "14.2.0")
    )
    if os.name == "nt":
        expected = {
            "nvidia/cu13/bin/x86_64" if path.startswith("nvidia/cu13") else
            path.rsplit("/", 1)[0] + "/bin"
            for path in expected
        }

    directories = cuda_runtime.cuda_component_library_dirs()

    assert {path.relative_to(site.resolve()).as_posix() for path in directories} == (
        expected
    )


@pytest.mark.parametrize(
    "family, root",
    [(13, "nvidia/cu13"), (12, "nvidia/cuda_runtime"), (11, "nvidia/cuda_runtime")],
)
def test_component_headers_define_cuda_path_for_each_family(
    monkeypatch, tmp_path: Path, family: int, root: str
) -> None:
    site = _install_fake_wheels(
        monkeypatch, tmp_path, (family,), (_CUPY_FOR_FAMILY[family], "13.6.0")
    )
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(cuda_runtime, "_configured_file_root", lambda: None)
    monkeypatch.setattr(cuda_runtime, "_system_cuda_roots", lambda: ())

    assert cuda_runtime.discover_cuda_path() == (site / root).resolve()


def test_cuda_home_is_used_when_no_wheel_headers_exist(
    monkeypatch, tmp_path: Path
) -> None:
    toolkit = tmp_path / "cuda-11.8"
    (toolkit / "include").mkdir(parents=True)
    (toolkit / "include" / "cuda_fp16.h").write_bytes(b"")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    monkeypatch.setattr(cuda_runtime, "_configured_file_root", lambda: None)
    monkeypatch.setattr(cuda_runtime, "_component_cuda_roots", lambda: ())
    monkeypatch.setattr(cuda_runtime, "_system_cuda_roots", lambda: ())

    assert cuda_runtime.discover_cuda_path() == toolkit.resolve()


@pytest.mark.skipif(os.name == "nt", reason="Linux soname preloading")
@pytest.mark.parametrize(
    "family, cupy_version, expected",
    [
        (13, "14.2.0", ["libnvJitLink.so.13", "libcufft.so.12"]),
        (12, "14.2.0", ["libnvJitLink.so.12", "libcufft.so.11"]),
        # CuPy 13 links CUDA libraries by soname, so all of them load first.
        (
            11,
            "13.6.0",
            [
                "libnvrtc.so.11.2",
                "libcudart.so.11.0",
                "libcublasLt.so.11",
                "libcublas.so.11",
                "libcufft.so.10",
                "libcusolver.so.11",
            ],
        ),
    ],
)
def test_linux_preload_matches_cupy_library_lookup(
    monkeypatch, tmp_path: Path, family: int, cupy_version: str, expected: list[str]
) -> None:
    _install_fake_wheels(
        monkeypatch, tmp_path, (family,), (_CUPY_FOR_FAMILY[family], cupy_version)
    )
    loaded: list[str] = []
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        cuda_runtime.ctypes, "CDLL", lambda path, mode=0: loaded.append(path) or path
    )
    monkeypatch.setattr(cuda_runtime, "_PRELOADED_LIBRARIES", [])

    cuda_runtime.preload_cuda_component_libraries()

    assert [Path(path).name for path in loaded] == expected


def test_windows_compiler_wrapper_accepts_cupy13_positional_arguments(
    monkeypatch,
) -> None:
    calls = []

    # CuPy 13 signature; its core passes arch through backend positionally.
    def compile_module(source, options=(), arch=None, cache_dir=None,
                       extra_source=None, backend="nvrtc", **kwargs):
        calls.append((options, arch, cache_dir, extra_source, backend, kwargs))
        return "compiled"

    fake_compiler = SimpleNamespace(_compile_module_with_cache=compile_module)
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(compiler=fake_compiler))
    monkeypatch.setattr(cuda_runtime.os, "name", "nt")
    monkeypatch.setattr(cuda_runtime, "_windows_short_path", lambda value: "SHORT")

    assert cuda_runtime.configure_cupy_compiler_paths(fake_cupy)
    result = fake_compiler._compile_module_with_cache(
        "kernel", ("-IC:\\Ω",), "86", None, "extra", "nvrtc", jitify=False
    )

    assert result == "compiled"
    assert calls == [(("-ISHORT",), "86", None, "extra", "nvrtc", {"jitify": False})]


@pytest.mark.parametrize(
    "version, extra",
    [
        (None, None),
        (13000, "gpu-cu13"),
        (14010, "gpu-cu13"),
        (12090, "gpu-cu12"),
        (12000, "gpu-cu12"),
        (11080, "gpu-cu11"),
        (11020, "gpu-cu11"),
        (11010, None),
        (10020, None),
    ],
)
def test_gpu_extra_follows_driver_cuda_version(version, extra) -> None:
    assert cuda_driver.gpu_extra_for_driver(version) == extra


def test_driver_probe_runs_without_cascade_installed() -> None:
    script = Path(cuda_driver.__file__)
    completed = subprocess.run(
        [sys.executable, "-I", str(script)],
        capture_output=True,
        encoding="utf-8", errors="replace",
        check=True,
        timeout=60,
    )

    assert re.fullmatch(r"\d+ (gpu-cu1[123]|none) (sm_\d+|none)\n", completed.stdout)


def test_gpu_preflight_hint_uninstalls_a_mismatched_cupy(monkeypatch) -> None:
    monkeypatch.setattr(
        backend, "installed_cupy_distributions", lambda: (("cupy-cuda13x", "14.2.0"),)
    )

    hint = backend._gpu_install_hint(12040)

    assert "pip uninstall -y cupy-cuda13x" in hint
    assert hint.index("uninstall") < hint.index("cascade-vascular[gpu-cu12]")
    assert "uninstall" not in backend._gpu_install_hint(13000)


@pytest.mark.parametrize(
    "driver, installed, expected",
    [
        (12040, [("cupy-cuda13x", "14.2.0")], ["580 or later", "uninstall -y cupy-cuda13x",
                                                "cascade-vascular[gpu-cu12]"]),
        (11080, [("cupy-cuda12x", "14.2.0")], ["CUDA 11.8", "cascade-vascular[gpu-cu11]"]),
        (11010, [("cupy-cuda11x", "13.6.0")], ["Update the NVIDIA driver"]),
        (12040, [("cupy-cuda12x", "14.2.0"), ("cupy-cuda13x", "14.2.0")],
         ["Several CuPy packages", "uninstall -y cupy-cuda12x cupy-cuda13x"]),
        (13020, [("cupy-cuda13x", "14.2.0")], None),
        (12040, [("cupy-cuda12x", "14.2.0")], None),
        (13020, [("cupy-cuda11x", "13.6.0")], None),
        (None, [("cupy-cuda13x", "14.2.0")], None),
        (12040, [], None),
    ],
)
def test_doctor_flags_cupy_variants_the_driver_cannot_run(
    driver, installed, expected
) -> None:
    issue = environment.cuda_variant_issue(driver, tuple(installed))

    if expected is None:
        assert issue is None
    else:
        for text in expected:
            assert text in issue


def test_doctor_reports_driver_and_cupy_variant(monkeypatch, capsys) -> None:
    monkeypatch.setattr(environment, "_DEPENDENCIES", ("numpy",))
    monkeypatch.setattr(environment, "configure_cuda_runtime", lambda: ())
    monkeypatch.setattr(environment, "cuda_driver_version", lambda: 12040)
    monkeypatch.setattr(
        environment,
        "cuda_devices",
        lambda: (cuda_driver.CudaDevice(0, (7, 0), 3),),
    )
    monkeypatch.setattr(
        environment, "foreign_component_distributions", lambda: ("nvidia-cufft",)
    )
    monkeypatch.setattr(
        environment,
        "installed_cupy_distributions",
        lambda: (("cupy-cuda13x", "14.2.0"),),
    )

    report = environment.collect_diagnostics(probe_gpu=False)
    assert report["cuda_driver_version"] == "12.4"
    assert report["cupy_packages"] == [{"name": "cupy-cuda13x", "version": "14.2.0"}]
    assert "gpu-cu12" in report["cuda_variant_issue"]

    assert environment.main(["--no-gpu-probe"]) == 0
    output = capsys.readouterr().out
    assert "NVIDIA driver: CUDA 12.4" in output
    assert "CuPy package: cupy-cuda13x 14.2.0" in output
    assert "WARNING: cupy-cuda13x needs a CUDA 13 driver" in output
    assert "CUDA device 0: sm_70, Exclusive_Process compute mode" in output
    assert "WARNING: cupy-cuda13x cannot compile kernels for GPU 0 (sm_70)" in output
    assert "pip uninstall -y nvidia-cufft" in output
    assert report["cuda_devices"] == [
        {"index": 0, "sm": "sm_70", "compute_mode": "Exclusive_Process"}
    ]


def test_run_provenance_records_driver_and_cuda_packages(monkeypatch) -> None:
    from cascade.exporting import run

    monkeypatch.setattr(run, "cuda_driver_version", lambda: 12040)
    run._platform_metadata.cache_clear()
    run._dependency_versions.cache_clear()
    try:
        assert run._platform_metadata()["cuda_driver_version"] == "12.4"
        versions = run._dependency_versions()
    finally:
        run._platform_metadata.cache_clear()
        run._dependency_versions.cache_clear()
    for key in ("cupy_cuda11x_version", "cupy_cuda12x_version",
                "cupy_cuda13x_version", "cuda_toolkit_version"):
        assert key in versions


@pytest.mark.parametrize(
    "version, compute, extra",
    [
        (13020, (8, 6), "gpu-cu13"),
        (13020, (7, 5), "gpu-cu13"),
        # CUDA 13 dropped Maxwell, Pascal and Volta: V100 and P100 need CUDA 12.
        (13020, (7, 0), "gpu-cu12"),
        (13020, (6, 0), "gpu-cu12"),
        (12040, (5, 0), "gpu-cu12"),
        # CUDA 12 dropped Kepler; CUDA 11.8 still targets sm_35.
        (13020, (3, 7), "gpu-cu11"),
        (11080, (3, 5), "gpu-cu11"),
        (13020, (3, 0), None),
        (11080, (8, 6), "gpu-cu11"),
    ],
)
def test_gpu_extra_follows_oldest_gpu_architecture(version, compute, extra) -> None:
    assert cuda_driver.gpu_extra_for_driver(version, compute) == extra


class _FakeDriver:
    """ctypes-style stand-in for libcuda's device and context queries."""

    def __init__(self, devices, *, initialized=True, current=0, primary=()):
        self.devices = devices  # [(major, minor, compute_mode)]
        self.initialized = initialized
        self.current = current
        self.primary = set(primary)
        self.init_calls = 0

    def cuDriverGetVersion(self, out):
        out._obj.value = 12040
        return 0

    def cuInit(self, _flags):
        self.init_calls += 1
        self.initialized = True
        return 0 if self.devices else 100

    def cuDeviceGetCount(self, out):
        if not self.initialized:
            return 3
        out._obj.value = len(self.devices)
        return 0

    def cuDeviceGet(self, out, ordinal):
        out._obj.value = ordinal
        return 0

    def cuDeviceGetAttribute(self, out, attribute, device):
        major, minor, mode = self.devices[device.value]
        out._obj.value = {75: major, 76: minor, 20: mode}[attribute]
        return 0

    def cuCtxGetCurrent(self, out):
        if not self.initialized:
            return 3
        out._obj.value = self.current or None
        return 0

    def cuDevicePrimaryCtxGetState(self, device, _flags, active):
        active._obj.value = int(device.value in self.primary)
        return 0


def test_cuda_devices_report_architecture_and_compute_mode(monkeypatch) -> None:
    fake = _FakeDriver([(8, 6, 0), (7, 0, 3)])
    monkeypatch.setattr(cuda_driver, "_driver_library", lambda: fake)

    devices = cuda_driver.cuda_devices()

    assert [device.sm for device in devices] == ["sm_86", "sm_70"]
    assert [device.compute_mode_name for device in devices] == [
        "Default",
        "Exclusive_Process",
    ]
    assert devices[1].exclusive_process
    assert cuda_driver.minimum_compute_capability(devices) == (7, 0)
    monkeypatch.setattr(cuda_driver, "_driver_library", lambda: _FakeDriver([]))
    assert cuda_driver.cuda_devices() == ()


@pytest.mark.parametrize(
    "fake, expected",
    [
        (_FakeDriver([(7, 0, 3)], initialized=False), False),
        (_FakeDriver([(7, 0, 3)]), False),
        (_FakeDriver([(7, 0, 3)], current=1234), True),
        (_FakeDriver([(8, 6, 0), (7, 0, 3)], primary={1}), True),
    ],
)
def test_context_check_never_initializes_the_driver(monkeypatch, fake, expected):
    monkeypatch.setattr(cuda_driver, "_driver_library", lambda: fake)

    assert cuda_driver.process_has_cuda_context() is expected
    assert fake.init_calls == 0


def _fail_subprocess(*_args, **_kwargs):
    raise AssertionError("the probe must not start a subprocess")


@pytest.mark.parametrize(
    "has_context, exclusive, will_use_gpu, in_process",
    [
        (True, False, False, True),
        (False, True, True, True),
        # doctor and other callers that will not use the GPU keep isolation.
        (False, True, False, False),
        (False, False, True, False),
    ],
)
def test_probe_runs_in_process_when_a_subprocess_cannot_open_the_gpu(
    monkeypatch, has_context, exclusive, will_use_gpu, in_process
) -> None:
    monkeypatch.setattr(backend, "process_has_cuda_context", lambda: has_context)
    monkeypatch.setattr(
        backend,
        "cuda_devices",
        lambda: (cuda_driver.CudaDevice(0, (7, 0), 3 if exclusive else 0),),
    )
    monkeypatch.setattr(
        backend,
        "_probe_in_process",
        lambda: backend.GpuProbeResult(True, "in-process"),
    )
    if in_process:
        monkeypatch.setattr(backend.subprocess, "run", _fail_subprocess)
    else:
        monkeypatch.setattr(backend, "cuda_subprocess_environment", lambda: {})
        monkeypatch.setattr(
            backend.subprocess,
            "run",
            lambda *_a, **_k: subprocess.CompletedProcess(
                [], 0, "CASCADE_GPU_READY|Tesla V100S|cupy=14.2.0\n", ""
            ),
        )

    result = backend.probe_gpu_runtime(will_use_gpu=will_use_gpu)

    assert result.ready
    assert (result.summary == "in-process") is in_process


def test_in_process_probe_reports_success_and_busy_devices(monkeypatch) -> None:
    monkeypatch.setattr(
        backend, "_PROBE_CODE", "print('CASCADE_GPU_READY|Tesla P100|cupy=13.6.0')"
    )
    assert backend._probe_in_process() == backend.GpuProbeResult(
        True, "Tesla P100 (cupy=13.6.0)"
    )

    monkeypatch.setattr(
        backend,
        "_PROBE_CODE",
        "raise RuntimeError('cudaErrorDevicesUnavailable: CUDA-capable device(s) "
        "is/are busy or unavailable')",
    )
    result = backend._probe_in_process()
    assert not result.ready and result.busy
    assert "Exclusive_Process" in result.detail
    assert "cudaErrorDevicesUnavailable" in result.detail


def test_subprocess_probe_explains_exclusive_process_busy_errors(monkeypatch) -> None:
    monkeypatch.setattr(backend, "process_has_cuda_context", lambda: False)
    monkeypatch.setattr(backend, "cuda_subprocess_environment", lambda: {})
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            [],
            1,
            "",
            "cupy_backends.cuda.api.runtime.CUDARuntimeError: "
            "cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or "
            "unavailable",
        ),
    )

    result = backend.probe_gpu_runtime()

    assert result.busy
    assert result.summary == "GPU is busy or unavailable"
    assert "another process is holding it" in result.detail


def test_busy_gpu_preflight_explains_exclusive_mode_without_install_hint(
    monkeypatch, capsys
) -> None:
    busy = backend.GpuProbeResult(False, "GPU is busy", "Exclusive_Process ...", True)
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    monkeypatch.setattr(backend, "probe_gpu_runtime", lambda **_: busy)

    with pytest.raises(RuntimeError) as excinfo:
        backend.require_gpu_runtime(_run_config(cext_mode="gpu", tissue_mode="cpu"))
    assert "busy or unavailable" in str(excinfo.value)
    assert "Exclusive_Process" in str(excinfo.value)
    assert "pip install" not in str(excinfo.value)

    assert backend.require_gpu_runtime(
        _run_config(cext_mode="auto", tissue_mode="auto")
    ) is None
    assert "(the GPU is busy)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "driver, compute, expected",
    [
        (13020, (7, 0), ["sm_70", "sm_75 or newer", "cascade-vascular[gpu-cu12]"]),
        (13020, (3, 7), ["sm_37", "CUDA 12", "cascade-vascular[gpu-cu11]"]),
        (13020, (3, 0), ["sm_30", "older than every CUDA version"]),
        (12040, (8, 6), ["supports up to CUDA 12.4", "cascade-vascular[gpu-cu12]"]),
    ],
)
def test_gpu_preflight_hint_accounts_for_gpu_architecture(
    monkeypatch, driver, compute, expected
) -> None:
    monkeypatch.setattr(backend, "installed_cupy_distributions", lambda: ())

    hint = backend._gpu_install_hint(driver, compute)

    for text in expected:
        assert text in hint


@pytest.mark.parametrize(
    "cupy, compute, expected",
    [
        ("cupy-cuda13x", (7, 0), ["GPU 0 (sm_70)", "sm_75 or newer",
                                   "cascade-vascular[gpu-cu12]"]),
        ("cupy-cuda12x", (7, 0), None),
        ("cupy-cuda12x", (3, 7), ["sm_50 or newer", "cascade-vascular[gpu-cu11]"]),
        ("cupy-cuda13x", (8, 6), None),
    ],
)
def test_doctor_flags_gpus_the_cupy_family_cannot_target(cupy, compute, expected):
    devices = (cuda_driver.CudaDevice(0, compute, 3),)

    issue = environment.cuda_device_issue(13020, devices, ((cupy, "14.2.0"),))

    if expected is None:
        assert issue is None
    else:
        for text in expected:
            assert text in issue


@pytest.mark.skipif(os.name == "nt", reason="Linux soname preloading")
def test_leftover_cuda13_wheels_cannot_supply_cupy_cuda12x_libraries(
    monkeypatch, tmp_path: Path
) -> None:
    # Pathfinder searches nvidia/cu13 first, so the cu12 libraries must already
    # be loaded; nothing may come from the leftover CUDA 13 wheels.
    _install_fake_wheels(monkeypatch, tmp_path, (13, 12), ("cupy-cuda12x", "14.2.0"))
    loaded: list[str] = []
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        cuda_runtime.ctypes, "CDLL", lambda path, mode=0: loaded.append(path) or path
    )
    monkeypatch.setattr(cuda_runtime, "_PRELOADED_LIBRARIES", [])

    assert cuda_runtime.foreign_component_distributions() == (
        "nvidia-cuda-runtime",
        "nvidia-cufft",
        "nvidia-nvjitlink",
    )
    cuda_runtime.preload_cuda_component_libraries()

    assert [Path(path).name for path in loaded] == [
        "libnvJitLink.so.12",
        "libnvrtc.so.12",
        "libcudart.so.12",
        "libcufft.so.11",
    ]
    assert not any("/cu13/" in path for path in loaded)


def test_gpu_subprocess_guard_skips_when_this_process_holds_an_exclusive_gpu(
    monkeypatch, request
) -> None:
    monkeypatch.setattr(cuda_driver, "process_has_cuda_context", lambda: True)
    monkeypatch.setattr(
        cuda_driver, "cuda_devices", lambda: (cuda_driver.CudaDevice(0, (7, 0), 3),)
    )

    with pytest.raises(pytest.skip.Exception, match="Exclusive_Process"):
        request.getfixturevalue("gpu_subprocess_guard")


class _UnusableCupy:
    """CuPy on a GPU another process holds exclusively: devices can be counted,
    but anything that would create a CUDA context fails."""

    def __init__(self, path: str = "cupy") -> None:
        self._path = path

    def __getattr__(self, name):
        path = f"{self._path}.{name}"
        if path in {"cupy.cuda", "cupy.cuda.runtime"}:
            return _UnusableCupy(path)
        if path == "cupy.cuda.runtime.getDeviceCount":
            return lambda: 1
        raise RuntimeError(f"cudaErrorDevicesUnavailable: {path} used")


def test_busy_gpu_auto_run_completes_on_cpu_without_touching_cupy(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import json

    from cascade.commands.main import main
    from cascade.configuration import solver_state as state

    monkeypatch.setenv("CASCADE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(state, "_cp", _UnusableCupy())
    monkeypatch.setattr(backend, "_WITHHELD_CUPY", None)
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    busy = backend.GpuProbeResult(False, "GPU is busy", "Exclusive_Process", True)
    monkeypatch.setattr(backend, "probe_gpu_runtime", lambda **_: busy)
    # Every acceleration stage is left on "auto".
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "simple",
            "simple": {"mode": "onechannel", "radius_cm": 0.02, "flow_ul_min": 2.0},
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "blood",
            "qin_target_ul_min": 2.0,
            "concentration_solver": "network_ext",
            "distance_sample_count": 8,
        },
        "outputs": {
            "out_dir": str(tmp_path / "run"),
            "write_paraview": False,
            "save_network": False,
        },
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(settings), encoding="utf-8")

    status = main(["run", "--settings", str(path)])

    captured = capsys.readouterr()
    assert status == 0, captured.out + captured.err
    assert "(the GPU is busy); using CPU-compatible solver paths" in captured.out
    assert (tmp_path / "run" / "summary.csv").is_file()
    assert backend.gpu_disabled_for_run()
    # The next run's preflight hands CuPy back to the solvers.
    backend.restore_gpu_for_run()
    assert isinstance(state._cp, _UnusableCupy)


def test_child_processes_write_and_parents_read_utf8_under_ascii_locale() -> None:
    from cascade.utils.console import utf8_child_environment

    # Force ASCII pipe output as well as the locale: Windows does not use
    # LC_ALL to select its pipe encoding. The child otherwise escapes "…" and "µ".
    hostile = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LC_", "LANG", "PYTHONUTF8", "PYTHONIOENCODING"))
    }
    hostile.update({
        "LC_ALL": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONIOENCODING": "ascii",
    })
    code = (
        "from cascade.utils.console import configure_console_error_handling\n"
        "configure_console_error_handling()\n"
        # Escaped so the command line itself stays ASCII.
        "print('Solving patches \\u2026 5 \\u00b5m')\n"
    )

    def run(environment):
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            check=True,
            timeout=60,
        ).stdout

    assert "\\u2026" in run(hostile)
    assert run(utf8_child_environment(hostile)) == "Solving patches … 5 µm\n"
    probe_environment = cuda_runtime.cuda_subprocess_environment()
    assert probe_environment["PYTHONIOENCODING"] == "utf-8"
    assert probe_environment["PYTHONUTF8"] == "1"


def test_geometry_only_run_never_touches_cupy(monkeypatch, tmp_path: Path) -> None:
    import json

    from cascade.commands.main import main
    from cascade.configuration import solver_state as state

    monkeypatch.setenv("CASCADE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(state, "_cp", _UnusableCupy())
    monkeypatch.setattr(backend, "_WITHHELD_CUPY", None)
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)

    def _no_probe(**_):
        raise AssertionError("geometry-only runs must not probe the GPU")

    monkeypatch.setattr(backend, "probe_gpu_runtime", _no_probe)
    settings = {
        "domain": {
            "type": "sphere",
            "side_length": 1.0,
            "radius": 0.5,
            "theta_resolution": 12,
            "phi_resolution": 12,
            "random_seed": 42,
        },
        "network": {
            "mode": "simple",
            "simple": {"mode": "onechannel", "radius_cm": 0.02, "flow_ul_min": 2.0},
        },
        "growth": {"enabled": False},
        "simulation": {"geometry_only": True, "distance_sample_count": 0},
        "outputs": {"out_dir": str(tmp_path / "run"), "save_network": False},
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(settings), encoding="utf-8")

    assert main(["run", "--settings", str(path)]) == 0
    assert backend.gpu_disabled_for_run()
    backend.restore_gpu_for_run()


def test_busy_gpu_sweep_completes_on_cpu_without_touching_cupy(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import json

    from cascade.commands.main import main
    from cascade.configuration import solver_state as state

    monkeypatch.setenv("CASCADE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(state, "_cp", _UnusableCupy())
    monkeypatch.setattr(backend, "_WITHHELD_CUPY", None)
    monkeypatch.setattr(backend, "_GPU_PROBE_SUCCESS", None)
    busy = backend.GpuProbeResult(False, "GPU is busy", "Exclusive_Process", True)
    monkeypatch.setattr(backend, "probe_gpu_runtime", lambda **_: busy)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "simple",
            "simple": {"mode": "onechannel", "radius_cm": 0.02, "flow_ul_min": 2.0},
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "water",
            "distance_sample_count": 0,
            "skip_tissue_oxygen": True,
        },
        "outputs": {"out_dir": str(tmp_path / "work"), "save_network": False},
        "sweep": {
            "target_terminal_counts": [1],
            "fluids": ["water"],
            "qin_target_ul_min_values": [1.0, 2.0],
            "distance_sample_counts": [0],
            "output_csv": str(tmp_path / "sweep.csv"),
            "work_dir": str(tmp_path / "work"),
            "save_final_network": False,
        },
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(settings), encoding="utf-8")

    status = main(["sweep", "--settings", str(path)])

    captured = capsys.readouterr()
    assert status == 0, captured.out + captured.err
    assert (tmp_path / "sweep.csv").is_file()
    backend.restore_gpu_for_run()
