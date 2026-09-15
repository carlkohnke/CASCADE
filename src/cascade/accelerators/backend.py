"""Discover CPU/GPU capabilities and select the requested execution backend.

Application code uses this module before entering a CUDA solver. Users should
run ``cascade doctor`` for the same compiled-kernel and cuFFT readiness check;
executing this module directly is a minimal GPU-only diagnostic.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


_PROBE_MARKER = "CASCADE_GPU_READY"
_PROBE_CODE = r"""
import cupy as cp

device_count = int(cp.cuda.runtime.getDeviceCount())
if device_count < 1:
    raise RuntimeError("CuPy did not find an NVIDIA CUDA device")

# A device-count check or cp.arange alone can succeed even when CUDA component
# libraries needed by CASCADE are missing. Exercise both an elementwise kernel
# and cuFFT, which is used by the production heart Cext path.
values = cp.full((4, 4), cp.int32(-1), dtype=cp.int32)
total = int(cp.sum(values).get())
if total != -16:
    raise RuntimeError(f"CUDA test kernel returned {total}, expected -16")

spectrum = cp.fft.fftn(cp.ones((4, 4), dtype=cp.float32))
fft_total = float(cp.abs(spectrum).sum().get())
if abs(fft_total - 16.0) > 1.0e-4:
    raise RuntimeError(f"cuFFT test returned {fft_total}, expected 16")

name = cp.cuda.runtime.getDeviceProperties(0)["name"]
if isinstance(name, bytes):
    name = name.decode("utf-8", errors="replace")
print(f"CASCADE_GPU_READY|{name}|cupy={cp.__version__}", flush=True)
"""


@dataclass(frozen=True)
class GpuProbeResult:
    ready: bool
    summary: str
    detail: str = ""


_DLL_DIRECTORY_HANDLES: list[Any] = []
_GPU_PROBE_SUCCESS: GpuProbeResult | None = None


def cuda_component_library_dirs() -> tuple[Path, ...]:
    """Return library directories supplied by NVIDIA CUDA component wheels."""
    candidates: list[Path] = []
    for distribution_name in (
        "nvidia-cuda-runtime",
        "nvidia-cufft",
        "nvidia-nvjitlink",
    ):
        try:
            distribution = metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError:
            continue
        for relative in (
            "nvidia/cu13/lib",
            "nvidia/cu13/bin",
            "nvidia/cufft/lib",
            "nvidia/cufft/bin",
            "nvidia/nvjitlink/lib",
            "nvidia/nvjitlink/bin",
        ):
            candidate = Path(distribution.locate_file(relative)).resolve()
            if candidate.is_dir() and candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def preload_cuda_component_libraries() -> tuple[str, ...]:
    """Make wheel-provided CUDA libraries visible to the current process."""
    directories = cuda_component_library_dirs()
    if not directories:
        return ()

    if os.name == "nt":
        add_directory = getattr(os, "add_dll_directory", None)
        if add_directory is not None:
            for directory in directories:
                _DLL_DIRECTORY_HANDLES.append(add_directory(str(directory)))
        return tuple(str(path) for path in directories)

    if not sys.platform.startswith("linux"):
        return tuple(str(path) for path in directories)

    # Load by absolute path so a user does not need to export LD_LIBRARY_PATH.
    # nvJitLink must be globally visible before cuFFT is loaded.
    for library_name in ("libnvJitLink.so.13", "libcufft.so.12"):
        library_path = next(
            (
                directory / library_name
                for directory in directories
                if (directory / library_name).is_file()
            ),
            None,
        )
        if library_path is None:
            continue
        try:
            ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load CUDA component library {library_path}: {exc}"
            ) from exc
    return tuple(str(path) for path in directories)


def gpu_requested(config: Any) -> bool:
    """Return whether this run can enter a CuPy-backed solver path."""
    simulation = config.simulation
    if bool(simulation.geometry_only):
        return False

    settings = config.runtime_settings
    solver = str(simulation.concentration_solver or "topdown").strip().lower()
    cext = settings.get("cext", {})
    tissue = settings.get("tissue", {})
    cext_mode = str(cext.get("accel_mode", "gpu")).strip().lower()
    tissue_mode = (
        str(simulation.tissue_accel or tissue.get("accel_mode", "gpu")).strip().lower()
    )

    if solver in {
        "network_ext",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "topdown_ext_treecode",
        "network_ext_hybrid_bg",
    } and cext_mode in {"gpu", "auto"}:
        return True
    return not bool(simulation.skip_tissue_oxygen) and tissue_mode in {"gpu", "auto"}


def probe_gpu_runtime(*, timeout_s: float = 30.0) -> GpuProbeResult:
    """Test CUDA in an isolated process so the caller keeps no GPU context."""
    probe_env = os.environ.copy()
    component_dirs = cuda_component_library_dirs()
    if component_dirs:
        variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        existing = probe_env.get(variable, "")
        prefix = os.pathsep.join(str(path) for path in component_dirs)
        probe_env[variable] = prefix if not existing else prefix + os.pathsep + existing
    if not probe_env.get("CUDA_PATH"):
        candidates = [Path(sys.prefix) / "targets" / "x86_64-linux"]
        try:
            runtime_dist = metadata.distribution("nvidia-cuda-runtime")
            candidates.append(Path(runtime_dist.locate_file("nvidia/cu13")))
        except metadata.PackageNotFoundError:
            pass
        source_root = Path(__file__).resolve().parents[2]
        config_roots = []
        if os.environ.get("CASCADE_CONFIG_DIR"):
            config_roots.append(Path(os.environ["CASCADE_CONFIG_DIR"]).expanduser())
        config_roots.extend(
            [
                source_root.parent / "CASCADE-workbench" / "config",
                source_root,
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "cascade",
            ]
        )
        for config_root in config_roots:
            source_config = config_root / ".cascade_cuda_path"
            if source_config.is_file():
                try:
                    candidates.append(
                        Path(
                            source_config.read_text(encoding="utf-8").strip()
                        ).expanduser()
                    )
                except OSError:
                    pass
        candidates.extend(
            Path(value)
            for value in ("/usr/local/cuda", "/usr/local/cuda-13", "/usr/local/cuda-12")
        )
        for candidate in candidates:
            if (candidate / "include" / "cuda_fp16.h").is_file():
                probe_env["CUDA_PATH"] = str(candidate)
                break
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=probe_env,
        )
    except subprocess.TimeoutExpired:
        return GpuProbeResult(
            False,
            "GPU check timed out",
            f"CuPy did not finish a test kernel within {timeout_s:g} seconds.",
        )
    except Exception as exc:
        return GpuProbeResult(False, "GPU check could not start", str(exc))

    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode == 0:
        marker = next(
            (
                line
                for line in completed.stdout.splitlines()
                if line.startswith(_PROBE_MARKER)
            ),
            _PROBE_MARKER,
        )
        fields = marker.split("|")
        device = fields[1] if len(fields) > 1 else "CUDA device"
        version = fields[2] if len(fields) > 2 else "CuPy ready"
        return GpuProbeResult(True, f"{device} ({version})")

    detail_lines = output.splitlines()
    detail = "\n".join(detail_lines[-12:]) if detail_lines else "Unknown CuPy error"
    return GpuProbeResult(False, "CuPy cannot compile a CUDA test kernel", detail)


def require_gpu_runtime(config: Any) -> GpuProbeResult | None:
    """Fail early with an actionable message when a configured GPU path is broken."""
    global _GPU_PROBE_SUCCESS
    if not gpu_requested(config):
        return None
    if _GPU_PROBE_SUCCESS is not None:
        return _GPU_PROBE_SUCCESS
    preload_cuda_component_libraries()
    result = probe_gpu_runtime()
    if result.ready:
        _GPU_PROBE_SUCCESS = result
        return result
    raise RuntimeError(
        "GPU preflight failed before vessel generation. This simulation requests a "
        "CUDA backend, but CuPy could not run CASCADE's kernel/FFT preflight.\n\n"
        "For this project's CUDA 13 environment, run:\n"
        f"  {sys.executable} -m pip install 'cascade-vascular[gpu-cu13]'\n\n"
        "Then relaunch CASCADE Studio. Alternatively, choose CPU-compatible solver "
        "settings in Advanced solver settings.\n\n"
        f"Technical detail:\n{result.detail}"
    )


if __name__ == "__main__":
    probe = probe_gpu_runtime()
    print(probe.summary, flush=True)
    if probe.detail:
        print(probe.detail, flush=True)
    raise SystemExit(0 if probe.ready else 1)
