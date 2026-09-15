"""Discover CPU/GPU capabilities and select the requested execution backend.

Application code uses this module before entering a CUDA solver. Users should
run ``cascade doctor`` for the same compiled-kernel and cuFFT readiness check;
executing this module directly is a minimal GPU-only diagnostic.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from cascade.accelerators.cuda_runtime import (
    configure_cuda_runtime,
    cuda_component_library_dirs,
    cuda_subprocess_environment,
    preload_cuda_component_libraries,
)

_PROBE_MARKER = "CASCADE_GPU_READY"
_PROBE_CODE = r"""
from cascade.accelerators.cuda_runtime import configure_cuda_runtime

configure_cuda_runtime()
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


_GPU_PROBE_SUCCESS: GpuProbeResult | None = None


def _configured_gpu_modes(config: Any) -> tuple[str, str, str]:
    """Return the solver and effective Cext/tissue acceleration policies."""
    simulation = config.simulation
    settings = config.runtime_settings
    solver = str(simulation.concentration_solver or "topdown").strip().lower()
    cext = settings.get("cext", {})
    tissue = settings.get("tissue", {})
    simulation_cext = simulation.cext or {}
    cext_mode = str(
        cext.get("accel_mode", simulation_cext.get("accel_mode", "auto"))
    ).strip().lower()
    tissue_mode = str(
        tissue.get("accel_mode", simulation.tissue_accel or "auto")
    ).strip().lower()
    return solver, cext_mode, tissue_mode


def gpu_requested(config: Any) -> bool:
    """Return whether this run can enter a CuPy-backed solver path."""
    simulation = config.simulation
    if bool(simulation.geometry_only):
        return False

    solver, cext_mode, tissue_mode = _configured_gpu_modes(config)

    if solver in {
        "network_ext",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "topdown_ext_treecode",
        "network_ext_hybrid_bg",
    } and cext_mode in {"gpu", "auto"}:
        return True
    return not bool(simulation.skip_tissue_oxygen) and tissue_mode in {"gpu", "auto"}


def _gpu_required(config: Any) -> bool:
    """Return whether CPU fallback would change an explicit run requirement."""
    simulation = config.simulation
    if bool(simulation.geometry_only):
        return False

    solver, cext_mode, tissue_mode = _configured_gpu_modes(config)

    # The general-network hybrid FFT formulation has no CPU implementation.
    if solver == "network_ext_hybrid_bg":
        return True
    if solver in {
        "network_ext",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "topdown_ext_treecode",
    } and cext_mode == "gpu":
        return True
    return not bool(simulation.skip_tissue_oxygen) and tissue_mode == "gpu"


def _select_cpu_for_auto_modes(config: Any) -> None:
    """Pin unresolved automatic modes to CPU after a failed CUDA preflight."""
    simulation = config.simulation
    settings = config.runtime_settings
    solver, cext_mode, tissue_mode = _configured_gpu_modes(config)

    if cext_mode == "auto" and solver in {
        "network_ext",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "topdown_ext_treecode",
    }:
        cext = settings.setdefault("cext", {})
        cext["accel_mode"] = "cpu"
        simulation_cext = simulation.cext
        if str(simulation_cext.get("accel_mode", "auto")).strip().lower() == "auto":
            simulation_cext["accel_mode"] = "cpu"

    if tissue_mode == "auto":
        simulation.tissue_accel = "cpu"
        settings.setdefault("tissue", {})["accel_mode"] = "cpu"


def probe_gpu_runtime(*, timeout_s: float = 30.0) -> GpuProbeResult:
    """Test CUDA in an isolated process so the caller keeps no GPU context."""
    probe_env = cuda_subprocess_environment()
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
    """Validate required CUDA paths and let automatic modes fall back to CPU."""
    global _GPU_PROBE_SUCCESS
    if not gpu_requested(config):
        return None
    if _GPU_PROBE_SUCCESS is not None:
        return _GPU_PROBE_SUCCESS
    configure_cuda_runtime()
    result = probe_gpu_runtime()
    if result.ready:
        _GPU_PROBE_SUCCESS = result
        return result
    if not _gpu_required(config):
        _select_cpu_for_auto_modes(config)
        print(
            "GPU auto-detection was unavailable; using CPU-compatible solver paths.",
            flush=True,
        )
        return None
    raise RuntimeError(
        "GPU preflight failed before vessel generation. This simulation requests a "
        "CUDA backend, but CuPy could not run CASCADE's kernel/FFT preflight.\n\n"
        "For this project's CUDA 13 environment, run:\n"
        f"  {sys.executable} -m pip install 'cascade-vascular[gpu-cu13]'\n\n"
        "Then relaunch CASCADE Studio. Alternatively, choose CPU-compatible solver "
        "settings in Advanced solver settings.\n\n"
        f"Technical detail:\n{result.detail}"
    )


__all__ = [
    "GpuProbeResult",
    "cuda_component_library_dirs",
    "gpu_requested",
    "preload_cuda_component_libraries",
    "probe_gpu_runtime",
    "require_gpu_runtime",
]


if __name__ == "__main__":
    probe = probe_gpu_runtime()
    print(probe.summary, flush=True)
    if probe.detail:
        print(probe.detail, flush=True)
    raise SystemExit(0 if probe.ready else 1)
