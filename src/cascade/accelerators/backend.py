"""Discover CPU/GPU capabilities and select the requested execution backend.

Application code uses this module before entering a CUDA solver. Users should
run ``cascade doctor`` for the same compiled-kernel and cuFFT readiness check;
executing this module directly is a minimal GPU-only diagnostic.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from cascade.accelerators.cuda_driver import (
    cuda_devices,
    cuda_driver_version,
    extra_cuda_major,
    family_min_compute_capability,
    format_cuda_version,
    gpu_extra_for_driver,
    minimum_compute_capability,
    process_has_cuda_context,
)
from cascade.accelerators.cuda_runtime import (
    configure_cuda_runtime,
    cuda_component_library_dirs,
    cuda_subprocess_environment,
    installed_cupy_distributions,
    preload_cuda_component_libraries,
)

_PROBE_MARKER = "CASCADE_GPU_READY"
_PROBE_CODE = r"""
from cascade.accelerators.cuda_runtime import configure_cuda_runtime
from cascade.accelerators.cuda_runtime import configure_cupy_compiler_paths

configure_cuda_runtime()
import cupy as cp
configure_cupy_compiler_paths(cp)

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
    busy: bool = False


_BUSY_MARKERS = (
    "cudaErrorDevicesUnavailable",
    "CUDA_ERROR_DEVICE_UNAVAILABLE",
    "busy or unavailable",
)
_BUSY_EXPLANATION = (
    "The GPU refused a new CUDA context. Clusters commonly run GPUs in "
    "Exclusive_Process compute mode (see `nvidia-smi -q -d COMPUTE`), where only "
    "one process may use a device at a time, so another process is holding it: "
    "another job, Studio's worker, or an earlier CASCADE or Python process on "
    "this node. Stop that process or request a separate GPU, then retry."
)


_GPU_PROBE_SUCCESS: GpuProbeResult | None = None


def _configured_gpu_modes(config: Any) -> tuple[str, str, str]:
    """Return the solver and effective Cext/tissue acceleration policies."""
    simulation = config.simulation
    settings = config.runtime_settings
    solver = str(simulation.concentration_solver or "topdown").strip().lower()
    cext = settings.get("cext", {})
    tissue = settings.get("tissue", {})
    simulation_cext = simulation.cext or {}
    cext_mode = (
        str(cext.get("accel_mode", simulation_cext.get("accel_mode", "auto")))
        .strip()
        .lower()
    )
    tissue_mode = (
        str(tissue.get("accel_mode", simulation.tissue_accel or "auto")).strip().lower()
    )
    return solver, cext_mode, tissue_mode


def gpu_requested(config: Any) -> bool:
    """Return whether this run can enter a CuPy-backed solver path."""
    simulation = config.simulation
    if bool(simulation.geometry_only):
        return False

    solver, cext_mode, tissue_mode = _configured_gpu_modes(config)

    flow_mode, transport_mode = _configured_graph_modes(config)
    if flow_mode in {"gpu", "gpu_amg", "auto"} or transport_mode in {"gpu", "auto"}:
        return True

    if solver in {
        "network_ext",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "topdown_ext_treecode",
        "network_ext_hybrid_bg",
        "network_ext_treecode",
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
    flow_mode, transport_mode = _configured_graph_modes(config)
    if flow_mode in {"gpu", "gpu_amg"} or transport_mode == "gpu":
        return True
    if solver in {"network_ext_hybrid_bg", "network_ext_treecode"}:
        return True
    if (
        solver
        in {
            "network_ext",
            "topdown_ext",
            "topdown_ext_hybrid_bg",
            "topdown_ext_treecode",
        }
        and cext_mode == "gpu"
    ):
        return True
    return not bool(simulation.skip_tissue_oxygen) and tissue_mode == "gpu"


def _select_cpu_for_auto_modes(config: Any) -> None:
    """Pin unresolved automatic modes to CPU after a failed CUDA preflight."""
    simulation = config.simulation
    settings = config.runtime_settings
    solver, cext_mode, tissue_mode = _configured_gpu_modes(config)

    flow_mode, transport_mode = _configured_graph_modes(config)
    if flow_mode == "auto":
        settings.setdefault("hemodynamics", {})["kirchhoff_solver"] = "spsolve"
        if "kirchhoff" in settings:
            settings["kirchhoff"]["solver"] = "spsolve"
    if transport_mode == "auto":
        settings.setdefault("oxygen", {})["network_transport_accel"] = "cpu"

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


def _configured_graph_modes(config: Any) -> tuple[str, str]:
    settings = config.runtime_settings
    flow = settings.get("hemodynamics", {})
    kirchhoff = settings.get("kirchhoff", {})
    oxygen = settings.get("oxygen", {})
    flow_mode = kirchhoff.get(
        "solver",
        kirchhoff.get(
            "KIRCHHOFF_SOLVER",
            flow.get("kirchhoff_solver", flow.get("KIRCHHOFF_SOLVER", "auto")),
        ),
    )
    transport_mode = oxygen.get(
        "network_transport_accel", oxygen.get("NETWORK_TRANSPORT_ACCEL", "auto")
    )
    return str(flow_mode).strip().lower(), str(transport_mode).strip().lower()


def _probe_result(succeeded: bool, stdout: str, output: str) -> GpuProbeResult:
    if succeeded:
        marker = next(
            (line for line in stdout.splitlines() if line.startswith(_PROBE_MARKER)),
            _PROBE_MARKER,
        )
        fields = marker.split("|")
        device = fields[1] if len(fields) > 1 else "CUDA device"
        version = fields[2] if len(fields) > 2 else "CuPy ready"
        return GpuProbeResult(True, f"{device} ({version})")

    detail_lines = output.splitlines()
    detail = "\n".join(detail_lines[-12:]) if detail_lines else "Unknown CuPy error"
    if any(marker in output for marker in _BUSY_MARKERS):
        return GpuProbeResult(
            False,
            "GPU is busy or unavailable",
            f"{_BUSY_EXPLANATION}\n\n{detail}",
            busy=True,
        )
    return GpuProbeResult(False, "CuPy cannot compile a CUDA test kernel", detail)


def _probe_in_process() -> GpuProbeResult:
    """Run the probe in this process, which then keeps its CUDA context."""
    import traceback

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(_PROBE_CODE, "<cascade-gpu-probe>", "exec"), {})
    except Exception:
        output = stdout.getvalue().strip() + "\n" + traceback.format_exc()
        return _probe_result(False, stdout.getvalue(), output.strip())
    return _probe_result(True, stdout.getvalue(), stdout.getvalue())


def _probe_must_run_in_process(will_use_gpu: bool) -> bool:
    # A subprocess cannot open a GPU this process holds in Exclusive_Process
    # mode, and one this process is about to use gains nothing from isolation.
    if process_has_cuda_context():
        return True
    return will_use_gpu and any(device.exclusive_process for device in cuda_devices())


def probe_gpu_runtime(
    *, timeout_s: float = 30.0, will_use_gpu: bool = False
) -> GpuProbeResult:
    """Test CUDA, normally in an isolated process so the caller keeps no context.

    The probe runs in this process instead when it already holds a CUDA context,
    or when ``will_use_gpu`` and a device is in Exclusive_Process mode.
    """
    if _probe_must_run_in_process(will_use_gpu):
        return _probe_in_process()
    probe_env = cuda_subprocess_environment()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
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
    return _probe_result(completed.returncode == 0, completed.stdout, output)


def _gpu_install_hint(
    driver_version: int | None, compute_capability: tuple[int, int] | None = None
) -> str:
    install = f"  {sys.executable} -m pip install 'cascade-vascular[{{extra}}]'\n"
    extra = gpu_extra_for_driver(driver_version, compute_capability)
    if driver_version is not None and extra is None:
        if gpu_extra_for_driver(driver_version) is not None:
            sm = "sm_{}{}".format(*compute_capability)
            return (
                f"This GPU ({sm}) is older than every CUDA version CASCADE's GPU "
                "packages support (CUDA 11.8 needs sm_35 or newer). Use "
                "CPU-compatible solver settings or a newer GPU.\n"
            )
        return (
            f"The installed NVIDIA driver supports only CUDA "
            f"{format_cuda_version(driver_version)}, which CASCADE's GPU packages do "
            "not support. Update the NVIDIA driver, then run:\n"
            + install.format(extra="gpu-cu13")
        )
    extra = extra or "gpu-cu13"
    # CuPy variants share one import name, so a different one must go first.
    stale = [
        name
        for name, _version in installed_cupy_distributions()
        if name != f"cupy-cuda{extra_cuda_major(extra)}x"
    ]
    commands = (
        f"  {sys.executable} -m pip uninstall -y {' '.join(stale)}\n" if stale else ""
    ) + install.format(extra=extra)
    if extra == "gpu-cu13":
        return "For this project's CUDA 13 environment, run:\n" + commands
    major = extra_cuda_major(extra)
    newer = family_min_compute_capability(major + 1)
    if compute_capability is not None and tuple(compute_capability) < newer:
        sm = "sm_{}{}".format(*compute_capability)
        return (
            f"This GPU ({sm}) is older than CUDA {major + 1} supports "
            f"(sm_{newer[0]}{newer[1]} or newer), so install the CUDA {major} "
            "packages:\n" + commands
        )
    return (
        "The installed NVIDIA driver supports up to CUDA "
        f"{format_cuda_version(driver_version)}, but the CUDA 13 packages need a "
        "newer driver (580 or later). Either update the driver, or install the "
        f"CUDA {major} packages instead:\n" + commands
    )


def require_gpu_runtime(config: Any) -> GpuProbeResult | None:
    """Validate required CUDA paths and let automatic modes fall back to CPU."""
    global _GPU_PROBE_SUCCESS
    if not gpu_requested(config):
        return None
    if _GPU_PROBE_SUCCESS is not None:
        return _GPU_PROBE_SUCCESS
    configure_cuda_runtime()
    result = probe_gpu_runtime(will_use_gpu=True)
    if result.ready:
        _GPU_PROBE_SUCCESS = result
        return result
    if not _gpu_required(config):
        _select_cpu_for_auto_modes(config)
        reason = " (the GPU is busy)" if result.busy else ""
        print(
            f"GPU auto-detection was unavailable{reason}; using CPU-compatible "
            "solver paths.",
            flush=True,
        )
        return None
    if result.busy:
        raise RuntimeError(
            "GPU preflight failed before vessel generation: the GPU is busy or "
            f"unavailable.\n\n{result.detail}"
        )
    hint = _gpu_install_hint(
        cuda_driver_version(), minimum_compute_capability(cuda_devices())
    )
    raise RuntimeError(
        "GPU preflight failed before vessel generation. This simulation requests a "
        "CUDA backend, but CuPy could not run CASCADE's kernel/FFT preflight.\n\n"
        f"{hint}\n"
        "Then relaunch CASCADE Studio. Alternatively, choose CPU-compatible solver "
        "settings in Advanced solver settings.\n\n"
        f"Technical detail:\n{result.detail}"
    )


__all__ = [
    "GpuProbeResult",
    "cuda_component_library_dirs",
    "cuda_driver_version",
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
