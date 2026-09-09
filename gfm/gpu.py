from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
from typing import Any


_PROBE_MARKER = "GFM_GPU_READY"
_PROBE_CODE = r"""
import cupy as cp

device_count = int(cp.cuda.runtime.getDeviceCount())
if device_count < 1:
    raise RuntimeError("CuPy did not find an NVIDIA CUDA device")

# This intentionally uses an elementwise operation. A device-count check or
# cp.arange alone can succeed even when the CUDA headers needed by NVRTC are
# missing; CASCADE's production kernels would then fail much later.
values = cp.full((4, 4), cp.int32(-1), dtype=cp.int32)
total = int(cp.sum(values).get())
if total != -16:
    raise RuntimeError(f"CUDA test kernel returned {total}, expected -16")

name = cp.cuda.runtime.getDeviceProperties(0)["name"]
if isinstance(name, bytes):
    name = name.decode("utf-8", errors="replace")
print(f"GFM_GPU_READY|{name}|cupy={cp.__version__}", flush=True)
"""


@dataclass(frozen=True)
class GpuProbeResult:
    ready: bool
    summary: str
    detail: str = ""


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
    tissue_mode = str(
        simulation.tissue_accel or tissue.get("accel_mode", "gpu")
    ).strip().lower()

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
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
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
    if not gpu_requested(config):
        return None
    result = probe_gpu_runtime()
    if result.ready:
        return result
    raise RuntimeError(
        "GPU preflight failed before vessel generation. This simulation requests a "
        "CUDA backend, but CuPy could not compile a test kernel.\n\n"
        "For this project's CUDA 13 environment, run:\n"
        f"  {sys.executable} -m pip install 'nvidia-cuda-runtime==13.2.*'\n\n"
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
