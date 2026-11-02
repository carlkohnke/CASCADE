"""Query the NVIDIA driver and GPUs, and choose the matching GPU extra.

This module uses only the standard library so installers can run it before
CASCADE or CuPy is installed::

    python -I src/cascade/accelerators/cuda_driver.py

prints the driver's CUDA version (``12040`` for CUDA 12.4, ``0`` without a
driver), the matching extra or ``none``, and the oldest visible GPU
architecture (``sm_70``) or ``none``.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

# Lowest driver CUDA version for each extra. cupy-cuda11x (CuPy 13) needs 11.2.
_GPU_EXTRAS = ((13000, "gpu-cu13"), (12000, "gpu-cu12"), (11020, "gpu-cu11"))
# Oldest GPU architecture each family's NVRTC and libraries target. CUDA 13
# dropped Maxwell, Pascal and Volta (sm_50-sm_72); CUDA 12 dropped Kepler;
# CUDA 11.8 still builds sm_35.
_FAMILY_MIN_COMPUTE = {13: (7, 5), 12: (5, 0), 11: (3, 5)}

_CUDA_SUCCESS = 0
_CUDA_ERROR_NOT_INITIALIZED = 3
_CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
COMPUTE_MODES = {
    0: "Default",
    1: "Exclusive_Thread",
    2: "Prohibited",
    3: "Exclusive_Process",
}
_EXCLUSIVE_PROCESS = 3


@dataclass(frozen=True)
class CudaDevice:
    index: int
    compute_capability: tuple[int, int]
    compute_mode: int

    @property
    def sm(self) -> str:
        return "sm_{}{}".format(*self.compute_capability)

    @property
    def compute_mode_name(self) -> str:
        return COMPUTE_MODES.get(self.compute_mode, str(self.compute_mode))

    @property
    def exclusive_process(self) -> bool:
        return self.compute_mode == _EXCLUSIVE_PROCESS


def _driver_library():
    names = (
        ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    )
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def cuda_driver_version() -> int | None:
    """Return the driver's CUDA version (e.g. 12040), or None without a driver."""
    library = _driver_library()
    if library is None:
        return None
    version = ctypes.c_int(0)
    try:
        status = library.cuDriverGetVersion(ctypes.byref(version))
    except (AttributeError, OSError):
        return None
    return int(version.value) if status == 0 and version.value > 0 else None


def _device_handles(library) -> list[ctypes.c_int]:
    count = ctypes.c_int(0)
    if library.cuDeviceGetCount(ctypes.byref(count)) != _CUDA_SUCCESS:
        return []
    handles = []
    for ordinal in range(count.value):
        device = ctypes.c_int(0)
        if library.cuDeviceGet(ctypes.byref(device), ordinal) == _CUDA_SUCCESS:
            handles.append(device)
    return handles


def cuda_devices() -> tuple[CudaDevice, ...]:
    """Return the visible GPUs' architectures and compute modes.

    This initializes the driver API (``cuInit``) but creates no CUDA context,
    so it is safe before a subprocess or another process opens the GPU.
    """
    library = _driver_library()
    if library is None:
        return ()
    try:
        if library.cuInit(0) != _CUDA_SUCCESS:
            return ()
        devices = []
        for ordinal, device in enumerate(_device_handles(library)):
            values = []
            for attribute in (
                _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                _CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
            ):
                value = ctypes.c_int(0)
                status = library.cuDeviceGetAttribute(
                    ctypes.byref(value), attribute, device
                )
                values.append(value.value if status == _CUDA_SUCCESS else None)
            if None in values:
                continue
            devices.append(CudaDevice(ordinal, (values[0], values[1]), values[2]))
    except (AttributeError, OSError):
        return ()
    return tuple(devices)


def process_has_cuda_context() -> bool:
    """Return whether this process already holds a CUDA context.

    The check never initializes the driver: a process that has not called
    ``cuInit`` cannot own a context.
    """
    library = _driver_library()
    if library is None:
        return False
    try:
        context = ctypes.c_void_p()
        status = library.cuCtxGetCurrent(ctypes.byref(context))
        if status == _CUDA_ERROR_NOT_INITIALIZED:
            return False
        if status == _CUDA_SUCCESS and context.value:
            return True
        # CuPy uses each device's primary context, possibly on another thread.
        for device in _device_handles(library):
            flags = ctypes.c_uint(0)
            active = ctypes.c_int(0)
            status = library.cuDevicePrimaryCtxGetState(
                device, ctypes.byref(flags), ctypes.byref(active)
            )
            if status == _CUDA_SUCCESS and active.value:
                return True
    except (AttributeError, OSError):
        return False
    return False


def format_cuda_version(version: int) -> str:
    """Format a CUDA version integer (12040) as nvidia-smi shows it (12.4)."""
    major, minor = divmod(int(version), 1000)
    return f"{major}.{minor // 10}"


def minimum_compute_capability(
    devices: tuple[CudaDevice, ...],
) -> tuple[int, int] | None:
    """Return the oldest architecture among ``devices``, or None without any."""
    return min((device.compute_capability for device in devices), default=None)


def family_min_compute_capability(cuda_major: int) -> tuple[int, int] | None:
    """Return the oldest GPU architecture a CUDA major version can target."""
    return _FAMILY_MIN_COMPUTE.get(cuda_major)


def extra_cuda_major(extra: str) -> int:
    """Return 12 for ``gpu-cu12``."""
    return int(extra.removeprefix("gpu-cu"))


def gpu_extra_for_driver(
    version: int | None, compute_capability: tuple[int, int] | None = None
) -> str | None:
    """Return the newest GPU extra the driver and every GPU can run, or None.

    ``compute_capability`` is the oldest visible device's; None skips that check.
    """
    if version is None:
        return None
    for minimum, extra in _GPU_EXTRAS:
        if version < minimum:
            continue
        oldest = _FAMILY_MIN_COMPUTE[extra_cuda_major(extra)]
        if compute_capability is None or tuple(compute_capability) >= oldest:
            return extra
    return None


def main() -> int:
    version = cuda_driver_version()
    compute = minimum_compute_capability(cuda_devices()) if version else None
    extra = gpu_extra_for_driver(version, compute)
    sm = "sm_{}{}".format(*compute) if compute else "none"
    print(f"{version or 0} {extra or 'none'} {sm}")
    return 0


__all__ = [
    "COMPUTE_MODES",
    "CudaDevice",
    "cuda_devices",
    "cuda_driver_version",
    "extra_cuda_major",
    "family_min_compute_capability",
    "format_cuda_version",
    "gpu_extra_for_driver",
    "minimum_compute_capability",
    "process_has_cuda_context",
]


if __name__ == "__main__":
    raise SystemExit(main())
