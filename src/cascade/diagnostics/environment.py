"""Inspect an installed CASCADE environment and optional GPU capabilities.

This module backs ``cascade doctor`` and ``cascade-doctor``. Use ``--json`` for
machine-readable output, ``--no-gpu-probe`` on CPU-only systems, or
``--require-gpu`` when CUDA readiness is mandatory.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from cascade import __version__
from cascade.accelerators.backend import probe_gpu_runtime
from cascade.accelerators.cuda_driver import (
    CudaDevice,
    cuda_devices,
    cuda_driver_version,
    family_min_compute_capability,
    format_cuda_version,
    gpu_extra_for_driver,
    minimum_compute_capability,
)
from cascade.accelerators.cuda_runtime import (
    configure_cuda_runtime,
    cupy_cuda_major,
    foreign_component_distributions,
    installed_cupy_distributions,
)
from cascade.utils.console import configure_console_error_handling

_DEPENDENCIES = (
    "svv",
    "numpy",
    "scipy",
    "numba",
    "pyvista",
    "vtk",
    "cupy",
    "PySide6",
)


def cuda_variant_issue(
    driver_version: int | None, cupy_distributions: tuple[tuple[str, str], ...]
) -> str | None:
    """Describe an installed CuPy variant that the NVIDIA driver cannot run."""
    pip = f"{sys.executable} -m pip"
    if len(cupy_distributions) > 1:
        names = ", ".join(name for name, _version in cupy_distributions)
        return (
            f"Several CuPy packages are installed ({names}); they overwrite each "
            f"other. Uninstall all of them with\n  {pip} uninstall -y "
            + " ".join(name for name, _version in cupy_distributions)
            + "\nthen reinstall the single extra matching the driver."
        )
    if driver_version is None or not cupy_distributions:
        return None
    name = cupy_distributions[0][0]
    cuda_major = cupy_cuda_major(name)
    if cuda_major is None:
        return None
    # cupy-cuda11x (CuPy 13) needs CUDA 11.2; later majors need their x.0 driver.
    if driver_version >= (11020 if cuda_major == 11 else cuda_major * 1000):
        return None
    driver = format_cuda_version(driver_version)
    extra = gpu_extra_for_driver(driver_version)
    if extra is None:
        return (
            f"{name} needs a CUDA {cuda_major} driver, but the NVIDIA driver "
            f"supports only CUDA {driver}. Update the NVIDIA driver."
        )
    update = " to 580 or later" if cuda_major == 13 else ""
    return (
        f"{name} needs a CUDA {cuda_major} driver, but the NVIDIA driver supports "
        f"up to CUDA {driver}. Update the driver{update}, or switch to the "
        f"matching packages:\n  {pip} uninstall -y {name}\n"
        f"  {pip} install 'cascade-vascular[{extra}]'"
    )


def cuda_device_issue(
    driver_version: int | None,
    devices: tuple[CudaDevice, ...],
    cupy_distributions: tuple[tuple[str, str], ...],
) -> str | None:
    """Describe GPUs older than the installed CuPy family can compile for."""
    if len(cupy_distributions) != 1:
        return None
    name = cupy_distributions[0][0]
    cuda_major = cupy_cuda_major(name)
    oldest = None if cuda_major is None else family_min_compute_capability(cuda_major)
    unsupported = [
        device for device in devices if oldest and device.compute_capability < oldest
    ]
    if not unsupported:
        return None
    listed = ", ".join(f"GPU {device.index} ({device.sm})" for device in unsupported)
    message = (
        f"{name} cannot compile kernels for {listed}: CUDA {cuda_major} supports "
        f"sm_{oldest[0]}{oldest[1]} or newer."
    )
    extra = gpu_extra_for_driver(driver_version, minimum_compute_capability(devices))
    if extra is None:
        return message + " No CASCADE GPU package supports this GPU and driver."
    pip = f"{sys.executable} -m pip"
    return (
        f"{message} Switch to the matching packages:\n  {pip} uninstall -y {name}\n"
        f"  {pip} install 'cascade-vascular[{extra}]'"
    )


def foreign_components_issue(names: tuple[str, ...]) -> str | None:
    """Describe CUDA component wheels left over from another CuPy family."""
    if not names:
        return None
    return (
        "CUDA component wheels from another CUDA family are installed ("
        + ", ".join(names)
        + "). CASCADE loads the installed CuPy's own libraries first, but removing "
        f"the leftovers avoids mixed toolkits:\n  {sys.executable} -m pip "
        "uninstall -y " + " ".join(names)
    )


def collect_diagnostics(*, probe_gpu: bool = True) -> dict[str, Any]:
    # CuPy 13 cannot locate component wheels itself; match the solver import.
    try:
        configure_cuda_runtime()
        cuda_runtime_error = None
    except RuntimeError as exc:
        cuda_runtime_error = str(exc)
    packages: dict[str, dict[str, str | None]] = {}
    for name in _DEPENDENCIES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            packages[name] = {"version": None, "path": None, "error": str(exc)}
            continue
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = None
        packages[name] = {
            "version": None if version is None else str(version),
            "path": str(getattr(module, "__file__", "")),
        }

    driver_version = cuda_driver_version()
    devices = cuda_devices() if driver_version is not None else ()
    cupy_distributions = installed_cupy_distributions()
    foreign = foreign_component_distributions()

    gpu = None
    if probe_gpu:
        result = probe_gpu_runtime()
        gpu = {
            "ready": result.ready,
            "summary": result.summary,
            "detail": result.detail,
        }

    return {
        "cascade_version": __version__,
        "cascade_path": str(Path(__file__).resolve().parents[1]),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_path": os.environ.get("CUDA_PATH"),
        "cuda_driver_version": (
            None if driver_version is None else format_cuda_version(driver_version)
        ),
        "cupy_packages": [
            {"name": name, "version": version} for name, version in cupy_distributions
        ],
        "cuda_devices": [
            {
                "index": device.index,
                "sm": device.sm,
                "compute_mode": device.compute_mode_name,
            }
            for device in devices
        ],
        "cuda_variant_issue": cuda_variant_issue(driver_version, cupy_distributions),
        "cuda_device_issue": cuda_device_issue(
            driver_version, devices, cupy_distributions
        ),
        "foreign_cuda_components": list(foreign),
        "foreign_cuda_components_issue": foreign_components_issue(foreign),
        "cuda_runtime_error": cuda_runtime_error,
        "packages": packages,
        "gpu": gpu,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    configure_console_error_handling()
    parser = argparse.ArgumentParser(
        description="Inspect the installed CASCADE runtime."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    parser.add_argument(
        "--no-gpu-probe", action="store_true", help="Skip the compiled CUDA check."
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Return failure unless CUDA is ready.",
    )
    args = parser.parse_args(argv)
    report = collect_diagnostics(probe_gpu=not args.no_gpu_probe)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"CASCADE {report['cascade_version']} ({report['cascade_path']})")
        print(f"Python {report['python']} ({report['python_executable']})")
        print(f"Platform {report['platform']}")
        for name, info in report["packages"].items():
            status = info.get("version") or "not installed"
            print(f"{name}: {status}")
        driver = report["cuda_driver_version"]
        print(f"NVIDIA driver: {'CUDA ' + driver if driver else 'not found'}")
        for device in report["cuda_devices"]:
            print(
                f"CUDA device {device['index']}: {device['sm']}, "
                f"{device['compute_mode']} compute mode"
            )
        for package in report["cupy_packages"]:
            print(f"CuPy package: {package['name']} {package['version']}")
        for warning in (
            report["cuda_variant_issue"],
            report["cuda_device_issue"],
            report["foreign_cuda_components_issue"],
            report["cuda_runtime_error"],
        ):
            if warning:
                print(f"WARNING: {warning}")
        if report["gpu"] is not None:
            print(f"GPU: {report['gpu']['summary']}")
            if report["gpu"].get("detail") and not report["gpu"]["ready"]:
                print(report["gpu"]["detail"])

    gpu = report["gpu"]
    if args.require_gpu and (gpu is None or not gpu["ready"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
