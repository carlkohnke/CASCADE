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


def collect_diagnostics(*, probe_gpu: bool = True) -> dict[str, Any]:
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
