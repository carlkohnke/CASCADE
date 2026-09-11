"""Optional scientific-runtime and hardware capability detection.

Scientific modules import ordinary dependencies directly. The shared solver
state imports only the capability flags and optional runtime handles
defined here so availability is detected once per process.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
import sys

try:
    from vtkmodules.vtkCommonCore import (
        vtkLogger as _vtkLogger,
        vtkObject as _vtkObject,
    )
except Exception:  # pragma: no cover
    _vtkLogger = None
    _vtkObject = None
else:
    try:
        # Suppress noisy VTK writer warnings (e.g., legacy vtkDataArray precision warning).
        _vtkObject.GlobalWarningDisplayOff()
        _vtkLogger.SetStderrVerbosity(_vtkLogger.VERBOSITY_ERROR)
    except Exception:
        pass

try:
    import_module("scipy")
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
try:
    import_module("scipy.sparse")
    import_module("scipy.sparse.linalg")
    _scipy_linalg = import_module("scipy.linalg")
    _HAVE_SCIPY_SPARSE = True
except Exception:  # pragma: no cover
    _scipy_linalg = None
    _HAVE_SCIPY_SPARSE = False
try:
    import_module("scipy.spatial")
    _HAVE_SCIPY_SPATIAL = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY_SPATIAL = False
try:
    from scipy import ndimage as _scipy_ndimage

    _HAVE_SCIPY_NDIMAGE = True
except Exception:  # pragma: no cover
    _scipy_ndimage = None
    _HAVE_SCIPY_NDIMAGE = False
try:
    from numba import get_num_threads, set_num_threads

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False
    get_num_threads = None  # type: ignore[assignment]
    set_num_threads = None  # type: ignore[assignment]

_compute_cext_batch_numba = None


def _configure_cupy_cuda_path() -> None:
    """Point CuPy/NVRTC at conda-installed CUDA headers when available."""
    if os.environ.get("CUDA_PATH"):
        return
    candidates = [Path(sys.prefix) / "targets" / "x86_64-linux"]
    source_root = Path(__file__).resolve().parents[3]
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
        local_cuda_path = config_root / ".cascade_cuda_path"
        if local_cuda_path.exists():
            try:
                configured = Path(
                    local_cuda_path.read_text(encoding="utf-8").strip()
                ).expanduser()
            except Exception:
                configured = None
            if configured is not None:
                candidates.append(configured)
    candidates.extend(
        Path(path)
        for path in ("/usr/local/cuda", "/usr/local/cuda-13", "/usr/local/cuda-12")
    )
    for target in candidates:
        if (target / "include" / "cuda_fp16.h").exists():
            os.environ["CUDA_PATH"] = str(target)
            return


_configure_cupy_cuda_path()
try:
    import cupy as _cp

    _HAVE_CUPY = True
except Exception:  # pragma: no cover
    _cp = None  # type: ignore[assignment]
    _HAVE_CUPY = False

try:
    from cascade.domain.svv import Domain
    from cascade.vessels.generation.svv_adapter import Tree
except ModuleNotFoundError as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    Domain = None  # type: ignore[assignment]
    Tree = None  # type: ignore[assignment]
else:
    _IMPORT_ERROR = None


__all__ = [
    "Domain",
    "Tree",
    "_HAVE_CUPY",
    "_HAVE_NUMBA",
    "_HAVE_SCIPY",
    "_HAVE_SCIPY_NDIMAGE",
    "_HAVE_SCIPY_SPARSE",
    "_HAVE_SCIPY_SPATIAL",
    "_IMPORT_ERROR",
    "_compute_cext_batch_numba",
    "_cp",
    "_scipy_linalg",
    "_scipy_ndimage",
    "get_num_threads",
    "set_num_threads",
]
