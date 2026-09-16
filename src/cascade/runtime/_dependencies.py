"""Optional scientific-runtime and hardware capability detection.

Scientific modules import ordinary dependencies directly. The shared solver
state imports only the capability flags and optional runtime handles
defined here so availability is detected once per process.
"""

from __future__ import annotations

from importlib import import_module

from cascade.accelerators.cuda_runtime import (
    configure_cuda_runtime,
    configure_cupy_compiler_paths,
)

try:
    from vtkmodules.vtkCommonCore import (
        vtkLogger as _vtkLogger,
    )
    from vtkmodules.vtkCommonCore import vtkObject as _vtkObject
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


configure_cuda_runtime()
try:
    import cupy as _cp

    configure_cupy_compiler_paths(_cp)

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
