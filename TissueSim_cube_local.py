"""
Tree flow + Greens concentration script.

Computes Kirchhoff flows on a single tree, then solves intravascular
concentration profiles and samples tissue points using the same Greens
model as forest_test_greens.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import math
import os
import json
from pathlib import Path
import sys
import threading
from time import perf_counter
import time
import traceback
import cProfile
import pstats
import random as _py_random

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from numbers import Number

try:
    from vtkmodules.vtkCommonCore import vtkLogger as _vtkLogger, vtkObject as _vtkObject
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
    profile
except NameError:  # pragma: no cover
    def profile(func):
        return func

try:
    from scipy.special import k0 as _bessel_k0, k1 as _bessel_k1
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _splinalg
    import scipy.linalg as _scipy_linalg
    _HAVE_SCIPY_SPARSE = True
except Exception:  # pragma: no cover
    _scipy_linalg = None
    _HAVE_SCIPY_SPARSE = False
try:
    from scipy.spatial import cKDTree as _cKDTree
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
    from numba import get_num_threads, njit, prange, set_num_threads
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False
    get_num_threads = None  # type: ignore[assignment]
    set_num_threads = None  # type: ignore[assignment]

def _configure_cupy_cuda_path() -> None:
    """Point CuPy/NVRTC at conda-installed CUDA headers when available."""
    if os.environ.get("CUDA_PATH"):
        return
    target = Path(sys.prefix) / "targets" / "x86_64-linux"
    if (target / "include" / "cuda_fp16.h").exists():
        os.environ["CUDA_PATH"] = str(target)


_configure_cupy_cuda_path()
try:
    import cupy as _cp
    _HAVE_CUPY = True
except Exception:  # pragma: no cover
    _cp = None  # type: ignore[assignment]
    _HAVE_CUPY = False

try:
    from gfm.svv_adapter import Domain, Tree
except ModuleNotFoundError as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    Domain = None  # type: ignore[assignment]
    Tree = None  # type: ignore[assignment]
else:
    _IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# (1,    2,    3,    4,    5,    8,    10,    15,    25,   35,    50,    75,      100,    150,    250,    350,   500,    750,    1000,    2000,    5000,    10000,  20000, 50000,   100000, )
N_EQUAL_BIFURCATIONS: int | None = None
EQUAL_TERMINAL_ONLY_ENABLE: bool = True
EQUAL_TERMINAL_BATCH_SIZE: int = 500
EQUAL_TERMINAL_N_CANDIDATES: int = 3
EQUAL_TERMINAL_ALPHA_MIN_DEG: float = 5.0
EQUAL_TERMINAL_ALPHA_MODE_DEG: float = 37.5
EQUAL_TERMINAL_ALPHA_MAX_DEG: float = 85.0
EQUAL_TERMINAL_PSI_STEP_DEG: float = 10.0
EQUAL_TERMINAL_PSI_MAX_DEG: float = 120.0
EQUAL_TERMINAL_DOMAIN_MARGIN: float = 0.0
EQUAL_TERMINAL_CHECK_MIDPOINT: bool = True
EQUAL_TERMINAL_DEFER_HNSW_UPDATES: bool = True
EQUAL_TERMINAL_RECORD_ADD_TIMES: bool = False
EQUAL_TERMINAL_REPORT_TIMINGS: bool = False
EQUAL_TERMINAL_REPORT_EVERY: int = 1
EQUAL_TERMINAL_DOMAIN_WORKERS: int = max(1, (os.cpu_count() or 1) - 2)
EQUAL_TERMINAL_DOMAIN_CHUNK: int = 0
EQUAL_TERMINAL_DOMAIN_PARALLEL_MIN_POINTS: int = 0
EQUAL_TERMINAL_LENGTH: float | None = None
EQUAL_TERMINAL_LENGTH_MODE: str = "gen_f_mix2"
EQUAL_TERMINAL_DENSITY_K: int = 3
EQUAL_TERMINAL_DENSITY_ALPHA: float = -0.17
EQUAL_TERMINAL_DENSITY_BETA: float = 0.6
EQUAL_TERMINAL_DENSITY_EPSILON: float = 0.99
EQUAL_TERMINAL_GEN_F_K: int = 3
EQUAL_TERMINAL_GEN_F_MIX_W: float = 0.19928806732283386
EQUAL_TERMINAL_GEN_F_MIX_MU1_LN: float = -1.4940805101469532
EQUAL_TERMINAL_GEN_F_MIX_SIG1_LN: float = 0.6459130145614477
EQUAL_TERMINAL_GEN_F_MIX_MU2_LN: float = -0.9535671641507659
EQUAL_TERMINAL_GEN_F_MIX_SIG2_LN: float = 0.34151613489071186
EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_INTERCEPT: float = 0.4795
EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_SLOPE: float = 0.688
EQUAL_TERMINAL_LENGTH_SCALE: float = 0.01
EQUAL_TERMINAL_LENGTH_POWER: float = -0.33
EQUAL_TERMINAL_LENGTH_MIN: float = 1e-4
EQUAL_TERMINAL_LENGTH_MAX: float = 1.0
EQUAL_TERMINAL_LENGTH_SHRINK: float = 0.7
EQUAL_BIFURCATION_T_MIN: float = 1e-3
EQUAL_BIFURCATION_RADIUS_FLOOR: float = 1e-4
EQUAL_BIFURCATION_LENGTH_FLOOR: float = 1e-4


TARGET_TERMINAL_COUNTS: tuple[int, ...] = (0,0, 1, 2, 3, 4, 5, 8, 10, 15, 25, 35, 50, 75, 100, 150, 250, 350, 500, 750, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,300000,400000,500000,650000,1000000,2000000,3000000,4000000,5000000)
# (0,0, 1, 2, 3, 4, 5, 8, 10, 15, 25, 35, 50, 75, 100, 150, 250, 350, 500, 750, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,300000,400000,500000,650000,1000000,2000000,3000000,4000000,5000000)


# (0, 1, 2, 3, 4, 5, 8, 10, 15, 25, 35, 50, 75, 100, 150, 250, 350, 500, 750, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,300000,400000,500000,650000,1000000,2000000,3000000,4000000,5000000)
# (0, 1, 2, 3, 4, 5, 8, 10, 15, 25, 35, 50, 75, 100, 150, 250, 350, 500, 750, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,300000,400000,500000)
# TARGET_TERMINAL_COUNTS: tuple[int, ...] = (100000, 200000,300000,400000,500000)

# 1,    2,    3,    4,    5,    8,    10,    15,    25,   35,    50,    75,      100,    150,    250,    350,   500,    750,    1000,    2000,    5000,    10000,  20000,
    # 1,    2,    3,    4,    5,    8,    10,    15,    25,   35,    50,    75,      100,    150,    250,    350,   500,    750,    1000,    2000,    5000,    10000,  20000,    50000,   100000, 
ROOT_LOCATION = np.array([[0.49, -0.49, -0.49]])  # scaled by cube side length at runtime
ROOT_DIR = np.array([[-0.49, 0.49, 0.49]])
# OUTPUT_CSV_NAME = "Cube_EXT_q_considers_Cext_Vmax_04_300kplus.csv"
OUTPUT_CSV_NAME = "Cube_testing.csv"
NONDIMENSIONAL_NUMBERS = False
NONDIMENSIONAL_CSV_NAME = "Cube_nondimensional_numbers.csv"
FLUID = "both"  # analysis mode: "water", "blood", or "both"
BUILD_FLUID = "blood"
ACTIVE_FLUID = FLUID

DISTANCE_SAMPLE_COUNT = 1000000
DISTANCE_CHUNK_SIZE = 256
COMPUTE_AVG_DISTANCE_TO_CHANNEL = False

# Large-tree distance computations (DNC / distance-to-channel stats) must not form
# (chunk_size x n_segments) dense distance matrices. Use KDTree candidates for big trees.
DISTANCE_USE_KDTREE_FOR_LARGE_TREES = True
DISTANCE_BRUTE_FORCE_MAX_SEGMENTS = 20_000
DISTANCE_KDTREE_MIN_CANDIDATES = 64
DISTANCE_KDTREE_CANDIDATE_MULT = 8
DISTANCE_KDTREE_MAX_CANDIDATES = 4096
# ROOT_PRESSURE = 133322.37
# TERMINAL_PRESSURE = 133200
# ROOT_PRESSURE = 113324.
# TERMINAL_PRESSURE = 133200
# TERMINAL_PRESSURE = 123660.0
# TERMINAL_PRESSURE = 40000.0
ROOT_PRESSURE = 66661.0
TERMINAL_PRESSURE = 40000.0
QIN_TARGET = 900.0  # uL/min target inlet flow rate
DLP_ANGLE_VALUES: Sequence[float] | float = (0,)
CUBE_SIDE_LENGTHS: Sequence[float] | float = (1.0,)
SCALE_NTERMS_BY_VOLUME = False
SCALE_Q_BY_VOLUME = True
SCALE_dP_BY_VOLUME = True
TRIALS_PER_COMBO = 1
BUILD_ON_PREVIOUS = True if TRIALS_PER_COMBO == 1 else False
CHECKER_PLOT = False
CHECKER_LINE_RESOLUTION = 24
CHECKER_TUBE_SIDES = 18
NEAREST_TISSUE_VESSELS = 250 ### was 1000 !!!
WINDOW_FACTOR = 6 ### was 8 !!!
TISSUE_KDTREE_CANDIDATE_MULT = 2
TISSUE_PARALLEL_WORKERS = max(os.cpu_count() or 1, 1)
CONC_USE_NUMBA = True
TISSUE_USE_NUMBA = True
TISSUE_STREAMING_ENABLED = False
TISSUE_STREAMING_MIN_POINTS = 100_000
TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS = 500_000
TISSUE_STREAMING_MIN_CHUNK_POINTS = 256
TISSUE_STREAMING_MAX_CHUNK_POINTS = 2048
TISSUE_STREAMING_PRUNE_BY_WINDOW = True
TISSUE_STREAMING_LOG_EVERY_CHUNKS = 25
TISSUE_STREAMING_CHUNK_WORKERS = max(os.cpu_count()-2 or 1, 1)
TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER = 1
SOLVER_TIMING_DETAILS = True
TISSUE_CACHE_CHUNK_SIZE = 512
TISSUE_KDTREE_WORKERS = max((os.cpu_count() or 1) - 2, 1)
TISSUE_ACCEL_MODE = "gpu"  # "cpu", "gpu", or "auto".
TISSUE_GPU_CHUNK_POINTS = 8192
TISSUE_GPU_MIN_CHUNK_POINTS = 512
TISSUE_GPU_VALIDATE_POINTS = 0
TISSUE_CEXT_CELL_LIST_ENABLE = True
TISSUE_CEXT_CELL_TARGET_OCCUPANCY = 4
TISSUE_CEXT_CELL_MIN_GRID = 16
TISSUE_CEXT_CELL_MAX_GRID = 256
TISSUE_CEXT_CELL_MAX_RAD_CELLS = 8
TISSUE_MIN_SEGMENT_LENGTH_SI = 1.0e-12
_LAST_TISSUE_TIMINGS: dict[str, float | str] = {}
_LAST_CONCENTRATION_TIMINGS: dict[str, float | str | int | list[float]] = {}
_LAST_CEXT_SOURCE_STATE: dict | None = None
_LAST_CEXT_FROZEN_STEP_TIMINGS: dict[str, float | str] = {}
CEXT_TRACE_ITERATIONS: set[int] | None = None
CEXT_TRACE_CALLBACK = None
CEXT_TRACE_TISSUE_ENABLED = False
CEXT_TRACE_TISSUE_POINTS = None
CEXT_TRACE_TISSUE_CACHE = None

CEXT_ACCEL_MODE = "gpu"  # "cpu", "gpu", or "auto".
CEXT_FROZEN_ACCEL_MODE = "gpu"  # "cpu", "gpu", or "auto".
CEXT_INIT_MODE = "decoupled_greens"  # "zero" or "decoupled_greens".
CEXT_LAMBDA_SOURCE = os.environ.get("SVV_CEXT_LAMBDA_SOURCE", "lambda_t").strip().lower()
CEXT_WINDOW_FACTOR = WINDOW_FACTOR
CEXT_VESS_COUPLING_MAX_ITER = 5  # Local-only screened Cext needs the safer converged default.
CEXT_VESS_COUPLING_TOL = 1.0e-3
CEXT_VESS_COUPLING_OMEGA = 1.0
CEXT_VESS_COUPLING_REL_TOL = 0.0
CEXT_VESS_COUPLING_ACCEL = "anderson"  # "none", "aitken", or "anderson".
CEXT_VESS_COUPLING_OMEGA_MIN = 0.025
CEXT_VESS_COUPLING_OMEGA_MAX = 1.4
CEXT_VESS_COUPLING_TRUST_ABS = 1.0e-3
CEXT_VESS_COUPLING_TRUST_REL = 0.4
CEXT_VESS_COUPLING_ANDERSON_DEPTH = 4
CEXT_VESS_COUPLING_ANDERSON_REG = 1.0e-10
CEXT_VESS_COUPLING_ANDERSON_START = 2
CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO = 0.9
CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS = 2
CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR = 1.5
CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR = 0.9
CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR = 2.0
CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR = 1.1
CEXT_VESS_COUPLING_BEST_STALL_ITERS = 20
CEXT_VESS_COUPLING_BEST_REVERT_FACTOR = 1.02
CEXT_VESS_COUPLING_STEP_REJECT_FACTOR = 1.02
CEXT_VESS_COUPLING_STEP_RETRY_FACTOR = 0.5
CEXT_ACTIVE_SET_ENABLE = True
CEXT_ACTIVE_SET_START = 6
CEXT_ACTIVE_SET_STABLE_ITERS = 3
CEXT_ACTIVE_SET_REL_TOL = 5.0e-3
CEXT_ACTIVE_SET_ABS_TOL = 2.5e-4
CEXT_ACTIVE_SET_REFRESH_PERIOD = 8
CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT = 1024
CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION = 0.0
CEXT_TARGET_ACTIVE_SET_ENABLE = True
CEXT_TARGET_ACTIVE_SET_START = 6
CEXT_TARGET_ACTIVE_SET_STABLE_ITERS = 3
CEXT_TARGET_ACTIVE_SET_REL_TOL = 5.0e-3
CEXT_TARGET_ACTIVE_SET_ABS_TOL = 2.5e-4
CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT = 1024
CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD = 1
CEXT_HYBRID_BG_GRID = 256
CEXT_HYBRID_BG_LAMBDA_BINS = 5
CEXT_HYBRID_BG_NEAR_RADIUS_MULT = 0.0  # Previous default: 4.0.
CEXT_HYBRID_BG_VCYCLES = 2
CEXT_HYBRID_BG_ASSIGNMENT = "tsc"  # "cic" or "tsc"
CEXT_HYBRID_BG_SOLVER = "auto"  # "auto", "fft", or "jacobi"
CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING = False
# Best 500k diagnostic FFT path: dynamic quantile lambda bins, TSC source deposit,
# CIC target sampling, finite-radius O2 correction, and grid-consistent self-subtraction.
CEXT_HYBRID_BG_MODE = os.environ.get("SVV_CEXT_HYBRID_BG_MODE", "fft").strip().lower()
CEXT_HYBRID_FFT_QUANTILE_BINS = str(os.environ.get("SVV_CEXT_HYBRID_FFT_QUANTILE_BINS", "true")).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_O2_CORRECTION = str(os.environ.get("SVV_CEXT_HYBRID_FFT_O2_CORRECTION", "true")).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_SELF_SUBTRACT = str(os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUBTRACT", "true")).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING = os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING", "cic").strip().lower()
CEXT_HYBRID_FFT_SELF_SUB_SCALE = float(os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_SCALE", "1.0"))
CEXT_HYBRID_FFT_BIN_EPOCH_CACHE = str(os.environ.get("SVV_CEXT_HYBRID_FFT_BIN_EPOCH_CACHE", "true")).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_RESPONSE_BATCHED = str(os.environ.get("SVV_CEXT_HYBRID_FFT_RESPONSE_BATCHED", "false")).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_O2_FUSED_IFFT = str(os.environ.get("SVV_CEXT_HYBRID_FFT_O2_FUSED_IFFT", "true")).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_SELF_SUB_FUSED = str(os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_FUSED", "false")).strip().lower() in ("1", "true", "yes", "on")
try:
    CEXT_HYBRID_FFT_O2_MOMENT_BATCH = max(int(os.environ.get("SVV_CEXT_HYBRID_FFT_O2_MOMENT_BATCH", "1")), 1)
except ValueError:
    CEXT_HYBRID_FFT_O2_MOMENT_BATCH = 1
CEXT_HYBRID_GPU_ITERATION_CACHE = str(os.environ.get("SVV_CEXT_HYBRID_GPU_ITERATION_CACHE", "false")).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_GPU_RUNTIME_WEIGHTS = str(os.environ.get("SVV_CEXT_HYBRID_GPU_RUNTIME_WEIGHTS", "true")).strip().lower() in ("1", "true", "yes", "on")
CEXT_LOCAL_ONLY_FAST_SELF = True
CEXT_TREECODE_THETA = 0.5
CEXT_TREECODE_ORDER = 1
CEXT_TREECODE_LEAF_NODES = 128
CEXT_TREECODE_LAMBDA_BINS = 8
CEXT_TREECODE_NEAR_RADIUS_MULT = 4.0
CEXT_TREECODE_GPU_LOCAL = True
CEXT_TAIL_SOLVER = "active_core_nk"  # "none" or "active_core_nk"
CEXT_TAIL_TRIGGER_START_ITER = 8
CEXT_TAIL_TRIGGER_STALL_ITERS = 6
CEXT_TAIL_TRIGGER_ACTIVE_COUNT = 4096
CEXT_TAIL_MAX_NONLINEAR_ITERS = 6
CEXT_TAIL_GMRES_RESTART = 32
CEXT_TAIL_GMRES_MAXITER = 96
CEXT_TAIL_TRIGGER_IMPROVEMENT_RATIO = 0.8
CEXT_TAIL_REBOUND_REL = 0.9
CEXT_TAIL_REBOUND_ARM_REL = 0.4
CEXT_TAIL_CORE_REL_THRESHOLD = 5.0e-3
CEXT_TAIL_CORE_ABS_THRESHOLD = 2.5e-4
CEXT_TREECODE_FREEZE_QREL_TOL = 2.5e-2
CEXT_GRID_CELL_FACTOR = 1.0
CEXT_STREAMING_TARGET_CANDIDATE_SLOTS = 500_000
CEXT_MAX_CELLS_PER_SEG = 128
CEXT_GPU_VALIDATE_SEGMENTS = 0
CEXT_APPROX_WINDOW_SCALE = 1.0
CEXT_MAX_CANDIDATES_PER_TARGET = 100
CEXT_PRECOMPUTE_WORKERS = max(1, os.cpu_count()-2 or 1)
CEXT_LOCAL_EXCLUDE_HOPS = 2
VESS_CONC_FLOOR = 1.0e-12
CEXT_FLOAT_DTYPE = np.float32
CEXT_INDEX_DTYPE = np.int32

# Expanded oxygen transport controls.
# c_iv_gl remains the cup/bulk lumen concentration for compatibility with the
# original topdown solver.  c_wall_gl is the true wall concentration used by
# q_line and by the finite-radius dipole source coefficient.
FINITE_RADIUS_O2_TERMS = str(os.environ.get("SVV_FINITE_RADIUS_O2_TERMS", "both")).strip().lower()
LUMEN_WALL_CLOSURE = str(os.environ.get("SVV_LUMEN_WALL_CLOSURE", "graetz")).strip().lower()
GRAETZ_N_RADIAL = int(os.environ.get("SVV_GRAETZ_N_RADIAL", "6"))
GRAETZ_N_MODES = int(os.environ.get("SVV_GRAETZ_N_MODES", "3"))
GRAETZ_MAX_FP_ITERS = int(os.environ.get("SVV_GRAETZ_MAX_FP_ITERS", "4"))
GRAETZ_FP_TOL = float(os.environ.get("SVV_GRAETZ_FP_TOL", "1e-5"))
GRAETZ_VELOCITY_PROFILE = str(os.environ.get("SVV_GRAETZ_VELOCITY_PROFILE", "poiseuille")).strip().lower()
GRAETZ_BI_CACHE_PER_DECADE = int(os.environ.get("SVV_GRAETZ_BI_CACHE_PER_DECADE", "16"))
GRAETZ_MIN_BI = float(os.environ.get("SVV_GRAETZ_MIN_BI", "1e-8"))
GRAETZ_MAX_BI = float(os.environ.get("SVV_GRAETZ_MAX_BI", "1e6"))
GRAETZ_DEBUG_DIAGNOSTICS = str(os.environ.get("SVV_GRAETZ_DEBUG_DIAGNOSTICS", "0")).strip().lower() in ("1", "true", "yes", "on")
LUMEN_DIFFUSIVITY_WATER_CM2_S = float(os.environ.get("SVV_LUMEN_DIFFUSIVITY_WATER_CM2_S", "3.2e-5"))
LUMEN_DIFFUSIVITY_BLOOD_CM2_S = float(os.environ.get("SVV_LUMEN_DIFFUSIVITY_BLOOD_CM2_S", "2.41e-5"))
LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S = (
    float(os.environ["SVV_LUMEN_DIFFUSIVITY_CM2_S"]) if "SVV_LUMEN_DIFFUSIVITY_CM2_S" in os.environ else None
)
LUMEN_DIFFUSIVITY_CM2_S = (
    float(LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S)
    if LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S is not None
    else LUMEN_DIFFUSIVITY_BLOOD_CM2_S
)
_GRAETZ_BASIS_TABLE_CACHE: dict[tuple, dict] = {}


def _lumen_diffusivity_cm2_s_for_fluid(fluid_mode: str | None = None) -> float:
    """Return lumen oxygen diffusivity; explicit CLI/env override still wins."""
    if LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S is not None:
        return float(LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S)
    mode = str(fluid_mode or "").strip().lower()
    if mode == "blood":
        return float(LUMEN_DIFFUSIVITY_BLOOD_CM2_S)
    return float(LUMEN_DIFFUSIVITY_WATER_CM2_S)

# Tissue cache storage dtype: using float32/int32 can significantly reduce peak memory for large runs.
# Set to True to force float64 everywhere in the tissue cache.
TISSUE_CACHE_FORCE_FLOAT64 = False
TISSUE_CACHE_FLOAT_DTYPE = np.float32
TISSUE_CACHE_INDEX_DTYPE = np.int32


def _normalize_tree_float_dtype(value: object | None) -> str:
    if value is None:
        return "float64"
    s = str(value).strip().lower()
    if s in ("float32", "f32", "32"):
        return "float32"
    if s in ("float64", "f64", "64"):
        return "float64"
    try:
        dt = np.dtype(value)
        return "float32" if dt == np.float32 else "float64"
    except Exception:
        return "float64"


def _normalize_tree_int_dtype(value: object | None) -> str:
    if value is None:
        return "int64"
    s = str(value).strip().lower()
    if s in ("int32", "i32", "32"):
        return "int32"
    if s in ("int64", "i64", "64"):
        return "int64"
    try:
        dt = np.dtype(value)
        return "int32" if dt == np.int32 else "int64"
    except Exception:
        return "int64"


TREE_DATA_DTYPE_STR = _normalize_tree_float_dtype(os.environ.get("SVV_TREE_DATA_DTYPE", "float64"))
TREE_INDEX_DTYPE_STR = _normalize_tree_int_dtype(os.environ.get("SVV_TREE_INDEX_DTYPE", "int64"))
TREE_DATA_DTYPE = np.float32 if TREE_DATA_DTYPE_STR == "float32" else np.float64
TREE_INDEX_DTYPE = np.int32 if TREE_INDEX_DTYPE_STR == "int32" else np.int64
TREE_FAST_CACHE_ENABLE = False

DEBUG_ADD_VESSEL = False
SAVE_TREES = False
USE_TREE_CACHE = True
TREE_CACHE_DIRNAME = "trees_cache"
TREE_CACHE_INDEX_NAME = "tree_index.csv"

VMAX_MM = 2e-16 * 2.2e13
VMAX_MM = 0.04
# VMAX_MM = 3.85e-17*6e+13  # mol/(m^3*s) (Michaelis-Menten Vmax for cell media depletion)
K_M_MM = 0.0069
# K_M_MM = 0.035954   # mol/m^3 (Michaelis-Menten constant)
AXIAL_BLOOD_STEPS = 5
OMEGA = 0.7

CONCENTRATION_SOLVER = "topdown_ext_hybrid_bg"  # topdown, network, topdown_ext, topdown_ext_hybrid_bg, or topdown_ext_treecode
HEMATOCRIT_MODEL = "pries_secomb"  # "uniform_tube" or "pries_secomb".
HEMATOCRIT_FLOW_ITERATIONS = 2
HEMATOCRIT_RELAXATION = 1.0
HEMATOCRIT_QTOL_NL_MIN = 1.0e-3
HEMATOCRIT_HDTOL = 1.0e-3
HEMATOCRIT_MIN = 0.0
HEMATOCRIT_MAX = 0.95
HEMATOCRIT_DIAGNOSTICS = True
NETFLOW_BIFPAR_1 = 0.964
NETFLOW_BIFPAR_2 = 6.98
NETFLOW_BIFPAR_3 = -13.29
NETFLOW_CPAR_1 = 0.80
NETFLOW_CPAR_2 = -0.075
NETFLOW_CPAR_3 = -11.0
NETFLOW_CPAR_4 = 12.0
NETFLOW_VISCPAR_1 = 6.0
NETFLOW_VISCPAR_2 = -0.085
NETFLOW_VISCPAR_3 = 3.2
NETFLOW_VISCPAR_4 = -2.44
NETFLOW_VISCPAR_5 = -0.06
NETFLOW_VISCPAR_6 = 0.645
NETFLOW_OPTW_UM = 1.1
NETFLOW_VPLAS_CP = 1.0466
NETFLOW_MCV_FL = 55.0
NETFLOW_MCV_CORR = (92.0 / NETFLOW_MCV_FL) ** (1.0 / 3.0)

# Sparse Kirchhoff solver selection.
# Options: "auto" | "cg" | "spsolve" | "gmres_ilu".
KIRCHHOFF_SOLVER = "tree"  # "tree" or one of the sparse solver options below.
KIRCHHOFF_BC_MODE = "legacy_equal_terminal_flow"  # "terminal_pressure" or "legacy_equal_terminal_flow".
KIRCHHOFF_VALIDATE_TREE = False
KIRCHHOFF_VALIDATE_SPARSE_SOLVER = "spsolve"
KIRCHHOFF_SPARSE_SOLVER = "spsolve"
KIRCHHOFF_CG_MIN_NODES = 50_000
KIRCHHOFF_CG_RTOL = 1e-10
KIRCHHOFF_CG_MAXITER = 10_000
KIRCHHOFF_GMRES_RTOL = 1e-4
KIRCHHOFF_GMRES_MAXITER = 1_000
KIRCHHOFF_GMRES_RESTART = 300
KIRCHHOFF_ILU_DROP_TOL = 1e-7
KIRCHHOFF_ILU_FILL_FACTOR = 300
# If enabled, solve S*A*S*y=S*b with S=diag(1/sqrt(diag(A))) then recover x=S*y.
# This can improve GMRES behavior on highly scaled Laplacians.
KIRCHHOFF_GMRES_EQUILIBRATE = True
KIRCHHOFF_GMRES_EQ_DIAG_FLOOR_REL = 1e-12
KIRCHHOFF_ILU_PERMC_SPECS: tuple[str, ...] = ("COLAMD", "MMD_AT_PLUS_A", "NATURAL")
KIRCHHOFF_ILU_SHIFT_RELS: tuple[float, ...] = (0.0, 1e-14, 1e-12)
KIRCHHOFF_GMRES_RETRY_UNSCALED_IF_EQ_FAIL = True
KIRCHHOFF_DIAGNOSTICS = True

CM_TO_M = 0.01
CM2_TO_M2 = 1e-4
CM3_TO_M3 = 1e-6
CM_TO_UM = 1e4
PA_TO_DYN_PER_CM2 = 10.0
DYN_PER_CM2_TO_PA = 0.1
# Legacy conversion for literature values quoted as mL O2 / dL blood.
# Do not apply this to O2_CAP_PER_HCT below; that value is already mol / m^3.
O2_ML_PER_DL_TO_MOL_PER_M3_STP = (1e-3 / 22.414) / 1e-4

CONC_MAX_FOR_NORMALIZATION = .14

EXTRAVASCULAR_CONCENTRATION = 0.0
# SOLUTE_DIFFUSIVITY = (3.2e-5) * (0.9 ** (4 / 3))
SOLUTE_DIFFUSIVITY = 2.41e-5
# TISSUE_DECAY_LENGTH = 0.2
POROSITY = 0.9
GL_ORDER = 5  # Gauss-Legendre points per segment for Greens integral (5, 9, or 20)
GL_ORDER_CEXT = 1  # Gauss-Legendre points per segment for explicit vessel Cext coupling

HD_DISCHARGE = 0.42
# Hemoglobin-bound O2 capacity in mol / m^3 blood per unit tube hematocrit.
# Therefore Chb_max = HT * O2_CAP_PER_HCT is already in mol / m^3.
O2_CAP_PER_HCT = 20.3
ALPHA_MMHG = 1.408e-3
P50_MMHG = 26.5
N_HILL = 2.7


def _diagnostic_stats(name: str, values: np.ndarray) -> str:
    arr = np.asarray(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{name}=empty"
    q01, q50, q99 = np.quantile(arr, [0.01, 0.5, 0.99])
    return (
        f"{name}[min={float(arr.min()):.3e} p01={float(q01):.3e} "
        f"p50={float(q50):.3e} p99={float(q99):.3e} max={float(arr.max()):.3e}]"
    )


def get_analysis_fluids() -> tuple[str, ...]:
    mode = str(FLUID).lower()
    if mode == "both":
        return ("water", "blood")
    if mode in ("water", "blood"):
        return (mode,)
    raise ValueError('FLUID must be "water", "blood", or "both".')


def get_concentration_inlet(fluid: str | None = None) -> float:
    mode = str(fluid or ACTIVE_FLUID).lower()
    if mode == "both":
        raise ValueError("Active fluid must be 'water' or 'blood'.")
    try:
        return float(CONCENTRATION_INLET_BY_FLUID[mode])
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown fluid mode: {mode}") from exc


def _process_tissue_chunk(start_idx: int, end_idx: int, data: dict) -> tuple[int, np.ndarray]:

    points_si = data["points_si"]
    starts_si = data["starts_si"]
    segment_vectors = data["segment_vectors"]
    seg_len = data["seg_len"]
    radii_si = data["radii_si"]
    nearest_idx = data["nearest_idx"]
    proj_raw = data["proj_raw"]
    d_center = data["d_center"]
    keep_mask = data["keep_mask"]
    cin_pos = data["cin_pos"]
    alpha_edge = data["alpha_edge"]
    flow_sign = data["flow_sign"]
    diffusivity_si = data["diffusivity_si"]
    km = data["km"]
    vmax = data["vmax"]
    window_factor = data["window_factor"]
    lam_ref = data["lam_ref"]
    gl_nodes = data["gl_nodes"]
    gl_weights = data["gl_weights"]

    # ramp_params = (.33,3.1,0.,1.57,0.,.443) # min_r, max_r, cs_min, cr_min, cs_max, cr_max
    # min_r_ratio=ramp_params[0],
    # max_r_ratio=ramp_params[1],
    # cs_mult_min=ramp_params[2],
    # cr_mult_min=ramp_params[3],
    # cs_mult_max=ramp_params[4],
    # cr_mult_max=ramp_params[5],

    chunk = points_si[start_idx:end_idx]
    if chunk.size == 0:
        return start_idx, np.zeros((0,), dtype=float)
    idx_local = nearest_idx[start_idx:end_idx]
    proj_local = proj_raw[start_idx:end_idx]
    d_local = d_center[start_idx:end_idx]
    safe_idx_local = np.where(idx_local >= 0, idx_local, 0)
    proj_flow = np.where(flow_sign[safe_idx_local] < 0.0, 1.0 - proj_local, proj_local)
    proj_flow = np.clip(proj_flow, 0.0, 1.0)
    out = np.zeros((len(chunk),), dtype=float)

    for row, point in enumerate(chunk):
        if not keep_mask[start_idx + row]:
            continue
        seg_idx = idx_local[row]
        if seg_idx.size == 0:
            continue
        proj_row = proj_flow[row]
        cap_max = 1e-6
        total = 0.0
        for local_i, seg_i in enumerate(seg_idx):
            if seg_i < 0:
                continue
            L = seg_len[seg_i]
            if L <= 0.0:
                continue
            s_star = float(proj_row[local_i] * L)
            Cc_star = cin_pos[seg_i] * float(np.exp(-alpha_edge[seg_i] * s_star))
            denom_gate = max(km + max(Cc_star, 1e-12), 1e-30)
            lam_gate = float(np.sqrt(diffusivity_si / max(vmax / denom_gate, 1e-30)))
            if d_local[row, local_i] > window_factor * lam_gate:
                continue
            if Cc_star > cap_max:
                cap_max = Cc_star

            halfW = window_factor * lam_ref
            s0 = max(0.0, s_star - halfW)
            s1 = min(L, s_star + halfW)
            if s1 <= s0 + 1e-15:
                continue

            mid = 0.5 * (s0 + s1)
            half = 0.5 * (s1 - s0)
            s = mid + half * gl_nodes
            t = s / L
            xs = starts_si[seg_i] + t[:, None] * segment_vectors[seg_i]
            r_vec = point - xs
            r = np.linalg.norm(r_vec, axis=1)
            valid_r = r > 1e-12
            if not np.any(valid_r):
                continue

            cc_s = cin_pos[seg_i] * np.exp(-alpha_edge[seg_i] * s)
            r_safe = np.where(valid_r, r, 1.0)
            # denom_loc = np.maximum(km + np.maximum(cc_s / 2.0, 1e-12), 1e-30)
            denom_loc = np.maximum(km + np.maximum(cc_s, 1e-12), 1e-30)

            lam_loc = np.sqrt(diffusivity_si / np.maximum(vmax / denom_loc, 1e-30))
            phi_loc = np.maximum(radii_si[seg_i] / np.maximum(lam_loc, 1e-30), 1e-12)
            r_over_lam = r_safe / np.maximum(lam_loc, 1e-30)
            k0_num = _k0_lookup(r_over_lam)
            k0_den = _k0_lookup(phi_loc)
            Ci_R = cc_s * (k0_num / np.maximum(k0_den, 1e-300))
            denom_corr = np.maximum(km + np.minimum(Ci_R, cc_s), 1e-30)



            # ramp_span = max(max_r_ratio - min_r_ratio, 1e-12)
            # t_ramp = np.clip((r_over_lam - min_r_ratio) / ramp_span, 0.0, 1.0)
            # c_low = cs_mult_min * cc_s + np.minimum(cr_mult_min * Ci_R,cc_s)
            # c_high = cs_mult_max * cc_s + np.minimum( cr_mult_max * Ci_R,cc_s)
            # c_eff = (1.0 - t_ramp) * c_low + t_ramp * c_high
            # denom_corr = np.maximum(km + np.maximum(c_eff, 1e-12), 1e-30)


            lam_corr = np.sqrt(diffusivity_si / np.maximum(vmax / denom_corr, 1e-30))
            phi = np.maximum(radii_si[seg_i] / np.maximum(lam_corr, 1e-30), 1e-12)
            ratio = _k_ratio(phi)
            wall_factor = (diffusivity_si / np.maximum(lam_corr, 1e-30)) * ratio
            q_s = (2.0 * np.pi * radii_si[seg_i]) * wall_factor * cc_s
            kernel = np.exp(-r_safe / np.maximum(lam_corr, 1e-30)) / (4.0 * np.pi * r_safe)
            integrand = q_s * (kernel / diffusivity_si)
            integrand[~valid_r] = 0.0

            total += half * float(np.sum(gl_weights * integrand))

        if total < 0.0 or not np.isfinite(total):
            total = 0.0
        if cap_max > -np.inf:
            out[row] = min(total, cap_max)
        else:
            out[row] = total
    return start_idx, out


def _prepare_tissue_geometry(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    *,
    max_nearby: int,
) -> dict:
    t_total = perf_counter()
    t0 = perf_counter()
    cache_float = np.float64 if TISSUE_CACHE_FORCE_FLOAT64 else TISSUE_CACHE_FLOAT_DTYPE
    points_si = (points * CM_TO_M).astype(cache_float, copy=False)
    starts_si = (starts * CM_TO_M).astype(cache_float, copy=False)
    ends_si = (ends * CM_TO_M).astype(cache_float, copy=False)
    radii_si = (radii * CM_TO_M).astype(cache_float, copy=False)

    segment_vectors = ends_si - starts_si
    seg_len_sq = np.sum(segment_vectors**2, axis=1)
    min_len_sq = float(TISSUE_MIN_SEGMENT_LENGTH_SI) * float(TISSUE_MIN_SEGMENT_LENGTH_SI)
    valid = np.isfinite(seg_len_sq) & (seg_len_sq > min_len_sq)
    starts_si = starts_si[valid]
    ends_si = ends_si[valid]
    radii_si = radii_si[valid]
    segment_vectors = segment_vectors[valid]
    seg_len_sq = seg_len_sq[valid]
    seg_len = np.sqrt(seg_len_sq)
    t_prepare = perf_counter() - t0
    if starts_si.size == 0:
        return {
            "valid_mask": valid,
            "points_si": points_si,
            "starts_si": starts_si,
            "ends_si": ends_si,
            "radii_si": radii_si,
            "segment_vectors": segment_vectors,
            "seg_len_sq": seg_len_sq,
            "seg_len": seg_len,
            "nearest_idx": np.empty((points.shape[0], 0), dtype=int),
            "proj_raw": np.empty((points.shape[0], 0), dtype=float),
            "d_center": np.empty((points.shape[0], 0), dtype=float),
            "keep_mask": np.zeros((points.shape[0],), dtype=bool),
        }

    nseg = starts_si.shape[0]
    candidate_k = min(nseg, max(max_nearby, max_nearby * TISSUE_KDTREE_CANDIDATE_MULT))
    max_nearby = min(max_nearby, candidate_k, nseg)
    if max_nearby <= 0:
        return {
            "valid_mask": valid,
            "points_si": points_si,
            "starts_si": starts_si,
            "ends_si": ends_si,
            "radii_si": radii_si,
            "segment_vectors": segment_vectors,
            "seg_len_sq": seg_len_sq,
            "seg_len": seg_len,
            "nearest_idx": np.empty((points.shape[0], 0), dtype=int),
            "proj_raw": np.empty((points.shape[0], 0), dtype=float),
            "d_center": np.empty((points.shape[0], 0), dtype=float),
            "keep_mask": np.ones((points.shape[0],), dtype=bool),
        }

    if _HAVE_SCIPY_SPATIAL:
        t0 = perf_counter()
        midpoints = 0.5 * (starts_si + ends_si)
        tree = _cKDTree(midpoints)
        t_kdtree = perf_counter() - t0
    else:
        tree = None
        t_kdtree = 0.0

    nearest_idx = np.zeros((points.shape[0], max_nearby), dtype=TISSUE_CACHE_INDEX_DTYPE)
    proj_raw = np.zeros((points.shape[0], max_nearby), dtype=cache_float)
    d_center = np.zeros((points.shape[0], max_nearby), dtype=cache_float)
    keep_mask = np.ones((points.shape[0],), dtype=bool)

    chunk_size = max(int(TISSUE_CACHE_CHUNK_SIZE), 1)
    t_query = 0.0
    t_refine = 0.0
    t_inside = 0.0
    for idx in range(0, len(points_si), chunk_size):
        chunk = points_si[idx: idx + chunk_size]
        if chunk.size == 0:
            continue
        if tree is not None:
            t0 = perf_counter()
            _, cand = _ckdtree_query(tree, chunk, k=candidate_k)
            t_query += perf_counter() - t0
            if cand.ndim == 1:
                cand = cand[:, None]
        else:
            cand = np.tile(np.arange(nseg, dtype=int), (len(chunk), 1))

        t0 = perf_counter()
        starts_c = starts_si[cand]
        seg_c = segment_vectors[cand]
        seg_len_sq_c = seg_len_sq[cand]
        diff = chunk[:, None, :] - starts_c
        proj = np.sum(diff * seg_c, axis=2) / seg_len_sq_c
        proj_clipped = np.clip(proj, 0.0, 1.0)
        closest = starts_c + proj_clipped[:, :, None] * seg_c
        dist = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
        sel = np.argpartition(dist, kth=max_nearby - 1, axis=1)[:, :max_nearby]
        nearest_idx[idx: idx + len(chunk)] = np.take_along_axis(cand, sel, axis=1)
        proj_raw[idx: idx + len(chunk)] = np.take_along_axis(proj, sel, axis=1)
        d_center[idx: idx + len(chunk)] = np.take_along_axis(dist, sel, axis=1)
        t_refine += perf_counter() - t0

        t0 = perf_counter()
        radius_local = np.minimum(
            radii_si[nearest_idx[idx: idx + len(chunk)]],
            np.sqrt(seg_len_sq[nearest_idx[idx: idx + len(chunk)]]),
        )
        proj_sel = proj_raw[idx: idx + len(chunk)]
        inside_any = np.any(
            (proj_sel >= 0.0) & (proj_sel <= 1.0) & (d_center[idx: idx + len(chunk)] <= radius_local),
            axis=1,
        )
        keep_mask[idx: idx + len(chunk)] = ~inside_any
        t_inside += perf_counter() - t0

    if SOLVER_TIMING_DETAILS:
        n_chunks = int(math.ceil(len(points_si) / max(chunk_size, 1)))
        print(
            "  Tissue geometry prep: "
            f"points={len(points_si)} nseg={nseg} candidate_k={candidate_k} keep_k={max_nearby} "
            f"chunk_points={chunk_size} chunks={n_chunks} kdtree_workers={TISSUE_KDTREE_WORKERS} "
            f"prepare={_fmt_seconds(t_prepare)} kdtree={_fmt_seconds(t_kdtree)} "
            f"query={_fmt_seconds(t_query)} refine={_fmt_seconds(t_refine)} "
            f"inside_mask={_fmt_seconds(t_inside)} total={_fmt_seconds(perf_counter() - t_total)}"
        )

    return {
        "valid_mask": valid,
        "points_si": points_si,
        "starts_si": starts_si,
        "ends_si": ends_si,
        "radii_si": radii_si,
        "segment_vectors": segment_vectors,
        "seg_len_sq": seg_len_sq,
        "seg_len": seg_len,
        "nearest_idx": nearest_idx,
        "proj_raw": proj_raw,
        "d_center": d_center,
        "keep_mask": keep_mask,
    }


def _build_tissue_geometry_context(
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
) -> dict:
    cache_float = np.float64 if TISSUE_CACHE_FORCE_FLOAT64 else TISSUE_CACHE_FLOAT_DTYPE
    starts_si_all = (starts * CM_TO_M).astype(cache_float, copy=False)
    ends_si_all = (ends * CM_TO_M).astype(cache_float, copy=False)
    radii_si_all = (radii * CM_TO_M).astype(cache_float, copy=False)

    segment_vectors_all = ends_si_all - starts_si_all
    seg_len_sq_all = np.sum(segment_vectors_all**2, axis=1)
    min_len_sq = float(TISSUE_MIN_SEGMENT_LENGTH_SI) * float(TISSUE_MIN_SEGMENT_LENGTH_SI)
    valid = np.isfinite(seg_len_sq_all) & (seg_len_sq_all > min_len_sq)
    starts_si = starts_si_all[valid]
    ends_si = ends_si_all[valid]
    radii_si = radii_si_all[valid]
    segment_vectors = segment_vectors_all[valid]
    seg_len_sq = seg_len_sq_all[valid]
    seg_len = np.sqrt(seg_len_sq)

    return {
        "valid_mask": valid,
        "starts_si": starts_si,
        "ends_si": ends_si,
        "radii_si": radii_si,
        "segment_vectors": segment_vectors,
        "seg_len_sq": seg_len_sq,
        "seg_len": seg_len,
        "tree": None,
        "cache_float": cache_float,
    }


def _ensure_tissue_context_kdtree(context: dict):
    tree = context.get("tree")
    if tree is None and _HAVE_SCIPY_SPATIAL and int(context["starts_si"].shape[0]) > 0:
        with _TISSUE_CONTEXT_KDTREE_LOCK:
            tree = context.get("tree")
            if tree is None:
                t0 = perf_counter()
                midpoints = 0.5 * (context["starts_si"] + context["ends_si"])
                tree = _cKDTree(midpoints)
                context["tree"] = tree
                print(f"  Tissue KDTree build: {_fmt_seconds(perf_counter() - t0)}")
    return tree


def _prepare_tissue_geometry_from_context(
    points: np.ndarray,
    context: dict,
    *,
    max_nearby: int,
) -> dict:
    cache_float = context.get("cache_float", np.float64 if TISSUE_CACHE_FORCE_FLOAT64 else TISSUE_CACHE_FLOAT_DTYPE)
    points_si = (points * CM_TO_M).astype(cache_float, copy=False)
    starts_si = context["starts_si"]
    ends_si = context["ends_si"]
    radii_si = context["radii_si"]
    segment_vectors = context["segment_vectors"]
    seg_len_sq = context["seg_len_sq"]
    seg_len = context["seg_len"]
    valid = context["valid_mask"]
    nseg = int(starts_si.shape[0])

    if nseg == 0:
        return {
            "valid_mask": valid,
            "points_si": points_si,
            "starts_si": starts_si,
            "ends_si": ends_si,
            "radii_si": radii_si,
            "segment_vectors": segment_vectors,
            "seg_len_sq": seg_len_sq,
            "seg_len": seg_len,
            "nearest_idx": np.empty((points.shape[0], 0), dtype=TISSUE_CACHE_INDEX_DTYPE),
            "proj_raw": np.empty((points.shape[0], 0), dtype=cache_float),
            "d_center": np.empty((points.shape[0], 0), dtype=cache_float),
            "keep_mask": np.zeros((points.shape[0],), dtype=bool),
        }

    candidate_k = min(nseg, max(max_nearby, max_nearby * TISSUE_KDTREE_CANDIDATE_MULT))
    max_nearby = min(max_nearby, candidate_k, nseg)
    if max_nearby <= 0:
        return {
            "valid_mask": valid,
            "points_si": points_si,
            "starts_si": starts_si,
            "ends_si": ends_si,
            "radii_si": radii_si,
            "segment_vectors": segment_vectors,
            "seg_len_sq": seg_len_sq,
            "seg_len": seg_len,
            "nearest_idx": np.empty((points.shape[0], 0), dtype=TISSUE_CACHE_INDEX_DTYPE),
            "proj_raw": np.empty((points.shape[0], 0), dtype=cache_float),
            "d_center": np.empty((points.shape[0], 0), dtype=cache_float),
            "keep_mask": np.ones((points.shape[0],), dtype=bool),
        }

    tree = _ensure_tissue_context_kdtree(context)
    nearest_idx = np.zeros((points.shape[0], max_nearby), dtype=TISSUE_CACHE_INDEX_DTYPE)
    proj_raw = np.zeros((points.shape[0], max_nearby), dtype=cache_float)
    d_center = np.zeros((points.shape[0], max_nearby), dtype=cache_float)
    keep_mask = np.ones((points.shape[0],), dtype=bool)

    chunk_size = max(int(TISSUE_CACHE_CHUNK_SIZE), 1)
    for idx in range(0, len(points_si), chunk_size):
        chunk = points_si[idx: idx + chunk_size]
        if chunk.size == 0:
            continue
        if tree is not None:
            _, cand = _ckdtree_query(tree, chunk, k=candidate_k)
            if cand.ndim == 1:
                cand = cand[:, None]
        else:
            cand = np.tile(np.arange(nseg, dtype=int), (len(chunk), 1))

        starts_c = starts_si[cand]
        seg_c = segment_vectors[cand]
        seg_len_sq_c = seg_len_sq[cand]
        diff = chunk[:, None, :] - starts_c
        proj = np.sum(diff * seg_c, axis=2) / seg_len_sq_c
        proj_clipped = np.clip(proj, 0.0, 1.0)
        closest = starts_c + proj_clipped[:, :, None] * seg_c
        dist = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
        sel = np.argpartition(dist, kth=max_nearby - 1, axis=1)[:, :max_nearby]
        nearest_idx[idx: idx + len(chunk)] = np.take_along_axis(cand, sel, axis=1)
        proj_raw[idx: idx + len(chunk)] = np.take_along_axis(proj, sel, axis=1)
        d_center[idx: idx + len(chunk)] = np.take_along_axis(dist, sel, axis=1)

        radius_local = np.minimum(
            radii_si[nearest_idx[idx: idx + len(chunk)]],
            np.sqrt(seg_len_sq[nearest_idx[idx: idx + len(chunk)]]),
        )
        proj_sel = proj_raw[idx: idx + len(chunk)]
        inside_any = np.any(
            (proj_sel >= 0.0) & (proj_sel <= 1.0) & (d_center[idx: idx + len(chunk)] <= radius_local),
            axis=1,
        )
        keep_mask[idx: idx + len(chunk)] = ~inside_any

    return {
        "valid_mask": valid,
        "points_si": points_si,
        "starts_si": starts_si,
        "ends_si": ends_si,
        "radii_si": radii_si,
        "segment_vectors": segment_vectors,
        "seg_len_sq": seg_len_sq,
        "seg_len": seg_len,
        "nearest_idx": nearest_idx,
        "proj_raw": proj_raw,
        "d_center": d_center,
        "keep_mask": keep_mask,
    }


def _compact_tissue_cache_by_influence(tissue_cache: dict, influence_radius: np.ndarray) -> dict:
    nearest_idx = tissue_cache["nearest_idx"]
    if nearest_idx.size == 0:
        return tissue_cache

    safe_idx = np.where(nearest_idx >= 0, nearest_idx, 0)
    valid = (nearest_idx >= 0) & (tissue_cache["d_center"] <= influence_radius[safe_idx])
    counts = np.sum(valid, axis=1)
    max_count = int(np.max(counts)) if counts.size else 0
    if max_count >= nearest_idx.shape[1] and np.all(valid):
        return tissue_cache

    out = dict(tissue_cache)
    idx_compact = np.full((nearest_idx.shape[0], max_count), -1, dtype=nearest_idx.dtype)
    proj_compact = np.zeros((nearest_idx.shape[0], max_count), dtype=tissue_cache["proj_raw"].dtype)
    d_compact = np.zeros((nearest_idx.shape[0], max_count), dtype=tissue_cache["d_center"].dtype)
    for row in range(nearest_idx.shape[0]):
        if counts[row] <= 0:
            continue
        cols = np.flatnonzero(valid[row])
        n = int(cols.size)
        idx_compact[row, :n] = nearest_idx[row, cols]
        proj_compact[row, :n] = tissue_cache["proj_raw"][row, cols]
        d_compact[row, :n] = tissue_cache["d_center"][row, cols]
    out["nearest_idx"] = idx_compact
    out["proj_raw"] = proj_compact
    out["d_center"] = d_compact
    return out


def _streaming_tissue_chunk_size(max_nearby: int) -> int:
    k = max(int(max_nearby), 1)
    target_slots = max(int(TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS), k)
    chunk = max(int(TISSUE_STREAMING_MIN_CHUNK_POINTS), target_slots // k)
    return int(min(max(chunk, 1), int(TISSUE_STREAMING_MAX_CHUNK_POINTS)))


def _compute_streaming_tissue_chunk(task: tuple[int, int, int, dict]) -> tuple[int, int, int, np.ndarray, np.ndarray, int, int, int]:
    chunk_i, start_idx, end_idx, state = task
    context = state["context"]
    points = state["points"]
    max_nearby = int(state["max_nearby"])
    influence_radius = state["influence_radius"]

    chunk_cache = _prepare_tissue_geometry_from_context(
        points[start_idx:end_idx],
        context,
        max_nearby=max_nearby,
    )
    before_candidates = int(np.count_nonzero(chunk_cache["nearest_idx"] >= 0))
    if state["prune_by_window"]:
        chunk_cache = _compact_tissue_cache_by_influence(chunk_cache, influence_radius)
    after_candidates = int(np.count_nonzero(chunk_cache["nearest_idx"] >= 0))

    out = None
    if state["used_numba"]:
        try:
            out = _tissue_kernel_numba(
                chunk_cache["points_si"],
                state["starts_si"],
                state["segment_vectors"],
                state["seg_len"],
                state["radii_si"],
                chunk_cache["nearest_idx"],
                chunk_cache["proj_raw"],
                chunk_cache["d_center"],
                chunk_cache["keep_mask"],
                state["cin_pos"],
                state["alpha_edge"],
                state["flow_sign"],
                float(state["diffusivity_si"]),
                float(state["km"]),
                float(state["vmax"]),
                float(state["window_factor"]),
                float(state["lam_ref"]),
                state["gl_nodes"],
                state["gl_weights"],
                _KRATIO_XS,
                _K0_LUT,
            )
        except Exception:
            print("WARNING: Numba streaming tissue kernel failed for one chunk; falling back to non-numba chunk.")
            traceback.print_exc()
            out = None

    if out is None:
        worker_data = {
            "points_si": chunk_cache["points_si"],
            "starts_si": state["starts_si"],
            "segment_vectors": state["segment_vectors"],
            "seg_len": state["seg_len"],
            "radii_si": state["radii_si"],
            "nearest_idx": chunk_cache["nearest_idx"],
            "proj_raw": chunk_cache["proj_raw"],
            "d_center": chunk_cache["d_center"],
            "keep_mask": chunk_cache["keep_mask"],
            "cin_pos": state["cin_pos"],
            "alpha_edge": state["alpha_edge"],
            "flow_sign": state["flow_sign"],
            "diffusivity_si": state["diffusivity_si"],
            "km": state["km"],
            "vmax": state["vmax"],
            "window_factor": state["window_factor"],
            "lam_ref": state["lam_ref"],
            "gl_nodes": state["gl_nodes"],
            "gl_weights": state["gl_weights"],
        }
        ranges = [
            (i, min(i + DISTANCE_CHUNK_SIZE, end_idx - start_idx))
            for i in range(0, end_idx - start_idx, DISTANCE_CHUNK_SIZE)
        ]
        out = np.zeros((end_idx - start_idx,), dtype=float)
        for r in ranges:
            local_start, local_out = _process_tissue_chunk(*r, worker_data)
            out[local_start: local_start + len(local_out)] = local_out

    keep_mask = chunk_cache["keep_mask"]
    kept = int(np.count_nonzero(keep_mask))
    return chunk_i, start_idx, end_idx, keep_mask, np.asarray(out, dtype=float), kept, before_candidates, after_candidates


CONCENTRATION_INLET_BY_FLUID = {
    "water": 0.2211 * 0.75,
    "blood": 0.14,
}



SAVE_RESULTS = False
RESULTS_FILE = "tree_greens_results.npz"
RESULTS_POINTS_CSV = "tree_greens_results.csv"
IMPORT_POINTS_CSV = "Profiles_NewTest30.csv"
USE_VALIDATION_CSV = False
VALIDATION_CSV_PATH = IMPORT_POINTS_CSV

PLOT_FLOW = False
PLOT_CONC_VESSELS = False
PLOT_CONC_VESSELS_POINTS = False
PLOT_MB_CONVERGENCE = False
PLOT_VALIDATION_SCATTER = False
PLOT_VIABILITY_POINTS = False
PLOT_HISTOGRAM = False
HISTOGRAM_NBINS = 32
HISTOGRAM_ALPHA = 0.7
GAUSSIAN_PLOT_XMAX_UM = 10000.0
GAUSSIAN_PLOT_NPTS = 1000
GAUSSIAN_LOG_XMIN_UM = 1
GAUSSIAN_COLORMAP_LOG10 = True
SAVE_DNC_GAUSSIAN_FITS = False
VIOLIN_DNC = False
VIOLIN_CSV_NAME = "dnc_moredP_cont.csv"
VIOLIN_BIN_WIDTH_UM = 10.0
VIOLIN_MAX_UM = 10000.0
VIOLIN_MIN_KEEP_UM = 0.1
PLOT_WINDOW_SIZE = (1088, 765)
PLOT_WINDOW_POSITION = (50, 50)
PLOT_ZOOM = 0.9
PLOT_TUBE_SIDES = 18
PLOT_LINE_RESOLUTION = 24
PLOT_POINT_SIZE = 14
POINT_OPACITY = 1.0
USE_TRICOLOR_CMAP = False
PLOT_CMAP = "jet"

TRICOLOR_CMAP = None  # initialized lazily to avoid matplotlib dependency
KRATIO_LUT_PATH = Path(__file__).with_name("besselks.npz")

CSV_FIELDNAMES = [
    "number_of_trees",
    "target_terminals",
    "dlp_angle",
    "cube_side_length",
    "fluid",
    "concentration_solver",
    "finite_radius_o2_terms",
    "lumen_wall_closure",
    "graetz_n_radial",
    "graetz_n_modes",
    "cext_accel_mode",
    "pressure_in_root",
    "pressure_out_terminals",
    "concentration_inlet",
    "extravascular_concentration",
    "solute_diffusivity",
    # "tissue_decay_length",
    "qin_target_uL_per_min",
    "distance_sample_count",
]
AGGREGATED_METRICS = [
    "total_volume",
    "total_flowrate",
    "avg_radius",
    "avg_length",
    "total_length",
    "avg_distance_to_channel",
    "total_segments",
    "terminal_segments",
    "Rnet",
    "dRnet",
    "Qmin_over_Qinlet",
    "C_LQ_over_Cmax",
    "C_tiss_over_Cmax",
    "Damkohler",
    "t_load_s",
    "t_assembly_s",
    "t_kirchhoff_s",
    "t_concentration_s",
    "t_cext_total_s",
    "t_cext_query_s",
    "t_cext_kernel_s",
    "t_cext_gpu_transfer_s",
    "t_cext_hybrid_deposit_s",
    "t_cext_hybrid_fft_s",
    "t_cext_hybrid_o2_terms_s",
    "t_cext_hybrid_self_subtract_s",
    "t_cext_hybrid_local_corr_s",
    "t_cext_hybrid_sample_s",
    "t_tissue_s",
    "t_tissue_geometry_s",
    "t_tissue_total_s",
    "t_tissue_kdtree_query_s",
    "t_tissue_gpu_refine_s",
    "t_tissue_gpu_transfer_s",
    "FracAbove50pct",
    "FracAbove25pct",
    "FracAbove10pct",
    "FracAbove5pct",
    "FracAbove1pct",
    "pressure_drop",
    "inlet_flow_ul_per_min",
    "radius_max",
    "radius_min",
]
for metric in AGGREGATED_METRICS:
    CSV_FIELDNAMES.extend([f"{metric}_mean", f"{metric}_std"])
CSV_FIELDNAMES.extend(["cext_concentration_mean", "cext_concentration_std"])

DNC_FIT_FIELDNAMES = [
    "dnc_gaussian_mu_um",
    "dnc_gaussian_sigma_um",
]
CSV_FIELDNAMES.extend(DNC_FIT_FIELDNAMES)

PI_PHI_GAMMA_FIELDNAMES = [
    "C0_plasma",
    "C_Hb_bound",
    "C0_eff",
    "Q0_inlet_m3_s",
    "Vmax",
    "Km",
    "Cstar",
    "D_m2_s",
    "Vt_m3",
    "l_d_m",
    "lambda_reaction_diffusion_m",
    "Pi2",
    "Phi2",
    "Gamma",
    "test4",
]
CSV_FIELDNAMES.extend(PI_PHI_GAMMA_FIELDNAMES)

TEST4_PARAMS = (
    2007.6397317172216,
    -2.638294847873327,
    -2.902324538409277,
    -1.15173409506264,
    244.11689279412852,
    2.124490859545908,
    -1.6571662897669492,
    -0.6379214646370855,
    0.07216793256660264,
    4.9628430042544665,
    -0.08512071658666168,
    -0.19360829893347317,
    -0.3943956301736953,
)

NONDIMENSIONAL_FIELDNAMES = [
    "run_name",
    "number_of_trees",
    "target_terminals",
    "total_segments",
    "dlp_angle",
    "cube_side_length_m",
    "fluid",
    "C0_plasma",
    "C_Hb_bound",
    "C0_eff",
    "Q0_inlet_m3_s",
    "Vmax",
    "Km",
    "Cstar",
    "Vt_m3",
    "l_d_m",
    "D_m2_s",
    "lambda_m",
    "avg_radius_m",
    "avg_length_m",
    "FracAbove5pct",
    "FracAbove1pct",
    "hematocrit_est",
    "severinghaus_saturation",
    "Pi",
    "Phi",
    "Da",
]

TREE_CACHE_FIELDNAMES = [
    "config_id",
    "target_terminals",
    "target_raw",
    "tree_path",
    "tree_data_dtype",
    "tree_index_dtype",
    "created_utc",
    "cube_side_length",
    "root_pressure",
    "terminal_pressure",
    "qin_target_ul_min",
    "inlet_flow_cm3_s",
    "terminal_flow_cm3_s",
    "scale_nterms_by_volume",
    "scale_q_by_volume",
    "scale_dp_by_volume",
    "build_fluid",
    "build_fluid_density",
    "build_kinematic_viscosity",
    "dlp_enable",
    "dlp_min_angle_deg",
    "dlp_min_adv",
    "dlp_build_dir",
    "dlp_parent_below",
    "dlp_build_cap_frac",
    "dlp_early_slab_frac",
    "dlp_early_quota_frac",
    "domain_shape",
    "domain_random_seed",
    "root_location",
    "root_direction",
    "root_location_scaled",
    "weighted_sampling",
    "ignore_collisions",
    "allow_inside_vessels",
    "n_closest_vessels",
    "n_points",
    "murray_exponent",
    "radius_exponent",
    "length_exponent",
    "max_nonconvex_count",
    "config_json",
]


# ---------------------------------------------------------------------------
# Debug helper (disabled; no-op to silence debug output)
# ---------------------------------------------------------------------------
def _dbg(msg: str) -> None:
    return


def _fmt_seconds(value: float | int | None) -> str:
    if value is None or not (isinstance(value, (int, float))):
        return "n/a"
    return f"{value:.3f}s"


def _ckdtree_query(tree, points: np.ndarray, *, k: int):
    workers = int(TISSUE_KDTREE_WORKERS)
    try:
        return tree.query(points, k=k, workers=workers)
    except TypeError:
        return tree.query(points, k=k)


def _cupy_device_available() -> bool:
    if not _HAVE_CUPY or _cp is None:
        return False
    try:
        return int(_cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _resolve_tissue_accel_mode(*, require_gpu: bool = False) -> str:
    mode = str(TISSUE_ACCEL_MODE or "cpu").strip().lower()
    if mode not in ("cpu", "gpu", "auto"):
        raise ValueError("--tissue-accel must be 'cpu', 'gpu', or 'auto'.")
    if mode == "cpu":
        return "cpu"
    if _cupy_device_available():
        return "gpu"
    if mode == "gpu" or require_gpu:
        raise RuntimeError(
            "Tissue GPU mode requested, but CuPy/CUDA is not available. "
            "Verify `import cupy` and `cupy.cuda.runtime.getDeviceCount()` in the svva2 environment."
        )
    if SOLVER_TIMING_DETAILS:
        print("WARNING: tissue-accel=auto requested, but CuPy/CUDA is unavailable; using CPU tissue path.")
    return "cpu"


# ---------------------------------------------------------------------------
# Progress tracking (for silent exits)
# ---------------------------------------------------------------------------
_LAST_PROGRESS: dict[str, object] = {
    "stage": "startup",
    "run_name": None,
    "output_csv": None,
    "side_len": None,
    "theta": None,
    "target": None,
    "trial": None,
    "fluid": None,
}


def _set_progress(stage: str, **kwargs: object) -> None:
    _LAST_PROGRESS["stage"] = stage
    for key, val in kwargs.items():
        if key in _LAST_PROGRESS:
            _LAST_PROGRESS[key] = val


def _progress_snapshot() -> str:
    items = []
    for key in ("stage", "run_name", "output_csv", "side_len", "theta", "target", "trial", "fluid"):
        val = _LAST_PROGRESS.get(key)
        if val is not None:
            items.append(f"{key}={val}")
    return ", ".join(items) if items else "no progress info"


def _write_trial_error_log(exc: BaseException, context: str) -> None:
    try:
        path = Path(__file__).with_name("TissueSim_trial_errors.log")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n---\n")
            handle.write(_progress_snapshot() + "\n")
            handle.write(context + "\n")
            handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


def _write_crash_log(exc: BaseException) -> None:
    try:
        path = Path(__file__).with_name("TissueSim_crash.log")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n---\n")
            handle.write(_progress_snapshot() + "\n")
            handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


def _require_domain_class() -> None:
    if Domain is None:
        raise ModuleNotFoundError(
            "svv.domain.domain could not be imported. "
            "Ensure the compiled extensions are built before running cdtest."
        ) from _IMPORT_ERROR


def _require_tree_class() -> None:
    if Tree is None:
        raise ModuleNotFoundError(
            "svv.tree.tree could not be imported. "
            "Ensure the compiled extensions are built before running cdtest."
        ) from _IMPORT_ERROR


# ---------------------------------------------------------------------------
# Domain + tree helpers
# ---------------------------------------------------------------------------
def build_domain(side_length: float = 1.0) -> Domain:
    _require_domain_class()
    side = float(side_length)
    cube = pv.Cube(x_length=side, y_length=side, z_length=side)
    domain = Domain(cube)
    domain.random_seed = 42
    np.random.seed(int(domain.random_seed))
    _py_random.seed(int(domain.random_seed))
    domain.create()
    domain.solve()
    domain.build()
    domain.set_random_generator()
    return domain


def grow_tree(
    domain: Domain,
    vessels_to_add: int,
    *,
    dlp_enable: bool,
    min_theta: float,
    side_length: Optional[float] = None,
    terminal_flow_override: Optional[float] = None,
    fluid: str | None = None,
    n_closest_vessels: int = 5,
    n_points: int = 7,
    weighted_sampling: bool = False,
    ignore_collisions: bool = False,
    allow_inside_vessels: bool = False,
) -> Tree:
    _require_tree_class()
    tree = Tree(data_dtype=TREE_DATA_DTYPE, index_dtype=TREE_INDEX_DTYPE)
    tree.set_domain(domain)
    tree.parameters.root_pressure = ROOT_PRESSURE
    if SCALE_dP_BY_VOLUME and side_length is not None:
        tree.parameters.terminal_pressure = ROOT_PRESSURE - (abs(ROOT_PRESSURE - TERMINAL_PRESSURE)) * (side_length**3)
        if tree.parameters.terminal_pressure < 0:
            raise ValueError("Terminal pressure is negative after scaling; aborting run.")
    else:
        tree.parameters.terminal_pressure = TERMINAL_PRESSURE
    fluid_mode = (fluid or ACTIVE_FLUID).lower()
    try:
        tree.parameters.fluid = fluid_mode
    except Exception:
        tree.fluid = fluid_mode
    if fluid_mode in {"water", "cell media", "media"}:
        tree.parameters.kinematic_viscosity = 0.6959 / 100
        tree.parameters.fluid_density = 0.99336
    elif fluid_mode == "blood":
        tree.parameters.fluid_density = 1.06
        mu_plasma_cgs = 0.012
        tree.parameters.kinematic_viscosity = mu_plasma_cgs / tree.parameters.fluid_density
    n_vessels = max(int(vessels_to_add), 1)
    total_terminals = max(n_vessels + 1, 1)
    if terminal_flow_override is not None:
        tree.parameters.terminal_flow = terminal_flow_override
    else:
        tree.parameters.terminal_flow = QIN_TARGET * 0.00001666666666 / total_terminals
    # tree.set_dlp(
    #     enable=dlp_enable,
    #     build_dir=(-1, 1, 1),
    #     min_adv=0.0001,
    #     min_angle_deg=min_theta,
    #     parent_below=True,
    #     build_cap_frac=None,
    # )
    scale = float(side_length) if side_length is not None else float(getattr(domain, "characteristic_length", 1.0))
    root_loc = ROOT_LOCATION * scale
    if ROOT_DIR is not None:
        tree.set_root(root_loc, ROOT_DIR)
    else:
        tree.set_root(root_loc)
    _apply_equal_bifurcation(tree)
    tree.n_add(
        n_vessels,
        n_closest_vessels=n_closest_vessels,
        n_points=n_points,
        use_random_int=not weighted_sampling,
        ignore_collisions=ignore_collisions,
        allow_inside_vessels=allow_inside_vessels,
        debug_add_vessel=DEBUG_ADD_VESSEL,
    )
    return tree


def _apply_equal_bifurcation(tree: Tree) -> None:
    if tree is None:
        return
    if N_EQUAL_BIFURCATIONS is None:
        return
    try:
        tree.parameters.n_equal_bifurcations = int(N_EQUAL_BIFURCATIONS)
        tree.parameters.equal_bifurcation_radius_floor = float(EQUAL_BIFURCATION_RADIUS_FLOOR)
        tree.parameters.equal_bifurcation_length_floor = float(EQUAL_BIFURCATION_LENGTH_FLOOR)

        # Heart-style terminal-only equal-bif method.
        tree.parameters.equal_bifurcation_terminal_only = bool(EQUAL_TERMINAL_ONLY_ENABLE)
        tree.parameters.equal_terminal_batch_size = int(EQUAL_TERMINAL_BATCH_SIZE)
        tree.parameters.equal_terminal_n_candidates = int(EQUAL_TERMINAL_N_CANDIDATES)
        tree.parameters.equal_terminal_alpha_min_deg = float(EQUAL_TERMINAL_ALPHA_MIN_DEG)
        tree.parameters.equal_terminal_alpha_mode_deg = float(EQUAL_TERMINAL_ALPHA_MODE_DEG)
        tree.parameters.equal_terminal_alpha_max_deg = float(EQUAL_TERMINAL_ALPHA_MAX_DEG)
        tree.parameters.equal_terminal_psi_step_deg = float(EQUAL_TERMINAL_PSI_STEP_DEG)
        tree.parameters.equal_terminal_psi_max_deg = float(EQUAL_TERMINAL_PSI_MAX_DEG)
        tree.parameters.equal_terminal_domain_margin = float(EQUAL_TERMINAL_DOMAIN_MARGIN)
        tree.parameters.equal_terminal_check_midpoint = bool(EQUAL_TERMINAL_CHECK_MIDPOINT)
        tree.parameters.equal_terminal_domain_workers = int(EQUAL_TERMINAL_DOMAIN_WORKERS)
        tree.parameters.equal_terminal_domain_chunk = int(EQUAL_TERMINAL_DOMAIN_CHUNK)
        tree.parameters.equal_terminal_domain_parallel_min_points = int(EQUAL_TERMINAL_DOMAIN_PARALLEL_MIN_POINTS)
        tree.parameters.equal_terminal_length = EQUAL_TERMINAL_LENGTH
        tree.parameters.equal_terminal_length_mode = str(EQUAL_TERMINAL_LENGTH_MODE or "power")
        tree.parameters.equal_terminal_total_length_log10_intercept = float(EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_INTERCEPT)
        tree.parameters.equal_terminal_total_length_log10_slope = float(EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_SLOPE)
        tree.parameters.equal_terminal_length_scale = float(EQUAL_TERMINAL_LENGTH_SCALE)
        tree.parameters.equal_terminal_length_power = float(EQUAL_TERMINAL_LENGTH_POWER)
        tree.parameters.equal_terminal_length_min = float(EQUAL_TERMINAL_LENGTH_MIN)
        tree.parameters.equal_terminal_length_max = float(EQUAL_TERMINAL_LENGTH_MAX)
        tree.parameters.equal_terminal_length_shrink = float(EQUAL_TERMINAL_LENGTH_SHRINK)
        tree.parameters.equal_terminal_density_k = int(EQUAL_TERMINAL_DENSITY_K)
        tree.parameters.equal_terminal_density_alpha = float(EQUAL_TERMINAL_DENSITY_ALPHA)
        tree.parameters.equal_terminal_density_beta = float(EQUAL_TERMINAL_DENSITY_BETA)
        tree.parameters.equal_terminal_density_epsilon = float(EQUAL_TERMINAL_DENSITY_EPSILON)
        tree.parameters.equal_terminal_gen_f_k = int(EQUAL_TERMINAL_GEN_F_K)
        tree.parameters.equal_terminal_gen_f_mix_w = float(EQUAL_TERMINAL_GEN_F_MIX_W)
        tree.parameters.equal_terminal_gen_f_mix_mu1_ln = float(EQUAL_TERMINAL_GEN_F_MIX_MU1_LN)
        tree.parameters.equal_terminal_gen_f_mix_sig1_ln = float(EQUAL_TERMINAL_GEN_F_MIX_SIG1_LN)
        tree.parameters.equal_terminal_gen_f_mix_mu2_ln = float(EQUAL_TERMINAL_GEN_F_MIX_MU2_LN)
        tree.parameters.equal_terminal_gen_f_mix_sig2_ln = float(EQUAL_TERMINAL_GEN_F_MIX_SIG2_LN)
        tree.parameters.equal_terminal_report_timings = bool(EQUAL_TERMINAL_REPORT_TIMINGS)
        tree.parameters.equal_terminal_report_every = int(EQUAL_TERMINAL_REPORT_EVERY)
    except Exception:
        return


def _apply_run_tree_params(tree, side_length: float | None, terminal_flow_override: float | None) -> None:
    if tree is None:
        return
    # Ensure build-related parameters reflect the current run.
    tree.parameters.root_pressure = ROOT_PRESSURE
    if SCALE_dP_BY_VOLUME and side_length is not None:
        tree.parameters.terminal_pressure = ROOT_PRESSURE - (abs(ROOT_PRESSURE - TERMINAL_PRESSURE)) * (side_length**3)
        if tree.parameters.terminal_pressure < 0:
            raise ValueError("Terminal pressure is negative after scaling; aborting run.")
    else:
        tree.parameters.terminal_pressure = TERMINAL_PRESSURE
    set_tree_fluid(tree, BUILD_FLUID)
    _apply_equal_bifurcation(tree)
    if terminal_flow_override is not None:
        tree.parameters.terminal_flow = terminal_flow_override


def set_tree_fluid(tree, fluid: str) -> None:
    fluid_mode = (fluid or ACTIVE_FLUID).lower()
    try:
        tree.parameters.fluid = fluid_mode
    except Exception:
        tree.fluid = fluid_mode
    if fluid_mode in {"water", "cell media", "media"}:
        tree.parameters.kinematic_viscosity = 0.6959 / 100
        tree.parameters.fluid_density = 0.99336
    elif fluid_mode == "blood":
        tree.parameters.fluid_density = 1.06
        mu_plasma_cgs = 0.012
        tree.parameters.kinematic_viscosity = mu_plasma_cgs / tree.parameters.fluid_density


def sample_domain_points(domain: Domain, n_points: int) -> np.ndarray:
    if n_points <= 0:
        return np.empty((0, domain.points.shape[1]))
    pts, _ = domain.get_interior_points(n_points, method="implicit_only", convex=True)
    return np.asarray(pts, dtype=float)


def has_dlp_feasible_parent(tree: Tree, point: np.ndarray, k: int = 10) -> bool:
    """
    Fast check: is there at least one nearby parent vessel that could satisfy DLP constraints
    to grow toward the given point?
    """
    if tree is None or tree.segment_count <= 0:
        return False
    parms = getattr(tree, "parameters", None)
    if not (parms and getattr(parms, "dlp_enable", False) and getattr(parms, "dlp_build_dir", None) is not None):
        return True
    bdir = np.asarray(getattr(parms, "dlp_build_dir", None), dtype=float)
    n = float(np.linalg.norm(bdir))
    if n <= 0.0:
        return False
    b_hat = bdir / n
    s_min = float(getattr(parms, "dlp_min_adv", 0.0))
    sin_theta_min = math.sin(math.radians(float(getattr(parms, "dlp_min_angle_deg", 0.0))))
    data = tree.data[:tree.segment_count, :]
    query_k = max(1, min(int(k), data.shape[0]))
    try:
        _, idx = tree.hnsw_tree.query(np.asarray(point, dtype=float).reshape(1, 3), k=query_k)
    except Exception:
        return False
    idx = np.atleast_1d(idx).reshape(-1)
    for ii in idx:
        parent_prox = data[int(ii), 0:3]
        u = np.asarray(point, dtype=float) - parent_prox
        adv = float(np.dot(u, b_hat))
        if adv < s_min:
            continue
        L = float(np.linalg.norm(u))
        if L <= 1e-12:
            continue
        if abs(adv) / L < sin_theta_min:
            continue
        return True
    return False


def _characteristic_length(tree: Tree, fallback_side_length: float | None = None) -> float:
    domain = getattr(tree, "domain", None)
    if domain is not None:
        char_len = getattr(domain, "characteristic_length", None)
        if char_len is not None and np.isfinite(char_len):
            return float(char_len)
        bounds = getattr(domain, "bounds", None) or getattr(domain, "bounding_box", None)
        if bounds is not None and len(bounds) >= 6:
            span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            return float(span)
    if fallback_side_length is not None and np.isfinite(fallback_side_length):
        return float(fallback_side_length)
    return 1.0


def compute_average_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> float:
    if points.size == 0 or starts.size == 0:
        return float("nan")

    points = np.asarray(points, dtype=float)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)

    segment_vectors = ends - starts
    seg_len_sq = np.sum(segment_vectors**2, axis=1)
    seg_len_sq[seg_len_sq <= 0] = 1e-12

    use_kdtree = (
        _HAVE_SCIPY_SPATIAL
        and DISTANCE_USE_KDTREE_FOR_LARGE_TREES
        and starts.shape[0] > DISTANCE_BRUTE_FORCE_MAX_SEGMENTS
    )

    min_dists: list[np.ndarray] = []
    if use_kdtree:
        midpoints = 0.5 * (starts + ends)
        kdtree = _cKDTree(midpoints)
        candidate_k = min(
            starts.shape[0],
            max(
                int(DISTANCE_KDTREE_MIN_CANDIDATES),
                min(
                    int(DISTANCE_KDTREE_MAX_CANDIDATES),
                    int(DISTANCE_KDTREE_MIN_CANDIDATES * DISTANCE_KDTREE_CANDIDATE_MULT),
                ),
            ),
        )
        for idx in range(0, len(points), DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            _, cand = kdtree.query(chunk, k=candidate_k)
            if cand.ndim == 1:
                cand = cand[:, None]
            starts_c = starts[cand]
            seg_c = segment_vectors[cand]
            seg_len_sq_c = seg_len_sq[cand]
            diff = chunk[:, None, :] - starts_c
            proj = np.sum(diff * seg_c, axis=2) / seg_len_sq_c
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts_c + proj[:, :, None] * seg_c
            distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            min_dists.append(np.min(distances, axis=1))
    else:
        for idx in range(0, len(points), DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            diff = chunk[:, None, :] - starts[None, :, :]
            proj = np.sum(diff * segment_vectors[None, :, :], axis=2) / seg_len_sq[None, :]
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts[None, :, :] + proj[:, :, None] * segment_vectors[None, :, :]
            distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            min_dists.append(np.min(distances, axis=1))

    if not min_dists:
        return float("nan")
    all_dists = np.concatenate(min_dists)
    return float(np.mean(all_dists))


def compute_distance_to_nearest_channel(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    if points.size == 0 or starts.size == 0:
        return np.empty((0,), dtype=float)

    points = np.asarray(points, dtype=float)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if radii is None:
        radii = np.zeros((starts.shape[0],), dtype=float)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if radii.size != starts.shape[0]:
        raise ValueError("radii must have the same length as starts/ends")
    radii = np.maximum(radii, 0.0)

    segment_vectors = ends - starts
    seg_len_sq = np.sum(segment_vectors**2, axis=1)
    seg_len_sq[seg_len_sq <= 0] = 1e-12

    use_kdtree = (
        _HAVE_SCIPY_SPATIAL
        and DISTANCE_USE_KDTREE_FOR_LARGE_TREES
        and starts.shape[0] > DISTANCE_BRUTE_FORCE_MAX_SEGMENTS
    )

    min_dists: list[np.ndarray] = []
    if use_kdtree:
        midpoints = 0.5 * (starts + ends)
        kdtree = _cKDTree(midpoints)
        candidate_k = min(
            starts.shape[0],
            max(
                int(DISTANCE_KDTREE_MIN_CANDIDATES),
                min(
                    int(DISTANCE_KDTREE_MAX_CANDIDATES),
                    int(DISTANCE_KDTREE_MIN_CANDIDATES * DISTANCE_KDTREE_CANDIDATE_MULT),
                ),
            ),
        )
        for idx in range(0, len(points), DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            _, cand = kdtree.query(chunk, k=candidate_k)
            if cand.ndim == 1:
                cand = cand[:, None]
            starts_c = starts[cand]
            seg_c = segment_vectors[cand]
            seg_len_sq_c = seg_len_sq[cand]
            radii_c = radii[cand]
            diff = chunk[:, None, :] - starts_c
            proj = np.sum(diff * seg_c, axis=2) / seg_len_sq_c
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts_c + proj[:, :, None] * seg_c
            d_center = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            d_wall = np.maximum(d_center - radii_c, 0.0)
            min_dists.append(np.min(d_wall, axis=1))
    else:
        for idx in range(0, len(points), DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            diff = chunk[:, None, :] - starts[None, :, :]
            proj = np.sum(diff * segment_vectors[None, :, :], axis=2) / seg_len_sq[None, :]
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts[None, :, :] + proj[:, :, None] * segment_vectors[None, :, :]
            distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            nearest_idx = np.argmin(distances, axis=1)
            row_idx = np.arange(distances.shape[0], dtype=int)
            d_center = distances[row_idx, nearest_idx]
            r_local = radii[nearest_idx]
            d_wall = np.maximum(d_center - r_local, 0.0)
            min_dists.append(d_wall)

    if not min_dists:
        return np.empty((0,), dtype=float)
    return np.concatenate(min_dists)


def _plot_cmap():
    global TRICOLOR_CMAP
    if not USE_TRICOLOR_CMAP:
        return PLOT_CMAP
    if TRICOLOR_CMAP is None:
        from matplotlib.colors import LinearSegmentedColormap
        TRICOLOR_CMAP = LinearSegmentedColormap.from_list(
            "tri_stoplight",
            [
                (0.0, "#0000ff"),
                (0.5, "#800080"),
                (1.0, "#ff0000"),
            ],
        )
    return TRICOLOR_CMAP


def _add_domain_outline(plotter: pv.Plotter, domain: Domain) -> None:
    if domain is None:
        return
    mesh = getattr(domain, "mesh", None)
    if mesh is not None:
        outline = mesh.outline()
        if outline is not None and outline.n_cells:
            plotter.add_mesh(outline, color="gray", opacity=0.6, line_width=2.0)
            return
    bounds = getattr(domain, "bounds", None) or getattr(domain, "bounding_box", None)
    if bounds is not None:
        plotter.add_mesh(pv.Cube(bounds=bounds).outline(), color="gray", opacity=0.4, line_width=2.0)


def _show_plotter(plotter: pv.Plotter) -> None:
    if PLOT_WINDOW_POSITION is not None:
        try:
            plotter.show(window_size=PLOT_WINDOW_SIZE, window_position=PLOT_WINDOW_POSITION)
            return
        except TypeError:
            pass
    plotter.show(window_size=PLOT_WINDOW_SIZE)


# ---------------------------------------------------------------------------
# Flow solver (Kirchhoff)
# ---------------------------------------------------------------------------
def _build_node_indices(geom: np.ndarray, decimals: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prox_points = geom[:, 0:3]
    dist_points = geom[:, 3:6]
    node_lookup: Dict[Tuple[float, float, float], int] = {}
    nodes: List[np.ndarray] = []

    def _get_id(point: np.ndarray) -> int:
        key = tuple(np.round(point, decimals=decimals))
        if key not in node_lookup:
            node_lookup[key] = len(nodes)
            nodes.append(point.copy())
        return node_lookup[key]

    prox_ids = np.array([_get_id(pt) for pt in prox_points], dtype=int)
    dist_ids = np.array([_get_id(pt) for pt in dist_points], dtype=int)
    return prox_ids, dist_ids, np.array(nodes)


def _normalize_kirchhoff_bc_mode(value: str | None = None) -> str:
    mode = str(value or KIRCHHOFF_BC_MODE).strip().lower().replace("-", "_")
    aliases = {
        "mixed": "terminal_pressure",
        "terminal_pressure": "terminal_pressure",
        "pressure": "terminal_pressure",
        "pressure_terminals": "terminal_pressure",
        "terminal_pressure_bc": "terminal_pressure",
        "legacy": "legacy_equal_terminal_flow",
        "legacy_equal_terminal_flow": "legacy_equal_terminal_flow",
        "equal_terminal_flow": "legacy_equal_terminal_flow",
        "equal_terminal_flows": "legacy_equal_terminal_flow",
        "current_outlets": "legacy_equal_terminal_flow",
        "neumann": "legacy_equal_terminal_flow",
        "tree_neumann": "legacy_equal_terminal_flow",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(
            "Unknown Kirchhoff BC mode "
            f"{value!r}; expected 'terminal_pressure' or 'legacy_equal_terminal_flow'."
        ) from exc


def _solve_kirchhoff_sparse(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
    *,
    num_nodes: int | None = None,
    sparse_solver: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prox_ids = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
    dist_ids = np.asarray(dist_ids, dtype=np.int64).reshape(-1)
    num_edges = int(prox_ids.size)
    if num_edges == 0:
        raise ValueError("No segments provided for Kirchhoff solve.")

    if num_nodes is None:
        num_nodes = int(max(int(prox_ids.max()), int(dist_ids.max())) + 1)
    if num_nodes <= 0:
        raise ValueError("No nodes provided for Kirchhoff solve.")

    edge_conductance = 1.0 / np.maximum(np.asarray(resistances, dtype=float).reshape(-1), 1e-30)

    rhs = np.zeros(num_nodes, dtype=float)
    if inlet_nodes:
        inlet_share = inlet_flow_cm3_s / max(len(inlet_nodes), 1)
        for node in inlet_nodes:
            rhs[int(node)] += inlet_share
    outlet_arr = np.asarray(list(outlet_nodes), dtype=np.int64).reshape(-1)
    outlet_arr = outlet_arr[(outlet_arr >= 0) & (outlet_arr < num_nodes)]
    bc_mode = _normalize_kirchhoff_bc_mode()
    dirichlet_mask = np.zeros(num_nodes, dtype=bool)
    anchor = -1
    if bc_mode == "legacy_equal_terminal_flow":
        if outlet_arr.size:
            outlet_share = inlet_flow_cm3_s / max(int(outlet_arr.size), 1)
            rhs[outlet_arr] -= outlet_share
        anchor = num_nodes - 1
        inlet_set = {int(node) for node in inlet_nodes}
        outlet_set = {int(node) for node in outlet_arr.tolist()}
        if (anchor in inlet_set) or (anchor in outlet_set):
            anchor = max(num_nodes - 2, 0)
        dirichlet_mask[int(anchor)] = True
    else:
        if outlet_arr.size == 0:
            raise ValueError("At least one terminal pressure node is required for the mixed Kirchhoff solve.")
        dirichlet_mask[outlet_arr] = True

    if _HAVE_SCIPY_SPARSE:
        # Build graph Laplacian directly from edges (avoid B @ G @ B.T intermediates).
        u = prox_ids
        v = dist_ids
        g = edge_conductance
        rows = np.concatenate([u, v, u, v])
        cols = np.concatenate([u, v, v, u])
        data = np.concatenate([g, g, -g, -g])
        laplacian = _sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

        mask = np.ones(num_nodes, dtype=bool)
        mask[dirichlet_mask] = False
        A = laplacian[mask][:, mask].tocsr()
        b = rhs[mask]

        solver = str(sparse_solver or KIRCHHOFF_SPARSE_SOLVER).strip().lower()
        use_cg = solver == "cg" or (solver == "auto" and A.shape[0] >= int(KIRCHHOFF_CG_MIN_NODES))
        use_gmres_ilu = solver == "gmres_ilu"
        gmres_failed_reason: str | None = None
        cg_failed_reason: str | None = None
        gmres_ilu_build_s = 0.0
        gmres_solve_s = 0.0
        gmres_iters = 0
        gmres_cycles = 0
        gmres_final_resid = float("nan")
        solver_used = "none"

        if KIRCHHOFF_DIAGNOSTICS:
            print(
                f"Kirchhoff diagnostics: n={A.shape[0]} nnz={int(A.nnz)} "
                f"{_diagnostic_stats('R', resistances)} "
                f"{_diagnostic_stats('G', edge_conductance)} "
                f"{_diagnostic_stats('diag(A)', A.diagonal())}"
            )

        x = None
        if use_gmres_ilu:
            try:
                gmres_systems: list[tuple[str, _sp.csr_matrix, np.ndarray, np.ndarray | None]] = []
                if KIRCHHOFF_GMRES_EQUILIBRATE:
                    diag_a = np.abs(A.diagonal()).astype(float, copy=False)
                    diag_pos = diag_a[diag_a > 0.0]
                    if diag_pos.size:
                        diag_floor = max(
                            float(np.median(diag_pos)) * float(KIRCHHOFF_GMRES_EQ_DIAG_FLOOR_REL),
                            1e-30,
                        )
                    else:
                        diag_floor = 1e-30
                    diag_safe = np.maximum(diag_a, diag_floor)
                    equil_scale = np.divide(
                        1.0,
                        np.sqrt(diag_safe),
                        out=np.zeros_like(diag_safe),
                        where=diag_safe > 0.0,
                    )
                    if not np.all(np.isfinite(equil_scale)) or np.any(equil_scale <= 0.0):
                        raise RuntimeError("computed non-finite diagonal equilibration scale")
                    S = _sp.diags(equil_scale)
                    A_eq = (S @ A @ S).tocsr()
                    b_eq = equil_scale * b
                    gmres_systems.append(("equilibrated", A_eq, b_eq, equil_scale))
                    if KIRCHHOFF_DIAGNOSTICS:
                        print(
                            f"Kirchhoff diagnostics: gmres_equilibrate=True "
                            f"diag_floor={diag_floor:.3e} "
                            f"{_diagnostic_stats('eq_scale', equil_scale)} "
                            f"{_diagnostic_stats('diag(A_eq)', A_eq.diagonal())}"
                        )
                if not gmres_systems or KIRCHHOFF_GMRES_RETRY_UNSCALED_IF_EQ_FAIL:
                    gmres_systems.append(("unscaled", A, b, None))

                gmres_failed_reason = "gmres_ilu did not run"
                for system_name, A_gmres, b_gmres, equil_scale in gmres_systems:
                    diag_ref_arr = np.abs(A_gmres.diagonal()).astype(float, copy=False)
                    diag_ref_pos = diag_ref_arr[diag_ref_arr > 0.0]
                    diag_ref = float(np.median(diag_ref_pos)) if diag_ref_pos.size else 1.0
                    if not np.isfinite(diag_ref) or diag_ref <= 0.0:
                        diag_ref = 1.0

                    ilu = None
                    attempt_reason: str | None = None
                    used_permc = "unknown"
                    used_shift = 0.0
                    for permc_spec in KIRCHHOFF_ILU_PERMC_SPECS:
                        for shift_rel in KIRCHHOFF_ILU_SHIFT_RELS:
                            shift_abs = float(max(shift_rel, 0.0)) * diag_ref
                            if shift_abs > 0.0:
                                A_ilu = A_gmres + (_sp.identity(A_gmres.shape[0], format="csr") * shift_abs)
                            else:
                                A_ilu = A_gmres
                            t_ilu = perf_counter()
                            try:
                                ilu_try = _splinalg.spilu(
                                    A_ilu.tocsc(),
                                    drop_tol=float(KIRCHHOFF_ILU_DROP_TOL),
                                    fill_factor=float(KIRCHHOFF_ILU_FILL_FACTOR),
                                    permc_spec=str(permc_spec),
                                )
                                gmres_ilu_build_s = perf_counter() - t_ilu
                                ilu = ilu_try
                                used_permc = str(permc_spec)
                                used_shift = float(shift_abs)
                                print(
                                    f"Kirchhoff gmres_ilu: ilu_build={gmres_ilu_build_s:.3f}s "
                                    f"(starting GMRES) n={A.shape[0]} system={system_name} "
                                    f"permc={used_permc} shift={used_shift:.3e}"
                                )
                                if KIRCHHOFF_DIAGNOSTICS:
                                    ilu_nnz = int(ilu.L.nnz + ilu.U.nnz)
                                    fill_ratio = float(ilu_nnz / max(int(A_gmres.nnz), 1))
                                    print(
                                        f"Kirchhoff diagnostics: ilu_nnz={ilu_nnz} "
                                        f"fill_ratio={fill_ratio:.3f}"
                                    )
                                break
                            except Exception as exc:
                                attempt_reason = f"{type(exc).__name__}: {exc}"
                                if KIRCHHOFF_DIAGNOSTICS:
                                    print(
                                        f"Kirchhoff diagnostics: ILU attempt failed "
                                        f"system={system_name} permc={permc_spec} "
                                        f"shift={shift_abs:.3e} reason={attempt_reason}"
                                    )
                        if ilu is not None:
                            break

                    if ilu is None:
                        gmres_failed_reason = (
                            f"ILU failed for system={system_name} "
                            f"after {len(KIRCHHOFF_ILU_PERMC_SPECS) * len(KIRCHHOFF_ILU_SHIFT_RELS)} attempts; "
                            f"last_error={attempt_reason}"
                        )
                        continue

                    M = _splinalg.LinearOperator(A_gmres.shape, matvec=ilu.solve, dtype=float)
                    gmres_resids: list[float] = []

                    def _gmres_callback(value: object) -> None:
                        try:
                            gmres_resids.append(float(value))
                        except Exception:
                            pass

                    t_gmres = perf_counter()
                    y, info = _splinalg.gmres(
                        A_gmres,
                        b_gmres,
                        M=M,
                        rtol=float(KIRCHHOFF_GMRES_RTOL),
                        atol=0.0,
                        maxiter=int(KIRCHHOFF_GMRES_MAXITER),
                        restart=int(KIRCHHOFF_GMRES_RESTART),
                        callback=_gmres_callback,
                        callback_type="pr_norm",
                    )
                    gmres_solve_s = perf_counter() - t_gmres
                    gmres_iters = len(gmres_resids)
                    gmres_cycles = int(math.ceil(gmres_iters / max(int(KIRCHHOFF_GMRES_RESTART), 1))) if gmres_iters else 0
                    if gmres_resids:
                        gmres_final_resid = float(gmres_resids[-1])
                    if info != 0 or y is None:
                        gmres_failed_reason = (
                            f"gmres returned info={info} "
                            f"on system={system_name} permc={used_permc} shift={used_shift:.3e}"
                        )
                        x = None
                        continue

                    y_arr = np.asarray(y, dtype=float).reshape(-1)
                    if equil_scale is not None:
                        x = equil_scale * y_arr
                    else:
                        x = y_arr
                    if not np.all(np.isfinite(x)):
                        gmres_failed_reason = (
                            f"gmres returned non-finite solution "
                            f"on system={system_name} permc={used_permc} shift={used_shift:.3e}"
                        )
                        x = None
                        continue

                    solver_used = "gmres_ilu"
                    break
            except Exception as exc:
                gmres_failed_reason = f"{type(exc).__name__}: {exc}"
                x = None
            status = "success" if x is not None else "failed"
            resid_text = (
                f"{gmres_final_resid:.3e}" if np.isfinite(gmres_final_resid) else "nan"
            )
            print(
                f"Kirchhoff gmres_ilu timings: gmres_solve={gmres_solve_s:.3f}s "
                f"iter={gmres_iters} cycles={gmres_cycles} final_pr_norm={resid_text} "
                f"status={status} n={A.shape[0]}"
            )

        if x is None and use_cg:
            try:
                diag = A.diagonal()
                inv_diag = np.divide(1.0, diag, out=np.zeros_like(diag), where=diag != 0.0)
                M = _splinalg.LinearOperator(A.shape, matvec=lambda z: inv_diag * z, dtype=float)
                x, info = _splinalg.cg(
                    A,
                    b,
                    M=M,
                    rtol=float(KIRCHHOFF_CG_RTOL),
                    atol=0.0,
                    maxiter=int(KIRCHHOFF_CG_MAXITER),
                )
                if info != 0 or x is None or not np.all(np.isfinite(x)):
                    cg_failed_reason = f"cg returned info={info}"
                    x = None
                else:
                    solver_used = "cg"
            except Exception as exc:
                cg_failed_reason = f"{type(exc).__name__}: {exc}"
                x = None

        if x is None:
            # Make solver fallback explicit in CLI output for large runs.
            if use_gmres_ilu and gmres_failed_reason is not None:
                print(
                    f"Kirchhoff fallback: gmres_ilu failed ({gmres_failed_reason}); "
                    f"using sparse LU (spsolve). n={A.shape[0]}"
                )
            elif use_cg and cg_failed_reason is not None:
                print(
                    f"Kirchhoff fallback: cg failed ({cg_failed_reason}); "
                    f"using sparse LU (spsolve). n={A.shape[0]}"
                )
            try:
                x = _splinalg.spsolve(A, b)
                solver_used = "spsolve"
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Failed to solve sparse Kirchhoff system. Check BCs and connectivity.") from exc

        if KIRCHHOFF_DIAGNOSTICS and x is not None:
            x_arr = np.asarray(x, dtype=float).reshape(-1)
            r = b - A.dot(x_arr)
            bnorm = float(np.linalg.norm(b))
            if bnorm > 0.0:
                rel_true_resid = float(np.linalg.norm(r) / bnorm)
            else:
                rel_true_resid = float(np.linalg.norm(r))
            print(
                f"Kirchhoff diagnostics: solver_used={solver_used} "
                f"bc={bc_mode} true_rel_resid={rel_true_resid:.3e}"
            )

        pressures = np.zeros(num_nodes, dtype=float)
        pressures[mask] = np.asarray(x, dtype=float).reshape(-1)
    else:
        # Dense fallback (only feasible for small problems).
        laplacian = np.zeros((num_nodes, num_nodes), dtype=float)
        for i in range(num_edges):
            ui = int(prox_ids[i])
            vi = int(dist_ids[i])
            gi = float(edge_conductance[i])
            laplacian[ui, ui] += gi
            laplacian[vi, vi] += gi
            laplacian[ui, vi] -= gi
            laplacian[vi, ui] -= gi

        mask = np.ones(num_nodes, dtype=bool)
        mask[dirichlet_mask] = False
        A = laplacian[np.ix_(mask, mask)]
        b = rhs[mask]
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Failed to solve Kirchhoff system. Check BCs and connectivity.") from exc
        pressures = np.zeros(num_nodes, dtype=float)
        pressures[mask] = x

    flows = edge_conductance * (pressures[prox_ids] - pressures[dist_ids])
    return pressures, flows, prox_ids, dist_ids, np.empty((0, 3), dtype=float)


if _HAVE_NUMBA:
    @njit(cache=True)
    def _fixed_terminal_flows_numba(
        order: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        inlet_flow_cm3_s: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        nseg = int(left_child.shape[0])
        downstream_terms = np.zeros((nseg,), dtype=np.int64)
        for oi in range(order.shape[0] - 1, -1, -1):
            seg = int(order[oi])
            if seg < 0 or seg >= nseg:
                continue
            left = int(left_child[seg])
            right = int(right_child[seg])
            has_left = left >= 0 and left < nseg
            has_right = right >= 0 and right < nseg
            if not has_left and not has_right:
                downstream_terms[seg] = 1
            else:
                total = 0
                if has_left:
                    total += int(downstream_terms[left])
                if has_right:
                    total += int(downstream_terms[right])
                downstream_terms[seg] = total
        total_terms = 0
        for seg in range(nseg):
            left = int(left_child[seg])
            right = int(right_child[seg])
            if not (left >= 0 and left < nseg) and not (right >= 0 and right < nseg):
                total_terms += 1
        if total_terms <= 0:
            total_terms = 1
        terminal_flow = float(inlet_flow_cm3_s) / float(total_terms)
        flows = np.empty((nseg,), dtype=np.float64)
        for seg in range(nseg):
            flows[seg] = float(downstream_terms[seg]) * terminal_flow
        return flows, downstream_terms, total_terms


    @njit(cache=True)
    def _pressures_from_tree_flows_numba(
        order: np.ndarray,
        prox_ids: np.ndarray,
        dist_ids: np.ndarray,
        flows: np.ndarray,
        resistances: np.ndarray,
        root_node: int,
        root_pressure: float,
        num_nodes: int,
    ) -> tuple[np.ndarray, int]:
        pressures = np.empty((num_nodes,), dtype=np.float64)
        for node_i in range(num_nodes):
            pressures[node_i] = np.nan
        if root_node >= 0 and root_node < num_nodes:
            pressures[root_node] = root_pressure
        assigned_edges = 0
        for oi in range(order.shape[0]):
            seg = int(order[oi])
            if seg < 0 or seg >= flows.shape[0]:
                continue
            u = int(prox_ids[seg])
            v = int(dist_ids[seg])
            q = float(flows[seg])
            r = float(resistances[seg])
            if u >= 0 and u < num_nodes and v >= 0 and v < num_nodes:
                pu_known = np.isfinite(pressures[u])
                pv_known = np.isfinite(pressures[v])
                if pu_known and not pv_known:
                    pressures[v] = pressures[u] - q * r
                    assigned_edges += 1
                elif pv_known and not pu_known:
                    pressures[u] = pressures[v] + q * r
                    assigned_edges += 1
                elif pu_known and pv_known:
                    assigned_edges += 1
        return pressures, assigned_edges


    @njit
    def _kirchhoff_tree_neumann_numba(
        prox_ids: np.ndarray,
        dist_ids: np.ndarray,
        resistances: np.ndarray,
        rhs: np.ndarray,
        root_node: int,
        anchor_node: int,
        num_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        num_edges = prox_ids.shape[0]

        degree = np.zeros(num_nodes, dtype=np.int64)
        for edge_i in range(num_edges):
            degree[int(prox_ids[edge_i])] += 1
            degree[int(dist_ids[edge_i])] += 1

        offsets = np.empty(num_nodes + 1, dtype=np.int64)
        total = 0
        offsets[0] = 0
        for node_i in range(num_nodes):
            total += degree[node_i]
            offsets[node_i + 1] = total

        cursor = offsets[:-1].copy()
        adj_to = np.empty(total, dtype=np.int64)
        adj_edge = np.empty(total, dtype=np.int64)
        for edge_i in range(num_edges):
            u = int(prox_ids[edge_i])
            v = int(dist_ids[edge_i])
            pos = cursor[u]
            adj_to[pos] = v
            adj_edge[pos] = edge_i
            cursor[u] = pos + 1
            pos = cursor[v]
            adj_to[pos] = u
            adj_edge[pos] = edge_i
            cursor[v] = pos + 1

        parent_node = np.full(num_nodes, -2, dtype=np.int64)
        parent_edge = np.full(num_nodes, -1, dtype=np.int64)
        order = np.empty(num_nodes, dtype=np.int64)
        stack = np.empty(num_nodes, dtype=np.int64)

        top = 1
        stack[0] = root_node
        parent_node[root_node] = -1
        n_order = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            order[n_order] = node
            n_order += 1
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != -2:
                    continue
                parent_node[nbr] = node
                parent_edge[nbr] = int(adj_edge[pos])
                stack[top] = nbr
                top += 1

        subtree_rhs = rhs.copy()
        flows = np.zeros(num_edges, dtype=np.float64)
        for order_i in range(n_order - 1, 0, -1):
            node = int(order[order_i])
            edge_i = int(parent_edge[node])
            parent = int(parent_node[node])
            sub_rhs = subtree_rhs[node]
            q_parent_to_node = -sub_rhs
            if int(prox_ids[edge_i]) == parent and int(dist_ids[edge_i]) == node:
                flows[edge_i] = q_parent_to_node
            else:
                flows[edge_i] = -q_parent_to_node
            subtree_rhs[parent] += sub_rhs

        pressures = np.empty(num_nodes, dtype=np.float64)
        visited = np.zeros(num_nodes, dtype=np.uint8)
        for node_i in range(num_nodes):
            pressures[node_i] = np.nan

        top = 1
        stack[0] = anchor_node
        visited[anchor_node] = 1
        pressures[anchor_node] = 0.0
        n_pressure = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            n_pressure += 1
            p_node = pressures[node]
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if visited[nbr] != 0:
                    continue
                edge_i = int(adj_edge[pos])
                if int(prox_ids[edge_i]) == node and int(dist_ids[edge_i]) == nbr:
                    pressures[nbr] = p_node - flows[edge_i] * resistances[edge_i]
                else:
                    pressures[nbr] = p_node + flows[edge_i] * resistances[edge_i]
                visited[nbr] = 1
                stack[top] = nbr
                top += 1

        return pressures, flows, n_order, n_pressure


    @njit
    def _kirchhoff_tree_mixed_numba(
        prox_ids: np.ndarray,
        dist_ids: np.ndarray,
        resistances: np.ndarray,
        outlet_nodes: np.ndarray,
        root_node: int,
        inlet_flow_cm3_s: float,
        num_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        num_edges = prox_ids.shape[0]

        degree = np.zeros(num_nodes, dtype=np.int64)
        for edge_i in range(num_edges):
            degree[int(prox_ids[edge_i])] += 1
            degree[int(dist_ids[edge_i])] += 1

        offsets = np.empty(num_nodes + 1, dtype=np.int64)
        total = 0
        offsets[0] = 0
        for node_i in range(num_nodes):
            total += degree[node_i]
            offsets[node_i + 1] = total

        cursor = offsets[:-1].copy()
        adj_to = np.empty(total, dtype=np.int64)
        adj_edge = np.empty(total, dtype=np.int64)
        for edge_i in range(num_edges):
            u = int(prox_ids[edge_i])
            v = int(dist_ids[edge_i])
            pos = cursor[u]
            adj_to[pos] = v
            adj_edge[pos] = edge_i
            cursor[u] = pos + 1
            pos = cursor[v]
            adj_to[pos] = u
            adj_edge[pos] = edge_i
            cursor[v] = pos + 1

        outlet_mask = np.zeros(num_nodes, dtype=np.uint8)
        for i in range(outlet_nodes.shape[0]):
            node = int(outlet_nodes[i])
            if 0 <= node < num_nodes:
                outlet_mask[node] = 1

        parent_node = np.full(num_nodes, -2, dtype=np.int64)
        parent_edge = np.full(num_nodes, -1, dtype=np.int64)
        order = np.empty(num_nodes, dtype=np.int64)
        stack = np.empty(num_nodes, dtype=np.int64)

        top = 1
        stack[0] = root_node
        parent_node[root_node] = -1
        n_order = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            order[n_order] = node
            n_order += 1
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != -2:
                    continue
                parent_node[nbr] = node
                parent_edge[nbr] = int(adj_edge[pos])
                stack[top] = nbr
                top += 1

        eq_g = np.zeros(num_nodes, dtype=np.float64)
        branch_g = np.zeros(num_edges, dtype=np.float64)
        for order_i in range(n_order - 1, -1, -1):
            node = int(order[order_i])
            if outlet_mask[node] != 0:
                eq_g[node] = 0.0
                continue
            gsum = 0.0
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != node:
                    continue
                edge_i = int(adj_edge[pos])
                r_edge = resistances[edge_i]
                if r_edge < 1.0e-300:
                    r_edge = 1.0e-300
                if outlet_mask[nbr] != 0:
                    bg = 1.0 / r_edge
                else:
                    child_g = eq_g[nbr]
                    if child_g > 0.0:
                        bg = 1.0 / (r_edge + 1.0 / child_g)
                    else:
                        bg = 0.0
                branch_g[edge_i] = bg
                gsum += bg
            eq_g[node] = gsum

        pressures = np.empty(num_nodes, dtype=np.float64)
        flows = np.zeros(num_edges, dtype=np.float64)
        visited = np.zeros(num_nodes, dtype=np.uint8)
        for node_i in range(num_nodes):
            pressures[node_i] = np.nan

        root_g = eq_g[root_node]
        if root_g <= 0.0:
            return pressures, flows, n_order, 0
        pressures[root_node] = inlet_flow_cm3_s / root_g
        visited[root_node] = 1
        top = 1
        stack[0] = root_node
        n_pressure = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            n_pressure += 1
            p_node = pressures[node]
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != node:
                    continue
                edge_i = int(adj_edge[pos])
                q_parent_to_child = branch_g[edge_i] * p_node
                if int(prox_ids[edge_i]) == node and int(dist_ids[edge_i]) == nbr:
                    flows[edge_i] = q_parent_to_child
                else:
                    flows[edge_i] = -q_parent_to_child
                child_p = p_node - q_parent_to_child * resistances[edge_i]
                if outlet_mask[nbr] != 0:
                    child_p = 0.0
                pressures[nbr] = child_p
                visited[nbr] = 1
                stack[top] = nbr
                top += 1

        return pressures, flows, n_order, n_pressure


def _kirchhoff_rhs_anchor_and_root(
    num_nodes: int,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
) -> tuple[np.ndarray, int, int]:
    rhs = np.zeros(num_nodes, dtype=float)
    inlet_arr = np.asarray(list(inlet_nodes), dtype=np.int64).reshape(-1)
    outlet_arr = np.asarray(list(outlet_nodes), dtype=np.int64).reshape(-1)

    if inlet_arr.size:
        inlet_share = float(inlet_flow_cm3_s) / float(inlet_arr.size)
        rhs[inlet_arr] += inlet_share
        root_node = int(inlet_arr[0])
    else:
        root_node = 0
    if outlet_arr.size:
        outlet_share = float(inlet_flow_cm3_s) / float(outlet_arr.size)
        rhs[outlet_arr] -= outlet_share

    anchor = max(num_nodes - 1, 0)
    if (
        (inlet_arr.size and bool(np.any(inlet_arr == anchor)))
        or (outlet_arr.size and bool(np.any(outlet_arr == anchor)))
    ):
        anchor = max(num_nodes - 2, 0)
    return rhs, root_node, anchor


def _kirchhoff_residual_norms(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    flows: np.ndarray,
    rhs: np.ndarray,
    num_nodes: int,
) -> tuple[float, float]:
    balance = np.bincount(prox_ids, weights=flows, minlength=num_nodes).astype(float, copy=False)
    balance -= np.bincount(dist_ids, weights=flows, minlength=num_nodes).astype(float, copy=False)
    resid = balance - rhs
    abs_norm = float(np.linalg.norm(resid))
    rhs_norm = float(np.linalg.norm(rhs))
    rel_norm = abs_norm / rhs_norm if rhs_norm > 0.0 else abs_norm
    return abs_norm, rel_norm


def solve_kirchhoff_tree(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
    *,
    num_nodes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not _HAVE_NUMBA:
        raise RuntimeError("Tree Kirchhoff solver requires numba; use a sparse Kirchhoff solver instead.")

    prox_ids = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
    dist_ids = np.asarray(dist_ids, dtype=np.int64).reshape(-1)
    resistances = np.asarray(resistances, dtype=float).reshape(-1)
    num_edges = int(prox_ids.size)
    if num_edges == 0:
        raise ValueError("No segments provided for Kirchhoff solve.")
    if num_nodes is None:
        num_nodes = int(max(int(prox_ids.max()), int(dist_ids.max())) + 1)
    if num_nodes <= 0:
        raise ValueError("No nodes provided for Kirchhoff solve.")

    inlet_arr = np.asarray(list(inlet_nodes), dtype=np.int64).reshape(-1)
    root_node = int(inlet_arr[0]) if inlet_arr.size else 0
    outlet_arr = np.asarray(list(outlet_nodes), dtype=np.int64).reshape(-1)
    outlet_arr = outlet_arr[(outlet_arr >= 0) & (outlet_arr < int(num_nodes))]
    bc_mode = _normalize_kirchhoff_bc_mode()
    t0 = perf_counter()
    rhs_diag = np.zeros(int(num_nodes), dtype=float)
    if bc_mode == "legacy_equal_terminal_flow":
        rhs, root_node, anchor = _kirchhoff_rhs_anchor_and_root(
            int(num_nodes),
            inlet_nodes,
            float(inlet_flow_cm3_s),
            outlet_arr,
        )
        rhs_diag = rhs
        pressures, flows, n_flow_nodes, n_pressure_nodes = _kirchhoff_tree_neumann_numba(
            prox_ids,
            dist_ids,
            np.maximum(resistances, 1e-30),
            rhs,
            int(root_node),
            int(anchor),
            int(num_nodes),
        )
        solver_used = "tree_neumann"
    else:
        if outlet_arr.size == 0:
            raise ValueError("At least one terminal pressure node is required for the mixed tree Kirchhoff solve.")
        pressures, flows, n_flow_nodes, n_pressure_nodes = _kirchhoff_tree_mixed_numba(
            prox_ids,
            dist_ids,
            np.maximum(resistances, 1e-30),
            outlet_arr,
            int(root_node),
            float(inlet_flow_cm3_s),
            int(num_nodes),
        )
        solver_used = "tree_mixed"
        if inlet_arr.size:
            inlet_share = float(inlet_flow_cm3_s) / float(inlet_arr.size)
            rhs_diag[inlet_arr] = inlet_share
    solve_s = perf_counter() - t0

    if n_flow_nodes != int(num_nodes) or n_pressure_nodes != int(num_nodes):
        raise RuntimeError(
            "Tree Kirchhoff solver could not traverse all nodes; graph may be disconnected or not tree-like. "
            f"flow_nodes={n_flow_nodes}/{num_nodes} pressure_nodes={n_pressure_nodes}/{num_nodes}"
        )
    if not np.all(np.isfinite(pressures)) or not np.all(np.isfinite(flows)):
        raise RuntimeError("Tree Kirchhoff solver produced non-finite pressures or flows.")

    if KIRCHHOFF_DIAGNOSTICS:
        balance = np.bincount(prox_ids, weights=flows, minlength=int(num_nodes)).astype(float, copy=False)
        balance -= np.bincount(dist_ids, weights=flows, minlength=int(num_nodes)).astype(float, copy=False)
        if bc_mode == "legacy_equal_terminal_flow":
            check_mask = np.ones(int(num_nodes), dtype=bool)
        else:
            check_mask = np.ones(int(num_nodes), dtype=bool)
            check_mask[outlet_arr] = False
        resid = balance[check_mask] - rhs_diag[check_mask]
        rhs_norm = float(np.linalg.norm(rhs_diag[check_mask]))
        rel_resid = float(np.linalg.norm(resid) / rhs_norm) if rhs_norm > 0.0 else float(np.linalg.norm(resid))
        print(
            f"Kirchhoff diagnostics: solver_used={solver_used} "
            f"bc={bc_mode} solve_time={solve_s:.3f}s true_rel_resid={rel_resid:.3e}"
        )

    return pressures, flows, prox_ids, dist_ids, np.empty((0, 3), dtype=float)


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float).reshape(-1) - np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(np.asarray(b, dtype=float).reshape(-1)))
    num = float(np.linalg.norm(diff))
    return num / denom if denom > 0.0 else num


def solve_kirchhoff(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
    *,
    num_nodes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    solver_mode = str(KIRCHHOFF_SOLVER).strip().lower()
    if solver_mode in ("tree", "tree_neumann", "tree-current-bc"):
        t_tree = perf_counter()
        tree_result = solve_kirchhoff_tree(
            prox_ids,
            dist_ids,
            resistances,
            inlet_nodes,
            inlet_flow_cm3_s,
            outlet_nodes,
            num_nodes=num_nodes,
        )
        t_tree = perf_counter() - t_tree
        if KIRCHHOFF_VALIDATE_TREE:
            t_sparse = perf_counter()
            sparse_result = _solve_kirchhoff_sparse(
                prox_ids,
                dist_ids,
                resistances,
                inlet_nodes,
                inlet_flow_cm3_s,
                outlet_nodes,
                num_nodes=num_nodes,
                sparse_solver=KIRCHHOFF_VALIDATE_SPARSE_SOLVER,
            )
            t_sparse = perf_counter() - t_sparse
            p_tree, q_tree = tree_result[0], tree_result[1]
            p_sparse, q_sparse = sparse_result[0], sparse_result[1]
            pressure_rel_l2 = _relative_l2(p_tree, p_sparse)
            flow_rel_l2 = _relative_l2(q_tree, q_sparse)
            pressure_max_abs = float(np.max(np.abs(p_tree - p_sparse))) if p_tree.size else 0.0
            flow_max_abs = float(np.max(np.abs(q_tree - q_sparse))) if q_tree.size else 0.0
            speedup = t_sparse / max(t_tree, 1e-30)
            print(
                "Kirchhoff tree validation: "
                f"tree={t_tree:.3f}s sparse={t_sparse:.3f}s speedup={speedup:.2f}x "
                f"flow_rel_l2={flow_rel_l2:.3e} flow_max_abs={flow_max_abs:.3e} "
                f"pressure_rel_l2={pressure_rel_l2:.3e} pressure_max_abs={pressure_max_abs:.3e}"
            )
        return tree_result

    return _solve_kirchhoff_sparse(
        prox_ids,
        dist_ids,
        resistances,
        inlet_nodes,
        inlet_flow_cm3_s,
        outlet_nodes,
        num_nodes=num_nodes,
        sparse_solver=solver_mode,
    )


def solve_kirchhoff_dirichlet(
    starts: np.ndarray,
    ends: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_pressure: float,
    outlet_nodes: Sequence[int],
    outlet_flow_per_node: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geom = np.zeros((starts.shape[0], 6), dtype=float)
    geom[:, 0:3] = starts
    geom[:, 3:6] = ends
    prox_ids, dist_ids, nodes = _build_node_indices(geom)

    num_edges = starts.shape[0]
    num_nodes = int(max(prox_ids.max(), dist_ids.max()) + 1) if num_edges > 0 else 0
    if num_nodes == 0:
        raise ValueError("No segments provided for Kirchhoff solve.")
    if not inlet_nodes:
        raise ValueError("No inlet nodes provided for Dirichlet solve.")

    edge_conductance = np.zeros(num_edges, dtype=float)
    edge_conductance[:] = 1.0 / np.maximum(resistances, 1e-30)

    rhs = np.zeros(num_nodes, dtype=float)
    if outlet_nodes:
        sink_share = float(outlet_flow_per_node)
        for node in outlet_nodes:
            rhs[int(node)] -= sink_share

    if _HAVE_SCIPY_SPARSE:
        rows = np.concatenate([prox_ids, dist_ids])
        cols = np.concatenate([np.arange(num_edges), np.arange(num_edges)])
        data = np.concatenate([np.ones(num_edges), -np.ones(num_edges)])
        B = _sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_edges)).tocsr()
        G = _sp.diags(edge_conductance, 0, shape=(num_edges, num_edges), format="csr")
        laplacian = (B @ G @ B.T).tolil()

        for node in inlet_nodes:
            laplacian[:, node] = 0.0
            laplacian[node, :] = 0.0
            laplacian[node, node] = 1.0
            rhs[node] = float(inlet_pressure)

        try:
            pressures = _splinalg.spsolve(laplacian.tocsr(), rhs)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to solve sparse Kirchhoff system. Check BCs and connectivity.") from exc
    else:
        B = np.zeros((num_nodes, num_edges), dtype=float)
        for idx in range(num_edges):
            B[prox_ids[idx], idx] = 1.0
            B[dist_ids[idx], idx] = -1.0

        BG = B * edge_conductance
        laplacian = BG @ B.T

        for node in inlet_nodes:
            laplacian[:, node] = 0.0
            laplacian[node, :] = 0.0
            laplacian[node, node] = 1.0
            rhs[node] = float(inlet_pressure)

        try:
            pressures = np.linalg.solve(laplacian, rhs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Failed to solve Kirchhoff system. Check BCs and connectivity.") from exc

    flows = edge_conductance * (pressures[prox_ids] - pressures[dist_ids])
    return pressures, flows, prox_ids, dist_ids, nodes


# ---------------------------------------------------------------------------
# Blood viscosity helpers (Fahraeus-Lindqvist)
# ---------------------------------------------------------------------------
def eta_rel_pries(d_um: np.ndarray, hd: np.ndarray) -> np.ndarray:
    d = np.asarray(d_um, dtype=float)
    hd = np.asarray(hd, dtype=float)
    A = 4.0 / (1.0 + np.exp(-0.593 * (d - 6.74)))
    term = 110.0 * np.exp(-1.424 * d) + 3.0 - 3.45 * np.exp(-0.035 * d)
    num = np.exp(hd) - 1.0
    den = np.exp(0.45 * A) - 1.0
    factor = np.divide(num, den, out=np.zeros_like(num), where=den != 0.0)
    return 1.0 + factor * term


def _use_netflow_blood_rheology() -> bool:
    try:
        return _normalize_hematocrit_model() == "pries_secomb"
    except Exception:
        return False


def netflow_viscor_cgs(d_um: np.ndarray, hd: np.ndarray) -> np.ndarray:
    """NetFlowV2 `viscor.cpp` apparent blood viscosity, returned in cgs units."""
    d = np.asarray(d_um, dtype=float)
    h = np.clip(np.asarray(hd, dtype=float), HEMATOCRIT_MIN, HEMATOCRIT_MAX)
    h = np.broadcast_to(h, d.shape)
    dcorr = np.maximum(d * float(NETFLOW_MCV_CORR), 1.0e-9)
    denom = np.maximum(dcorr - float(NETFLOW_OPTW_UM), 1.0e-9)
    geom_fac = (dcorr / denom) ** 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inv_term = 1.0 / (1.0 + (10.0 ** NETFLOW_CPAR_3) * (dcorr ** NETFLOW_CPAR_4))
        c = (NETFLOW_CPAR_1 + np.exp(NETFLOW_CPAR_2 * dcorr)) * (-1.0 + inv_term) + inv_term
        eta45 = (
            NETFLOW_VISCPAR_1 * np.exp(NETFLOW_VISCPAR_2 * dcorr)
            + NETFLOW_VISCPAR_3
            + NETFLOW_VISCPAR_4 * np.exp(NETFLOW_VISCPAR_5 * (dcorr ** NETFLOW_VISCPAR_6))
        )
        hdfac_den = (1.0 - 0.45) ** c - 1.0
        hdfac = np.divide((1.0 - h) ** c - 1.0, hdfac_den, out=np.zeros_like(dcorr), where=hdfac_den != 0.0)
        etarel = (1.0 + (eta45 - 1.0) * hdfac * geom_fac) * geom_fac
    mu_cgs = etarel * NETFLOW_VPLAS_CP * 0.01
    return np.where(np.isfinite(mu_cgs) & (mu_cgs > 0.0), mu_cgs, NETFLOW_VPLAS_CP * 0.01)


if _HAVE_NUMBA:
    @njit
    def _netflow_viscor_cgs_numba(d_um: np.ndarray, hd: np.ndarray) -> np.ndarray:
        n = d_um.shape[0]
        out = np.empty(n, dtype=np.float64)
        fallback = NETFLOW_VPLAS_CP * 0.01
        ten_cpar3 = 10.0 ** NETFLOW_CPAR_3
        for i in range(n):
            d = d_um[i]
            if not np.isfinite(d) or d <= 0.0:
                d = 1.0e-9
            h = hd[i]
            if not np.isfinite(h):
                h = HD_DISCHARGE
            if h < HEMATOCRIT_MIN:
                h = HEMATOCRIT_MIN
            elif h > HEMATOCRIT_MAX:
                h = HEMATOCRIT_MAX

            dcorr = d * NETFLOW_MCV_CORR
            if dcorr < 1.0e-9:
                dcorr = 1.0e-9
            denom = dcorr - NETFLOW_OPTW_UM
            if denom < 1.0e-9:
                denom = 1.0e-9
            geom_fac = (dcorr / denom) ** 2.0
            inv_term = 1.0 / (1.0 + ten_cpar3 * (dcorr ** NETFLOW_CPAR_4))
            c = (NETFLOW_CPAR_1 + np.exp(NETFLOW_CPAR_2 * dcorr)) * (-1.0 + inv_term) + inv_term
            eta45 = (
                NETFLOW_VISCPAR_1 * np.exp(NETFLOW_VISCPAR_2 * dcorr)
                + NETFLOW_VISCPAR_3
                + NETFLOW_VISCPAR_4 * np.exp(NETFLOW_VISCPAR_5 * (dcorr ** NETFLOW_VISCPAR_6))
            )
            hdfac_den = (1.0 - 0.45) ** c - 1.0
            if hdfac_den != 0.0:
                hdfac = ((1.0 - h) ** c - 1.0) / hdfac_den
            else:
                hdfac = 0.0
            etarel = (1.0 + (eta45 - 1.0) * hdfac * geom_fac) * geom_fac
            mu = etarel * fallback
            if np.isfinite(mu) and mu > 0.0:
                out[i] = mu
            else:
                out[i] = fallback
        return out


def _segment_permeation_rate(radius: float, diffusivity: float) -> float:
    if not np.isfinite(radius) or radius <= 0.0:
        return 0.0
    return float(3.66 * diffusivity / (radius * radius))


def tube_hematocrit(radius_cm: float, hd: float = HD_DISCHARGE) -> float:
    if radius_cm <= 0.0 or not np.isfinite(radius_cm):
        return 0.0
    d_um = 2.0 * radius_cm * 1.0e4
    ratio = hd + (1.0 - hd) * (1.0 + 1.7 * np.exp(-0.415 * d_um) - 0.6 * np.exp(-0.011 * d_um))
    return float(hd * ratio)


def segment_viscosity_from_radius(
    radii_cm: np.ndarray,
    mu_base: float,
    fluid: str,
    hd: float = HD_DISCHARGE,
) -> np.ndarray:
    fluid_mode = (fluid or ACTIVE_FLUID).lower()
    if fluid_mode != "blood":
        return np.full_like(radii_cm, float(mu_base))
    return segment_viscosity_from_radius_hd(radii_cm, mu_base, fluid, np.full_like(radii_cm, float(hd), dtype=float))


def segment_viscosity_from_radius_hd(
    radii_cm: np.ndarray,
    mu_base: float,
    fluid: str,
    hd: np.ndarray,
) -> np.ndarray:
    fluid_mode = (fluid or ACTIVE_FLUID).lower()
    if fluid_mode != "blood":
        return np.full_like(radii_cm, float(mu_base), dtype=float)
    d_um = 2.0 * radii_cm * 1.0e4
    hd_arr = np.asarray(hd, dtype=float)
    if hd_arr.ndim == 0:
        hd_arr = np.full_like(d_um, float(hd_arr), dtype=float)
    elif hd_arr.shape[0] != d_um.shape[0]:
        hd_arr = np.full_like(d_um, float(HD_DISCHARGE), dtype=float)
    if _use_netflow_blood_rheology():
        if _HAVE_NUMBA:
            return _netflow_viscor_cgs_numba(
                np.asarray(d_um, dtype=np.float64),
                np.asarray(hd_arr, dtype=np.float64),
            )
        return netflow_viscor_cgs(d_um, hd_arr)
    eta_rel = eta_rel_pries(d_um, hd_arr)
    return mu_base * eta_rel


def compute_segment_viscosity(
    tree,
    hd_root: float = HD_DISCHARGE,
    hd_per_segment: np.ndarray | None = None,
    data_override: np.ndarray | None = None,
) -> np.ndarray:
    if data_override is not None:
        data = np.asarray(data_override, dtype=float)
        nseg = data.shape[0]
    else:
        nseg = int(getattr(tree, "segment_count", 0))
        if nseg <= 0:
            return np.empty((0,), dtype=float)
        data = np.asarray(tree.data[:nseg], dtype=float)

    radii_cm = data[:, 21]
    d_um = 2.0 * radii_cm * 1.0e4

    if hd_per_segment is None:
        HD = np.full(nseg, float(hd_root), dtype=float)
    else:
        HD = np.asarray(hd_per_segment, dtype=float)
        if HD.shape[0] != nseg:
            HD = np.full(nseg, float(hd_root), dtype=float)

    rho = getattr(getattr(tree, "parameters", None), "fluid_density", 1.06)
    nu = getattr(getattr(tree, "parameters", None), "kinematic_viscosity", 0.012 / 1.06)
    mu_base = rho * nu
    fluid_mode = getattr(tree, "fluid", None) or getattr(getattr(tree, "parameters", None), "fluid", None) or ACTIVE_FLUID
    return segment_viscosity_from_radius_hd(radii_cm, mu_base, fluid_mode, HD)


def _update_resistance_variable_mu_py(data: np.ndarray, idx: np.ndarray, gamma: float, mu_seg: np.ndarray) -> None:
    vessels = list(range(data.shape[0]))
    max_depth = float(np.nanmax(data[:, 26]))
    while vessels:
        tmp = []
        for i in vessels:
            if data[i, 26] != max_depth:
                tmp.append(i)
                continue

            local_mu = mu_seg[i]
            if np.isnan(data[i, 15:17]).all():
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20]
                data[i, 27] = 0.0
            elif np.isnan(data[i, 15]):
                right = idx[i, 1]
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20] + data[right, 25]
                data[i, 23] = 0.0
                data[i, 24] = 1.0
                data[i, 27] = data[right, 20] + data[right, 27]
            elif np.isnan(data[i, 16]):
                left = idx[i, 0]
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20] + data[left, 25]
                data[i, 23] = 1.0
                data[i, 24] = 0.0
                data[i, 27] = data[left, 20] + data[left, 27]
            else:
                left = idx[i, 0]
                right = idx[i, 1]
                lr = ((data[left, 22] * data[left, 25]) / (data[right, 22] * data[right, 25])) ** 0.25
                lbif = (1.0 + lr ** (-gamma)) ** (-1.0 / gamma)
                rbif = (1.0 + lr ** gamma) ** (-1.0 / gamma)
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20] + (
                    (lbif ** 4 / data[left, 25]) + (rbif ** 4 / data[right, 25])
                ) ** -1.0
                data[i, 23] = lbif
                data[i, 24] = rbif
                data[i, 27] = lbif ** 2 * (data[left, 20] + data[left, 27]) + rbif ** 2 * (
                    data[right, 20] + data[right, 27]
                )

        vessels = tmp
        max_depth -= 1.0


def apply_fahraeus_lindqvist_resistance(tree, *, fluid: str | None = None, hd_root: float = HD_DISCHARGE) -> None:
    fluid_mode = (fluid or getattr(getattr(tree, "parameters", None), "fluid", None) or ACTIVE_FLUID).lower()
    if fluid_mode != "blood":
        _dbg("F-L skipped: fluid_mode != blood")
        return

    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        _dbg("F-L skipped: nseg <= 0")
        return

    data = np.asarray(tree.data[:nseg], dtype=float)
    idx = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(np.int64, copy=False)
    original_root_flow = float(data[0, 22]) if data.size else float("nan")

    try:
        hd_existing = getattr(tree, "discharge_hematocrit", None)
    except Exception:
        hd_existing = None

    mu_seg = compute_segment_viscosity(
        tree,
        hd_root=hd_root,
        hd_per_segment=hd_existing,
        data_override=data,
    )

    _dbg(f"F-L: nseg={nseg}, mu_base*eta_rel sample={mu_seg[:3] if mu_seg.size else 'empty'}")

    _update_resistance_variable_mu_py(data, idx, float(getattr(tree.parameters, "murray_exponent", 3.0)), mu_seg)
    _dbg(f"F-L: updated resistances sample={data[:3,25] if data.shape[0]>=3 else data[:,25]}")
    flows = np.full(nseg, np.nan, dtype=float)
    delta_p = float(tree.parameters.root_pressure) - float(tree.parameters.terminal_pressure)
    root_R = data[0, 25]
    if root_R > 0:
        flows[0] = delta_p / root_R
        stack = [0]
        while stack:
            i = stack.pop()
            f_i = flows[i]
            left = idx[i, 0]
            right = idx[i, 1]
            has_left = not np.isnan(data[i, 15])
            has_right = not np.isnan(data[i, 16])
            if has_left and not has_right:
                flows[left] = f_i
                stack.append(left)
            elif has_right and not has_left:
                flows[right] = f_i
                stack.append(right)
            elif has_left and has_right:
                Rl = data[left, 25]
                Rr = data[right, 25]
                lbif = data[i, 23]
                rbif = data[i, 24]
                denom = 0.0
                if Rl > 0.0:
                    denom += (lbif**4) / Rl
                if Rr > 0.0:
                    denom += (rbif**4) / Rr
                if denom > 0.0:
                    flows[left] = f_i * ((lbif**4) / Rl) / denom if Rl > 0 else 0.0
                    flows[right] = f_i * ((rbif**4) / Rr) / denom if Rr > 0 else 0.0
                else:
                    flows[left] = flows[right] = f_i * 0.5
                stack.append(left)
                stack.append(right)
    n_terms = max(int(getattr(tree, "n_terminals", 0)) - 1, 1)
    desired_root_flow = original_root_flow
    if not np.isfinite(desired_root_flow) or desired_root_flow == 0.0:
        desired_root_flow = float(tree.parameters.terminal_flow) * float(n_terms)
    if np.isfinite(desired_root_flow) and np.isfinite(flows[0]) and flows[0] != 0.0:
        scale = desired_root_flow / flows[0]
        flows *= scale
    _dbg(f"F-L: updated flows sample={flows[:3] if flows.size else 'empty'}")

    try:
        if flows.size == data.shape[0]:
            data[:, 22] = flows
        tree.data[:nseg] = data
    except Exception:
        pass


def _normalize_hematocrit_model(value: str | None = None) -> str:
    mode = str(value or HEMATOCRIT_MODEL).strip().lower()
    if mode in ("uniform", "uniform_tube", "diameter", "diameter_only"):
        return "uniform_tube"
    if mode in ("pries", "pries_secomb", "phase_separation", "plasma_skimming"):
        return "pries_secomb"
    raise ValueError("hematocrit model must be 'uniform_tube' or 'pries_secomb'.")


def _tube_hematocrit_from_hd_radius(radii_cm: np.ndarray, hd: np.ndarray) -> np.ndarray:
    radii_arr = np.asarray(radii_cm, dtype=float)
    hd_arr = np.clip(np.asarray(hd, dtype=float), HEMATOCRIT_MIN, HEMATOCRIT_MAX)
    d_um = 2.0 * radii_arr * 1.0e4
    ratio = hd_arr + (1.0 - hd_arr) * (1.0 + 1.7 * np.exp(-0.415 * d_um) - 0.6 * np.exp(-0.011 * d_um))
    ht = hd_arr * ratio
    return np.where((radii_arr > 0.0) & np.isfinite(radii_arr), ht, 0.0)


if _HAVE_NUMBA:
    @njit
    def _phase_fraction_pries_numba(
        q_frac: float,
        d_parent_um: float,
        d_current_um: float,
        d_sibling_um: float,
        hd_parent: float,
        h_min: float,
        h_max: float,
    ) -> float:
        if q_frac <= 0.0:
            return 0.0
        if q_frac >= 1.0:
            return 1.0
        hp = hd_parent
        if hp < h_min:
            hp = h_min
        elif hp > h_max:
            hp = h_max
        dp = d_parent_um if d_parent_um > 1.0e-9 else 1.0e-9
        dc = d_current_um if d_current_um > 1.0e-9 else 1.0e-9
        ds = d_sibling_um if d_sibling_um > 1.0e-9 else 1.0e-9
        x0 = NETFLOW_BIFPAR_1 * (1.0 - hp) / dp
        if x0 < 0.0:
            x0 = 0.0
        elif x0 > 0.49:
            x0 = 0.49
        if q_frac <= x0:
            return 0.0
        if q_frac >= 1.0 - x0:
            return 1.0
        diaquot = (dc * dc) / (ds * ds)
        asym = (diaquot - 1.0) / (diaquot + 1.0)
        A = NETFLOW_BIFPAR_3 * asym * (1.0 - hp) / dp
        B = 1.0 + NETFLOW_BIFPAR_2 * (1.0 - hp) / dp
        z = (q_frac - x0) / (1.0 - 2.0 * x0)
        if z < 1.0e-12:
            z = 1.0e-12
        elif z > 1.0 - 1.0e-12:
            z = 1.0 - 1.0e-12
        y = A + B * np.log(z / (1.0 - z))
        if y > 50.0:
            return 1.0
        if y < -50.0:
            return 0.0
        return 1.0 / (1.0 + np.exp(-y))


    @njit
    def _propagate_hematocrit_pries_numba(
        order: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        flows_abs: np.ndarray,
        diam_um: np.ndarray,
        hd_root: float,
        h_min: float,
        h_max: float,
    ) -> np.ndarray:
        nseg = flows_abs.shape[0]
        hd = np.empty(nseg, dtype=np.float64)
        for i in range(nseg):
            hd[i] = np.nan
        root_val = hd_root
        if root_val < h_min:
            root_val = h_min
        elif root_val > h_max:
            root_val = h_max
        for oi in range(order.shape[0]):
            idx = int(order[oi])
            if idx < 0 or idx >= nseg:
                continue
            hp = hd[idx]
            if not np.isfinite(hp):
                hp = root_val
                hd[idx] = hp
            left = int(left_child[idx])
            right = int(right_child[idx])
            has_left = left >= 0 and left < nseg
            has_right = right >= 0 and right < nseg
            if not has_left and not has_right:
                continue
            if has_left and not has_right:
                hd[left] = hp
                continue
            if has_right and not has_left:
                hd[right] = hp
                continue

            ql = flows_abs[left]
            qr = flows_abs[right]
            if not np.isfinite(ql) or ql < 0.0:
                ql = 0.0
            if not np.isfinite(qr) or qr < 0.0:
                qr = 0.0
            qsum = ql + qr
            if qsum <= 1.0e-300:
                hd[left] = hp
                hd[right] = hp
                continue
            fq_l = ql / qsum
            if fq_l <= 1.0e-12:
                fe_l = 0.0
            elif fq_l >= 1.0 - 1.0e-12:
                fe_l = 1.0
            else:
                fe_l = _phase_fraction_pries_numba(
                    fq_l,
                    diam_um[idx],
                    diam_um[left],
                    diam_um[right],
                    hp,
                    h_min,
                    h_max,
                )
            fq_r = 1.0 - fq_l
            fe_r = 1.0 - fe_l
            if fq_l > 1.0e-12:
                h_l = hp * fe_l / fq_l
            else:
                h_l = h_min
            if fq_r > 1.0e-12:
                h_r = hp * fe_r / fq_r
            else:
                h_r = h_min
            if h_l < h_min:
                h_l = h_min
            elif h_l > h_max:
                h_l = h_max
            if h_r < h_min:
                h_r = h_min
            elif h_r > h_max:
                h_r = h_max
            hd[left] = h_l
            hd[right] = h_r
        for i in range(nseg):
            if not np.isfinite(hd[i]):
                hd[i] = root_val
        return hd


def _topdown_order_for_tree_data(data: np.ndarray, parents: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    nseg = int(data.shape[0])
    if parents is None:
        parents = np.nan_to_num(data[:, 17], nan=-1.0).astype(np.int64)
    parent_rows = np.arange(nseg, dtype=np.int64)
    nonroot = parents >= 0
    if np.all(parents[nonroot] < parent_rows[nonroot]):
        return parent_rows, "index"
    if _HAVE_NUMBA:
        left_child = np.nan_to_num(data[:, 15], nan=-1.0).astype(np.int64)
        right_child = np.nan_to_num(data[:, 16], nan=-1.0).astype(np.int64)
        roots = np.flatnonzero(parents < 0).astype(np.int64)
        order_candidate, n_order = _topdown_order_from_children_numba(left_child, right_child, roots, nseg)
        if int(n_order) == nseg:
            return order_candidate, "children_dfs"
        depths = np.asarray(data[:, 26], dtype=float)
        return np.argsort(depths, kind="mergesort").astype(np.int64, copy=False), f"depth_sort_after_children_dfs_{int(n_order)}"
    depths = np.asarray(data[:, 26], dtype=float)
    return np.argsort(depths, kind="mergesort").astype(np.int64, copy=False), "depth_sort"


def _hematocrit_context_for_tree(tree) -> dict[str, np.ndarray | str | int]:
    nseg = int(getattr(tree, "segment_count", 0))
    cached = getattr(tree, "_hematocrit_context", None)
    if isinstance(cached, dict) and int(cached.get("nseg", -1)) == nseg:
        return cached

    data = np.asarray(tree.data[:nseg])
    parents = np.nan_to_num(data[:, 17], nan=-1.0).astype(np.int64)
    left_child = np.nan_to_num(data[:, 15], nan=-1.0).astype(np.int64)
    right_child = np.nan_to_num(data[:, 16], nan=-1.0).astype(np.int64)
    order, order_mode = _topdown_order_for_tree_data(data, parents)
    radii = np.asarray(data[:, 21], dtype=float)
    context: dict[str, np.ndarray | str | int] = {
        "nseg": int(nseg),
        "parents": parents,
        "left_child": left_child,
        "right_child": right_child,
        "order": np.asarray(order, dtype=np.int64),
        "order_mode": str(order_mode),
        "radii": radii,
        "diam_um": np.asarray(2.0 * radii * 1.0e4, dtype=np.float64),
    }
    try:
        tree._hematocrit_context = context
    except Exception:
        pass
    return context


def _store_tree_hematocrit_cache(
    tree,
    HD: np.ndarray,
    HT: np.ndarray,
    *,
    model: str,
    flows: np.ndarray | None = None,
    fixed_flow_bc: bool = False,
) -> None:
    try:
        hd_arr = np.asarray(HD, dtype=np.float32)
        ht_arr = np.asarray(HT, dtype=np.float32)
        tree.discharge_hematocrit = hd_arr
        tree.tube_hematocrit = ht_arr
        # Chb_max stays in mol / m^3: HT is unitless and O2_CAP_PER_HCT is
        # already mol / m^3 per unit hematocrit.
        tree.Chb_max = ht_arr * float(O2_CAP_PER_HCT)
        tree._hematocrit_cache_nseg = int(hd_arr.shape[0])
        tree._hematocrit_cache_model = _normalize_hematocrit_model(model)
        tree._hematocrit_cache_flows_id = id(flows) if flows is not None else None
        tree._hematocrit_cache_fixed_flow_bc = bool(fixed_flow_bc)
    except Exception:
        pass


def _get_tree_hematocrit_cache(
    tree,
    nseg: int,
    *,
    model: str,
    flows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if int(getattr(tree, "_hematocrit_cache_nseg", -1)) != int(nseg):
            return None
        if getattr(tree, "_hematocrit_cache_model", None) != _normalize_hematocrit_model(model):
            return None
        flow_id = getattr(tree, "_hematocrit_cache_flows_id", None)
        fixed_flow_bc = bool(getattr(tree, "_hematocrit_cache_fixed_flow_bc", False))
        if not fixed_flow_bc and flows is not None and flow_id != id(flows):
            return None
        HD = np.asarray(getattr(tree, "discharge_hematocrit"), dtype=np.float32)
        HT = np.asarray(getattr(tree, "tube_hematocrit"), dtype=np.float32)
        if HD.shape[0] != nseg or HT.shape[0] != nseg:
            return None
        if not (np.all(np.isfinite(HD)) and np.all(np.isfinite(HT))):
            return None
        return HD, HT
    except Exception:
        return None


def compute_tree_hematocrit(
    tree,
    hd_root: float = HD_DISCHARGE,
    *,
    flows: np.ndarray | None = None,
    model: str | None = None,
    order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty

    context = _hematocrit_context_for_tree(tree)
    radii = np.asarray(context["radii"], dtype=float)
    mode = _normalize_hematocrit_model(model)

    hd = float(hd_root)
    if mode == "pries_secomb" and flows is not None and _HAVE_NUMBA:
        if order is None:
            order = np.asarray(context["order"], dtype=np.int64)
        HD = _propagate_hematocrit_pries_numba(
            np.asarray(order, dtype=np.int64),
            np.asarray(context["left_child"], dtype=np.int64),
            np.asarray(context["right_child"], dtype=np.int64),
            np.abs(np.asarray(flows, dtype=float)),
            np.asarray(context["diam_um"], dtype=np.float64),
            hd,
            float(HEMATOCRIT_MIN),
            float(HEMATOCRIT_MAX),
        )
    elif mode == "pries_secomb" and flows is not None and not _HAVE_NUMBA:
        raise RuntimeError("Pries-Secomb hematocrit propagation requires numba in this implementation.")
    else:
        HD = np.full(nseg, np.clip(hd, HEMATOCRIT_MIN, HEMATOCRIT_MAX), dtype=float)

    HT = _tube_hematocrit_from_hd_radius(radii, HD)

    return HD, HT


if _HAVE_NUMBA:
    @njit
    def _topdown_order_from_children_numba(
        left_child: np.ndarray,
        right_child: np.ndarray,
        roots: np.ndarray,
        nseg: int,
    ) -> tuple[np.ndarray, int]:
        order = np.empty(nseg, dtype=np.int64)
        stack = np.empty(nseg, dtype=np.int64)
        visited = np.zeros(nseg, dtype=np.uint8)
        top = 0
        for i in range(roots.shape[0] - 1, -1, -1):
            root = int(roots[i])
            if root >= 0 and root < nseg:
                stack[top] = root
                top += 1
        n_order = 0
        while top > 0:
            top -= 1
            idx = int(stack[top])
            if idx < 0 or idx >= nseg:
                continue
            if visited[idx] != 0:
                continue
            visited[idx] = 1
            order[n_order] = idx
            n_order += 1

            right = int(right_child[idx])
            if right >= 0 and right < nseg and visited[right] == 0:
                stack[top] = right
                top += 1
            left = int(left_child[idx])
            if left >= 0 and left < nseg and visited[left] == 0:
                stack[top] = left
                top += 1
        return order, n_order


def _solve_channel_concentrations_topdown(
    tree: Tree,
    flows: np.ndarray,
    *,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    fluid: str,
) -> Tuple[np.ndarray, np.ndarray]:
    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty

    t_total = perf_counter()
    t0 = perf_counter()
    data = np.asarray(tree.data[:nseg])
    hct_context = _hematocrit_context_for_tree(tree)
    lengths = np.asarray(data[:, 20], dtype=float)
    radii = np.asarray(hct_context["radii"], dtype=float)
    flows = np.asarray(flows, dtype=float)

    diffusivity_si = diffusivity * CM2_TO_M2
    lengths_si = lengths * CM_TO_M
    radii_si = radii * CM_TO_M
    flows_si = flows * CM3_TO_M3
    t_arrays = perf_counter() - t0

    fluid_mode = (fluid or getattr(getattr(tree, "parameters", None), "fluid", None) or ACTIVE_FLUID).lower()
    t_hema = 0.0
    hct_source = "none"
    HD = np.empty((0,), dtype=float)
    HT = np.empty((0,), dtype=float)
    if fluid_mode == "blood":
        t0 = perf_counter()
        cached_hct = _get_tree_hematocrit_cache(
            tree,
            nseg,
            model=HEMATOCRIT_MODEL,
            flows=flows,
        )
        if cached_hct is not None:
            HD, HT = cached_hct
            hct_source = "cache"
        else:
            HD, HT = compute_tree_hematocrit(
                tree,
                hd_root=HD_DISCHARGE,
                flows=flows,
                model=HEMATOCRIT_MODEL,
                order=np.asarray(hct_context["order"], dtype=np.int64),
            )
            _store_tree_hematocrit_cache(
                tree,
                HD,
                HT,
                model=HEMATOCRIT_MODEL,
                flows=flows,
                fixed_flow_bc=False,
            )
            hct_source = "computed"
        Chb_max = np.asarray(HT, dtype=float) * float(O2_CAP_PER_HCT)
        t_hema = perf_counter() - t0
    else:
        Chb_max = np.zeros_like(radii)

    t0 = perf_counter()
    parents = np.asarray(hct_context["parents"], dtype=np.int64)
    order = np.asarray(hct_context["order"], dtype=np.int64)
    order_mode = str(hct_context["order_mode"])
    t_order = perf_counter() - t0

    if _HAVE_NUMBA and CONC_USE_NUMBA:
        t0 = perf_counter()
        cin, cout = _solve_channel_concentrations_topdown_numba(
            order,
            parents,
            flows_si.astype(np.float64),
            radii_si.astype(np.float64),
            lengths_si.astype(np.float64),
            float(diffusivity_si),
            float(vmax),
            float(km),
            float(inlet_concentration),
            np.asarray(Chb_max, dtype=np.float64),
            bool(fluid_mode == "blood"),
            int(AXIAL_BLOOD_STEPS),
        )
        t_kernel = perf_counter() - t0
    else:
        t0 = perf_counter()
        cin = np.full(nseg, np.nan, dtype=float)
        cout = np.full(nseg, np.nan, dtype=float)
        root_mask = parents < 0
        cin[root_mask] = float(inlet_concentration)
        for idx in order:
            if not np.isfinite(cin[idx]):
                parent = parents[idx] if 0 <= idx < parents.size else -1
                if 0 <= parent < nseg and np.isfinite(cout[parent]):
                    cin[idx] = cout[parent]
                else:
                    cin[idx] = float(inlet_concentration)

            cin_local = float(max(cin[idx], 0.0))
            cin[idx] = cin_local
            if fluid_mode == "blood":
                decay = _blood_greens_decay_factor(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                    Chb_max[idx],
                )
            else:
                decay = _greens_decay_factor(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                )
            cout[idx] = cin_local * decay
        t_kernel = perf_counter() - t0

    if SOLVER_TIMING_DETAILS:
        hd_text = ""
        if fluid_mode == "blood" and HD.size:
            hd_q = np.quantile(HD[np.isfinite(HD)], [0.01, 0.5, 0.99]) if np.any(np.isfinite(HD)) else [float("nan")] * 3
            ht_q = np.quantile(HT[np.isfinite(HT)], [0.01, 0.5, 0.99]) if np.any(np.isfinite(HT)) else [float("nan")] * 3
            hd_text = (
                f" hematocrit_model={_normalize_hematocrit_model()} "
                f"source={hct_source} "
                f"HD[p01={hd_q[0]:.3f} p50={hd_q[1]:.3f} p99={hd_q[2]:.3f}] "
                f"HT[p01={ht_q[0]:.3f} p50={ht_q[1]:.3f} p99={ht_q[2]:.3f}]"
            )
        print(
            "  Concentration topdown: "
            f"nseg={nseg} fluid={fluid_mode} order={order_mode} axial_steps={AXIAL_BLOOD_STEPS if fluid_mode == 'blood' else 0} "
            f"arrays={_fmt_seconds(t_arrays)} hematocrit={_fmt_seconds(t_hema)} "
            f"order={_fmt_seconds(t_order)} kernel={_fmt_seconds(t_kernel)} "
            f"total={_fmt_seconds(perf_counter() - t_total)}{hd_text}"
        )

    return cin, cout


def _resolve_concentration_solver(value: str | None) -> str:
    if value is None:
        value = CONCENTRATION_SOLVER
    mode = str(value).strip().lower()
    if mode in ("topdown", "network", "topdown_ext", "topdown_ext_hybrid_bg", "topdown_ext_treecode"):
        return mode
    raise ValueError("concentration solver must be 'topdown', 'network', 'topdown_ext', 'topdown_ext_hybrid_bg', or 'topdown_ext_treecode'.")


def _solve_channel_concentrations(
    tree: Tree,
    flows: np.ndarray,
    inlet_nodes: Sequence[int],
    outlet_nodes: Optional[Sequence[int]],
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    prox_ids: np.ndarray | None = None,
    dist_ids: np.ndarray | None = None,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    fluid: str,
    solver: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mode = _resolve_concentration_solver(solver)
    global _LAST_CONCENTRATION_TIMINGS, _LAST_CEXT_SOURCE_STATE
    _LAST_CONCENTRATION_TIMINGS = _default_cext_timing_details()
    _LAST_CEXT_SOURCE_STATE = None
    if mode == "topdown":
        return _solve_channel_concentrations_topdown(
            tree,
            flows,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    if mode == "topdown_ext":
        return _solve_channel_concentrations_topdown_ext(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    if mode == "topdown_ext_hybrid_bg":
        return _solve_channel_concentrations_topdown_ext_hybrid_bg(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    if mode == "topdown_ext_treecode":
        return _solve_channel_concentrations_topdown_ext_treecode(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    cin, cout, _, _ = solve_network_concentrations(
        starts,
        ends,
        radii,
        lengths,
        flows,
        inlet_nodes,
        outlet_nodes,
        inlet_concentration,
        fluid=fluid,
        prox_ids=prox_ids,
        dist_ids=dist_ids,
        diffusivity=diffusivity,
        vmax=vmax,
        km=km,
    )
    return cin, cout


# ---------------------------------------------------------------------------
# Greens-based concentration model
# ---------------------------------------------------------------------------
def _build_k_ratio_lut(
    xmin: float = 1e-5,
    xmax: float = 5e2,
    n: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.exp(np.linspace(np.log(xmin), np.log(xmax), n))
    if _HAVE_SCIPY:
        k0_vals = _bessel_k0(xs)
        k1_vals = _bessel_k1(xs)
        ratio = k1_vals / np.maximum(k0_vals, 1e-30)
    else:
        gamma = 0.5772156649015329
        small = xs < 1e-2
        large = xs > 8.0
        invphi = 1.0 / np.maximum(xs, 1e-30)
        r_small = invphi / np.maximum(-(np.log(xs / 2.0) + gamma), 1e-8)
        r_large = 1.0 + 0.5 * invphi + 0.375 * (invphi ** 2)
        r_mid = (1.0 + 0.5658 * xs + 0.1373 * xs * xs) / (1.0 + 1.0361 * xs + 0.5454 * xs * xs)
        ratio = np.where(small, r_small, np.where(large, r_large, r_mid))
        k0_vals = np.empty_like(xs)
        k1_vals = ratio * 0.0
    return xs, k0_vals, k1_vals, ratio


def _load_k_ratio_lut(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        data = np.load(path)
    except Exception:
        return None
    if "xs" not in data:
        return None
    xs = np.asarray(data["xs"], dtype=float)
    ratio = np.asarray(data.get("ratio", data.get("ys", np.array([]))), dtype=float)
    k0_vals = np.asarray(data.get("k0", np.array([])), dtype=float)
    k1_vals = np.asarray(data.get("k1", np.array([])), dtype=float)
    if xs.ndim != 1 or ratio.ndim != 1 or xs.size < 2 or xs.size != ratio.size:
        return None
    if k0_vals.ndim != 1 or k0_vals.size != xs.size:
        k0_vals = np.empty_like(xs)
    if k1_vals.ndim != 1 or k1_vals.size != xs.size:
        k1_vals = np.empty_like(xs)
    return xs, k0_vals, k1_vals, ratio


_KRATIO_LUT = _load_k_ratio_lut(KRATIO_LUT_PATH)
if _KRATIO_LUT is None:
    _KRATIO_XS, _K0_LUT, _K1_LUT, _KRATIO_YS = _build_k_ratio_lut()
else:
    _KRATIO_XS, _K0_LUT, _K1_LUT, _KRATIO_YS = _KRATIO_LUT


def _k0est(x: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(math.erf)
    long_est = np.sqrt(np.pi / (2.0 * x) * np.exp(-x))
    short_est = -np.log(x / 2.0) - 0.5772 + ((x ** 2) / 4.0) * (np.log(x / 2.0) + 0.0772)
    asymp = 0.5 * (1.0 - vec_erf(3.0 * (x - 0.5)))
    return asymp * short_est + (1.0 - asymp) * long_est


@profile
def _k_ratio(phi: np.ndarray | float) -> np.ndarray:
    phi_arr = np.asarray(phi, dtype=float)
    phi_arr = np.maximum(phi_arr, 1e-12)
    flat = phi_arr.ravel()
    out = np.interp(flat, _KRATIO_XS, _KRATIO_YS, left=_KRATIO_YS[0], right=_KRATIO_YS[-1])
    return out.reshape(phi_arr.shape)


def _k0_lookup(x: np.ndarray) -> np.ndarray:
    if _K0_LUT.size:
        flat = np.asarray(x, dtype=float).ravel()
        out = np.interp(flat, _KRATIO_XS, _K0_LUT, left=_K0_LUT[0], right=_K0_LUT[-1])
        return out.reshape(np.asarray(x).shape)
    if _HAVE_SCIPY:
        return _bessel_k0(x)
    return _k0est(x)


def _k1_lookup(x: np.ndarray) -> np.ndarray:
    if _K1_LUT.size:
        flat = np.asarray(x, dtype=float).ravel()
        out = np.interp(flat, _KRATIO_XS, _K1_LUT, left=_K1_LUT[0], right=_K1_LUT[-1])
        return out.reshape(np.asarray(x).shape)
    if _HAVE_SCIPY:
        return _bessel_k1(x)
    return _k0est(x) * 0.0


if _HAVE_NUMBA:
    @njit(cache=True)
    def _interp_scalar(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[xs.shape[0] - 1]:
            return ys[ys.shape[0] - 1]
        idx = np.searchsorted(xs, x) - 1
        if idx < 0:
            idx = 0
        if idx >= xs.shape[0] - 1:
            idx = xs.shape[0] - 2
        x0 = xs[idx]
        x1 = xs[idx + 1]
        y0 = ys[idx]
        y1 = ys[idx + 1]
        if x1 == x0:
            return y0
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


    @njit(cache=True)
    def _k0_lookup_numba(x_arr: np.ndarray, xs: np.ndarray, k0_lut: np.ndarray) -> np.ndarray:
        out = np.empty_like(x_arr)
        for i in range(x_arr.size):
            out[i] = _interp_scalar(x_arr[i], xs, k0_lut)
        return out


    @njit(cache=True)
    def _k_ratio_scalar_numba(phi: float) -> float:
        x = phi
        if x < 1e-12:
            x = 1e-12
        return _interp_scalar(x, _KRATIO_XS, _KRATIO_YS)


    @njit(cache=True)
    def _lambda_if_scalar_numba(c_iv: float, diffusivity_si: float, vmax: float, km: float) -> float:
        c_local = c_iv if c_iv > 0.0 else 0.0
        denom = km + c_local
        if denom < 1e-30:
            denom = 1e-30
        return np.sqrt(diffusivity_si / max(vmax / denom, 1e-30))


    @njit(cache=True)
    def _interfacial_transfer_coeff_scalar_numba(
        radius_si: float,
        lambda_if: float,
        diffusivity_si: float,
        xs_lut: np.ndarray,
        ys_lut: np.ndarray,
    ) -> float:
        lambda_local = max(lambda_if, 1e-30)
        phi = radius_si / lambda_local
        if phi < 1e-12:
            phi = 1e-12
        ratio = _interp_scalar(phi, xs_lut, ys_lut)
        return (2.0 * np.pi * radius_si) * (diffusivity_si / lambda_local) * ratio


    @njit(cache=True)
    def _greens_decay_ratio_numba(
        flow: float,
        radius: float,
        length: float,
        diffusivity: float,
        vmax: float,
        km: float,
        cin: float,
    ) -> float:
        if length <= 0.0 or radius <= 0.0:
            return 1.0
        flow_mag = abs(flow)
        if flow_mag <= 0.0:
            return 1.0
        cin_pos = cin if cin > 0.0 else 0.0
        denom = km + cin_pos
        if denom < 1e-30:
            denom = 1e-30
        k1 = vmax / denom
        if k1 < 1e-30:
            k1 = 1e-30
        lam = (diffusivity / k1) ** 0.5
        if lam < 1e-30:
            lam = 1e-30
        phi = radius / lam
        ratio = _k_ratio_scalar_numba(phi)
        beta = (2.0 * np.pi * radius / (flow_mag if flow_mag > 1e-30 else 1e-30)) * (diffusivity / lam) * ratio
        exponent = -beta * length
        if exponent < -150.0:
            exponent = -150.0
        elif exponent > 50.0:
            exponent = 50.0
        return np.exp(exponent)


    @njit(cache=True)
    def _severinghaus_dSdP_scalar(P: float) -> float:
        den = (P * P * P + 150.0 * P + 23400.0)
        return 70200.0 * (P * P + 50.0) / (den * den)


    @njit(cache=True)
    def _blood_greens_decay_ratio_numba(
        flow: float,
        radius: float,
        length: float,
        diffusivity: float,
        vmax: float,
        km: float,
        cin: float,
        ccap: float,
        steps: int,
    ) -> float:
        if length <= 0.0 or radius <= 0.0:
            return 1.0
        flow_mag = abs(flow)
        if flow_mag <= 0.0:
            return 1.0
        cin_pos = cin if cin > 0.0 else 0.0
        denom = km + cin_pos
        if denom < 1e-30:
            denom = 1e-30
        k1 = vmax / denom
        if k1 < 1e-30:
            k1 = 1e-30
        lam = (diffusivity / k1) ** 0.5
        if lam < 1e-30:
            lam = 1e-30
        phi = radius / lam
        ratio = _k_ratio_scalar_numba(phi)
        base = (2.0 * np.pi * radius / (flow_mag if flow_mag > 1e-30 else 1e-30)) * (diffusivity / lam) * ratio
        if steps < 1:
            steps = 1
        ds = length / steps
        c_local = cin_pos
        cap = ccap if ccap > 0.0 else 0.0
        for _ in range(steps):
            P_mmHg = c_local / ALPHA_MMHG
            buffer = 1.0 + cap * _severinghaus_dSdP_scalar(P_mmHg) / ALPHA_MMHG
            if buffer < 1e-30:
                buffer = 1e-30
            beta_local = base / buffer
            exponent = -beta_local * ds
            if exponent < -150.0:
                exponent = -150.0
            elif exponent > 50.0:
                exponent = 50.0
            c_local *= np.exp(exponent)
        denom_cin = cin if cin > 1e-30 else 1e-30
        return c_local / denom_cin


    @njit(cache=True)
    def _solve_channel_concentrations_topdown_numba(
        order: np.ndarray,
        parents: np.ndarray,
        flows_si: np.ndarray,
        radii_si: np.ndarray,
        lengths_si: np.ndarray,
        diffusivity_si: float,
        vmax: float,
        km: float,
        inlet_concentration: float,
        chb_max: np.ndarray,
        is_blood: bool,
        axial_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        nseg = parents.shape[0]
        cin = np.empty(nseg, dtype=np.float64)
        cout = np.empty(nseg, dtype=np.float64)
        for i in range(nseg):
            cin[i] = np.nan
            cout[i] = np.nan
        for i in range(nseg):
            if parents[i] < 0:
                cin[i] = inlet_concentration
        for oi in range(order.shape[0]):
            idx = int(order[oi])
            if idx < 0 or idx >= nseg:
                continue
            if not np.isfinite(cin[idx]):
                p = int(parents[idx])
                if p >= 0 and p < nseg and np.isfinite(cout[p]):
                    cin[idx] = cout[p]
                else:
                    cin[idx] = inlet_concentration
            cin_local = cin[idx]
            if not np.isfinite(cin_local) or cin_local < 0.0:
                cin_local = 0.0
            cin[idx] = cin_local
            if is_blood:
                decay = _blood_greens_decay_ratio_numba(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                    chb_max[idx],
                    axial_steps,
                )
            else:
                decay = _greens_decay_ratio_numba(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                )
            cout[idx] = cin_local * decay
        return cin, cout


    @njit(cache=True, parallel=True)
    def _tissue_kernel_numba(
        points_si: np.ndarray,
        starts_si: np.ndarray,
        segment_vectors: np.ndarray,
        seg_len: np.ndarray,
        radii_si: np.ndarray,
        nearest_idx: np.ndarray,
        proj_raw: np.ndarray,
        d_center: np.ndarray,
        keep_mask: np.ndarray,
        cin_pos: np.ndarray,
        alpha_edge: np.ndarray,
        flow_sign: np.ndarray,
        diffusivity_si: float,
        km: float,
        vmax: float,
        window_factor: float,
        lam_ref: float,
        gl_nodes: np.ndarray,
        gl_weights: np.ndarray,
        xs_lut: np.ndarray,
        k0_lut: np.ndarray,
    ) -> np.ndarray:
        n_points = points_si.shape[0]
        out = np.zeros(n_points, dtype=np.float64)
        for row in prange(n_points):
            if not keep_mask[row]:
                continue
            seg_idx = nearest_idx[row]
            if seg_idx.size == 0:
                continue
            proj_row = proj_raw[row]
            d_local = d_center[row]
            cap_max = 1e-6
            total = 0.0
            for local_i in range(seg_idx.size):
                seg_i = seg_idx[local_i]
                if seg_i < 0:
                    continue
                L = seg_len[seg_i]
                if L <= 0.0:
                    continue
                proj = proj_row[local_i]
                if flow_sign[seg_i] < 0.0:
                    proj = 1.0 - proj
                if proj < 0.0:
                    proj = 0.0
                if proj > 1.0:
                    proj = 1.0
                s_star = proj * L
                Cc_star = cin_pos[seg_i] * np.exp(-alpha_edge[seg_i] * s_star)
                denom_gate = km + max(Cc_star, 1e-12)
                if denom_gate < 1e-30:
                    denom_gate = 1e-30
                lam_gate = np.sqrt(diffusivity_si / max(vmax / denom_gate, 1e-30))
                if d_local[local_i] > window_factor * lam_gate:
                    continue
                if Cc_star > cap_max:
                    cap_max = Cc_star

                halfW = window_factor * lam_ref
                s0 = s_star - halfW
                if s0 < 0.0:
                    s0 = 0.0
                s1 = s_star + halfW
                if s1 > L:
                    s1 = L
                if s1 <= s0 + 1e-15:
                    continue

                mid = 0.5 * (s0 + s1)
                half = 0.5 * (s1 - s0)
                total_local = 0.0
                for g in range(gl_nodes.shape[0]):
                    s = mid + half * gl_nodes[g]
                    t = s / L
                    xs = starts_si[seg_i] + t * segment_vectors[seg_i]
                    r_vec0 = points_si[row, 0] - xs[0]
                    r_vec1 = points_si[row, 1] - xs[1]
                    r_vec2 = points_si[row, 2] - xs[2]
                    r = np.sqrt(r_vec0 * r_vec0 + r_vec1 * r_vec1 + r_vec2 * r_vec2)
                    if r <= 1e-12:
                        continue
                    cc_s = cin_pos[seg_i] * np.exp(-alpha_edge[seg_i] * s)
                    denom_loc = km + max(cc_s, 1e-12)
                    if denom_loc < 1e-30:
                        denom_loc = 1e-30
                    lam_loc = np.sqrt(diffusivity_si / max(vmax / denom_loc, 1e-30))
                    phi_loc = radii_si[seg_i] / max(lam_loc, 1e-30)
                    if phi_loc < 1e-12:
                        phi_loc = 1e-12
                    r_over_lam = r / max(lam_loc, 1e-30)
                    k0_num = _interp_scalar(r_over_lam, xs_lut, k0_lut)
                    k0_den = _interp_scalar(phi_loc, xs_lut, k0_lut)
                    if k0_den < 1e-300:
                        k0_den = 1e-300
                    Ci_R = cc_s * (k0_num / k0_den)

                    denom_corr = km + min(3.0 * Ci_R, cc_s)
                    if denom_corr < 1e-30:
                        denom_corr = 1e-30
                    lam_corr = np.sqrt(diffusivity_si / max(vmax / denom_corr, 1e-30))
                    phi = radii_si[seg_i] / max(lam_corr, 1e-30)
                    if phi < 1e-12:
                        phi = 1e-12
                    ratio = _interp_scalar(phi, _KRATIO_XS, _KRATIO_YS)
                    wall_factor = (diffusivity_si / max(lam_corr, 1e-30)) * ratio
                    q_s = (2.0 * np.pi * radii_si[seg_i]) * wall_factor * cc_s
                    kernel = np.exp(-r / max(lam_corr, 1e-30)) / (4.0 * np.pi * r)
                    integrand = q_s * (kernel / diffusivity_si)
                    total_local += gl_weights[g] * integrand

                total += half * total_local

            if total < 0.0 or not np.isfinite(total):
                total = 0.0
            if cap_max > -np.inf:
                out[row] = min(total, cap_max)
            else:
                out[row] = total
        return out


def _greens_lambda_char(diffusivity: float, vmax: float, km: float, cin: float) -> float:
    denom = max(km + max(cin, 0.0), 1e-30)
    k1 = vmax / denom
    return float(np.sqrt(diffusivity / max(k1, 1e-30)))


def _greens_decay_factor(
    flow: float,
    radius: float,
    length: float,
    diffusivity: float,
    vmax: float,
    km: float,
    cin: float,
) -> float:
    if length <= 0.0 or radius <= 0.0:
        return 1.0
    flow_mag = abs(flow)
    if flow_mag <= 0.0:
        return 1.0
    lam = _greens_lambda_char(diffusivity, vmax, km, cin)
    phi = radius / max(lam, 1e-30)
    ratio = float(_k_ratio(phi))
    beta = (2.0 * np.pi * radius / max(flow_mag, 1e-30)) * (diffusivity / max(lam, 1e-30)) * ratio
    exponent = np.clip(-beta * length, -150.0, 50.0)
    return float(np.exp(exponent))


def _greens_segment_params(
    cin: np.ndarray,
    radii: np.ndarray,
    flows: np.ndarray,
    diffusivity: float,
    vmax: float,
    km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cin_pos = np.maximum(cin, 0.0)
    denom = np.maximum(km + cin_pos,  1e-30)
    k1 = vmax / denom
    lam = np.sqrt(diffusivity / np.maximum(k1, 1e-30))
    phi = radii / np.maximum(lam, 1e-30)
    ratio = _k_ratio(phi)
    flow_mag = np.maximum(np.abs(flows), 1e-30)
    beta = (2.0 * np.pi * radii / flow_mag) * (diffusivity / np.maximum(lam, 1e-30)) * ratio
    return lam, ratio, beta


def _hill_dS_dC(
    cin: float,
    *,
    alpha: float = ALPHA_MMHG,
    p50: float = P50_MMHG,
    n_hill: float = N_HILL,
) -> float:
    cin_pos = max(cin, 0.0)
    alpha_n = alpha**n_hill
    p50_n = p50**n_hill
    denom = alpha_n * p50_n + cin_pos**n_hill
    return float((n_hill * alpha_n * p50_n * cin_pos ** (n_hill - 1.0)) / max(denom * denom, 1e-30))


def severinghaus_saturation(P_mmHg: np.ndarray | float) -> np.ndarray | float:
    P = np.asarray(P_mmHg, dtype=float)
    num = P ** 3 + 150.0 * P
    den = num + 23400.0
    S = num / den
    if np.isscalar(P_mmHg):
        return float(S)
    return S


def severinghaus_dSdP(P_mmHg: np.ndarray | float) -> np.ndarray | float:
    P = np.asarray(P_mmHg, dtype=float)
    den = (P ** 3 + 150.0 * P + 23400.0)
    dSdP = 70200.0 * (P ** 2 + 50.0) / (den ** 2)
    if np.isscalar(P_mmHg):
        return float(dSdP)
    return dSdP


def segment_O2_capacity(radius_cm: float, hd: float = HD_DISCHARGE) -> float:
    HT = tube_hematocrit(radius_cm, hd=hd)
    # O2_CAP_PER_HCT is already mol / m^3 per unit hematocrit.
    return float(HT * O2_CAP_PER_HCT)


def segment_O2_capacity_from_HT(HT: float) -> float:
    # O2_CAP_PER_HCT is already mol / m^3 per unit hematocrit.
    return float(HT * O2_CAP_PER_HCT)


def buffer_factor_B(C_plasma: float, Chb_max: float) -> float:
    C_plasma = max(C_plasma, 1e-12)
    P_mmHg = C_plasma / ALPHA_MMHG
    dSdP = severinghaus_dSdP(P_mmHg)
    return float(1.0 + (Chb_max / ALPHA_MMHG) * dSdP)


def _blood_greens_decay_factor(
    flow: float,
    radius: float,
    length: float,
    diffusivity: float,
    vmax: float,
    km: float,
    cin: float,
    ccap: float,
) -> float:
    if length <= 0.0 or radius <= 0.0:
        return 1.0
    flow_mag = abs(flow)
    if flow_mag <= 0.0:
        return 1.0
    lam = _greens_lambda_char(diffusivity, vmax, km, cin)
    phi = radius / max(lam, 1e-30)
    ratio = float(_k_ratio(phi))
    base = (2.0 * np.pi * radius / max(flow_mag, 1e-30)) * (diffusivity / max(lam, 1e-30)) * ratio
    steps = max(int(AXIAL_BLOOD_STEPS), 1)
    ds = length / steps
    c_local = float(max(cin, 0.0))
    for _ in range(steps):
        buffer = 1.0 + max(ccap, 0.0) * severinghaus_dSdP(c_local / ALPHA_MMHG) / ALPHA_MMHG
        beta_local = base / max(buffer, 1e-30)
        c_local *= float(np.exp(np.clip(-beta_local * ds, -150.0, 50.0)))
    return c_local / max(cin, 1e-30)


def _lambda_if_from_civ(
    c_iv: np.ndarray | float,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> np.ndarray:
    c_local = np.maximum(np.asarray(c_iv, dtype=float), 0.0)
    denom = np.maximum(km + c_local, 1e-30)
    return np.sqrt(diffusivity_si / np.maximum(vmax / denom, 1e-30))


def _lambda_tissue_from_civ_ctissue(
    c_iv: np.ndarray | float,
    c_tissue: np.ndarray | float,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> np.ndarray:
    c_iv_arr = np.maximum(np.asarray(c_iv, dtype=float), 0.0)
    c_tissue_arr = np.maximum(np.asarray(c_tissue, dtype=float), 0.0)
    effective_term = np.sqrt(np.maximum(km + c_iv_arr, 1e-30) * np.maximum(km + c_tissue_arr, 1e-30))
    return np.sqrt(diffusivity_si / np.maximum(vmax / effective_term, 1e-30))


def _normalize_cext_lambda_source(value: str | None = None) -> str:
    mode = str(CEXT_LAMBDA_SOURCE if value is None else value).strip().lower()
    aliases = {
        "vv": "lambda_vv",
        "lambda_vv": "lambda_vv",
        "vessel": "lambda_vv",
        "vessel_vessel": "lambda_vv",
        "if": "lambda_vv",
        "lambda_if": "lambda_vv",
        "wall": "lambda_vv",
        "t": "lambda_t",
        "lambda_t": "lambda_t",
        "tissue": "lambda_t",
        "lambda_tissue": "lambda_t",
    }
    if mode not in aliases:
        raise ValueError("SVV_CEXT_LAMBDA_SOURCE must be 'lambda_vv' or 'lambda_t'.")
    return aliases[mode]


def _cext_green_lambda_from_fields(
    c_wall: np.ndarray,
    c_ext: np.ndarray,
    lambda_wall: np.ndarray,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> np.ndarray:
    if _normalize_cext_lambda_source() == "lambda_vv":
        return np.maximum(np.asarray(lambda_wall, dtype=float), 1e-30)
    return np.maximum(
        _lambda_tissue_from_civ_ctissue(c_wall, c_ext, diffusivity_si, vmax, km),
        1e-30,
    )


def _interfacial_transfer_coefficient(
    radius_si: np.ndarray | float,
    lambda_if: np.ndarray | float,
    diffusivity_si: float,
) -> np.ndarray:
    radius_arr = np.maximum(np.asarray(radius_si, dtype=float), 0.0)
    lambda_arr = np.maximum(np.asarray(lambda_if, dtype=float), 1e-30)
    phi = np.maximum(radius_arr / lambda_arr, 1e-12)
    ratio = _k_ratio(phi)
    return (2.0 * np.pi * radius_arr) * (diffusivity_si / lambda_arr) * ratio


def _finite_radius_o2_term_flags() -> tuple[bool, bool]:
    mode = str(FINITE_RADIUS_O2_TERMS or "none").strip().lower()
    if mode in ("0", "false", "off", "none"):
        return False, False
    if mode in ("monopole", "mono", "monopole2"):
        return True, False
    if mode in ("dipole", "dipole2"):
        return False, True
    if mode in ("both", "all", "monopole+dipole"):
        return True, True
    raise ValueError("--finite-radius-o2-terms must be one of none, monopole, dipole, both.")


def _graetz_velocity_profile(rho: np.ndarray, profile_name: str) -> np.ndarray:
    profile = str(profile_name or "poiseuille").strip().lower()
    rho_arr = np.asarray(rho, dtype=float)
    if profile == "plug":
        return np.ones_like(rho_arr, dtype=float)
    if profile == "poiseuille":
        return 2.0 * np.maximum(1.0 - rho_arr * rho_arr, 0.0)
    raise ValueError("--graetz-profile must be 'poiseuille' or 'plug'.")


def _graetz_build_matrices(
    Bi: float,
    n_radial: int,
    profile_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = max(int(n_radial), 2)
    rho_nodes = np.linspace(0.0, 1.0, n, dtype=float)
    K = np.zeros((n, n), dtype=float)
    M = np.zeros((n, n), dtype=float)
    cup = np.zeros((n,), dtype=float)
    gp = np.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], dtype=float)
    gw = np.array([1.0, 1.0], dtype=float)
    for elem in range(n - 1):
        x0 = float(rho_nodes[elem])
        x1 = float(rho_nodes[elem + 1])
        h = max(x1 - x0, 1e-30)
        ids = (elem, elem + 1)
        for xi, wi in zip(gp, gw):
            rho = 0.5 * (x0 + x1) + 0.5 * h * float(xi)
            jac = 0.5 * h * float(wi)
            N0 = (x1 - rho) / h
            N1 = (rho - x0) / h
            dN = np.array([-1.0 / h, 1.0 / h], dtype=float)
            N = np.array([N0, N1], dtype=float)
            v = float(_graetz_velocity_profile(np.array([rho], dtype=float), profile_name)[0])
            stiff_w = rho * jac
            mass_w = rho * v * jac
            for a in range(2):
                ia = ids[a]
                cup[ia] += 2.0 * mass_w * N[a]
                for b in range(2):
                    ib = ids[b]
                    K[ia, ib] += stiff_w * dN[a] * dN[b]
                    M[ia, ib] += mass_w * N[a] * N[b]
    K[-1, -1] += max(float(Bi), 0.0)
    cup_sum = float(np.sum(cup))
    if cup_sum > 0.0:
        cup /= cup_sum
    return K, M, rho_nodes, cup


def _graetz_get_basis(
    Bi: float,
    n_radial: int,
    n_modes: int,
    profile_name: str,
) -> dict:
    if _scipy_linalg is None:
        raise RuntimeError("Graetz closure requires scipy.linalg in the active Python environment.")
    n = max(int(n_radial), 2)
    nm = max(1, min(int(n_modes), n))
    K, M, rho, cup = _graetz_build_matrices(float(Bi), n, profile_name)
    eigvals, eigvecs = _scipy_linalg.eigh(K, M, check_finite=False)
    order = np.argsort(eigvals)
    mu2 = np.maximum(np.asarray(eigvals[order[:nm]], dtype=float), 0.0)
    phi = np.asarray(eigvecs[:, order[:nm]], dtype=float)
    for j in range(phi.shape[1]):
        norm = float(math.sqrt(max(phi[:, j].T @ M @ phi[:, j], 1e-30)))
        phi[:, j] /= norm
    return {
        "Bi": float(Bi),
        "rho": rho,
        "K": K,
        "M": M,
        "mu2": mu2,
        "phi": phi,
        "project": phi.T @ M,
        "cup_weights": cup,
    }


def _graetz_get_basis_table(
    n_radial: int,
    n_modes: int,
    profile_name: str,
    *,
    bi_per_decade: int | None = None,
    min_bi: float | None = None,
    max_bi: float | None = None,
) -> dict:
    n = max(int(n_radial), 2)
    nm = max(1, min(int(n_modes), n))
    profile = str(profile_name or "poiseuille").strip().lower()
    per_decade = max(int(bi_per_decade if bi_per_decade is not None else GRAETZ_BI_CACHE_PER_DECADE), 1)
    bi_min = max(float(min_bi if min_bi is not None else GRAETZ_MIN_BI), 1e-30)
    bi_max = max(float(max_bi if max_bi is not None else GRAETZ_MAX_BI), bi_min)
    key_min = int(round(math.log10(bi_min) * per_decade))
    key_max = int(round(math.log10(bi_max) * per_decade))
    cache_key = (profile, n, nm, per_decade, key_min, key_max)
    cached = _GRAETZ_BASIS_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    n_keys = key_max - key_min + 1
    mu2_table = np.zeros((n_keys, nm), dtype=np.float32)
    phi_table = np.zeros((n_keys, n, nm), dtype=np.float32)
    project_table = np.zeros((n_keys, nm, n), dtype=np.float32)
    cup_table = np.zeros((n_keys, n), dtype=np.float32)
    bi_values = np.zeros((n_keys,), dtype=np.float32)
    for idx, key in enumerate(range(key_min, key_max + 1)):
        Bi = 10.0 ** (float(key) / float(per_decade))
        basis = _graetz_get_basis(Bi, n, nm, profile)
        bi_values[idx] = np.float32(Bi)
        mu2_table[idx, :] = np.asarray(basis["mu2"], dtype=np.float32)
        phi_table[idx, :, :] = np.asarray(basis["phi"], dtype=np.float32)
        project_table[idx, :, :] = np.asarray(basis["project"], dtype=np.float32)
        cup_table[idx, :] = np.asarray(basis["cup_weights"], dtype=np.float32)
    table = {
        "profile": profile,
        "n_radial": n,
        "n_modes": nm,
        "per_decade": per_decade,
        "key_min": key_min,
        "key_max": key_max,
        "min_bi": bi_min,
        "max_bi": bi_max,
        "bi_values": bi_values,
        "mu2": mu2_table,
        "phi": phi_table,
        "project": project_table,
        "cup_weights": cup_table,
    }
    _GRAETZ_BASIS_TABLE_CACHE[cache_key] = table
    return table


def _resolve_cext_accel_mode(*, require_gpu: bool = False) -> str:
    mode = str(CEXT_ACCEL_MODE or "cpu").strip().lower()
    if mode not in ("cpu", "gpu", "auto"):
        raise ValueError("--cext-accel must be 'cpu', 'gpu', or 'auto'.")
    if mode == "auto":
        if _cupy_device_available():
            return "gpu"
        return "cpu"
    if mode == "gpu" and not _cupy_device_available():
        if require_gpu:
            raise RuntimeError(
                "Cext GPU mode requested but no compatible CuPy CUDA device is available. "
                "Verify `import cupy` in the svva2 environment."
            )
        return "cpu"
    return mode


def _resolve_cext_frozen_accel_mode(*, require_gpu: bool = False) -> str:
    mode = str(CEXT_FROZEN_ACCEL_MODE or "cpu").strip().lower()
    if mode not in ("cpu", "gpu", "auto"):
        raise ValueError("--cext-frozen-accel must be 'cpu', 'gpu', or 'auto'.")
    if mode == "auto":
        if _cupy_device_available():
            return "gpu"
        return "cpu"
    if mode == "gpu" and not _cupy_device_available():
        if require_gpu:
            raise RuntimeError(
                "Frozen Cext GPU mode requested but no compatible CuPy CUDA device is available. "
                "Verify `import cupy` in the svva2 environment."
            )
        return "cpu"
    return mode


def _build_topdown_level_slices(order: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order_arr = np.asarray(order, dtype=np.int64).reshape(-1)
    parents_arr = np.asarray(parents, dtype=np.int64).reshape(-1)
    nseg = int(parents_arr.shape[0])
    if nseg <= 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, np.zeros((1,), dtype=np.int32), empty
    if _HAVE_NUMBA:
        return _build_topdown_level_slices_numba(order_arr, parents_arr)
    depth = np.zeros((nseg,), dtype=np.int32)
    for idx in order_arr:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= nseg:
            continue
        parent = int(parents_arr[idx_i])
        if parent >= 0 and parent < nseg:
            depth[idx_i] = np.int32(depth[parent] + 1)
        else:
            depth[idx_i] = np.int32(0)
    max_depth = int(np.max(depth)) if depth.size else 0
    counts = np.zeros((max_depth + 1,), dtype=np.int32)
    for idx in order_arr:
        idx_i = int(idx)
        if 0 <= idx_i < nseg:
            counts[int(depth[idx_i])] += 1
    level_offsets = np.zeros((counts.size + 1,), dtype=np.int32)
    np.cumsum(counts, out=level_offsets[1:])
    write_pos = level_offsets[:-1].copy()
    level_order = np.empty((nseg,), dtype=np.int32)
    for idx in order_arr:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= nseg:
            continue
        d = int(depth[idx_i])
        pos = int(write_pos[d])
        level_order[pos] = np.int32(idx_i)
        write_pos[d] = np.int32(pos + 1)
    return level_order, level_offsets, depth


if _HAVE_NUMBA:
    @njit(cache=True)
    def _build_topdown_level_slices_numba(
        order_arr: np.ndarray,
        parents_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nseg = int(parents_arr.shape[0])
        if nseg <= 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.zeros((1,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )
        depth = np.zeros((nseg,), dtype=np.int32)
        max_depth = 0
        for ii in range(order_arr.shape[0]):
            idx_i = int(order_arr[ii])
            if idx_i < 0 or idx_i >= nseg:
                continue
            parent = int(parents_arr[idx_i])
            if parent >= 0 and parent < nseg:
                depth[idx_i] = np.int32(depth[parent] + 1)
            else:
                depth[idx_i] = np.int32(0)
            if int(depth[idx_i]) > max_depth:
                max_depth = int(depth[idx_i])

        counts = np.zeros((max_depth + 1,), dtype=np.int32)
        for ii in range(order_arr.shape[0]):
            idx_i = int(order_arr[ii])
            if idx_i >= 0 and idx_i < nseg:
                counts[int(depth[idx_i])] += np.int32(1)

        level_offsets = np.zeros((counts.shape[0] + 1,), dtype=np.int32)
        total = 0
        for ii in range(counts.shape[0]):
            total += int(counts[ii])
            level_offsets[ii + 1] = np.int32(total)

        write_pos = level_offsets[:-1].copy()
        level_order = np.empty((nseg,), dtype=np.int32)
        for ii in range(order_arr.shape[0]):
            idx_i = int(order_arr[ii])
            if idx_i < 0 or idx_i >= nseg:
                continue
            d = int(depth[idx_i])
            pos = int(write_pos[d])
            level_order[pos] = np.int32(idx_i)
            write_pos[d] = np.int32(pos + 1)
        return level_order, level_offsets, depth

    @njit(cache=True)
    def _build_local_exclusion_arrays_numba(
        parents: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        max_local: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        nseg = int(parents.shape[0])
        out_idx = np.full((nseg, max_local), -1, dtype=np.int32)
        out_count = np.zeros((nseg,), dtype=np.uint8)
        for seg in range(nseg):
            buf = np.full((max_local,), -1, dtype=np.int32)
            count = 0

            def add_unique(value: int, count_local: int) -> int:
                if value < 0 or value >= nseg:
                    return count_local
                for ii in range(count_local):
                    if buf[ii] == value:
                        return count_local
                if count_local < max_local:
                    buf[count_local] = np.int32(value)
                    return count_local + 1
                return count_local

            def add_immediate(node: int, count_local: int) -> int:
                if node < 0 or node >= nseg:
                    return count_local
                parent = int(parents[node])
                count_local = add_unique(parent, count_local)
                left = int(left_child[node])
                right = int(right_child[node])
                count_local = add_unique(left, count_local)
                count_local = add_unique(right, count_local)
                if parent >= 0 and parent < nseg:
                    pl = int(left_child[parent])
                    pr = int(right_child[parent])
                    if pl != node:
                        count_local = add_unique(pl, count_local)
                    if pr != node:
                        count_local = add_unique(pr, count_local)
                return count_local

            count = add_unique(seg, count)
            count = add_immediate(seg, count)
            seed_count = count
            for ii in range(seed_count):
                count = add_immediate(int(buf[ii]), count)
            out_count[seg] = np.uint8(count)
            for ii in range(count):
                out_idx[seg, ii] = buf[ii]
        return out_idx, out_count


    @njit(cache=True, parallel=True)
    def _solve_topdown_ext_frozen_numba(
        level_order: np.ndarray,
        level_offsets: np.ndarray,
        parents: np.ndarray,
        flows_si: np.ndarray,
        radii_si: np.ndarray,
        lengths_si: np.ndarray,
        gl_t: np.ndarray,
        c_ext_gl: np.ndarray,
        diffusivity_si: float,
        vmax: float,
        km: float,
        inlet_concentration: float,
        chb_max: np.ndarray,
        is_blood: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nseg = int(parents.shape[0])
        m = int(gl_t.shape[0])
        cin_seg = np.full((nseg,), inlet_concentration, dtype=np.float32)
        cout_seg = np.full((nseg,), inlet_concentration, dtype=np.float32)
        c_iv_gl = np.full((nseg, m), inlet_concentration, dtype=np.float32)
        for level_idx in range(level_offsets.shape[0] - 1):
            start = int(level_offsets[level_idx])
            end = int(level_offsets[level_idx + 1])
            for pos in prange(start, end):
                seg_idx = int(level_order[pos])
                if seg_idx < 0 or seg_idx >= nseg:
                    continue
                parent = int(parents[seg_idx])
                cin_local = inlet_concentration
                if parent >= 0 and parent < nseg:
                    cin_local = float(cout_seg[parent])
                if cin_local < VESS_CONC_FLOOR:
                    cin_local = VESS_CONC_FLOOR
                cin_seg[seg_idx] = np.float32(cin_local)
                flow_mag_si = abs(float(flows_si[seg_idx]))
                if flow_mag_si <= 1e-30:
                    flow_mag_si = 1e-30
                radius_si = float(radii_si[seg_idx])
                length_si = float(lengths_si[seg_idx])
                c_running = float(cin_local)
                prev_s = 0.0
                for node_idx in range(m):
                    s_target = float(gl_t[node_idx]) * length_si
                    ds_step = s_target - prev_s
                    if ds_step < 0.0:
                        ds_step = 0.0
                    c_ext_local = float(c_ext_gl[seg_idx, node_idx])
                    if c_ext_local < 0.0:
                        c_ext_local = 0.0
                    lambda_if_up = _lambda_if_scalar_numba(c_running, diffusivity_si, vmax, km)
                    k_if_up = _interfacial_transfer_coeff_scalar_numba(radius_si, lambda_if_up, diffusivity_si, _KRATIO_XS, _KRATIO_YS)
                    beta_if = k_if_up / flow_mag_si
                    if is_blood > 0:
                        buffer = 1.0 + max(float(chb_max[seg_idx]), 0.0) * _severinghaus_dSdP_scalar(c_running / ALPHA_MMHG) / ALPHA_MMHG
                        if buffer < 1e-30:
                            buffer = 1e-30
                        beta_if = beta_if / buffer
                    exponent = -beta_if * ds_step
                    if exponent < -150.0:
                        exponent = -150.0
                    elif exponent > 50.0:
                        exponent = 50.0
                    c_node = c_ext_local + (c_running - c_ext_local) * math.exp(exponent)
                    if c_node < VESS_CONC_FLOOR:
                        c_node = VESS_CONC_FLOOR
                    c_iv_gl[seg_idx, node_idx] = np.float32(c_node)
                    c_running = c_node
                    prev_s = s_target
                ds_tail = length_si - prev_s
                if ds_tail < 0.0:
                    ds_tail = 0.0
                c_ext_tail = 0.0
                if m > 0:
                    c_ext_tail = float(c_ext_gl[seg_idx, m - 1])
                    if c_ext_tail < 0.0:
                        c_ext_tail = 0.0
                lambda_if_tail = _lambda_if_scalar_numba(c_running, diffusivity_si, vmax, km)
                k_if_tail = _interfacial_transfer_coeff_scalar_numba(radius_si, lambda_if_tail, diffusivity_si, _KRATIO_XS, _KRATIO_YS)
                beta_tail = k_if_tail / flow_mag_si
                if is_blood > 0:
                    buffer_tail = 1.0 + max(float(chb_max[seg_idx]), 0.0) * _severinghaus_dSdP_scalar(c_running / ALPHA_MMHG) / ALPHA_MMHG
                    if buffer_tail < 1e-30:
                        buffer_tail = 1e-30
                    beta_tail = beta_tail / buffer_tail
                exponent_tail = -beta_tail * ds_tail
                if exponent_tail < -150.0:
                    exponent_tail = -150.0
                elif exponent_tail > 50.0:
                    exponent_tail = 50.0
                c_out = c_ext_tail + (c_running - c_ext_tail) * math.exp(exponent_tail)
                if c_out < VESS_CONC_FLOOR:
                    c_out = VESS_CONC_FLOOR
                cout_seg[seg_idx] = np.float32(c_out)
        return cin_seg, cout_seg, c_iv_gl


def _solve_topdown_ext_frozen_python(
    level_order: np.ndarray,
    level_offsets: np.ndarray,
    parents: np.ndarray,
    flows_si: np.ndarray,
    radii_si: np.ndarray,
    lengths_si: np.ndarray,
    gl_t: np.ndarray,
    c_ext_gl: np.ndarray,
    diffusivity_si: float,
    vmax: float,
    km: float,
    inlet_concentration: float,
    chb_max: np.ndarray,
    is_blood: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nseg = int(np.asarray(parents).shape[0])
    m = int(np.asarray(gl_t).shape[0])
    cin_seg = np.zeros((nseg,), dtype=np.float32)
    cout_seg = np.zeros((nseg,), dtype=np.float32)
    c_iv_gl = np.zeros((nseg, m), dtype=np.float32)
    for level_idx in range(int(np.asarray(level_offsets).shape[0]) - 1):
        start_pos = int(level_offsets[level_idx])
        end_pos = int(level_offsets[level_idx + 1])
        for pos in range(start_pos, end_pos):
            seg_idx = int(level_order[pos])
            parent = int(parents[seg_idx])
            cin_local = float(inlet_concentration) if parent < 0 else float(cout_seg[parent])
            cin_local = max(cin_local, VESS_CONC_FLOOR)
            cin_seg[seg_idx] = np.float32(cin_local)
            flow_mag_si = max(abs(float(flows_si[seg_idx])), 1e-30)
            length_si = float(lengths_si[seg_idx])
            radius_si = float(radii_si[seg_idx])
            c_running = cin_local
            prev_s = 0.0
            for node_idx, gl_t_i in enumerate(np.asarray(gl_t, dtype=float)):
                s_target = float(gl_t_i) * length_si
                ds_step = max(s_target - prev_s, 0.0)
                c_ext_local = max(float(c_ext_gl[seg_idx, node_idx]), 0.0)
                lambda_if_up = float(_lambda_if_from_civ(c_running, float(diffusivity_si), float(vmax), float(km)))
                k_if_up = float(_interfacial_transfer_coefficient(radius_si, lambda_if_up, float(diffusivity_si)))
                beta_if = k_if_up / flow_mag_si
                if is_blood:
                    buffer = 1.0 + max(float(chb_max[seg_idx]), 0.0) * severinghaus_dSdP(c_running / ALPHA_MMHG) / ALPHA_MMHG
                    beta_if = beta_if / max(float(buffer), 1e-30)
                c_node = c_ext_local + (c_running - c_ext_local) * float(np.exp(np.clip(-beta_if * ds_step, -150.0, 50.0)))
                c_node = max(c_node, VESS_CONC_FLOOR)
                c_iv_gl[seg_idx, node_idx] = np.float32(c_node)
                c_running = c_node
                prev_s = s_target
            ds_tail = max(length_si - prev_s, 0.0)
            c_ext_tail = max(float(c_ext_gl[seg_idx, -1]), 0.0) if m > 0 else 0.0
            lambda_tail = float(_lambda_if_from_civ(c_running, float(diffusivity_si), float(vmax), float(km)))
            k_if_tail = float(_interfacial_transfer_coefficient(radius_si, lambda_tail, float(diffusivity_si)))
            beta_tail = k_if_tail / flow_mag_si
            if is_blood:
                buffer_tail = 1.0 + max(float(chb_max[seg_idx]), 0.0) * severinghaus_dSdP(c_running / ALPHA_MMHG) / ALPHA_MMHG
                beta_tail = beta_tail / max(float(buffer_tail), 1e-30)
            cout_seg[seg_idx] = np.float32(
                max(
                    c_ext_tail + (c_running - c_ext_tail) * float(np.exp(np.clip(-beta_tail * ds_tail, -150.0, 50.0))),
                    VESS_CONC_FLOOR,
                )
            )
    return cin_seg, cout_seg, c_iv_gl


    @njit(cache=True)
    def _cext_is_local_excluded_numba(source_seg: int, row: np.ndarray, count: int) -> int:
        for idx in range(count):
            if int(row[idx]) == source_seg:
                return 1
        return 0


    @njit(cache=True, parallel=True)
    def _compute_cext_batch_numba(
        target_seg_ids: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        gl_points_si: np.ndarray,
        midpoints_si: np.ndarray,
        radii_si: np.ndarray,
        lambda_iv_gl: np.ndarray,
        q_weighted_gl: np.ndarray,
        seg_cap_gl: np.ndarray,
        exclude_idx: np.ndarray,
        exclude_count: np.ndarray,
        diffusivity_si: float,
        window_factor: float,
    ) -> np.ndarray:
        batch_n = int(target_seg_ids.shape[0])
        m = int(gl_points_si.shape[1])
        out = np.zeros((batch_n, m), dtype=np.float32)
        for flat_idx in prange(batch_n * m):
            batch_idx = flat_idx // m
            target_node = flat_idx - batch_idx * m
            target_seg = int(target_seg_ids[batch_idx])
            target_point_x = float(gl_points_si[target_seg, target_node, 0])
            target_point_y = float(gl_points_si[target_seg, target_node, 1])
            target_point_z = float(gl_points_si[target_seg, target_node, 2])
            target_lambda = float(lambda_iv_gl[target_seg, target_node])
            if target_lambda < 1e-30:
                target_lambda = 1e-30
            target_radius = float(radii_si[target_seg])
            total = 0.0
            cap_max = 0.0
            row_start = int(row_ptr[batch_idx])
            row_end = int(row_ptr[batch_idx + 1])
            excl_count = int(exclude_count[target_seg])
            for pos in range(row_start, row_end):
                source_seg = int(col_idx[pos])
                if _cext_is_local_excluded_numba(source_seg, exclude_idx[target_seg], excl_count) != 0:
                    continue
                radius_sum = target_radius + float(radii_si[source_seg])
                dx_mid = float(midpoints_si[source_seg, 0]) - float(midpoints_si[target_seg, 0])
                dy_mid = float(midpoints_si[source_seg, 1]) - float(midpoints_si[target_seg, 1])
                dz_mid = float(midpoints_si[source_seg, 2]) - float(midpoints_si[target_seg, 2])
                coarse_r = math.sqrt(dx_mid * dx_mid + dy_mid * dy_mid + dz_mid * dz_mid + radius_sum * radius_sum)
                source_lambda_max = 1e-30
                for source_node in range(m):
                    source_lambda_probe = float(lambda_iv_gl[source_seg, source_node])
                    if source_lambda_probe > source_lambda_max:
                        source_lambda_max = source_lambda_probe
                coarse_lambda = target_lambda if target_lambda >= source_lambda_max else source_lambda_max
                if coarse_r > window_factor * coarse_lambda:
                    continue
                seg_cap = float(seg_cap_gl[source_seg])
                seg_contributed = 0
                for source_node in range(m):
                    source_lambda = float(lambda_iv_gl[source_seg, source_node])
                    if source_lambda < 1e-30:
                        source_lambda = 1e-30
                    source_point_x = float(gl_points_si[source_seg, source_node, 0])
                    source_point_y = float(gl_points_si[source_seg, source_node, 1])
                    source_point_z = float(gl_points_si[source_seg, source_node, 2])
                    dx = source_point_x - target_point_x
                    dy = source_point_y - target_point_y
                    dz = source_point_z - target_point_z
                    r = math.sqrt(dx * dx + dy * dy + dz * dz + radius_sum * radius_sum)
                    if r > window_factor * source_lambda:
                        continue
                    if r < 1e-30:
                        r = 1e-30
                    kernel = math.exp(-r / source_lambda) / (4.0 * math.pi * diffusivity_si * r)
                    total += float(q_weighted_gl[source_seg, source_node]) * kernel
                    seg_contributed = 1
                if seg_contributed == 1 and seg_cap > cap_max:
                    cap_max = seg_cap
            if total < 0.0 or not np.isfinite(total):
                total = 0.0
            if cap_max > 0.0 and total > cap_max:
                total = cap_max
            out[batch_idx, target_node] = np.float32(total)
        return out


_CEXT_GPU_KERNEL = None
_CEXT_GPU_DIRECT_KERNEL = None
_CEXT_GPU_LOCAL_CORR_KERNEL = None
_CEXT_ITERATION_CACHE_KERNEL = None
_CEXT_HYBRID_BG_KERNEL = None
_CEXT_HYBRID_DEPOSIT_KERNEL = None
_CEXT_FFT_MOMENT_DEPOSIT_KERNEL = None
_CEXT_FFT_MOMENT_DEPOSIT_BATCH_KERNEL = None
_CEXT_FFT_DISCRETE_SELF_KERNEL = None
_CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL = None
_CEXT_TREECODE_KERNEL = None
_CEXT_TREECODE_DEPOSIT_KERNEL = None
_CEXT_TREECODE_UPSWEEP_KERNEL = None
_FROZEN_TOPDOWN_GPU_KERNEL = None
_FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL = None
_CEXT_TISSUE_CELL_GPU_KERNEL = None


def _get_cext_gpu_kernel():
    global _CEXT_GPU_KERNEL
    if _CEXT_GPU_KERNEL is not None:
        return _CEXT_GPU_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __device__ float interp_lut(float x, const float* xs, const float* ys, int n) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid;
        else hi = mid;
    }
    float x0 = xs[lo];
    float x1 = xs[lo + 1];
    float y0 = ys[lo];
    float y1 = ys[lo + 1];
    if (x1 == x0) return y0;
    float t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

extern "C" __global__ void cext_kernel(
    const int* target_seg_ids,
    const int* row_ptr,
    const int* col_idx,
    const float* gl_points_si,
    const float* midpoints_si,
    const float* segment_vectors,
    const float* radii_si,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    const int* exclude_idx,
    const unsigned char* exclude_count,
    int gl_order,
    int exclude_width,
    int batch_n,
    float diffusivity_si,
    float window_factor,
    float* out
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = batch_n * gl_order;
    if (flat >= total_targets) return;
    int batch_idx = flat / gl_order;
    int target_node = flat - batch_idx * gl_order;
    int target_seg = target_seg_ids[batch_idx];
    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float target_lambda = lambda_iv_gl[target_seg * gl_order + target_node];
    if (target_lambda < 1.0e-30f) target_lambda = 1.0e-30f;
    float target_radius = radii_si[target_seg];
    float target_mid_x = midpoints_si[target_seg * 3 + 0];
    float target_mid_y = midpoints_si[target_seg * 3 + 1];
    float target_mid_z = midpoints_si[target_seg * 3 + 2];

    float total = 0.0f;
    float cap_max = 0.0f;
    int row_start = row_ptr[batch_idx];
    int row_end = row_ptr[batch_idx + 1];
    int excl_n = (int)exclude_count[target_seg];
    for (int pos = row_start; pos < row_end; ++pos) {
        int source_seg = col_idx[pos];
        int excluded = 0;
        for (int ei = 0; ei < excl_n; ++ei) {
            if (exclude_idx[target_seg * exclude_width + ei] == source_seg) {
                excluded = 1;
                break;
            }
        }
        if (excluded) continue;
        float radius_sum = target_radius + radii_si[source_seg];
        float mdx = midpoints_si[source_seg * 3 + 0] - target_mid_x;
        float mdy = midpoints_si[source_seg * 3 + 1] - target_mid_y;
        float mdz = midpoints_si[source_seg * 3 + 2] - target_mid_z;
        float coarse_r = sqrtf(mdx * mdx + mdy * mdy + mdz * mdz + radius_sum * radius_sum);
        float source_lambda_max = 1.0e-30f;
        for (int source_node = 0; source_node < gl_order; ++source_node) {
            float source_lambda_probe = lambda_iv_gl[source_seg * gl_order + source_node];
            if (source_lambda_probe > source_lambda_max) source_lambda_max = source_lambda_probe;
        }
        float coarse_lambda = target_lambda;
        if (source_lambda_max > coarse_lambda) coarse_lambda = source_lambda_max;
        if (coarse_r > window_factor * coarse_lambda) continue;
        float seg_cap = seg_cap_gl[source_seg];
        int seg_contributed = 0;
        for (int source_node = 0; source_node < gl_order; ++source_node) {
            float source_lambda = lambda_iv_gl[source_seg * gl_order + source_node];
            if (source_lambda < 1.0e-30f) source_lambda = 1.0e-30f;
            float sx = gl_points_si[(source_seg * gl_order + source_node) * 3 + 0];
            float sy = gl_points_si[(source_seg * gl_order + source_node) * 3 + 1];
            float sz = gl_points_si[(source_seg * gl_order + source_node) * 3 + 2];
            float dx = tx - sx;
            float dy = ty - sy;
            float dz = tz - sz;
            float r_center2 = dx * dx + dy * dy + dz * dz;
            float r = sqrtf(r_center2 + radius_sum * radius_sum);
            if (r > window_factor * source_lambda) continue;
            if (r < 1.0e-30f) r = 1.0e-30f;
            float kernel = expf(-r / source_lambda) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
            int source_flat = source_seg * gl_order + source_node;
            total += q_weighted_gl[source_flat] * kernel;
            float o2_weight = mono2_weight_gl[source_flat] + dipole2_weight_gl[source_flat];
            if (o2_weight != 0.0f) {
                float vx = segment_vectors[source_seg * 3 + 0];
                float vy = segment_vectors[source_seg * 3 + 1];
                float vz = segment_vectors[source_seg * 3 + 2];
                float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
                if (vlen > 1.0e-30f) {
                    float tdotr = (vx * dx + vy * dy + vz * dz) / vlen;
                    float mu2 = (tdotr * tdotr) / (r * r);
                    if (mu2 > 1.0f) mu2 = 1.0f;
                    float inv_r = 1.0f / r;
                    float inv_l = 1.0f / source_lambda;
                    float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                              + (1.0f - mu2) * inv_l * inv_l) * kernel;
                    total += o2_weight * pH;
                }
            }
            seg_contributed = 1;
        }
        if (seg_contributed && seg_cap > cap_max) cap_max = seg_cap;
    }
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    if (cap_max > 0.0f && total > cap_max) total = cap_max;
    out[target_seg * gl_order + target_node] = total;
}
'''
    _CEXT_GPU_KERNEL = _cp.RawKernel(code, "cext_kernel")
    return _CEXT_GPU_KERNEL


def _get_cext_gpu_direct_kernel():
    global _CEXT_GPU_DIRECT_KERNEL
    if _CEXT_GPU_DIRECT_KERNEL is not None:
        return _CEXT_GPU_DIRECT_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_direct_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* midpoints_si,
    const float* segment_vectors,
    const float* radii_si,
    const float* reach_si,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    const int* exclude_idx,
    const unsigned char* exclude_count,
    const int* cell_ptr,
    const int* cell_seg_ids,
    int cell_origin_x,
    int cell_origin_y,
    int cell_origin_z,
    int dim_x,
    int dim_y,
    int dim_z,
    float cell_size,
    float max_reach_si,
    float reach_scale,
    int gl_order,
    int exclude_width,
    int target_n,
    float diffusivity_si,
    float window_factor,
    float* out_total,
    float* out_cap
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_n * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];

    float target_mid_x = midpoints_si[target_seg * 3 + 0];
    float target_mid_y = midpoints_si[target_seg * 3 + 1];
    float target_mid_z = midpoints_si[target_seg * 3 + 2];
    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float target_lambda = lambda_iv_gl[target_seg * gl_order + target_node];
    if (target_lambda < 1.0e-30f) target_lambda = 1.0e-30f;
    float target_radius = radii_si[target_seg];
    float target_reach = reach_scale * reach_si[target_seg];
    float query_reach = target_reach + reach_scale * max_reach_si;

    int lo_x = (int)floorf((target_mid_x - query_reach) / cell_size) - cell_origin_x;
    int lo_y = (int)floorf((target_mid_y - query_reach) / cell_size) - cell_origin_y;
    int lo_z = (int)floorf((target_mid_z - query_reach) / cell_size) - cell_origin_z;
    int hi_x = (int)floorf((target_mid_x + query_reach) / cell_size) - cell_origin_x;
    int hi_y = (int)floorf((target_mid_y + query_reach) / cell_size) - cell_origin_y;
    int hi_z = (int)floorf((target_mid_z + query_reach) / cell_size) - cell_origin_z;
    if (lo_x < 0) lo_x = 0;
    if (lo_y < 0) lo_y = 0;
    if (lo_z < 0) lo_z = 0;
    if (hi_x >= dim_x) hi_x = dim_x - 1;
    if (hi_y >= dim_y) hi_y = dim_y - 1;
    if (hi_z >= dim_z) hi_z = dim_z - 1;

    float total = 0.0f;
    float cap_max = 0.0f;
    int excl_n = (int)exclude_count[target_seg];
    for (int ix = lo_x; ix <= hi_x; ++ix) {
        for (int iy = lo_y; iy <= hi_y; ++iy) {
            for (int iz = lo_z; iz <= hi_z; ++iz) {
                int cell_flat = ix + dim_x * (iy + dim_y * iz);
                int row_start = cell_ptr[cell_flat];
                int row_end = cell_ptr[cell_flat + 1];
                for (int pos = row_start; pos < row_end; ++pos) {
                    int source_seg = cell_seg_ids[pos];
                    int excluded = 0;
                    for (int ei = 0; ei < excl_n; ++ei) {
                        if (exclude_idx[target_seg * exclude_width + ei] == source_seg) {
                            excluded = 1;
                            break;
                        }
                    }
                    if (excluded) continue;
                    float source_reach = reach_scale * reach_si[source_seg];
                    float radius_sum = target_radius + radii_si[source_seg];
                    float mdx = midpoints_si[source_seg * 3 + 0] - target_mid_x;
                    float mdy = midpoints_si[source_seg * 3 + 1] - target_mid_y;
                    float mdz = midpoints_si[source_seg * 3 + 2] - target_mid_z;
                    float coarse_r2 = mdx * mdx + mdy * mdy + mdz * mdz;
                    float pair_limit = target_reach + source_reach;
                    if (coarse_r2 > pair_limit * pair_limit) continue;
                    float coarse_r = sqrtf(coarse_r2 + radius_sum * radius_sum);
                    float source_lambda_max = 1.0e-30f;
                    for (int source_node = 0; source_node < gl_order; ++source_node) {
                        float source_lambda_probe = lambda_iv_gl[source_seg * gl_order + source_node];
                        if (source_lambda_probe > source_lambda_max) source_lambda_max = source_lambda_probe;
                    }
                    float coarse_lambda = target_lambda;
                    if (source_lambda_max > coarse_lambda) coarse_lambda = source_lambda_max;
                    if (coarse_r > window_factor * coarse_lambda) continue;
                    float seg_cap = seg_cap_gl[source_seg];
                    int seg_contributed = 0;
                    for (int source_node = 0; source_node < gl_order; ++source_node) {
                        float source_lambda = lambda_iv_gl[source_seg * gl_order + source_node];
                        if (source_lambda < 1.0e-30f) source_lambda = 1.0e-30f;
                        float sx = gl_points_si[(source_seg * gl_order + source_node) * 3 + 0];
                        float sy = gl_points_si[(source_seg * gl_order + source_node) * 3 + 1];
                        float sz = gl_points_si[(source_seg * gl_order + source_node) * 3 + 2];
                        float dx = tx - sx;
                        float dy = ty - sy;
                        float dz = tz - sz;
                        float r_center2 = dx * dx + dy * dy + dz * dz;
                        float r = sqrtf(r_center2 + radius_sum * radius_sum);
                        if (r > window_factor * source_lambda) continue;
                        if (r < 1.0e-30f) r = 1.0e-30f;
                        float kernel = expf(-r / source_lambda) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                        int source_flat = source_seg * gl_order + source_node;
                        total += q_weighted_gl[source_flat] * kernel;
                        float o2_weight = mono2_weight_gl[source_flat] + dipole2_weight_gl[source_flat];
                        if (o2_weight != 0.0f) {
                            float vx = segment_vectors[source_seg * 3 + 0];
                            float vy = segment_vectors[source_seg * 3 + 1];
                            float vz = segment_vectors[source_seg * 3 + 2];
                            float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
                            if (vlen > 1.0e-30f) {
                                float tdotr = (vx * dx + vy * dy + vz * dz) / vlen;
                                float mu2 = (tdotr * tdotr) / (r * r);
                                if (mu2 > 1.0f) mu2 = 1.0f;
                                float inv_r = 1.0f / r;
                                float inv_l = 1.0f / source_lambda;
                                float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                                          + (1.0f - mu2) * inv_l * inv_l) * kernel;
                                total += o2_weight * pH;
                            }
                        }
                        seg_contributed = 1;
                    }
                    if (seg_contributed && seg_cap > cap_max) cap_max = seg_cap;
                }
            }
        }
    }
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out_total[target_idx * gl_order + target_node] = total;
    out_cap[target_idx * gl_order + target_node] = cap_max;
}
'''
    _CEXT_GPU_DIRECT_KERNEL = _cp.RawKernel(code, "cext_direct_kernel")
    return _CEXT_GPU_DIRECT_KERNEL


def _get_frozen_topdown_gpu_kernel():
    global _FROZEN_TOPDOWN_GPU_KERNEL
    if _FROZEN_TOPDOWN_GPU_KERNEL is not None:
        return _FROZEN_TOPDOWN_GPU_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __device__ float interp_lut(float x, const float* xs, const float* ys, int n) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    float x0 = xs[lo];
    float x1 = xs[lo + 1];
    float y0 = ys[lo];
    float y1 = ys[lo + 1];
    if (x1 == x0) return y0;
    float t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

extern "C" __global__ void frozen_topdown_kernel(
    const int* seg_ids,
    int n_level,
    const int* parents,
    const float* flows_si,
    const float* radii_si,
    const float* lengths_si,
    const float* gl_t,
    const float* c_ext_gl,
    int gl_order,
    float diffusivity_si,
    float vmax,
    float km,
    float inlet_concentration,
    const float* chb_max,
    int is_blood,
    float alpha_mmhg,
    float vess_floor,
    const float* xs_lut,
    const float* ratio_lut,
    int lut_n,
    float* cin_seg,
    float* cout_seg,
    float* c_iv_gl
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_level) return;
    int seg_idx = seg_ids[idx];
    int parent = parents[seg_idx];
    float cin_local = inlet_concentration;
    if (parent >= 0) cin_local = cout_seg[parent];
    if (cin_local < vess_floor) cin_local = vess_floor;
    cin_seg[seg_idx] = cin_local;

    float flow_mag_si = fabsf(flows_si[seg_idx]);
    if (flow_mag_si <= 1.0e-30f) flow_mag_si = 1.0e-30f;
    float radius_si = radii_si[seg_idx];
    float length_si = lengths_si[seg_idx];
    float c_running = cin_local;
    float prev_s = 0.0f;

    for (int node_idx = 0; node_idx < gl_order; ++node_idx) {
        float s_target = gl_t[node_idx] * length_si;
        float ds_step = s_target - prev_s;
        if (ds_step < 0.0f) ds_step = 0.0f;
        float c_ext_local = c_ext_gl[seg_idx * gl_order + node_idx];
        if (c_ext_local < 0.0f) c_ext_local = 0.0f;

        float c_pos = c_running;
        if (c_pos < 0.0f) c_pos = 0.0f;
        float denom = km + c_pos;
        if (denom < 1.0e-30f) denom = 1.0e-30f;
        float k1 = vmax / denom;
        if (k1 < 1.0e-30f) k1 = 1.0e-30f;
        float lambda_if = sqrtf(diffusivity_si / k1);
        if (lambda_if < 1.0e-30f) lambda_if = 1.0e-30f;
        float phi = radius_si / lambda_if;
        if (phi < 1.0e-12f) phi = 1.0e-12f;
        float ratio = interp_lut(phi, xs_lut, ratio_lut, lut_n);
        float k_if = (2.0f * 3.14159265358979323846f * radius_si) * (diffusivity_si / lambda_if) * ratio;
        float beta_if = k_if / flow_mag_si;
        if (is_blood > 0) {
            float P = c_running / alpha_mmhg;
            float den = (P * P * P + 150.0f * P + 23400.0f);
            float dsdP = 70200.0f * (P * P + 50.0f) / (den * den);
            float buffer = 1.0f + fmaxf(chb_max[seg_idx], 0.0f) * dsdP / alpha_mmhg;
            if (buffer < 1.0e-30f) buffer = 1.0e-30f;
            beta_if /= buffer;
        }
        float exponent = -beta_if * ds_step;
        if (exponent < -150.0f) exponent = -150.0f;
        else if (exponent > 50.0f) exponent = 50.0f;
        float c_node = c_ext_local + (c_running - c_ext_local) * expf(exponent);
        if (c_node < vess_floor) c_node = vess_floor;
        c_iv_gl[seg_idx * gl_order + node_idx] = c_node;
        c_running = c_node;
        prev_s = s_target;
    }

    float ds_tail = length_si - prev_s;
    if (ds_tail < 0.0f) ds_tail = 0.0f;
    float c_ext_tail = 0.0f;
    if (gl_order > 0) {
        c_ext_tail = c_ext_gl[seg_idx * gl_order + gl_order - 1];
        if (c_ext_tail < 0.0f) c_ext_tail = 0.0f;
    }
    float c_pos = c_running;
    if (c_pos < 0.0f) c_pos = 0.0f;
    float denom = km + c_pos;
    if (denom < 1.0e-30f) denom = 1.0e-30f;
    float k1 = vmax / denom;
    if (k1 < 1.0e-30f) k1 = 1.0e-30f;
    float lambda_if = sqrtf(diffusivity_si / k1);
    if (lambda_if < 1.0e-30f) lambda_if = 1.0e-30f;
    float phi = radius_si / lambda_if;
    if (phi < 1.0e-12f) phi = 1.0e-12f;
    float ratio = interp_lut(phi, xs_lut, ratio_lut, lut_n);
    float k_if = (2.0f * 3.14159265358979323846f * radius_si) * (diffusivity_si / lambda_if) * ratio;
    float beta_tail = k_if / flow_mag_si;
    if (is_blood > 0) {
        float P = c_running / alpha_mmhg;
        float den = (P * P * P + 150.0f * P + 23400.0f);
        float dsdP = 70200.0f * (P * P + 50.0f) / (den * den);
        float buffer = 1.0f + fmaxf(chb_max[seg_idx], 0.0f) * dsdP / alpha_mmhg;
        if (buffer < 1.0e-30f) buffer = 1.0e-30f;
        beta_tail /= buffer;
    }
    float exponent_tail = -beta_tail * ds_tail;
    if (exponent_tail < -150.0f) exponent_tail = -150.0f;
    else if (exponent_tail > 50.0f) exponent_tail = 50.0f;
    float c_out = c_ext_tail + (c_running - c_ext_tail) * expf(exponent_tail);
    if (c_out < vess_floor) c_out = vess_floor;
    cout_seg[seg_idx] = c_out;
}
'''
    _FROZEN_TOPDOWN_GPU_KERNEL = _cp.RawKernel(code, "frozen_topdown_kernel")
    return _FROZEN_TOPDOWN_GPU_KERNEL


def _solve_topdown_ext_frozen_gpu(
    context: dict,
    c_ext_gl: np.ndarray,
    *,
    diffusivity_si: float,
    vmax: float,
    km: float,
    inlet_concentration: float,
    chb_max: np.ndarray,
    fluid_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    kernel = _get_frozen_topdown_gpu_kernel()
    gl_order = int(np.asarray(context["gl_t"]).shape[0])
    nseg = int(np.asarray(context["parents"]).shape[0])

    t_upload = perf_counter()
    c_ext_g = _cp.asarray(np.asarray(c_ext_gl, dtype=np.float32))
    chb_max_g = _cp.asarray(np.asarray(chb_max, dtype=np.float32))
    cin_seg_g = _cp.full((nseg,), np.float32(inlet_concentration), dtype=_cp.float32)
    cout_seg_g = _cp.full((nseg,), np.float32(inlet_concentration), dtype=_cp.float32)
    c_iv_gl_g = _cp.full((nseg, gl_order), np.float32(inlet_concentration), dtype=_cp.float32)
    _cp.cuda.Stream.null.synchronize()
    upload_time = perf_counter() - t_upload

    threads = 128
    t_kernel = perf_counter()
    for level_idx in range(int(np.asarray(context["level_offsets"]).shape[0]) - 1):
        start = int(context["level_offsets"][level_idx])
        stop = int(context["level_offsets"][level_idx + 1])
        n_level = max(stop - start, 0)
        if n_level <= 0:
            continue
        seg_ids_g = static["level_order"][start:stop]
        blocks = (n_level + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                seg_ids_g,
                np.int32(n_level),
                static["parents"],
                static["flows_si"],
                static["radii_si"],
                static["lengths_si"],
                static["gl_t"],
                c_ext_g.ravel(),
                np.int32(gl_order),
                np.float32(diffusivity_si),
                np.float32(vmax),
                np.float32(km),
                np.float32(inlet_concentration),
                chb_max_g,
                np.int32(1 if str(fluid_mode).lower() == "blood" else 0),
                np.float32(ALPHA_MMHG),
                np.float32(VESS_CONC_FLOOR),
                static["xs_lut"],
                static["ratio_lut"],
                np.int32(int(np.asarray(_KRATIO_XS).size)),
                cin_seg_g,
                cout_seg_g,
                c_iv_gl_g.ravel(),
            ),
        )
    _cp.cuda.Stream.null.synchronize()
    kernel_time = perf_counter() - t_kernel

    t_download = perf_counter()
    cin_seg = _cp.asnumpy(cin_seg_g)
    cout_seg = _cp.asnumpy(cout_seg_g)
    c_iv_gl = _cp.asnumpy(c_iv_gl_g)
    _cp.cuda.Stream.null.synchronize()
    download_time = perf_counter() - t_download

    timings = {
        "upload_s": float(upload_time),
        "kernel_s": float(kernel_time),
        "download_s": float(download_time),
    }
    return cin_seg, cout_seg, c_iv_gl, timings


def _ensure_graetz_gpu_basis(context: dict) -> dict:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    if int(GRAETZ_N_RADIAL) > 32 or int(GRAETZ_N_MODES) > 16:
        raise RuntimeError("GPU Graetz closure supports up to 32 radial nodes and 16 modes in this v1 kernel.")
    table = _graetz_get_basis_table(
        int(GRAETZ_N_RADIAL),
        int(GRAETZ_N_MODES),
        str(GRAETZ_VELOCITY_PROFILE),
    )
    key = (
        table["profile"],
        int(table["n_radial"]),
        int(table["n_modes"]),
        int(table["per_decade"]),
        int(table["key_min"]),
        int(table["key_max"]),
    )
    cached = context.get("graetz_gpu_basis")
    if isinstance(cached, dict) and cached.get("key") == key:
        return cached
    cached = {
        "key": key,
        "mu2": _cp.asarray(np.asarray(table["mu2"], dtype=np.float32)),
        "phi": _cp.asarray(np.asarray(table["phi"], dtype=np.float32)),
        "project": _cp.asarray(np.asarray(table["project"], dtype=np.float32)),
        "cup_weights": _cp.asarray(np.asarray(table["cup_weights"], dtype=np.float32)),
        "key_min": int(table["key_min"]),
        "key_max": int(table["key_max"]),
        "per_decade": int(table["per_decade"]),
        "min_bi": float(table["min_bi"]),
        "max_bi": float(table["max_bi"]),
        "n_radial": int(table["n_radial"]),
        "n_modes": int(table["n_modes"]),
    }
    context["graetz_gpu_basis"] = cached
    return cached


def _get_frozen_topdown_graetz_gpu_kernel():
    global _FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL
    if _FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL is not None:
        return _FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __device__ float interp_lut_graetz(float x, const float* xs, const float* ys, int n) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid;
        else hi = mid;
    }
    float x0 = xs[lo];
    float x1 = xs[lo + 1];
    float y0 = ys[lo];
    float y1 = ys[lo + 1];
    if (x1 == x0) return y0;
    float t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

extern "C" __device__ int graetz_basis_index(
    float Bi,
    int key_min,
    int key_max,
    int per_decade,
    float min_bi,
    float max_bi
) {
    float b = fminf(fmaxf(Bi, min_bi), max_bi);
    int key = (int)lrintf(log10f(b) * (float)per_decade);
    if (key < key_min) key = key_min;
    if (key > key_max) key = key_max;
    return key - key_min;
}

extern "C" __device__ float lambda_from_wall_graetz(
    float c_wall,
    float diffusivity_si,
    float vmax,
    float km
) {
    float c = fmaxf(c_wall, 0.0f);
    float denom = fmaxf(km + c, 1.0e-30f);
    float rate = fmaxf(vmax / denom, 1.0e-30f);
    return fmaxf(sqrtf(diffusivity_si / rate), 1.0e-30f);
}

extern "C" __device__ float severinghaus_buffer_graetz(
    float c_bulk,
    float chb_max_local,
    float alpha_mmhg,
    int is_blood
) {
    if (is_blood <= 0) return 1.0f;
    float P = c_bulk / alpha_mmhg;
    float den = (P * P * P + 150.0f * P + 23400.0f);
    float dsdP = 70200.0f * (P * P + 50.0f) / fmaxf(den * den, 1.0e-30f);
    float buffer = 1.0f + fmaxf(chb_max_local, 0.0f) * dsdP / alpha_mmhg;
    return fmaxf(buffer, 1.0e-30f);
}

extern "C" __device__ void graetz_propagate_step(
    const float* profile,
    float* out_profile,
    float c_ext,
    float ds_step,
    float radius_si,
    float flow_mag_si,
    float lumen_diffusivity_si,
    float buffer,
    int basis_idx,
    int n_radial,
    int n_modes,
    const float* mu2_table,
    const float* phi_table,
    const float* project_table,
    const float* cup_table,
    float vess_floor,
    float* out_bulk,
    float* out_wall
) {
    float pi = 3.14159265358979323846f;
    float U = flow_mag_si / fmaxf(pi * radius_si * radius_si, 1.0e-30f);
    float xi = lumen_diffusivity_si * fmaxf(ds_step, 0.0f) / fmaxf(buffer * radius_si * radius_si * U, 1.0e-30f);
    float coeff[16];
    for (int m = 0; m < n_modes; ++m) {
        float acc = 0.0f;
        for (int j = 0; j < n_radial; ++j) {
            float y = profile[j] - c_ext;
            acc += project_table[(basis_idx * n_modes + m) * n_radial + j] * y;
        }
        float arg = mu2_table[basis_idx * n_modes + m] * xi;
        if (arg > 80.0f) arg = 80.0f;
        coeff[m] = acc * expf(-arg);
    }
    float bulk = 0.0f;
    for (int j = 0; j < n_radial; ++j) {
        float y = 0.0f;
        for (int m = 0; m < n_modes; ++m) {
            y += phi_table[(basis_idx * n_radial + j) * n_modes + m] * coeff[m];
        }
        float val = c_ext + y;
        if (val < vess_floor) val = vess_floor;
        out_profile[j] = val;
        bulk += cup_table[basis_idx * n_radial + j] * val;
    }
    if (bulk < vess_floor) bulk = vess_floor;
    *out_bulk = bulk;
    *out_wall = out_profile[n_radial - 1];
}

extern "C" __global__ void frozen_topdown_graetz_kernel(
    const int* seg_ids,
    int n_level,
    const int* parents,
    const float* flows_si,
    const float* radii_si,
    const float* lengths_si,
    const float* gl_t,
    const float* c_ext_gl,
    int gl_order,
    float diffusivity_si,
    float lumen_diffusivity_si,
    float vmax,
    float km,
    float inlet_concentration,
    const float* chb_max,
    int is_blood,
    float alpha_mmhg,
    float vess_floor,
    const float* xs_lut,
    const float* ratio_lut,
    int lut_n,
    int key_min,
    int key_max,
    int per_decade,
    float min_bi,
    float max_bi,
    int n_radial,
    int n_modes,
    int max_fp_iters,
    float fp_tol,
    const float* mu2_table,
    const float* phi_table,
    const float* project_table,
    const float* cup_table,
    float* cin_seg,
    float* cout_seg,
    float* c_bulk_gl,
    float* c_wall_gl,
    int debug_enabled,
    int* debug_gl_fp_iters,
    float* debug_gl_fp_resid,
    float* debug_gl_buffer,
    int* debug_tail_fp_iters,
    float* debug_tail_fp_resid,
    float* debug_tail_buffer
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_level) return;
    int seg_idx = seg_ids[idx];
    int parent = parents[seg_idx];
    float cin_local = inlet_concentration;
    if (parent >= 0) cin_local = cout_seg[parent];
    if (cin_local < vess_floor) cin_local = vess_floor;
    cin_seg[seg_idx] = cin_local;

    float flow_mag_si = fabsf(flows_si[seg_idx]);
    float radius_si = radii_si[seg_idx];
    float length_si = lengths_si[seg_idx];
    float profile[32];
    float trial[32];
    float best[32];
    for (int j = 0; j < n_radial; ++j) profile[j] = cin_local;
    float bulk_running = cin_local;
    float prev_s = 0.0f;
    int use_wellmixed = (flow_mag_si <= 1.0e-30f || radius_si <= 1.0e-30f || length_si <= 0.0f || lumen_diffusivity_si <= 1.0e-30f);

    for (int node_idx = 0; node_idx < gl_order; ++node_idx) {
        float s_target = gl_t[node_idx] * length_si;
        float ds_step = s_target - prev_s;
        if (ds_step < 0.0f) ds_step = 0.0f;
        float c_ext_local = c_ext_gl[seg_idx * gl_order + node_idx];
        if (c_ext_local < 0.0f) c_ext_local = 0.0f;
        float c_wall = bulk_running;
        float c_bulk = bulk_running;
        if (use_wellmixed) {
            c_bulk_gl[seg_idx * gl_order + node_idx] = c_bulk;
            c_wall_gl[seg_idx * gl_order + node_idx] = c_wall;
            prev_s = s_target;
            continue;
        }
        float wall_guess = fmaxf(profile[n_radial - 1], 0.0f);
        float lambda_guess = lambda_from_wall_graetz(wall_guess, diffusivity_si, vmax, km);
        float accepted_bulk = bulk_running;
        float accepted_wall = wall_guess;
        float final_rel = 0.0f;
        float step_buffer = severinghaus_buffer_graetz(
            bulk_running,
            chb_max[seg_idx],
            alpha_mmhg,
            is_blood
        );
        int used_fp_iters = 0;
        for (int fp = 0; fp < max_fp_iters; ++fp) {
            float phi_l = fmaxf(radius_si / fmaxf(lambda_guess, 1.0e-30f), 1.0e-12f);
            float ratio = interp_lut_graetz(phi_l, xs_lut, ratio_lut, lut_n);
            float k_if = (2.0f * 3.14159265358979323846f * radius_si)
                       * (diffusivity_si / fmaxf(lambda_guess, 1.0e-30f)) * ratio;
            float Bi = k_if / fmaxf(2.0f * 3.14159265358979323846f * lumen_diffusivity_si, 1.0e-30f);
            int basis_idx = graetz_basis_index(Bi, key_min, key_max, per_decade, min_bi, max_bi);
            graetz_propagate_step(
                profile,
                trial,
                c_ext_local,
                ds_step,
                radius_si,
                flow_mag_si,
                lumen_diffusivity_si,
                step_buffer,
                basis_idx,
                n_radial,
                n_modes,
                mu2_table,
                phi_table,
                project_table,
                cup_table,
                vess_floor,
                &c_bulk,
                &c_wall
            );
            float lambda_new = lambda_from_wall_graetz(fmaxf(c_wall, 0.0f), diffusivity_si, vmax, km);
            for (int j = 0; j < n_radial; ++j) best[j] = trial[j];
            accepted_bulk = c_bulk;
            accepted_wall = c_wall;
            used_fp_iters = fp + 1;
            final_rel = fabsf(lambda_new - lambda_guess) / fmaxf(lambda_guess, 1.0e-30f);
            if (final_rel < fp_tol) break;
            lambda_guess = 0.5f * lambda_guess + 0.5f * lambda_new;
            lambda_guess = fmaxf(lambda_guess, 1.0e-30f);
        }
        for (int j = 0; j < n_radial; ++j) profile[j] = best[j];
        bulk_running = accepted_bulk;
        c_bulk_gl[seg_idx * gl_order + node_idx] = accepted_bulk;
        c_wall_gl[seg_idx * gl_order + node_idx] = accepted_wall;
        if (debug_enabled > 0) {
            int out_idx = seg_idx * gl_order + node_idx;
            debug_gl_fp_iters[out_idx] = used_fp_iters;
            debug_gl_fp_resid[out_idx] = final_rel;
            debug_gl_buffer[out_idx] = step_buffer;
        }
        prev_s = s_target;
    }

    float ds_tail = length_si - prev_s;
    if (ds_tail < 0.0f) ds_tail = 0.0f;
    float c_ext_tail = 0.0f;
    if (gl_order > 0) c_ext_tail = fmaxf(c_ext_gl[seg_idx * gl_order + gl_order - 1], 0.0f);
    if (!use_wellmixed && ds_tail > 0.0f) {
        float tail_wall_guess = fmaxf(profile[n_radial - 1], 0.0f);
        float tail_lambda_guess = lambda_from_wall_graetz(tail_wall_guess, diffusivity_si, vmax, km);
        float tail_buffer = severinghaus_buffer_graetz(
            bulk_running,
            chb_max[seg_idx],
            alpha_mmhg,
            is_blood
        );
        float tail_bulk = bulk_running;
        float tail_wall = tail_wall_guess;
        float accepted_tail_bulk = tail_bulk;
        float accepted_tail_wall = tail_wall;
        float tail_rel = 0.0f;
        int tail_used_fp_iters = 0;
        for (int fp = 0; fp < max_fp_iters; ++fp) {
            float phi_l = fmaxf(radius_si / fmaxf(tail_lambda_guess, 1.0e-30f), 1.0e-12f);
            float ratio = interp_lut_graetz(phi_l, xs_lut, ratio_lut, lut_n);
            float k_if = (2.0f * 3.14159265358979323846f * radius_si)
                       * (diffusivity_si / fmaxf(tail_lambda_guess, 1.0e-30f)) * ratio;
            float Bi = k_if / fmaxf(2.0f * 3.14159265358979323846f * lumen_diffusivity_si, 1.0e-30f);
            int basis_idx = graetz_basis_index(Bi, key_min, key_max, per_decade, min_bi, max_bi);
            graetz_propagate_step(
                profile,
                trial,
                c_ext_tail,
                ds_tail,
                radius_si,
                flow_mag_si,
                lumen_diffusivity_si,
                tail_buffer,
                basis_idx,
                n_radial,
                n_modes,
                mu2_table,
                phi_table,
                project_table,
                cup_table,
                vess_floor,
                &tail_bulk,
                &tail_wall
            );
            float tail_lambda_new = lambda_from_wall_graetz(fmaxf(tail_wall, 0.0f), diffusivity_si, vmax, km);
            for (int j = 0; j < n_radial; ++j) best[j] = trial[j];
            accepted_tail_bulk = tail_bulk;
            accepted_tail_wall = tail_wall;
            tail_used_fp_iters = fp + 1;
            tail_rel = fabsf(tail_lambda_new - tail_lambda_guess) / fmaxf(tail_lambda_guess, 1.0e-30f);
            if (tail_rel < fp_tol) break;
            tail_lambda_guess = 0.5f * tail_lambda_guess + 0.5f * tail_lambda_new;
            tail_lambda_guess = fmaxf(tail_lambda_guess, 1.0e-30f);
        }
        for (int j = 0; j < n_radial; ++j) profile[j] = best[j];
        bulk_running = accepted_tail_bulk;
        if (debug_enabled > 0) {
            debug_tail_fp_iters[seg_idx] = tail_used_fp_iters;
            debug_tail_fp_resid[seg_idx] = tail_rel;
            debug_tail_buffer[seg_idx] = tail_buffer;
        }
    }
    cout_seg[seg_idx] = fmaxf(bulk_running, vess_floor);
}
'''
    _FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL = _cp.RawKernel(code, "frozen_topdown_graetz_kernel")
    return _FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL


def _solve_topdown_ext_frozen_graetz_gpu(
    context: dict,
    c_ext_gl: np.ndarray,
    *,
    diffusivity_si: float,
    vmax: float,
    km: float,
    inlet_concentration: float,
    chb_max: np.ndarray,
    fluid_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    basis = _ensure_graetz_gpu_basis(context)
    kernel = _get_frozen_topdown_graetz_gpu_kernel()
    gl_order = int(np.asarray(context["gl_t"]).shape[0])
    nseg = int(np.asarray(context["parents"]).shape[0])
    lumen_diffusivity_si = float(_lumen_diffusivity_cm2_s_for_fluid(fluid_mode)) * CM2_TO_M2

    t_upload = perf_counter()
    c_ext_g = _cp.asarray(np.asarray(c_ext_gl, dtype=np.float32))
    chb_max_g = _cp.asarray(np.asarray(chb_max, dtype=np.float32))
    cin_seg_g = _cp.full((nseg,), np.float32(inlet_concentration), dtype=_cp.float32)
    cout_seg_g = _cp.full((nseg,), np.float32(inlet_concentration), dtype=_cp.float32)
    c_bulk_gl_g = _cp.full((nseg, gl_order), np.float32(inlet_concentration), dtype=_cp.float32)
    c_wall_gl_g = _cp.full((nseg, gl_order), np.float32(inlet_concentration), dtype=_cp.float32)
    debug_enabled = bool(GRAETZ_DEBUG_DIAGNOSTICS)
    if debug_enabled:
        debug_gl_fp_iters_g = _cp.zeros((nseg, gl_order), dtype=_cp.int32)
        debug_gl_fp_resid_g = _cp.zeros((nseg, gl_order), dtype=_cp.float32)
        debug_gl_buffer_g = _cp.zeros((nseg, gl_order), dtype=_cp.float32)
        debug_tail_fp_iters_g = _cp.zeros((nseg,), dtype=_cp.int32)
        debug_tail_fp_resid_g = _cp.zeros((nseg,), dtype=_cp.float32)
        debug_tail_buffer_g = _cp.zeros((nseg,), dtype=_cp.float32)
    else:
        debug_gl_fp_iters_g = _cp.zeros((1,), dtype=_cp.int32)
        debug_gl_fp_resid_g = _cp.zeros((1,), dtype=_cp.float32)
        debug_gl_buffer_g = _cp.zeros((1,), dtype=_cp.float32)
        debug_tail_fp_iters_g = _cp.zeros((1,), dtype=_cp.int32)
        debug_tail_fp_resid_g = _cp.zeros((1,), dtype=_cp.float32)
        debug_tail_buffer_g = _cp.zeros((1,), dtype=_cp.float32)
    _cp.cuda.Stream.null.synchronize()
    upload_time = perf_counter() - t_upload

    threads = 128
    t_kernel = perf_counter()
    for level_idx in range(int(np.asarray(context["level_offsets"]).shape[0]) - 1):
        start = int(context["level_offsets"][level_idx])
        stop = int(context["level_offsets"][level_idx + 1])
        n_level = max(stop - start, 0)
        if n_level <= 0:
            continue
        seg_ids_g = static["level_order"][start:stop]
        blocks = (n_level + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                seg_ids_g,
                np.int32(n_level),
                static["parents"],
                static["flows_si"],
                static["radii_si"],
                static["lengths_si"],
                static["gl_t"],
                c_ext_g.ravel(),
                np.int32(gl_order),
                np.float32(diffusivity_si),
                np.float32(lumen_diffusivity_si),
                np.float32(vmax),
                np.float32(km),
                np.float32(inlet_concentration),
                chb_max_g,
                np.int32(1 if str(fluid_mode).lower() == "blood" else 0),
                np.float32(ALPHA_MMHG),
                np.float32(VESS_CONC_FLOOR),
                static["xs_lut"],
                static["ratio_lut"],
                np.int32(int(np.asarray(_KRATIO_XS).size)),
                np.int32(basis["key_min"]),
                np.int32(basis["key_max"]),
                np.int32(basis["per_decade"]),
                np.float32(basis["min_bi"]),
                np.float32(basis["max_bi"]),
                np.int32(basis["n_radial"]),
                np.int32(basis["n_modes"]),
                np.int32(max(int(GRAETZ_MAX_FP_ITERS), 1)),
                np.float32(float(GRAETZ_FP_TOL)),
                basis["mu2"].ravel(),
                basis["phi"].ravel(),
                basis["project"].ravel(),
                basis["cup_weights"].ravel(),
                cin_seg_g,
                cout_seg_g,
                c_bulk_gl_g.ravel(),
                c_wall_gl_g.ravel(),
                np.int32(1 if debug_enabled else 0),
                debug_gl_fp_iters_g.ravel(),
                debug_gl_fp_resid_g.ravel(),
                debug_gl_buffer_g.ravel(),
                debug_tail_fp_iters_g.ravel(),
                debug_tail_fp_resid_g.ravel(),
                debug_tail_buffer_g.ravel(),
            ),
        )
    _cp.cuda.Stream.null.synchronize()
    kernel_time = perf_counter() - t_kernel

    t_download = perf_counter()
    cin_seg = _cp.asnumpy(cin_seg_g)
    cout_seg = _cp.asnumpy(cout_seg_g)
    c_bulk_gl = _cp.asnumpy(c_bulk_gl_g)
    c_wall_gl = _cp.asnumpy(c_wall_gl_g)
    _cp.cuda.Stream.null.synchronize()
    download_time = perf_counter() - t_download

    timings = {
        "upload_s": float(upload_time),
        "kernel_s": float(kernel_time),
        "download_s": float(download_time),
        "graetz_debug_diagnostics": bool(debug_enabled),
        "graetz_tail_buffer_patch": "enabled",
        "graetz_blood_buffer_active": bool(str(fluid_mode).lower() == "blood"),
    }
    if debug_enabled:
        gl_fp = _cp.asnumpy(debug_gl_fp_iters_g)
        gl_resid = _cp.asnumpy(debug_gl_fp_resid_g)
        gl_buffer = _cp.asnumpy(debug_gl_buffer_g)
        tail_fp = _cp.asnumpy(debug_tail_fp_iters_g)
        tail_resid = _cp.asnumpy(debug_tail_fp_resid_g)
        tail_buffer = _cp.asnumpy(debug_tail_buffer_g)
        gl_mask = gl_fp > 0
        tail_mask = tail_fp > 0
        timings.update(
            {
                "graetz_gl_fp_iters_mean": float(np.mean(gl_fp[gl_mask])) if np.any(gl_mask) else 0.0,
                "graetz_gl_fp_iters_max": float(np.max(gl_fp[gl_mask])) if np.any(gl_mask) else 0.0,
                "graetz_gl_fp_resid_mean": float(np.mean(gl_resid[gl_mask])) if np.any(gl_mask) else 0.0,
                "graetz_gl_fp_resid_max": float(np.max(gl_resid[gl_mask])) if np.any(gl_mask) else 0.0,
                "graetz_gl_buffer_mean": float(np.mean(gl_buffer[gl_mask])) if np.any(gl_mask) else 1.0,
                "graetz_gl_buffer_max": float(np.max(gl_buffer[gl_mask])) if np.any(gl_mask) else 1.0,
                "graetz_tail_fp_iters_mean": float(np.mean(tail_fp[tail_mask])) if np.any(tail_mask) else 0.0,
                "graetz_tail_fp_iters_max": float(np.max(tail_fp[tail_mask])) if np.any(tail_mask) else 0.0,
                "graetz_tail_fp_resid_mean": float(np.mean(tail_resid[tail_mask])) if np.any(tail_mask) else 0.0,
                "graetz_tail_fp_resid_max": float(np.max(tail_resid[tail_mask])) if np.any(tail_mask) else 0.0,
                "graetz_tail_buffer_mean": float(np.mean(tail_buffer[tail_mask])) if np.any(tail_mask) else 1.0,
                "graetz_tail_buffer_max": float(np.max(tail_buffer[tail_mask])) if np.any(tail_mask) else 1.0,
            }
        )
    return cin_seg, cout_seg, c_bulk_gl, c_wall_gl, timings


def _build_local_exclusion_lists(
    parents: np.ndarray,
    left_child: np.ndarray,
    right_child: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parents_arr = np.asarray(parents, dtype=np.int32)
    left_arr = np.asarray(left_child, dtype=np.int32)
    right_arr = np.asarray(right_child, dtype=np.int32)
    max_local = 32
    if _HAVE_NUMBA:
        return _build_local_exclusion_arrays_numba(parents_arr, left_arr, right_arr, max_local)
    nseg = int(parents_arr.shape[0])
    out_idx = np.full((nseg, max_local), -1, dtype=np.int32)
    out_count = np.zeros((nseg,), dtype=np.uint8)
    for seg in range(nseg):
        seen: set[int] = set()

        def add_immediate(node: int) -> None:
            if node < 0 or node >= nseg:
                return
            seen.add(node)
            parent = int(parents_arr[node])
            if 0 <= parent < nseg:
                seen.add(parent)
                pl = int(left_arr[parent])
                pr = int(right_arr[parent])
                if pl >= 0:
                    seen.add(pl)
                if pr >= 0:
                    seen.add(pr)
            left = int(left_arr[node])
            right = int(right_arr[node])
            if left >= 0:
                seen.add(left)
            if right >= 0:
                seen.add(right)

        add_immediate(seg)
        for node in list(seen):
            add_immediate(int(node))
        values = sorted(int(v) for v in seen if 0 <= int(v) < nseg)[:max_local]
        out_count[seg] = np.uint8(len(values))
        if values:
            out_idx[seg, : len(values)] = np.asarray(values, dtype=np.int32)
    return out_idx, out_count


def _cext_initial_chunk_targets() -> int:
    slots = max(int(CEXT_STREAMING_TARGET_CANDIDATE_SLOTS), 1)
    per_target_guess = max(int(GL_ORDER_CEXT) * 32, 1)
    return max(1, slots // per_target_guess)


def _cext_state_cache_key(
    *,
    fluid_mode: str,
    inlet_concentration: float,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> tuple[str, int, float, float, float, float]:
    return (
        str(fluid_mode).lower(),
        int(GL_ORDER_CEXT),
        float(inlet_concentration),
        float(diffusivity_si),
        float(vmax),
        float(km),
    )


def _get_tree_cext_state_cache(
    tree: Tree,
    cache_key: tuple[str, int, float, float, float, float],
) -> dict | None:
    cache_root = getattr(tree, "_cext_state_cache", None)
    if not isinstance(cache_root, dict):
        return None
    cached = cache_root.get(cache_key)
    return cached if isinstance(cached, dict) else None


def _store_tree_cext_state_cache(
    tree: Tree,
    cache_key: tuple[str, int, float, float, float, float],
    ext_state: dict,
) -> None:
    cache_root = getattr(tree, "_cext_state_cache", None)
    if not isinstance(cache_root, dict):
        cache_root = {}
        setattr(tree, "_cext_state_cache", cache_root)
    cache_root[cache_key] = {
        "c_ext_gl": np.asarray(ext_state["c_ext_gl"], dtype=np.float32).copy(),
        "c_iv_gl": np.asarray(ext_state["c_iv_gl"], dtype=np.float32).copy(),
        "c_bulk_gl": np.asarray(ext_state.get("c_bulk_gl", ext_state["c_iv_gl"]), dtype=np.float32).copy(),
        "c_wall_gl": np.asarray(ext_state.get("c_wall_gl", ext_state["c_iv_gl"]), dtype=np.float32).copy(),
        "cin_seg": np.asarray(ext_state["cin_seg"], dtype=np.float32).copy(),
        "cout_seg": np.asarray(ext_state["cout_seg"], dtype=np.float32).copy(),
    }


def _clear_cext_runtime_state(tree: Tree | None = None) -> None:
    """Drop large per-fluid Cext arrays after tissue sampling consumes them."""
    global _LAST_CEXT_SOURCE_STATE
    _LAST_CEXT_SOURCE_STATE = None
    if tree is not None:
        cache_root = getattr(tree, "_cext_state_cache", None)
        if isinstance(cache_root, dict):
            cache_root.clear()


def _initialize_cext_state(
    tree: Tree,
    *,
    cache_key: tuple[str, int, float, float, float, float],
    nseg: int,
    inlet_concentration: float,
    vmax: float,
    km: float,
) -> dict:
    gl_order = int(GL_ORDER_CEXT)
    ext_state = {
        "c_ext_gl": np.zeros((nseg, gl_order), dtype=np.float32),
        "c_iv_gl": np.full((nseg, gl_order), float(inlet_concentration), dtype=np.float32),
        "c_bulk_gl": np.full((nseg, gl_order), float(inlet_concentration), dtype=np.float32),
        "c_wall_gl": np.full((nseg, gl_order), float(inlet_concentration), dtype=np.float32),
        "cin_seg": np.full((nseg,), float(inlet_concentration), dtype=np.float32),
        "cout_seg": np.full((nseg,), float(inlet_concentration), dtype=np.float32),
        "vmax": float(vmax),
        "km": float(km),
        "window_factor": float(CEXT_WINDOW_FACTOR),
    }
    cached = _get_tree_cext_state_cache(tree, cache_key)
    if not cached:
        return ext_state
    prefix = min(
        int(nseg),
        int(np.asarray(cached.get("c_ext_gl", ()), dtype=np.float32).shape[0]),
        int(np.asarray(cached.get("c_iv_gl", ()), dtype=np.float32).shape[0]),
        int(np.asarray(cached.get("cin_seg", ()), dtype=np.float32).shape[0]),
        int(np.asarray(cached.get("cout_seg", ()), dtype=np.float32).shape[0]),
    )
    if prefix <= 0:
        return ext_state
    ext_state["c_ext_gl"][:prefix] = np.asarray(cached["c_ext_gl"], dtype=np.float32)[:prefix]
    ext_state["c_iv_gl"][:prefix] = np.asarray(cached["c_iv_gl"], dtype=np.float32)[:prefix]
    ext_state["c_bulk_gl"][:prefix] = np.asarray(cached.get("c_bulk_gl", cached["c_iv_gl"]), dtype=np.float32)[:prefix]
    ext_state["c_wall_gl"][:prefix] = np.asarray(cached.get("c_wall_gl", cached["c_iv_gl"]), dtype=np.float32)[:prefix]
    ext_state["cin_seg"][:prefix] = np.asarray(cached["cin_seg"], dtype=np.float32)[:prefix]
    ext_state["cout_seg"][:prefix] = np.asarray(cached["cout_seg"], dtype=np.float32)[:prefix]
    return ext_state


def _build_cext_iteration_cache(context: dict, ext_state: dict) -> dict:
    diffusivity_si = float(context["diffusivity_si"])
    c_bulk_floor = np.maximum(np.asarray(ext_state["c_iv_gl"], dtype=float), VESS_CONC_FLOOR)
    c_wall_raw = ext_state.get("c_wall_gl")
    if c_wall_raw is None:
        c_wall_floor = c_bulk_floor.copy()
    else:
        c_wall_floor = np.maximum(np.asarray(c_wall_raw, dtype=float), VESS_CONC_FLOOR)
    c_ext_pos = np.maximum(np.asarray(ext_state["c_ext_gl"], dtype=float), 0.0)
    lambda_if_gl = _lambda_if_from_civ(c_wall_floor, diffusivity_si, float(ext_state["vmax"]), float(ext_state["km"]))
    lambda_iv_gl = _cext_green_lambda_from_fields(
        c_wall_floor,
        c_ext_pos,
        lambda_if_gl,
        diffusivity_si,
        float(ext_state["vmax"]),
        float(ext_state["km"]),
    )
    k_if_gl = _interfacial_transfer_coefficient(
        np.asarray(context["radii_si"], dtype=float)[:, None],
        lambda_if_gl,
        diffusivity_si,
    )
    q_line_gl = k_if_gl * (c_wall_floor - c_ext_pos)
    ds_gl = np.asarray(context["ds_gl"], dtype=float)
    q_weighted_gl = q_line_gl * ds_gl
    seg_cap_gl = np.max(np.maximum(c_bulk_floor, c_wall_floor), axis=1)
    include_mono2, include_dipole2 = _finite_radius_o2_term_flags()
    a2 = np.asarray(context["radii_si"], dtype=float)[:, None] ** 2
    mono2_weight_gl = (0.25 * a2 * q_weighted_gl) if include_mono2 else np.zeros_like(q_weighted_gl)
    dipole2_weight_gl = (
        a2 * np.pi * diffusivity_si * c_wall_floor * ds_gl
        if include_dipole2
        else np.zeros_like(q_weighted_gl)
    )
    cache = {
        "c_bulk_gl": np.asarray(c_bulk_floor, dtype=np.float32),
        "c_wall_gl": np.asarray(c_wall_floor, dtype=np.float32),
        "lambda_if_gl": np.asarray(lambda_if_gl, dtype=np.float32),
        "lambda_iv_gl": np.asarray(lambda_iv_gl, dtype=np.float32),
        "k_if_gl": np.asarray(k_if_gl, dtype=np.float32),
        "q_line_gl": np.asarray(q_line_gl, dtype=np.float32),
        "q_weighted_gl": np.asarray(q_weighted_gl, dtype=np.float32),
        "seg_cap_gl": np.asarray(seg_cap_gl, dtype=np.float32),
        "mono2_weight_gl": np.asarray(mono2_weight_gl, dtype=np.float32),
        "dipole2_weight_gl": np.asarray(dipole2_weight_gl, dtype=np.float32),
    }
    ext_state.update(cache)
    ext_state["_lambda_bin_epoch"] = int(ext_state.get("_lambda_bin_epoch", 0)) + 1
    return cache


def _get_cext_iteration_cache_kernel():
    global _CEXT_ITERATION_CACHE_KERNEL
    if _CEXT_ITERATION_CACHE_KERNEL is not None:
        return _CEXT_ITERATION_CACHE_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __device__ float cext_cache_interp_lut(float x, const float* xs, const float* ys, int n) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    float x0 = xs[lo];
    float x1 = xs[lo + 1];
    float y0 = ys[lo];
    float y1 = ys[lo + 1];
    if (x1 == x0) return y0;
    float t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

extern "C" __global__ void cext_iteration_cache_kernel(
    const float* c_bulk_in,
    const float* c_wall_in,
    const float* c_ext_gl,
    const float* radii_si,
    const float* ds_gl,
    const float* xs_lut,
    const float* ratio_lut,
    int lut_n,
    int nseg,
    int gl_order,
    float diffusivity_si,
    float vmax,
    float km,
    float vess_floor,
    int use_lambda_tissue,
    int include_mono2,
    int include_dipole2,
    float* q_weighted_gl,
    float* mono2_weight_gl,
    float* dipole2_weight_gl,
    float* lambda_iv_gl,
    float* seg_cap_gl
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nseg * gl_order;
    if (flat >= total) return;
    int seg_idx = flat / gl_order;
    int node_idx = flat - seg_idx * gl_order;

    float c_bulk = c_bulk_in[flat];
    if (!(c_bulk >= vess_floor) || !isfinite(c_bulk)) c_bulk = vess_floor;
    float c_wall = c_wall_in[flat];
    if (!(c_wall >= vess_floor) || !isfinite(c_wall)) c_wall = vess_floor;
    float c_ext = c_ext_gl[flat];
    if (!(c_ext > 0.0f) || !isfinite(c_ext)) c_ext = 0.0f;

    float denom_if = km + fmaxf(c_wall, 0.0f);
    if (denom_if < 1.0e-30f) denom_if = 1.0e-30f;
    float k1_if = vmax / denom_if;
    if (k1_if < 1.0e-30f) k1_if = 1.0e-30f;
    float lambda_if = sqrtf(diffusivity_si / k1_if);
    if (lambda_if < 1.0e-30f) lambda_if = 1.0e-30f;

    float lambda_iv = lambda_if;
    if (use_lambda_tissue != 0) {
        float eff = sqrtf(fmaxf(km + fmaxf(c_wall, 0.0f), 1.0e-30f) * fmaxf(km + c_ext, 1.0e-30f));
        float k1_t = vmax / fmaxf(eff, 1.0e-30f);
        if (k1_t < 1.0e-30f) k1_t = 1.0e-30f;
        lambda_iv = sqrtf(diffusivity_si / k1_t);
        if (lambda_iv < 1.0e-30f) lambda_iv = 1.0e-30f;
    }

    float radius = radii_si[seg_idx];
    if (!(radius > 0.0f) || !isfinite(radius)) radius = 0.0f;
    float phi = radius / lambda_if;
    if (phi < 1.0e-12f) phi = 1.0e-12f;
    float ratio = cext_cache_interp_lut(phi, xs_lut, ratio_lut, lut_n);
    float k_if = (2.0f * 3.14159265358979323846f * radius) * (diffusivity_si / lambda_if) * ratio;
    float q_line = k_if * (c_wall - c_ext);
    float ds = ds_gl[flat];
    float q_weight = q_line * ds;
    float a2 = radius * radius;

    q_weighted_gl[flat] = q_weight;
    lambda_iv_gl[flat] = lambda_iv;
    mono2_weight_gl[flat] = include_mono2 != 0 ? 0.25f * a2 * q_weight : 0.0f;
    dipole2_weight_gl[flat] = include_dipole2 != 0 ? a2 * 3.14159265358979323846f * diffusivity_si * c_wall * ds : 0.0f;

    if (node_idx == 0) {
        float cap = vess_floor;
        for (int j = 0; j < gl_order; ++j) {
            int idx = seg_idx * gl_order + j;
            float cb = c_bulk_in[idx];
            if (!(cb >= vess_floor) || !isfinite(cb)) cb = vess_floor;
            float cw = c_wall_in[idx];
            if (!(cw >= vess_floor) || !isfinite(cw)) cw = vess_floor;
            cap = fmaxf(cap, fmaxf(cb, cw));
        }
        seg_cap_gl[seg_idx] = cap;
    }
}
'''
    _CEXT_ITERATION_CACHE_KERNEL = _cp.RawKernel(code, "cext_iteration_cache_kernel")
    return _CEXT_ITERATION_CACHE_KERNEL


def _build_cext_iteration_cache_gpu(context: dict, ext_state: dict, runtime_state: dict) -> None:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    gl_shape = tuple(np.asarray(ext_state["c_iv_gl"], dtype=np.float32).shape)
    if runtime_state.get("cache_c_bulk_gl") is None or tuple(runtime_state["cache_c_bulk_gl"].shape) != gl_shape:
        runtime_state["cache_c_bulk_gl"] = _cp.empty(gl_shape, dtype=_cp.float32)
        runtime_state["cache_c_wall_gl"] = _cp.empty(gl_shape, dtype=_cp.float32)
    runtime_state["cache_c_bulk_gl"].set(np.asarray(ext_state["c_iv_gl"], dtype=np.float32))
    runtime_state["cache_c_wall_gl"].set(
        np.asarray(ext_state.get("c_wall_gl", ext_state["c_iv_gl"]), dtype=np.float32)
    )

    include_mono2, include_dipole2 = _finite_radius_o2_term_flags()
    gl_order = int(gl_shape[1]) if len(gl_shape) >= 2 else 1
    nseg = int(gl_shape[0]) if len(gl_shape) >= 1 else 0
    total = int(nseg * gl_order)
    if total <= 0:
        ext_state["lambda_iv_gl"] = np.zeros(gl_shape, dtype=np.float32)
        ext_state["_lambda_bin_epoch"] = int(ext_state.get("_lambda_bin_epoch", 0)) + 1
        return
    threads = 256
    blocks = (total + threads - 1) // threads
    kernel = _get_cext_iteration_cache_kernel()
    kernel(
        (blocks,),
        (threads,),
        (
            runtime_state["cache_c_bulk_gl"].ravel(),
            runtime_state["cache_c_wall_gl"].ravel(),
            runtime_state["c_ext_gl"].ravel(),
            static["radii_si"],
            static["ds_gl"].ravel(),
            static["xs_lut"],
            static["ratio_lut"],
            np.int32(int(np.asarray(_KRATIO_XS).size)),
            np.int32(nseg),
            np.int32(gl_order),
            np.float32(float(context["diffusivity_si"])),
            np.float32(float(ext_state["vmax"])),
            np.float32(float(ext_state["km"])),
            np.float32(float(VESS_CONC_FLOOR)),
            np.int32(1 if _normalize_cext_lambda_source() == "lambda_t" else 0),
            np.int32(1 if include_mono2 else 0),
            np.int32(1 if include_dipole2 else 0),
            runtime_state["q_weighted_gl"].ravel(),
            runtime_state["mono2_weight_gl"].ravel(),
            runtime_state["dipole2_weight_gl"].ravel(),
            runtime_state["lambda_iv_gl"].ravel(),
            runtime_state["seg_cap_gl"],
        ),
    )
    _cp.cuda.Stream.null.synchronize()
    ext_state["lambda_iv_gl"] = _cp.asnumpy(runtime_state["lambda_iv_gl"])
    ext_state["_lambda_bin_epoch"] = int(ext_state.get("_lambda_bin_epoch", 0)) + 1


def validate_cext_tissue_flux_consistency(cext_state: dict, *, rtol: float = 1e-5, atol: float = 1e-8) -> dict[str, float | bool]:
    """Check that q_line uses the wall concentration; c_iv_gl is bulk/cup."""
    c_wall = np.maximum(
        np.asarray(cext_state.get("c_wall_gl", cext_state["c_iv_gl"]), dtype=np.float32),
        np.float32(VESS_CONC_FLOOR),
    )
    c_ext = np.maximum(np.asarray(cext_state["c_ext_gl"], dtype=np.float32), np.float32(0.0))
    k_if = np.asarray(cext_state["k_if_gl"], dtype=np.float32)
    q_line = np.asarray(cext_state["q_line_gl"], dtype=np.float32)
    expected = k_if * (c_wall - c_ext)
    diff = q_line - expected
    max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    denom = float(np.linalg.norm(expected.reshape(-1))) if expected.size else 0.0
    rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / max(denom, 1e-30)) if diff.size else 0.0
    return {
        "ok": bool(max_abs <= float(atol) + float(rtol) * float(np.nanmax(np.abs(expected))) if expected.size else True),
        "max_abs": max_abs,
        "rel_l2": rel_l2,
    }


def _snapshot_cext_source_state(
    context: dict,
    ext_state: dict,
    *,
    solver: str,
    backend: str,
) -> dict:
    c_ext_gl = np.asarray(ext_state["c_ext_gl"], dtype=np.float32)
    q_weighted_gl = np.asarray(ext_state.get("q_weighted_gl", np.zeros_like(c_ext_gl)), dtype=np.float32)
    return {
        "solver": str(solver),
        "backend": str(backend),
        "gl_points_si": np.asarray(context["gl_points_si"], dtype=np.float32).copy(),
        "diffusivity_si": float(context["diffusivity_si"]),
        "window_factor": float(ext_state.get("window_factor", CEXT_WINDOW_FACTOR)),
        "c_iv_gl": np.asarray(ext_state["c_iv_gl"], dtype=np.float32).copy(),
        "c_bulk_gl": np.asarray(ext_state.get("c_bulk_gl", ext_state["c_iv_gl"]), dtype=np.float32).copy(),
        "c_wall_gl": np.asarray(ext_state.get("c_wall_gl", ext_state["c_iv_gl"]), dtype=np.float32).copy(),
        "c_ext_gl": c_ext_gl.copy(),
        "lambda_iv_gl": np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32).copy(),
        "k_if_gl": np.asarray(ext_state["k_if_gl"], dtype=np.float32).copy(),
        "q_line_gl": np.asarray(ext_state["q_line_gl"], dtype=np.float32).copy(),
        "q_weighted_gl": q_weighted_gl.copy(),
        "mono2_weight_gl": np.asarray(ext_state.get("mono2_weight_gl", np.zeros_like(q_weighted_gl)), dtype=np.float32).copy(),
        "dipole2_weight_gl": np.asarray(ext_state.get("dipole2_weight_gl", np.zeros_like(q_weighted_gl)), dtype=np.float32).copy(),
        "seg_cap_gl": np.asarray(ext_state["seg_cap_gl"], dtype=np.float32).copy(),
        "segment_vectors": np.asarray(context["segment_vectors"], dtype=np.float32).copy(),
    }


def _set_last_cext_source_state(
    context: dict,
    ext_state: dict,
    *,
    solver: str,
    backend: str,
) -> None:
    global _LAST_CEXT_SOURCE_STATE
    _LAST_CEXT_SOURCE_STATE = _snapshot_cext_source_state(
        context,
        ext_state,
        solver=solver,
        backend=backend,
    )


def _coerce_candidate_batches_for_gpu(context: dict) -> list[dict]:
    batches = list(context.get("candidate_batches", []))
    if _cp is None:
        return batches
    for batch in batches:
        gpu_batch = batch.get("gpu")
        if isinstance(gpu_batch, dict):
            continue
        batch["gpu"] = {
            "target_seg_ids": _cp.asarray(np.asarray(batch["target_seg_ids"], dtype=np.int32)),
            "row_ptr": _cp.asarray(np.asarray(batch["row_ptr"], dtype=np.int32)),
            "col_idx": _cp.asarray(np.asarray(batch["col_idx"], dtype=np.int32)),
        }
    return batches


def _maybe_trim_candidate_list(
    ordered: list[int],
    *,
    target_mid: np.ndarray,
    midpoints_si: np.ndarray,
) -> list[int]:
    max_candidates = int(CEXT_MAX_CANDIDATES_PER_TARGET)
    if max_candidates <= 0 or len(ordered) <= max_candidates:
        return ordered
    ranked = sorted(
        ordered,
        key=lambda source_i: float(np.dot(midpoints_si[source_i] - target_mid, midpoints_si[source_i] - target_mid)),
    )
    return ranked[:max_candidates]


def _maybe_trim_candidate_array(
    candidates: np.ndarray,
    *,
    target_mid: np.ndarray,
    midpoints_si: np.ndarray,
) -> np.ndarray:
    candidate_ids = np.asarray(candidates, dtype=np.int32)
    max_candidates = int(CEXT_MAX_CANDIDATES_PER_TARGET)
    if candidate_ids.size == 0:
        return candidate_ids
    if max_candidates <= 0 or candidate_ids.size <= max_candidates:
        return np.sort(candidate_ids)
    delta = np.asarray(midpoints_si[candidate_ids] - target_mid[None, :], dtype=float)
    dist2 = np.einsum("ij,ij->i", delta, delta)
    keep_idx = np.argsort(dist2, kind="stable")[:max_candidates]
    return np.asarray(candidate_ids[keep_idx], dtype=np.int32)


def _build_cext_geometry_context(
    tree: Tree,
    flows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    build_candidate_index: bool = True,
) -> dict:
    hct_context = _hematocrit_context_for_tree(tree)
    parents = np.asarray(hct_context["parents"], dtype=np.int32)
    left_child = np.asarray(hct_context["left_child"], dtype=np.int32)
    right_child = np.asarray(hct_context["right_child"], dtype=np.int32)
    order = np.asarray(hct_context["order"], dtype=np.int64)
    level_order, level_offsets, depth = _build_topdown_level_slices(order, parents)

    flow_starts = np.asarray(starts, dtype=float).copy()
    flow_ends = np.asarray(ends, dtype=float).copy()
    flip = np.asarray(flows, dtype=float) < 0.0
    flow_starts[flip] = np.asarray(ends, dtype=float)[flip]
    flow_ends[flip] = np.asarray(starts, dtype=float)[flip]

    flow_starts_si = np.asarray(flow_starts * CM_TO_M, dtype=CEXT_FLOAT_DTYPE)
    flow_ends_si = np.asarray(flow_ends * CM_TO_M, dtype=CEXT_FLOAT_DTYPE)
    flow_vectors_si = np.asarray(flow_ends_si - flow_starts_si, dtype=CEXT_FLOAT_DTYPE)
    midpoints_si = np.asarray(0.5 * (flow_starts_si + flow_ends_si), dtype=CEXT_FLOAT_DTYPE)
    flows_si = np.asarray(np.asarray(flows, dtype=float) * CM3_TO_M3, dtype=CEXT_FLOAT_DTYPE)
    lengths_si = np.asarray(np.maximum(np.asarray(lengths, dtype=float), 0.0) * CM_TO_M, dtype=CEXT_FLOAT_DTYPE)
    radii_si = np.asarray(np.maximum(np.asarray(radii, dtype=float), 0.0) * CM_TO_M, dtype=CEXT_FLOAT_DTYPE)
    gl_nodes, gl_weights = _get_gl_nodes_weights(GL_ORDER_CEXT)
    gl_t = np.asarray(0.5 * (gl_nodes + 1.0), dtype=CEXT_FLOAT_DTYPE)
    gl_weights_arr = np.asarray(gl_weights, dtype=CEXT_FLOAT_DTYPE)
    gl_points_si = np.asarray(
        flow_starts_si[:, None, :] + gl_t[None, :, None] * flow_vectors_si[:, None, :],
        dtype=CEXT_FLOAT_DTYPE,
    )
    ds_gl = np.asarray(0.5 * lengths_si[:, None] * gl_weights_arr[None, :], dtype=CEXT_FLOAT_DTYPE)

    exclude_idx, exclude_count = _build_local_exclusion_lists(parents, left_child, right_child)

    diffusivity_si = float(diffusivity * CM2_TO_M2)
    lambda_inlet = float(_lambda_if_from_civ(float(inlet_concentration), diffusivity_si, vmax, km))
    window = max(float(CEXT_WINDOW_FACTOR), 0.0)
    cell_size = float(CEXT_GRID_CELL_FACTOR) * window * lambda_inlet
    reach_si = np.asarray(0.5 * lengths_si + radii_si + window * lambda_inlet, dtype=CEXT_FLOAT_DTYPE)

    gpu_direct_cell_origin = np.zeros((3,), dtype=np.int32)
    gpu_direct_cell_dims = np.zeros((3,), dtype=np.int32)
    gpu_direct_cell_ptr = np.zeros((1,), dtype=np.int32)
    gpu_direct_cell_seg_ids = np.zeros((0,), dtype=np.int32)
    gpu_direct_cell_flat_sorted = np.zeros((0,), dtype=np.int64)
    gpu_direct_home_cell_flat = np.zeros((midpoints_si.shape[0],), dtype=np.int64)
    gpu_direct_available = False
    gpu_direct_n_cells = 0
    if build_candidate_index and cell_size > 0.0 and np.isfinite(cell_size) and midpoints_si.shape[0] > 0:
        home_idx = np.floor(np.asarray(midpoints_si, dtype=float) / cell_size).astype(np.int32)
        cell_origin = np.min(home_idx, axis=0).astype(np.int32)
        shifted = np.asarray(home_idx - cell_origin[None, :], dtype=np.int32)
        cell_dims = np.max(shifted, axis=0).astype(np.int32) + np.int32(1)
        n_cells = int(cell_dims[0]) * int(cell_dims[1]) * int(cell_dims[2])
        if n_cells > 0:
            flat_ids = (
                shifted[:, 0].astype(np.int64)
                + np.int64(cell_dims[0]) * (
                    shifted[:, 1].astype(np.int64)
                    + np.int64(cell_dims[1]) * shifted[:, 2].astype(np.int64)
                )
            )
            order_home = np.argsort(flat_ids, kind="stable")
            sorted_flat = np.asarray(flat_ids[order_home], dtype=np.int64)
            counts = np.bincount(sorted_flat, minlength=n_cells)
            cell_ptr = np.zeros((n_cells + 1,), dtype=np.int32)
            cell_ptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64).astype(np.int32)
            gpu_direct_cell_origin = np.asarray(cell_origin, dtype=np.int32)
            gpu_direct_cell_dims = np.asarray(cell_dims, dtype=np.int32)
            gpu_direct_cell_ptr = np.asarray(cell_ptr, dtype=np.int32)
            gpu_direct_cell_seg_ids = np.asarray(order_home, dtype=np.int32)
            gpu_direct_cell_flat_sorted = np.asarray(sorted_flat, dtype=np.int64)
            gpu_direct_home_cell_flat = np.asarray(flat_ids, dtype=np.int64)
            gpu_direct_available = True
            gpu_direct_n_cells = int(n_cells)

    candidate_kdtree = None
    grid: dict[tuple[int, int, int], np.ndarray] = {}
    overflow_segments = np.empty((0,), dtype=np.int32)
    if build_candidate_index and _HAVE_SCIPY_SPATIAL and midpoints_si.shape[0] > 0:
        candidate_kdtree = _cKDTree(np.asarray(midpoints_si, dtype=float))
    elif build_candidate_index and cell_size > 0.0 and np.isfinite(cell_size):
        builders: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        overflow: list[int] = []
        for seg_idx in range(int(lengths_si.shape[0])):
            reach = float(reach_si[seg_idx])
            mid = midpoints_si[seg_idx]
            lo = np.floor((mid - reach) / cell_size).astype(np.int64)
            hi = np.floor((mid + reach) / cell_size).astype(np.int64)
            nx = int(hi[0] - lo[0] + 1)
            ny = int(hi[1] - lo[1] + 1)
            nz = int(hi[2] - lo[2] + 1)
            n_cells = max(nx * ny * nz, 0)
            if n_cells <= 0:
                continue
            if n_cells > int(CEXT_MAX_CELLS_PER_SEG):
                overflow.append(seg_idx)
                continue
            for ix in range(int(lo[0]), int(hi[0]) + 1):
                for iy in range(int(lo[1]), int(hi[1]) + 1):
                    for iz in range(int(lo[2]), int(hi[2]) + 1):
                        builders[(ix, iy, iz)].append(seg_idx)
        grid = {key: np.asarray(values, dtype=np.int32) for key, values in builders.items()}
        overflow_segments = np.asarray(overflow, dtype=np.int32)

    return {
        "parents": parents,
        "left_child": left_child,
        "right_child": right_child,
        "order": np.asarray(order, dtype=np.int32),
        "level_order": level_order,
        "level_offsets": level_offsets,
        "depth": depth,
        "flow_starts_si": flow_starts_si,
        "flow_ends_si": flow_ends_si,
        "flow_vectors_si": flow_vectors_si,
        "segment_vectors": flow_vectors_si,
        "midpoints_si": midpoints_si,
        "flows_si": flows_si,
        "lengths_si": lengths_si,
        "radii_si": radii_si,
        "gl_t": gl_t,
        "gl_weights": gl_weights_arr,
        "gl_points_si": gl_points_si,
        "ds_gl": ds_gl,
        "exclude_idx": exclude_idx,
        "exclude_count": exclude_count,
        "diffusivity_si": diffusivity_si,
        "lambda_inlet": float(lambda_inlet),
        "cell_size": float(cell_size),
        "reach_si": reach_si,
        "max_reach_si": float(np.max(reach_si)) if reach_si.size else 0.0,
        "candidate_query_mode": "kdtree" if candidate_kdtree is not None else "grid",
        "candidate_kdtree": candidate_kdtree,
        "gpu_direct_available": bool(gpu_direct_available),
        "gpu_direct_cell_origin": gpu_direct_cell_origin,
        "gpu_direct_cell_dims": gpu_direct_cell_dims,
        "gpu_direct_cell_ptr": gpu_direct_cell_ptr,
        "gpu_direct_cell_seg_ids": gpu_direct_cell_seg_ids,
        "gpu_direct_cell_flat_sorted": gpu_direct_cell_flat_sorted,
        "gpu_direct_home_cell_flat": gpu_direct_home_cell_flat,
        "gpu_direct_n_cells": int(gpu_direct_n_cells),
        "grid": grid,
        "overflow_segments": overflow_segments,
    }


def _build_cext_gpu_direct_source_plan(
    context: dict,
    source_mask: np.ndarray | None,
) -> dict:
    nseg = int(np.asarray(context["midpoints_si"]).shape[0])
    if source_mask is None:
        return {
            "uses_static": True,
            "cell_ptr": np.asarray(context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)), dtype=np.int32),
            "cell_seg_ids": np.asarray(context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
            "active_count": nseg,
        }
    mask = np.asarray(source_mask, dtype=bool).reshape(-1)
    if mask.size != nseg:
        raise ValueError("source_mask size does not match Cext geometry segment count.")
    source_ids = np.flatnonzero(mask).astype(np.int32, copy=False)
    return _build_cext_gpu_direct_source_plan_from_ids(context, source_ids)


def _build_cext_gpu_direct_source_plan_from_ids(
    context: dict,
    source_seg_ids: np.ndarray | None,
) -> dict:
    nseg = int(np.asarray(context["midpoints_si"]).shape[0])
    if source_seg_ids is None:
        return {
            "uses_static": True,
            "cell_ptr": np.asarray(context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)), dtype=np.int32),
            "cell_seg_ids": np.asarray(context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
            "active_count": nseg,
        }
    source_ids = np.asarray(source_seg_ids, dtype=np.int32).reshape(-1)
    if source_ids.size == 0:
        n_cells = int(context.get("gpu_direct_n_cells", 0))
        return {
            "uses_static": False,
            "cell_ptr": np.zeros((max(n_cells, 0) + 1,), dtype=np.int32),
            "cell_seg_ids": np.zeros((0,), dtype=np.int32),
            "active_count": 0,
        }
    source_ids = np.unique(source_ids)
    active_count = int(source_ids.size)
    if active_count >= nseg:
        return {
            "uses_static": True,
            "cell_ptr": np.asarray(context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)), dtype=np.int32),
            "cell_seg_ids": np.asarray(context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32),
            "active_count": nseg,
        }
    n_cells = int(context.get("gpu_direct_n_cells", 0))
    if n_cells <= 0:
        return {
            "uses_static": False,
            "cell_ptr": np.zeros((1,), dtype=np.int32),
            "cell_seg_ids": np.zeros((0,), dtype=np.int32),
            "active_count": active_count,
        }
    home_flat = np.asarray(context.get("gpu_direct_home_cell_flat", np.zeros((nseg,), dtype=np.int64)), dtype=np.int64)
    active_flat = np.asarray(home_flat[source_ids], dtype=np.int64)
    order = np.argsort(active_flat, kind="stable")
    cell_seg_ids = np.asarray(source_ids[order], dtype=np.int32)
    active_flat = np.asarray(active_flat[order], dtype=np.int64)
    counts = np.bincount(active_flat, minlength=n_cells) if active_flat.size else np.zeros((n_cells,), dtype=np.int64)
    cell_ptr = np.zeros((n_cells + 1,), dtype=np.int32)
    if counts.size:
        cell_ptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64).astype(np.int32)
    return {
        "uses_static": False,
        "cell_ptr": cell_ptr,
        "cell_seg_ids": cell_seg_ids,
        "active_count": active_count,
    }


def _build_cext_gpu_direct_target_plan(
    context: dict,
    target_seg_ids: np.ndarray | None,
) -> dict:
    nseg = int(np.asarray(context["midpoints_si"]).shape[0])
    if target_seg_ids is None:
        return {
            "uses_static": True,
            "target_seg_ids": np.arange(nseg, dtype=np.int32),
            "target_count": nseg,
        }
    target_ids = np.asarray(target_seg_ids, dtype=np.int32).reshape(-1)
    if target_ids.size == 0:
        return {
            "uses_static": False,
            "target_seg_ids": np.zeros((0,), dtype=np.int32),
            "target_count": 0,
        }
    target_ids = np.unique(target_ids)
    if int(target_ids.size) >= nseg:
        return {
            "uses_static": True,
            "target_seg_ids": np.arange(nseg, dtype=np.int32),
            "target_count": nseg,
        }
    return {
        "uses_static": False,
        "target_seg_ids": np.asarray(target_ids, dtype=np.int32),
        "target_count": int(target_ids.size),
    }


def _build_cext_include_lists_kdtree(
    context: dict,
    target_seg_ids: np.ndarray,
    *,
    max_slots: int,
    query_workers: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    t0 = perf_counter()
    candidate_kdtree = context.get("candidate_kdtree")
    if candidate_kdtree is None:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), perf_counter() - t0, False
    reach_si = np.asarray(context["reach_si"], dtype=float)
    max_reach = float(context.get("max_reach_si", 0.0))
    midpoints_si = np.asarray(context["midpoints_si"], dtype=float)
    exclude_idx = np.asarray(context["exclude_idx"], dtype=np.int32)
    exclude_count = np.asarray(context["exclude_count"], dtype=np.uint8)
    reach_scale = max(float(CEXT_APPROX_WINDOW_SCALE), 0.0)
    target_ids = np.asarray(target_seg_ids, dtype=np.int32)
    row_ptr = [0]
    col_values: list[int] = []
    max_slots_i = max(int(max_slots), 1)
    too_many = False
    if target_ids.size == 0:
        return np.asarray(row_ptr, dtype=np.int32), np.zeros((0,), dtype=np.int32), perf_counter() - t0, False

    query_radii = np.asarray(reach_scale * reach_si[target_ids] + reach_scale * max_reach, dtype=float)
    target_midpoints = np.asarray(midpoints_si[target_ids], dtype=float)
    neighbor_lists = candidate_kdtree.query_ball_point(
        target_midpoints,
        r=query_radii,
        workers=max(int(query_workers), 1),
    )

    for idx, target_seg in enumerate(target_ids):
        target_i = int(target_seg)
        neighbors = np.asarray(neighbor_lists[idx], dtype=np.int32)
        if neighbors.size == 0:
            row_ptr.append(row_ptr[-1])
            continue
        local_count = int(exclude_count[target_i])
        if local_count > 0:
            excluded = np.asarray(exclude_idx[target_i, :local_count], dtype=np.int32)
            neighbors = neighbors[~np.isin(neighbors, excluded)]
        if neighbors.size > 0:
            target_mid = np.asarray(midpoints_si[target_i], dtype=float)
            target_reach = float(reach_scale * reach_si[target_i])
            delta = np.asarray(midpoints_si[neighbors] - target_mid[None, :], dtype=float)
            dist2 = np.einsum("ij,ij->i", delta, delta)
            pair_limit = np.square(target_reach + reach_scale * reach_si[neighbors])
            neighbors = neighbors[dist2 <= pair_limit]
        ordered = _maybe_trim_candidate_array(neighbors, target_mid=np.asarray(midpoints_si[target_i], dtype=float), midpoints_si=midpoints_si)
        next_size = row_ptr[-1] + int(ordered.size)
        if next_size > max_slots_i and target_ids.size > 1:
            too_many = True
            break
        if ordered.size:
            col_values.extend(int(val) for val in ordered)
        row_ptr.append(next_size)

    if too_many:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), perf_counter() - t0, True
    return (
        np.asarray(row_ptr, dtype=np.int32),
        np.asarray(col_values, dtype=np.int32),
        perf_counter() - t0,
        False,
    )


def _build_cext_include_lists(
    context: dict,
    target_seg_ids: np.ndarray,
    *,
    max_slots: int,
    query_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    if context.get("candidate_kdtree") is not None:
        return _build_cext_include_lists_kdtree(
            context,
            target_seg_ids,
            max_slots=max_slots,
            query_workers=query_workers,
        )
    t0 = perf_counter()
    grid = context["grid"]
    overflow_segments = np.asarray(context["overflow_segments"], dtype=np.int32)
    cell_size = float(context["cell_size"])
    reach_si = np.asarray(context["reach_si"], dtype=float)
    midpoints_si = np.asarray(context["midpoints_si"], dtype=float)
    exclude_idx = np.asarray(context["exclude_idx"], dtype=np.int32)
    exclude_count = np.asarray(context["exclude_count"], dtype=np.uint8)
    reach_scale = max(float(CEXT_APPROX_WINDOW_SCALE), 0.0)
    row_ptr = [0]
    col_values: list[int] = []
    max_slots_i = max(int(max_slots), 1)
    too_many = False

    for target_seg in np.asarray(target_seg_ids, dtype=np.int32):
        target_i = int(target_seg)
        target_mid = midpoints_si[target_i]
        target_reach = float(reach_scale * reach_si[target_i])
        exclude_local = {
            int(val)
            for val in exclude_idx[target_i, : int(exclude_count[target_i])]
            if int(val) >= 0
        }
        candidates: set[int] = set()

        if cell_size > 0.0 and grid:
            lo = np.floor((target_mid - target_reach) / cell_size).astype(np.int64)
            hi = np.floor((target_mid + target_reach) / cell_size).astype(np.int64)
            for ix in range(int(lo[0]), int(hi[0]) + 1):
                for iy in range(int(lo[1]), int(hi[1]) + 1):
                    for iz in range(int(lo[2]), int(hi[2]) + 1):
                        segs = grid.get((ix, iy, iz))
                        if segs is None:
                            continue
                        for source_seg in np.asarray(segs, dtype=np.int32):
                            source_i = int(source_seg)
                            if source_i in exclude_local:
                                continue
                            dx = midpoints_si[source_i] - target_mid
                            source_reach = float(reach_scale * reach_si[source_i])
                            if float(np.dot(dx, dx)) > float((target_reach + source_reach) ** 2):
                                continue
                            candidates.add(source_i)
            for source_seg in overflow_segments:
                source_i = int(source_seg)
                if source_i in exclude_local:
                    continue
                dx = midpoints_si[source_i] - target_mid
                source_reach = float(reach_scale * reach_si[source_i])
                if float(np.dot(dx, dx)) > float((target_reach + source_reach) ** 2):
                    continue
                candidates.add(source_i)

        ordered = _maybe_trim_candidate_list(sorted(candidates), target_mid=target_mid, midpoints_si=midpoints_si)
        next_size = row_ptr[-1] + len(ordered)
        if next_size > max_slots_i and target_seg_ids.size > 1:
            too_many = True
            break
        col_values.extend(ordered)
        row_ptr.append(next_size)

    if too_many:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), perf_counter() - t0, True
    return (
        np.asarray(row_ptr, dtype=np.int32),
        np.asarray(col_values, dtype=np.int32),
        perf_counter() - t0,
        False,
    )


def _build_cext_candidate_batches_for_range(
    context: dict,
    start_idx: int,
    stop_idx: int,
    *,
    max_slots: int,
    query_workers: int = 1,
) -> tuple[int, list[dict], float]:
    cursor = int(start_idx)
    local_batches: list[dict] = []
    query_time = 0.0
    while cursor < int(stop_idx):
        batch_stop = min(int(stop_idx), cursor + _cext_initial_chunk_targets())
        target_seg_ids = np.arange(cursor, batch_stop, dtype=np.int32)
        row_ptr, col_idx, query_t, too_many = _build_cext_include_lists(
            context,
            target_seg_ids,
            max_slots=max_slots,
            query_workers=query_workers,
        )
        query_time += query_t
        if too_many and target_seg_ids.size > 1:
            mid = cursor + max(int(target_seg_ids.size // 2), 1)
            _, left_batches, left_time = _build_cext_candidate_batches_for_range(
                context,
                cursor,
                mid,
                max_slots=max_slots,
                query_workers=query_workers,
            )
            _, right_batches, right_time = _build_cext_candidate_batches_for_range(
                context,
                mid,
                batch_stop,
                max_slots=max_slots,
                query_workers=query_workers,
            )
            query_time += left_time + right_time
            local_batches.extend(left_batches)
            local_batches.extend(right_batches)
        else:
            local_batches.append(
                {
                    "target_start": int(cursor),
                    "target_stop": int(batch_stop),
                    "target_seg_ids": np.asarray(target_seg_ids, dtype=np.int32),
                    "row_ptr": np.asarray(row_ptr, dtype=np.int32),
                    "col_idx": np.asarray(col_idx, dtype=np.int32),
                }
            )
        cursor = batch_stop
    return int(start_idx), local_batches, float(query_time)


def _build_cext_candidate_batches(context: dict) -> tuple[list[dict], float]:
    nseg = int(np.asarray(context["midpoints_si"]).shape[0])
    if nseg <= 0:
        return [], 0.0
    chunk_targets = _cext_initial_chunk_targets()
    target_ranges = max(1, min(int(CEXT_PRECOMPUTE_WORKERS), nseg))
    range_span = max(chunk_targets, int(math.ceil(float(nseg) / float(target_ranges))))
    ranges = [(start, min(start + range_span, nseg)) for start in range(0, nseg, range_span)]
    workers = min(int(CEXT_PRECOMPUTE_WORKERS), len(ranges))
    if workers <= 1 or len(ranges) <= 1:
        _, batches, query_time = _build_cext_candidate_batches_for_range(
            context,
            0,
            nseg,
            max_slots=CEXT_STREAMING_TARGET_CANDIDATE_SLOTS,
            query_workers=max(int(CEXT_PRECOMPUTE_WORKERS), 1),
        )
        return batches, float(query_time)

    wall_t0 = perf_counter()
    ordered_chunks: list[tuple[int, list[dict]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _build_cext_candidate_batches_for_range,
                context,
                start,
                stop,
                max_slots=CEXT_STREAMING_TARGET_CANDIDATE_SLOTS,
                query_workers=1,
            ): start
            for start, stop in ranges
        }
        for future in as_completed(futures):
            start_idx, batches, local_query_time = future.result()
            ordered_chunks.append((int(start_idx), batches))
    ordered_chunks.sort(key=lambda item: item[0])
    merged: list[dict] = []
    for _, batches in ordered_chunks:
        merged.extend(batches)
    return merged, float(perf_counter() - wall_t0)


def _ensure_cext_candidate_batches(context: dict) -> tuple[list[dict], float]:
    batches = context.get("candidate_batches")
    if isinstance(batches, list):
        return batches, float(context.get("candidate_build_time_s", 0.0))
    candidate_batches, candidate_query_time = _build_cext_candidate_batches(context)
    context["candidate_batches"] = candidate_batches
    context["candidate_build_time_s"] = float(candidate_query_time)
    return candidate_batches, float(candidate_query_time)


def _split_cext_candidate_batch(context: dict, batch: dict) -> list[dict]:
    target_seg_ids = np.asarray(batch["target_seg_ids"], dtype=np.int32)
    if target_seg_ids.size <= 1:
        return [batch]
    mid = int(target_seg_ids.size // 2)
    first_ids = np.asarray(target_seg_ids[:mid], dtype=np.int32)
    second_ids = np.asarray(target_seg_ids[mid:], dtype=np.int32)
    split_batches: list[dict] = []
    for seg_ids in (first_ids, second_ids):
        row_ptr, col_idx, _, _ = _build_cext_include_lists(
            context,
            seg_ids,
            max_slots=CEXT_STREAMING_TARGET_CANDIDATE_SLOTS,
            query_workers=1,
        )
        split_batches.append(
            {
                "target_start": int(seg_ids[0]),
                "target_stop": int(seg_ids[-1]) + 1,
                "target_seg_ids": np.asarray(seg_ids, dtype=np.int32),
                "row_ptr": np.asarray(row_ptr, dtype=np.int32),
                "col_idx": np.asarray(col_idx, dtype=np.int32),
            }
        )
    return split_batches


def _compute_cext_batch_cpu(
    context: dict,
    ext_state: dict,
    target_seg_ids: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    if target_seg_ids.size == 0:
        return np.zeros((0, gl_order), dtype=np.float32)
    if _HAVE_NUMBA:
        return _compute_cext_batch_numba(
            np.asarray(target_seg_ids, dtype=np.int32),
            np.asarray(row_ptr, dtype=np.int32),
            np.asarray(col_idx, dtype=np.int32),
            np.asarray(context["gl_points_si"], dtype=np.float32),
            np.asarray(context["midpoints_si"], dtype=np.float32),
            np.asarray(context["radii_si"], dtype=np.float32),
            np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32),
            np.asarray(ext_state["q_weighted_gl"], dtype=np.float32),
            np.asarray(ext_state["seg_cap_gl"], dtype=np.float32),
            np.asarray(context["exclude_idx"], dtype=np.int32),
            np.asarray(context["exclude_count"], dtype=np.uint8),
            float(context["diffusivity_si"]),
            float(ext_state["window_factor"]),
        )

    out = np.zeros((int(target_seg_ids.size), gl_order), dtype=np.float32)
    radii_si = np.asarray(context["radii_si"], dtype=float)
    gl_points_si = np.asarray(context["gl_points_si"], dtype=float)
    midpoints_si = np.asarray(context["midpoints_si"], dtype=float)
    exclude_idx = np.asarray(context["exclude_idx"], dtype=np.int32)
    exclude_count = np.asarray(context["exclude_count"], dtype=np.uint8)
    lambda_iv_gl = np.asarray(ext_state["lambda_iv_gl"], dtype=float)
    q_weighted_gl = np.asarray(ext_state["q_weighted_gl"], dtype=float)
    seg_cap_gl = np.asarray(ext_state["seg_cap_gl"], dtype=float)
    diffusivity_si = float(context["diffusivity_si"])
    window_factor = float(ext_state["window_factor"])
    for batch_idx, target_seg in enumerate(np.asarray(target_seg_ids, dtype=np.int32)):
        target_i = int(target_seg)
        exclude_local = {
            int(v)
            for v in exclude_idx[target_i, : int(exclude_count[target_i])]
            if int(v) >= 0
        }
        for target_node in range(gl_order):
            target_point = gl_points_si[target_i, target_node]
            target_lambda = max(float(lambda_iv_gl[target_i, target_node]), 1e-30)
            total = 0.0
            cap_max = 0.0
            for pos in range(int(row_ptr[batch_idx]), int(row_ptr[batch_idx + 1])):
                source_i = int(col_idx[pos])
                if source_i in exclude_local:
                    continue
                radius_sum = float(radii_si[target_i] + radii_si[source_i])
                coarse_r = math.sqrt(float(np.dot(midpoints_si[source_i] - midpoints_si[target_i], midpoints_si[source_i] - midpoints_si[target_i])) + radius_sum * radius_sum)
                source_lambda_max = max(float(np.max(lambda_iv_gl[source_i])), 1e-30)
                coarse_lambda = max(target_lambda, source_lambda_max)
                if coarse_r > window_factor * coarse_lambda:
                    continue
                seg_cap = float(seg_cap_gl[source_i])
                seg_contributed = False
                for source_node in range(gl_order):
                    source_lambda = max(float(lambda_iv_gl[source_i, source_node]), 1e-30)
                    source_point = gl_points_si[source_i, source_node]
                    r = math.sqrt(float(np.dot(source_point - target_point, source_point - target_point)) + radius_sum * radius_sum)
                    if r > window_factor * source_lambda:
                        continue
                    r = max(r, 1e-30)
                    kernel = math.exp(-r / source_lambda) / (4.0 * math.pi * diffusivity_si * r)
                    total += float(q_weighted_gl[source_i, source_node]) * kernel
                    seg_contributed = True
                if seg_contributed:
                    cap_max = max(cap_max, seg_cap)
            if total < 0.0 or not np.isfinite(total):
                total = 0.0
            if cap_max > 0.0:
                total = min(total, cap_max)
            out[batch_idx, target_node] = np.float32(total)
    return out


def _ensure_cext_gpu_static(context: dict) -> dict:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = context.get("gpu_static")
    if static is not None:
        return static
    static = {
        "gl_points_si": _cp.asarray(np.asarray(context["gl_points_si"], dtype=np.float32)),
        "midpoints_si": _cp.asarray(np.asarray(context["midpoints_si"], dtype=np.float32)),
        "segment_vectors": _cp.asarray(np.asarray(context["segment_vectors"], dtype=np.float32)),
        "radii_si": _cp.asarray(np.asarray(context["radii_si"], dtype=np.float32)),
        "lengths_si": _cp.asarray(np.asarray(context["lengths_si"], dtype=np.float32)),
        "ds_gl": _cp.asarray(np.asarray(context["ds_gl"], dtype=np.float32)),
        "gl_t": _cp.asarray(np.asarray(context["gl_t"], dtype=np.float32)),
        "flows_si": _cp.asarray(np.asarray(context["flows_si"], dtype=np.float32)),
        "parents": _cp.asarray(np.asarray(context["parents"], dtype=np.int32)),
        "level_order": _cp.asarray(np.asarray(context["level_order"], dtype=np.int32)),
        "level_offsets": _cp.asarray(np.asarray(context["level_offsets"], dtype=np.int32)),
        "exclude_idx": _cp.asarray(np.asarray(context["exclude_idx"], dtype=np.int32)),
        "exclude_count": _cp.asarray(np.asarray(context["exclude_count"], dtype=np.uint8)),
        "xs_lut": _cp.asarray(np.asarray(_KRATIO_XS, dtype=np.float32)),
        "ratio_lut": _cp.asarray(np.asarray(_KRATIO_YS, dtype=np.float32)),
        "reach_si": _cp.asarray(np.asarray(context["reach_si"], dtype=np.float32)),
        "gpu_direct_cell_ptr": _cp.asarray(np.asarray(context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)), dtype=np.int32)),
        "gpu_direct_cell_seg_ids": _cp.asarray(np.asarray(context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)),
        "all_target_seg_ids": _cp.arange(int(np.asarray(context["midpoints_si"]).shape[0]), dtype=_cp.int32),
    }
    context["gpu_static"] = static
    return static


def _context_uses_cext_gpu_direct(context: dict) -> bool:
    return bool(context.get("candidate_query_mode") == "gpu_direct")


def _compute_cext_batch_gpu_into(
    context: dict,
    batch: dict,
    lambda_iv_g,
    q_weighted_g,
    mono2_weight_g,
    dipole2_weight_g,
    seg_cap_g,
    c_ext_new_g,
) -> None:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    kernel = _get_cext_gpu_kernel()
    gpu_batch = batch.get("gpu")
    if not isinstance(gpu_batch, dict):
        raise RuntimeError("Missing cached GPU candidate batch.")
    target_ids_g = gpu_batch["target_seg_ids"]
    row_ptr_g = gpu_batch["row_ptr"]
    col_idx_g = gpu_batch["col_idx"]
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    threads = 128
    blocks = (int(target_ids_g.size) * gl_order + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            target_ids_g,
            row_ptr_g,
            col_idx_g,
            static["gl_points_si"].ravel(),
            static["midpoints_si"].ravel(),
            static["segment_vectors"].ravel(),
            static["radii_si"],
            lambda_iv_g.ravel(),
            q_weighted_g.ravel(),
            mono2_weight_g.ravel(),
            dipole2_weight_g.ravel(),
            seg_cap_g,
            static["exclude_idx"].ravel(),
            static["exclude_count"],
            np.int32(gl_order),
            np.int32(context["exclude_idx"].shape[1]),
            np.int32(target_ids_g.size),
            np.float32(context["diffusivity_si"]),
            np.float32(float(CEXT_WINDOW_FACTOR)),
            c_ext_new_g.ravel(),
        ),
    )


def _compute_cext_direct_gpu(
    context: dict,
    ext_state: dict,
    *,
    source_plan: dict | None = None,
    target_plan: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    kernel = _get_cext_gpu_direct_kernel()
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    plan = source_plan if isinstance(source_plan, dict) else None
    active_count = int(plan.get("active_count", nseg)) if plan is not None else nseg
    target_cfg = target_plan if isinstance(target_plan, dict) else None
    target_count = int(target_cfg.get("target_count", nseg)) if target_cfg is not None else nseg
    if active_count <= 0 or target_count <= 0:
        zeros = np.zeros(np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape, dtype=np.float32)
        return zeros, zeros.copy(), 0.0, 0.0

    t_transfer = perf_counter()
    lambda_iv_g = _cp.asarray(np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32))
    q_weighted_g = _cp.asarray(np.asarray(ext_state["q_weighted_gl"], dtype=np.float32))
    mono2_weight_g = _cp.asarray(np.asarray(ext_state.get("mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32))
    dipole2_weight_g = _cp.asarray(np.asarray(ext_state.get("dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32))
    seg_cap_g = _cp.asarray(np.asarray(ext_state["seg_cap_gl"], dtype=np.float32))
    c_ext_total_g = _cp.zeros((target_count, gl_order), dtype=_cp.float32)
    c_ext_cap_g = _cp.zeros((target_count, gl_order), dtype=_cp.float32)
    if plan is None or bool(plan.get("uses_static", False)):
        cell_ptr_g = static["gpu_direct_cell_ptr"]
        cell_seg_ids_g = static["gpu_direct_cell_seg_ids"]
    else:
        if "gpu_cell_ptr" not in plan:
            plan["gpu_cell_ptr"] = _cp.asarray(np.asarray(plan.get("cell_ptr", np.zeros((1,), dtype=np.int32)), dtype=np.int32))
            plan["gpu_cell_seg_ids"] = _cp.asarray(np.asarray(plan.get("cell_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32))
        cell_ptr_g = plan["gpu_cell_ptr"]
        cell_seg_ids_g = plan["gpu_cell_seg_ids"]
    if target_cfg is None or bool(target_cfg.get("uses_static", False)):
        target_seg_ids = np.asarray(np.arange(nseg, dtype=np.int32), dtype=np.int32)
        target_seg_ids_g = static["all_target_seg_ids"]
    else:
        target_seg_ids = np.asarray(target_cfg.get("target_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        if "gpu_target_seg_ids" not in target_cfg:
            target_cfg["gpu_target_seg_ids"] = _cp.asarray(target_seg_ids, dtype=_cp.int32)
        target_seg_ids_g = target_cfg["gpu_target_seg_ids"]
    _cp.cuda.Stream.null.synchronize()
    transfer_total = perf_counter() - t_transfer

    cell_origin = np.asarray(context.get("gpu_direct_cell_origin", np.zeros((3,), dtype=np.int32)), dtype=np.int32)
    cell_dims = np.asarray(context.get("gpu_direct_cell_dims", np.zeros((3,), dtype=np.int32)), dtype=np.int32)
    threads = 128
    blocks = (target_count * gl_order + threads - 1) // threads
    t_kernel = perf_counter()
    kernel(
        (blocks,),
        (threads,),
        (
            target_seg_ids_g,
            static["gl_points_si"].ravel(),
            static["midpoints_si"].ravel(),
            static["segment_vectors"].ravel(),
            static["radii_si"],
            static["reach_si"],
            lambda_iv_g.ravel(),
            q_weighted_g.ravel(),
            mono2_weight_g.ravel(),
            dipole2_weight_g.ravel(),
            seg_cap_g,
            static["exclude_idx"].ravel(),
            static["exclude_count"],
            cell_ptr_g,
            cell_seg_ids_g,
            np.int32(cell_origin[0]),
            np.int32(cell_origin[1]),
            np.int32(cell_origin[2]),
            np.int32(cell_dims[0]),
            np.int32(cell_dims[1]),
            np.int32(cell_dims[2]),
            np.float32(float(context["cell_size"])),
            np.float32(float(context.get("max_reach_si", 0.0))),
            np.float32(max(float(CEXT_APPROX_WINDOW_SCALE), 0.0)),
            np.int32(gl_order),
            np.int32(int(np.asarray(context["exclude_idx"]).shape[1])),
            np.int32(target_count),
            np.float32(float(context["diffusivity_si"])),
            np.float32(float(CEXT_WINDOW_FACTOR)),
            c_ext_total_g.ravel(),
            c_ext_cap_g.ravel(),
        ),
    )
    _cp.cuda.Stream.null.synchronize()
    kernel_total = perf_counter() - t_kernel

    t_download = perf_counter()
    c_ext_total_local = _cp.asnumpy(c_ext_total_g)
    c_ext_cap_local = _cp.asnumpy(c_ext_cap_g)
    _cp.cuda.Stream.null.synchronize()
    transfer_total += perf_counter() - t_download
    c_ext_total = np.zeros(np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape, dtype=np.float32)
    c_ext_cap = np.zeros_like(c_ext_total)
    c_ext_total[np.asarray(target_seg_ids, dtype=np.int32)] = np.asarray(c_ext_total_local, dtype=np.float32)
    c_ext_cap[np.asarray(target_seg_ids, dtype=np.int32)] = np.asarray(c_ext_cap_local, dtype=np.float32)
    return c_ext_total, c_ext_cap, kernel_total, transfer_total


def _ensure_cext_hybrid_bg_context(context: dict) -> dict:
    hybrid = context.get("hybrid_bg_context")
    grid_n = max(int(CEXT_HYBRID_BG_GRID), 16)
    bg_mode = _resolve_cext_hybrid_bg_mode()
    configured_near_radius_mult = (
        0.0 if bg_mode in ("local_only_nlambda", "fft") else max(float(CEXT_HYBRID_BG_NEAR_RADIUS_MULT), 0.0)
    )
    assignment = str(CEXT_HYBRID_BG_ASSIGNMENT).strip().lower()
    if assignment not in ("cic", "tsc"):
        assignment = "tsc"
    if isinstance(hybrid, dict):
        if (
            int(hybrid.get("grid_n", -1)) == grid_n
            and int(hybrid.get("lambda_bins", -1)) == max(int(CEXT_HYBRID_BG_LAMBDA_BINS), 1)
            and float(hybrid.get("near_radius_mult", -1.0)) == float(configured_near_radius_mult)
            and str(hybrid.get("assignment", "")) == assignment
            and str(hybrid.get("bg_mode", "")) == bg_mode
        ):
            return hybrid
    gl_points = np.asarray(context["gl_points_si"], dtype=np.float32).reshape(-1, 3)
    if gl_points.size:
        mins = np.min(gl_points, axis=0)
        maxs = np.max(gl_points, axis=0)
    else:
        mins = np.zeros((3,), dtype=np.float32)
        maxs = np.ones((3,), dtype=np.float32)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins)) if gl_points.size else 1.0
    span = max(span, 1.0e-8)
    pad = max(float(context.get("max_reach_si", 0.0)), 0.1 * span)
    side = span + 2.0 * pad
    h = side / float(grid_n)
    near_radius_mult = float(configured_near_radius_mult)
    near_radius = near_radius_mult * h
    pad = max(pad, 2.0 * near_radius)
    side = span + 2.0 * pad
    h = side / float(grid_n)
    near_radius = near_radius_mult * h
    origin = np.asarray(center - 0.5 * side, dtype=np.float32)
    kfreq = 2.0 * np.pi * np.fft.fftfreq(grid_n, d=h)
    k2 = (
        kfreq[:, None, None] ** 2
        + kfreq[None, :, None] ** 2
        + kfreq[None, None, :] ** 2
    ).astype(np.float32)
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    node_seg_ids = np.repeat(np.arange(nseg, dtype=np.int32), gl_order)
    hybrid = {
        "grid_n": int(grid_n),
        "lambda_bins": max(int(CEXT_HYBRID_BG_LAMBDA_BINS), 1),
        "near_radius_mult": float(near_radius_mult),
        "assignment": assignment,
        "bg_mode": bg_mode,
        "origin": origin,
        "side": float(side),
        "spacing": float(h),
        "near_radius_si": float(near_radius),
        "k2": np.asarray(k2, dtype=np.float32),
        "node_seg_ids": np.asarray(node_seg_ids, dtype=np.int32),
        "gl_points_flat": np.asarray(gl_points, dtype=np.float32),
        "lambda_bin_edges": None,
        "lambda_bin_centers": None,
        "gpu_static": None,
    }
    context["hybrid_bg_context"] = hybrid
    return hybrid


def _cext_hybrid_axis_assignment_cpu(coord: np.ndarray, grid_n: int, assignment: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    coord_arr = np.asarray(coord, dtype=np.float32)
    if assignment == "tsc":
        center = np.floor(coord_arr + np.float32(0.5)).astype(np.int32)
        idx_list: list[np.ndarray] = []
        weight_list: list[np.ndarray] = []
        for offset in (-1, 0, 1):
            idx = center + np.int32(offset)
            dist = np.abs(coord_arr - idx.astype(np.float32))
            weight = np.where(
                dist < np.float32(0.5),
                np.float32(0.75) - dist * dist,
                np.where(
                    dist < np.float32(1.5),
                    np.float32(0.5) * (np.float32(1.5) - dist) * (np.float32(1.5) - dist),
                    np.float32(0.0),
                ),
            ).astype(np.float32)
            idx_list.append(np.asarray(idx, dtype=np.int32))
            weight_list.append(np.asarray(weight, dtype=np.float32))
        return idx_list, weight_list
    base = np.floor(coord_arr).astype(np.int32)
    frac = (coord_arr - base.astype(np.float32)).astype(np.float32)
    return [
        np.asarray(base, dtype=np.int32),
        np.asarray(base + np.int32(1), dtype=np.int32),
    ], [
        np.asarray(np.float32(1.0) - frac, dtype=np.float32),
        np.asarray(frac, dtype=np.float32),
    ]


def _cext_hybrid_build_stencil_metadata(hybrid: dict) -> dict[str, np.ndarray | int]:
    points = np.asarray(hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    n_nodes = int(points.shape[0])
    grid_n = int(hybrid["grid_n"])
    assignment = str(hybrid["assignment"])
    spacing = float(hybrid["spacing"])
    origin = np.asarray(hybrid["origin"], dtype=np.float32)
    if n_nodes <= 0:
        return {
            "stencil_flat_idx": np.zeros((0, 0), dtype=np.int32),
            "stencil_weight": np.zeros((0, 0), dtype=np.float32),
            "stencil_n": 0,
            "grid_cells": int(grid_n * grid_n * grid_n),
        }
    coords = ((points - origin[None, :]) / np.float32(spacing)) - np.float32(0.5)
    idx_x, w_x = _cext_hybrid_axis_assignment_cpu(coords[:, 0], grid_n, assignment)
    idx_y, w_y = _cext_hybrid_axis_assignment_cpu(coords[:, 1], grid_n, assignment)
    idx_z, w_z = _cext_hybrid_axis_assignment_cpu(coords[:, 2], grid_n, assignment)
    stencil_n = int(len(idx_x) * len(idx_y) * len(idx_z))
    flat_idx = np.full((n_nodes, stencil_n), -1, dtype=np.int32)
    weights = np.zeros((n_nodes, stencil_n), dtype=np.float32)
    slot = 0
    for ax, wx in zip(idx_x, w_x):
        for ay, wy in zip(idx_y, w_y):
            for az, wz in zip(idx_z, w_z):
                weight = np.asarray(wx * wy * wz, dtype=np.float32)
                valid = (
                    (ax >= 0)
                    & (ax < grid_n)
                    & (ay >= 0)
                    & (ay < grid_n)
                    & (az >= 0)
                    & (az < grid_n)
                    & np.isfinite(weight)
                    & (weight != np.float32(0.0))
                )
                if np.any(valid):
                    flat = (((ax[valid] * grid_n) + ay[valid]) * grid_n + az[valid]).astype(np.int32, copy=False)
                    flat_idx[valid, slot] = np.asarray(flat, dtype=np.int32)
                    weights[valid, slot] = np.asarray(weight[valid], dtype=np.float32)
                slot += 1
    return {
        "stencil_flat_idx": np.asarray(flat_idx, dtype=np.int32),
        "stencil_weight": np.asarray(weights, dtype=np.float32),
        "stencil_n": int(stencil_n),
        "grid_cells": int(grid_n * grid_n * grid_n),
    }


def _cext_hybrid_build_stencil_metadata_gpu(hybrid: dict) -> dict[str, object | int]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    points = _cp.asarray(np.asarray(hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32))
    n_nodes = int(points.shape[0])
    grid_n = int(hybrid["grid_n"])
    grid_cells = int(grid_n * grid_n * grid_n)
    assignment = str(hybrid["assignment"])
    stencil_n = 27 if assignment == "tsc" else 8
    flat_idx = _cp.full((n_nodes, stencil_n), _cp.int32(-1), dtype=_cp.int32)
    weights = _cp.zeros((n_nodes, stencil_n), dtype=_cp.float32)
    if n_nodes <= 0 or grid_cells <= 0:
        return {
            "stencil_flat_idx": flat_idx,
            "stencil_weight": weights,
            "stencil_n": int(stencil_n),
            "grid_cells": int(grid_cells),
        }

    origin_g = _cp.asarray(np.asarray(hybrid["origin"], dtype=np.float32))
    spacing = _cp.float32(float(hybrid["spacing"]))
    coords = ((points - origin_g[None, :]) / spacing) - _cp.float32(0.5)
    if assignment == "tsc":
        axis_idx = []
        axis_weight = []
        for axis in range(3):
            center = _cp.floor(coords[:, axis] + _cp.float32(0.5)).astype(_cp.int32)
            idx_list = []
            weight_list = []
            for offset in (-1, 0, 1):
                idx = center + np.int32(offset)
                dist = _cp.abs(coords[:, axis] - idx.astype(_cp.float32))
                weight = _cp.where(
                    dist < _cp.float32(0.5),
                    _cp.float32(0.75) - dist * dist,
                    _cp.where(
                        dist < _cp.float32(1.5),
                        _cp.float32(0.5) * (_cp.float32(1.5) - dist) * (_cp.float32(1.5) - dist),
                        _cp.float32(0.0),
                    ),
                ).astype(_cp.float32)
                idx_list.append(idx)
                weight_list.append(weight)
            axis_idx.append(idx_list)
            axis_weight.append(weight_list)
    else:
        axis_idx = []
        axis_weight = []
        for axis in range(3):
            base = _cp.floor(coords[:, axis]).astype(_cp.int32)
            frac = (coords[:, axis] - base.astype(_cp.float32)).astype(_cp.float32)
            axis_idx.append([base, base + np.int32(1)])
            axis_weight.append([
                (_cp.float32(1.0) - frac).astype(_cp.float32),
                frac.astype(_cp.float32),
            ])

    slot = 0
    for ax, wx in zip(axis_idx[0], axis_weight[0]):
        for ay, wy in zip(axis_idx[1], axis_weight[1]):
            for az, wz in zip(axis_idx[2], axis_weight[2]):
                weight = (wx * wy * wz).astype(_cp.float32)
                valid = (
                    (ax >= 0)
                    & (ax < grid_n)
                    & (ay >= 0)
                    & (ay < grid_n)
                    & (az >= 0)
                    & (az < grid_n)
                    & _cp.isfinite(weight)
                    & (weight != _cp.float32(0.0))
                )
                flat = (((ax * np.int32(grid_n)) + ay) * np.int32(grid_n) + az).astype(_cp.int32)
                flat_idx[:, slot] = _cp.where(valid, flat, _cp.int32(-1))
                weights[:, slot] = _cp.where(valid, weight, _cp.float32(0.0))
                slot += 1

    return {
        "stencil_flat_idx": flat_idx,
        "stencil_weight": weights,
        "stencil_n": int(stencil_n),
        "grid_cells": int(grid_cells),
    }


def _cext_hybrid_build_local_node_cells(hybrid: dict) -> dict[str, np.ndarray | int]:
    points = np.asarray(hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    grid_n = int(hybrid["grid_n"])
    grid_cells = int(grid_n * grid_n * grid_n)
    spacing = float(hybrid["spacing"])
    origin = np.asarray(hybrid["origin"], dtype=np.float32)
    n_nodes = int(points.shape[0])
    node_cell_flat = np.full((n_nodes,), -1, dtype=np.int32)
    if n_nodes <= 0 or grid_cells <= 0 or spacing <= 0.0:
        return {
            "local_cell_ptr": np.zeros((grid_cells + 1,), dtype=np.int32),
            "local_node_ids": np.zeros((0,), dtype=np.int32),
            "local_node_cell_flat": node_cell_flat,
            "local_rad_cells": 0,
        }
    coords = np.floor((np.asarray(points, dtype=np.float32) - origin[None, :]) / np.float32(spacing)).astype(np.int32)
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < grid_n)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < grid_n)
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < grid_n)
    )
    if np.any(valid):
        flat = (
            (coords[valid, 0].astype(np.int64) * np.int64(grid_n) + coords[valid, 1].astype(np.int64))
            * np.int64(grid_n)
            + coords[valid, 2].astype(np.int64)
        )
        valid_node_ids = np.flatnonzero(valid).astype(np.int32, copy=False)
        node_cell_flat[valid_node_ids] = np.asarray(flat, dtype=np.int32)
        order = np.argsort(flat, kind="stable")
        sorted_flat = np.asarray(flat[order], dtype=np.int64)
        local_node_ids = np.asarray(valid_node_ids[order], dtype=np.int32)
        counts = np.bincount(sorted_flat, minlength=grid_cells)
    else:
        local_node_ids = np.zeros((0,), dtype=np.int32)
        counts = np.zeros((grid_cells,), dtype=np.int64)
    local_cell_ptr = np.zeros((grid_cells + 1,), dtype=np.int32)
    local_cell_ptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64).astype(np.int32)
    rad_cells = int(np.ceil(max(float(hybrid.get("near_radius_si", 0.0)), 0.0) / max(spacing, 1.0e-30)))
    return {
        "local_cell_ptr": np.asarray(local_cell_ptr, dtype=np.int32),
        "local_node_ids": np.asarray(local_node_ids, dtype=np.int32),
        "local_node_cell_flat": np.asarray(node_cell_flat, dtype=np.int32),
        "local_rad_cells": int(max(rad_cells, 0)),
    }


def _build_cext_hybrid_local_source_plan(hybrid: dict, source_mask: np.ndarray | None) -> dict:
    node_seg_ids = np.asarray(hybrid.get("node_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    node_cell_flat = np.asarray(hybrid.get("local_node_cell_flat", np.zeros((node_seg_ids.size,), dtype=np.int32)), dtype=np.int32)
    grid_n = int(hybrid.get("grid_n", 0))
    grid_cells = int(grid_n * grid_n * grid_n)
    if source_mask is None or node_seg_ids.size <= 0 or node_cell_flat.size != node_seg_ids.size or grid_cells <= 0:
        return {"kind": "hybrid_local_static", "uses_static": True, "active_count": int(node_seg_ids.size)}
    mask = np.asarray(source_mask, dtype=bool).reshape(-1)
    if mask.size <= 0 or int(np.count_nonzero(mask)) >= int(mask.size):
        return {"kind": "hybrid_local_static", "uses_static": True, "active_count": int(node_seg_ids.size)}
    valid_node_mask = (
        (node_seg_ids >= 0)
        & (node_seg_ids < mask.size)
        & mask[np.clip(node_seg_ids, 0, max(mask.size - 1, 0))]
        & (node_cell_flat >= 0)
        & (node_cell_flat < grid_cells)
    )
    node_ids = np.flatnonzero(valid_node_mask).astype(np.int32, copy=False)
    if node_ids.size <= 0:
        return {
            "kind": "hybrid_local_active",
            "uses_static": False,
            "cell_ptr": np.zeros((grid_cells + 1,), dtype=np.int32),
            "node_ids": np.zeros((0,), dtype=np.int32),
            "active_count": 0,
        }
    flat = np.asarray(node_cell_flat[node_ids], dtype=np.int64)
    order = np.argsort(flat, kind="stable")
    sorted_flat = np.asarray(flat[order], dtype=np.int64)
    sorted_node_ids = np.asarray(node_ids[order], dtype=np.int32)
    counts = np.bincount(sorted_flat, minlength=grid_cells)
    cell_ptr = np.zeros((grid_cells + 1,), dtype=np.int32)
    cell_ptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64).astype(np.int32)
    return {
        "kind": "hybrid_local_active",
        "uses_static": False,
        "cell_ptr": np.asarray(cell_ptr, dtype=np.int32),
        "node_ids": np.asarray(sorted_node_ids, dtype=np.int32),
        "active_count": int(node_ids.size),
    }




def _resolve_cext_hybrid_bg_mode() -> str:
    mode = str(CEXT_HYBRID_BG_MODE or "local_only_nlambda").strip().lower()
    aliases = {
        "local": "local_only_nlambda",
        "local_only_screened": "local_only_nlambda",
        "nlambda": "local_only_nlambda",
        "n_lambda": "local_only_nlambda",
        "pure_fft": "fft",
        "grid": "fft",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("local_only_nlambda", "local_only", "fft", "hybrid"):
        raise ValueError(
            "CEXT_HYBRID_BG_MODE must be one of local_only_nlambda, local_only, fft, or hybrid."
        )
    return mode


def _ensure_cext_hybrid_local_only_gpu_static(context: dict, hybrid: dict) -> dict:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = hybrid.get("gpu_static")
    if isinstance(static, dict) and "local_node_cell_flat" in static:
        return static
    grid_n = int(hybrid["grid_n"])
    if float(hybrid.get("near_radius_si", 0.0)) > 0.0:
        local_cell_meta = _cext_hybrid_build_local_node_cells(hybrid)
    else:
        n_nodes = int(np.asarray(hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32))).shape[0])
        grid_cells = int(grid_n * grid_n * grid_n)
        local_cell_meta = {
            "local_cell_ptr": np.zeros((grid_cells + 1,), dtype=np.int32),
            "local_node_ids": np.zeros((0,), dtype=np.int32),
            "local_node_cell_flat": np.full((n_nodes,), -1, dtype=np.int32),
            "local_rad_cells": 0,
        }
    hybrid["local_node_cell_flat"] = np.asarray(local_cell_meta["local_node_cell_flat"], dtype=np.int32)
    hybrid["local_rad_cells"] = int(local_cell_meta["local_rad_cells"])
    grid_cells = int(grid_n * grid_n * grid_n)
    static = {
        "origin": _cp.asarray(np.asarray(hybrid["origin"], dtype=np.float32)),
        "k2": _cp.asarray(np.zeros((1,), dtype=np.float32)),
        "node_seg_ids": _cp.asarray(np.asarray(hybrid["node_seg_ids"], dtype=np.int32)),
        "gl_points_flat": _cp.asarray(np.asarray(hybrid["gl_points_flat"], dtype=np.float32)),
        "all_target_seg_ids": _cp.arange(int(np.asarray(context["midpoints_si"]).shape[0]), dtype=_cp.int32),
        "grid_shape": (grid_n, grid_n, grid_n),
        "stencil_flat_idx": _cp.asarray(np.zeros((1, 1), dtype=np.int32)),
        "stencil_weight": _cp.asarray(np.ones((1, 1), dtype=np.float32)),
        "stencil_n": 1,
        "grid_cells": grid_cells,
        "local_cell_ptr": _cp.asarray(np.asarray(local_cell_meta["local_cell_ptr"], dtype=np.int32)),
        "local_node_ids": _cp.asarray(np.asarray(local_cell_meta["local_node_ids"], dtype=np.int32)),
        "local_node_cell_flat": _cp.asarray(np.asarray(local_cell_meta["local_node_cell_flat"], dtype=np.int32)),
        "local_rad_cells": int(local_cell_meta["local_rad_cells"]),
    }
    hybrid["gpu_static"] = static
    return static


def _ensure_cext_hybrid_local_only_runtime_state(context: dict, hybrid: dict, ext_state: dict) -> dict:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gl_shape = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape
    nseg = int(gl_shape[0]) if len(gl_shape) >= 1 else 0
    grid_shape = (0, int(hybrid["grid_n"]), int(hybrid["grid_n"]), int(hybrid["grid_n"]))
    state = hybrid.get("runtime_state")
    if isinstance(state, dict):
        if tuple(state.get("gl_shape", ())) == tuple(gl_shape) and bool(state.get("local_only_lean", False)):
            return state
    state = {
        "gl_shape": tuple(gl_shape),
        "grid_shape": tuple(grid_shape),
        "stencil_n": int(static.get("stencil_n", 1)),
        "local_only_lean": True,
        "q_weighted_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "mono2_weight_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "dipole2_weight_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "lambda_iv_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "seg_cap_gl": _cp.zeros((nseg,), dtype=_cp.float32),
        "c_ext_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "active_source_mask_g": _cp.ones((nseg,), dtype=_cp.uint8),
        "local_total_g": _cp.zeros(gl_shape, dtype=_cp.float32),
        "local_cap_g": _cp.zeros(gl_shape, dtype=_cp.float32),
    }
    hybrid["runtime_state"] = state
    return state

def _ensure_cext_hybrid_bg_gpu_static(context: dict, hybrid: dict) -> dict:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    if _resolve_cext_hybrid_bg_mode().startswith("local_only"):
        return _ensure_cext_hybrid_local_only_gpu_static(context, hybrid)
    static = hybrid.get("gpu_static")
    if isinstance(static, dict) and "local_node_cell_flat" not in static:
        return static
    grid_n = int(hybrid["grid_n"])
    stencil_meta = _cext_hybrid_build_stencil_metadata_gpu(hybrid)
    if float(hybrid.get("near_radius_si", 0.0)) > 0.0:
        local_cell_meta = _cext_hybrid_build_local_node_cells(hybrid)
    else:
        n_nodes = int(np.asarray(hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32))).shape[0])
        grid_cells = int(grid_n * grid_n * grid_n)
        local_cell_meta = {
            "local_cell_ptr": np.zeros((grid_cells + 1,), dtype=np.int32),
            "local_node_ids": np.zeros((0,), dtype=np.int32),
            "local_node_cell_flat": np.full((n_nodes,), -1, dtype=np.int32),
            "local_rad_cells": 0,
        }
    hybrid["local_node_cell_flat"] = np.asarray(local_cell_meta["local_node_cell_flat"], dtype=np.int32)
    hybrid["local_rad_cells"] = int(local_cell_meta["local_rad_cells"])
    static = {
        "origin": _cp.asarray(np.asarray(hybrid["origin"], dtype=np.float32)),
        "k2": _cp.asarray(np.asarray(hybrid["k2"], dtype=np.float32)),
        "node_seg_ids": _cp.asarray(np.asarray(hybrid["node_seg_ids"], dtype=np.int32)),
        "gl_points_flat": _cp.asarray(np.asarray(hybrid["gl_points_flat"], dtype=np.float32)),
        "all_target_seg_ids": _cp.arange(int(np.asarray(context["midpoints_si"]).shape[0]), dtype=_cp.int32),
        "grid_shape": (grid_n, grid_n, grid_n),
        "stencil_flat_idx": _cp.asarray(stencil_meta["stencil_flat_idx"], dtype=_cp.int32),
        "stencil_weight": _cp.asarray(stencil_meta["stencil_weight"], dtype=_cp.float32),
        "stencil_n": int(stencil_meta["stencil_n"]),
        "grid_cells": int(stencil_meta["grid_cells"]),
        "local_cell_ptr": _cp.asarray(np.asarray(local_cell_meta["local_cell_ptr"], dtype=np.int32)),
        "local_node_ids": _cp.asarray(np.asarray(local_cell_meta["local_node_ids"], dtype=np.int32)),
        "local_rad_cells": int(local_cell_meta["local_rad_cells"]),
    }
    hybrid["gpu_static"] = static
    return static


def _ensure_cext_hybrid_bg_runtime_state(context: dict, hybrid: dict, ext_state: dict) -> dict:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    if _resolve_cext_hybrid_bg_mode().startswith("local_only"):
        return _ensure_cext_hybrid_local_only_runtime_state(context, hybrid, ext_state)
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gl_shape = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape
    nseg = int(gl_shape[0]) if len(gl_shape) >= 1 else 0
    gl_order = int(gl_shape[1]) if len(gl_shape) >= 2 else 0
    n_bins = max(int(hybrid["lambda_bins"]), 1)
    grid_n = int(hybrid["grid_n"])
    grid_shape = (n_bins, grid_n, grid_n, grid_n)
    state = hybrid.get("runtime_state")
    if isinstance(state, dict):
        if (
            tuple(state.get("gl_shape", ())) == tuple(gl_shape)
            and tuple(state.get("grid_shape", ())) == tuple(grid_shape)
            and int(state.get("stencil_n", -1)) == int(static["stencil_n"])
        ):
            return state
    state = {
        "gl_shape": tuple(gl_shape),
        "grid_shape": tuple(grid_shape),
        "stencil_n": int(static["stencil_n"]),
        "q_weighted_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "mono2_weight_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "dipole2_weight_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "lambda_iv_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "seg_cap_gl": _cp.zeros((nseg,), dtype=_cp.float32),
        "c_ext_gl": _cp.zeros(gl_shape, dtype=_cp.float32),
        "active_source_mask_g": _cp.ones((nseg,), dtype=_cp.uint8),
        "active_mass_grids_g": _cp.zeros(grid_shape, dtype=_cp.float32),
        "bg_sample_g": _cp.zeros(gl_shape, dtype=_cp.float32),
        "local_total_g": _cp.zeros(gl_shape, dtype=_cp.float32),
        "local_cap_g": _cp.zeros(gl_shape, dtype=_cp.float32),
    }
    hybrid["runtime_state"] = state
    return state


def _sync_cext_hybrid_bg_runtime_state(ext_state: dict, runtime_state: dict) -> None:
    runtime_state["q_weighted_gl"].set(np.asarray(ext_state["q_weighted_gl"], dtype=np.float32))
    if "mono2_weight_gl" in runtime_state:
        runtime_state["mono2_weight_gl"].set(np.asarray(ext_state.get("mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32))
    if "dipole2_weight_gl" in runtime_state:
        runtime_state["dipole2_weight_gl"].set(np.asarray(ext_state.get("dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32))
    runtime_state["lambda_iv_gl"].set(np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32))
    runtime_state["seg_cap_gl"].set(np.asarray(ext_state["seg_cap_gl"], dtype=np.float32))
    runtime_state["c_ext_gl"].set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))


def _resolve_cext_hybrid_bg_solver_mode() -> str:
    mode = str(CEXT_HYBRID_BG_SOLVER or "auto").strip().lower()
    if mode not in ("auto", "fft", "jacobi"):
        mode = "auto"
    return mode


def _cext_hybrid_init_lambda_bins(hybrid: dict, ext_state: dict) -> tuple[np.ndarray, np.ndarray]:
    edges = hybrid.get("lambda_bin_edges")
    centers = hybrid.get("lambda_bin_centers")
    epoch = int(ext_state.get("_lambda_bin_epoch", -1))
    quantile_fft = (
        bool(CEXT_HYBRID_FFT_QUANTILE_BINS)
        and str(hybrid.get("bg_mode", _resolve_cext_hybrid_bg_mode())).strip().lower() == "fft"
    )
    if (
        bool(CEXT_HYBRID_FFT_BIN_EPOCH_CACHE)
        and quantile_fft
        and edges is not None
        and centers is not None
        and int(hybrid.get("lambda_bin_epoch", -2)) == epoch
    ):
        edge_arr = np.asarray(edges, dtype=np.float32)
        center_arr = np.asarray(centers, dtype=np.float32)
        if _cp is not None and isinstance(hybrid.get("gpu_static"), dict):
            static = hybrid["gpu_static"]
            if int(static.get("lambda_bin_epoch", -2)) != epoch:
                static["lambda_bin_edges"] = _cp.asarray(edge_arr)
                static["lambda_bin_centers"] = _cp.asarray(center_arr)
                static["lambda_bin_epoch"] = int(epoch)
        return edge_arr, center_arr
    if not quantile_fft and edges is not None and centers is not None:
        return np.asarray(edges, dtype=np.float32), np.asarray(centers, dtype=np.float32)
    lambda_gl = np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32)
    finite = lambda_gl[np.isfinite(lambda_gl) & (lambda_gl > 0.0)]
    if finite.size <= 0:
        finite = np.asarray([1.0e-6], dtype=np.float32)
    lam_min = max(float(np.min(finite)), 1.0e-8)
    lam_max = max(float(np.max(finite)), lam_min * (1.0 + 1.0e-6))
    n_bins = max(int(hybrid["lambda_bins"]), 1)
    if quantile_fft and n_bins > 1 and finite.size > 1:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges64 = np.quantile(np.asarray(finite, dtype=np.float64), quantiles)
        edges64[0] = lam_min
        edges64[-1] = lam_max
        min_step = max(lam_max, lam_min) * 1.0e-7
        for idx in range(1, edges64.size):
            if edges64[idx] <= edges64[idx - 1]:
                edges64[idx] = edges64[idx - 1] + min_step
        if edges64[-1] > lam_max:
            edges64 = np.linspace(lam_min, max(float(edges64[-1]), lam_max), n_bins + 1)
        edges = np.asarray(edges64, dtype=np.float32)
        centers = np.sqrt(np.maximum(edges[:-1], 1.0e-30) * np.maximum(edges[1:], 1.0e-30)).astype(np.float32)
        policy = "dynamic_quantile"
    elif n_bins <= 1:
        centers = np.asarray([math.sqrt(lam_min * lam_max)], dtype=np.float32)
        edges = np.asarray([lam_min, lam_max], dtype=np.float32)
        policy = "dynamic_quantile" if quantile_fft else "geomspace"
    else:
        if lam_max <= lam_min * (1.0 + 1.0e-6):
            pad = 1.0e-3
            lo = max(lam_min / (1.0 + pad), 1.0e-8)
            hi = max(lam_max * (1.0 + pad), lo * (1.0 + 1.0e-6))
        else:
            lo = lam_min
            hi = lam_max
        edges = np.geomspace(lo, hi, n_bins + 1).astype(np.float32)
        centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
        policy = "geomspace"
    hybrid["lambda_bin_edges"] = np.asarray(edges, dtype=np.float32)
    hybrid["lambda_bin_centers"] = np.asarray(centers, dtype=np.float32)
    edge_arr = np.asarray(edges, dtype=np.float32)
    center_arr = np.asarray(centers, dtype=np.float32)
    hybrid["lambda_bin_policy"] = policy
    hybrid["lambda_bin_epoch"] = int(epoch)
    hybrid["lambda_bin_edges_hash"] = hashlib.sha256(edge_arr.tobytes()).hexdigest()[:16]
    hybrid["lambda_bin_meta"] = {
        "lambda_bin_policy": policy,
        "lambda_bin_edges_hash": hybrid["lambda_bin_edges_hash"],
        "lambda_bin_edge_min": float(edge_arr[0]) if edge_arr.size else float("nan"),
        "lambda_bin_edge_max": float(edge_arr[-1]) if edge_arr.size else float("nan"),
        "lambda_bin_count": int(max(edge_arr.size - 1, 0)),
        "lambda_bin_center_min": float(np.min(center_arr)) if center_arr.size else float("nan"),
        "lambda_bin_center_max": float(np.max(center_arr)) if center_arr.size else float("nan"),
    }
    if _cp is not None and isinstance(hybrid.get("gpu_static"), dict):
        hybrid["gpu_static"]["lambda_bin_edges"] = _cp.asarray(np.asarray(edges, dtype=np.float32))
        hybrid["gpu_static"]["lambda_bin_centers"] = _cp.asarray(np.asarray(centers, dtype=np.float32))
        hybrid["gpu_static"]["lambda_bin_epoch"] = int(epoch)
    return np.asarray(edges, dtype=np.float32), np.asarray(centers, dtype=np.float32)


def _cext_hybrid_axis_assignment_gpu(coord_g, grid_n: int, assignment: str):
    if assignment == "tsc":
        center = _cp.floor(coord_g + _cp.float32(0.5)).astype(_cp.int32)
        idx_list = []
        weight_list = []
        for offset in (-1, 0, 1):
            idx = center + np.int32(offset)
            dist = _cp.abs(coord_g - idx.astype(_cp.float32))
            weight = _cp.where(
                dist < _cp.float32(0.5),
                _cp.float32(0.75) - dist * dist,
                _cp.where(
                    dist < _cp.float32(1.5),
                    _cp.float32(0.5) * (_cp.float32(1.5) - dist) * (_cp.float32(1.5) - dist),
                    _cp.float32(0.0),
                ),
            ).astype(_cp.float32)
            idx_list.append(idx)
            weight_list.append(weight)
        return idx_list, weight_list
    base = _cp.floor(coord_g).astype(_cp.int32)
    frac = (coord_g - base.astype(_cp.float32)).astype(_cp.float32)
    return [base, base + np.int32(1)], [(_cp.float32(1.0) - frac).astype(_cp.float32), frac.astype(_cp.float32)]


def _get_cext_hybrid_deposit_kernel():
    global _CEXT_HYBRID_DEPOSIT_KERNEL
    if _CEXT_HYBRID_DEPOSIT_KERNEL is not None:
        return _CEXT_HYBRID_DEPOSIT_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_hybrid_deposit_kernel(
    const float* q_weighted_gl,
    const float* lambda_iv_gl,
    const int* node_seg_ids,
    const unsigned char* active_source_mask,
    const int* stencil_flat_idx,
    const float* stencil_weight,
    const float* lambda_bin_edges,
    int n_bins,
    int grid_cells,
    int stencil_n,
    int n_nodes,
    float* out_mass
) {
    int node_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_idx >= n_nodes) return;
    int seg_idx = node_seg_ids[node_idx];
    if (active_source_mask[seg_idx] == 0) return;
    float q = q_weighted_gl[node_idx];
    float lam = lambda_iv_gl[node_idx];
    if (!isfinite(q) || !isfinite(lam) || q == 0.0f || lam <= 0.0f) return;
    int bin_idx = 0;
    for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
        if (lam > lambda_bin_edges[edge_idx]) {
            bin_idx = edge_idx;
        } else {
            break;
        }
    }
    int base = bin_idx * grid_cells;
    int stencil_base = node_idx * stencil_n;
    for (int slot = 0; slot < stencil_n; ++slot) {
        int flat = stencil_flat_idx[stencil_base + slot];
        if (flat < 0) continue;
        float weight = stencil_weight[stencil_base + slot];
        if (!(weight != 0.0f) || !isfinite(weight)) continue;
        atomicAdd(&out_mass[base + flat], q * weight);
    }
}
'''
    _CEXT_HYBRID_DEPOSIT_KERNEL = _cp.RawKernel(code, "cext_hybrid_deposit_kernel")
    return _CEXT_HYBRID_DEPOSIT_KERNEL


def _cext_hybrid_deposit_sources_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    *,
    source_mask: np.ndarray | None = None,
    runtime_state: dict | None = None,
    out_mass_grids_g=None,
) -> tuple[object, float]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    state = runtime_state if isinstance(runtime_state, dict) else _ensure_cext_hybrid_bg_runtime_state(context, hybrid, ext_state)
    edges, _ = _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    if "lambda_bin_edges" not in static:
        static["lambda_bin_edges"] = _cp.asarray(np.asarray(edges, dtype=np.float32))
    grid_n = int(hybrid["grid_n"])
    n_bins = max(int(np.asarray(edges).size - 1), 1)
    if "q_weighted_gl" in ext_state:
        n_nodes = int(np.asarray(ext_state["q_weighted_gl"], dtype=np.float32).size)
    else:
        n_nodes = int(state["q_weighted_gl"].size)
    if out_mass_grids_g is None:
        out_mass = state["active_mass_grids_g"]
    else:
        out_mass = out_mass_grids_g
    t0 = perf_counter()
    out_mass.fill(_cp.float32(0.0))
    if source_mask is None:
        state["active_source_mask_g"].fill(np.uint8(1))
    else:
        state["active_source_mask_g"].set(np.asarray(source_mask, dtype=np.uint8).reshape(-1))
    if n_nodes <= 0:
        return out_mass, 0.0
    kernel = _get_cext_hybrid_deposit_kernel()
    threads = 256
    blocks = (n_nodes + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            state["q_weighted_gl"].ravel(),
            state["lambda_iv_gl"].ravel(),
            static["node_seg_ids"],
            state["active_source_mask_g"],
            static["stencil_flat_idx"].ravel(),
            static["stencil_weight"].ravel(),
            static["lambda_bin_edges"],
            np.int32(n_bins),
            np.int32(int(static["grid_cells"])),
            np.int32(int(static["stencil_n"])),
            np.int32(n_nodes),
            out_mass.ravel(),
        ),
    )
    _cp.cuda.Stream.null.synchronize()
    return out_mass, perf_counter() - t0


def _cext_hybrid_solve_background_fft_gpu(
    context: dict,
    hybrid: dict,
    mass_grids_g,
    *,
    init_phi_grids_g=None,
) -> tuple[object, float, str]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    lambda_centers = np.asarray(hybrid.get("lambda_bin_centers", np.asarray([1.0], dtype=np.float32)), dtype=np.float32)
    if "lambda_bin_centers" not in static:
        static["lambda_bin_centers"] = _cp.asarray(np.asarray(lambda_centers, dtype=np.float32))
    spacing = float(hybrid["spacing"])
    diffusivity_si = float(context["diffusivity_si"])
    cell_vol = spacing ** 3
    mass_arr = _cp.asarray(mass_grids_g, dtype=_cp.float32)
    rhs_scale = np.float32(max(cell_vol * diffusivity_si, 1.0e-30))
    lambda_centers_g = static["lambda_bin_centers"]
    solver_mode = _resolve_cext_hybrid_bg_solver_mode()
    if solver_mode in ("auto", "fft"):
        try:
            t0 = perf_counter()
            phi_grids = _cp.empty_like(mass_arr)
            if bool(CEXT_HYBRID_FFT_RESPONSE_BATCHED):
                try:
                    rhs_hat = _cp.fft.fftn(mass_arr / rhs_scale, axes=(1, 2, 3))
                    lam = _cp.maximum(lambda_centers_g, _cp.float32(1.0e-8)).reshape((-1, 1, 1, 1))
                    denom = static["k2"][None, :, :, :] + _cp.reciprocal(lam * lam)
                    phi_grids[...] = _cp.real(_cp.fft.ifftn(rhs_hat / denom, axes=(1, 2, 3))).astype(_cp.float32)
                    del rhs_hat, lam, denom
                except Exception:
                    if str(os.environ.get("SVV_CEXT_HYBRID_FFT_RESPONSE_BATCHED_STRICT", "false")).strip().lower() in ("1", "true", "yes", "on"):
                        raise
                    for bin_idx in range(int(mass_arr.shape[0])):
                        rhs = mass_arr[bin_idx] / rhs_scale
                        rhs_hat = _cp.fft.fftn(rhs, axes=(0, 1, 2))
                        lam = _cp.maximum(lambda_centers_g[bin_idx], _cp.float32(1.0e-8))
                        denom = static["k2"] + _cp.reciprocal(lam * lam)
                        phi_hat = rhs_hat / denom
                        phi_grids[bin_idx] = _cp.real(_cp.fft.ifftn(phi_hat, axes=(0, 1, 2))).astype(_cp.float32)
                        del rhs, rhs_hat, denom, phi_hat
            else:
                for bin_idx in range(int(mass_arr.shape[0])):
                    rhs = mass_arr[bin_idx] / rhs_scale
                    rhs_hat = _cp.fft.fftn(rhs, axes=(0, 1, 2))
                    lam = _cp.maximum(lambda_centers_g[bin_idx], _cp.float32(1.0e-8))
                    denom = static["k2"] + _cp.reciprocal(lam * lam)
                    phi_hat = rhs_hat / denom
                    phi_grids[bin_idx] = _cp.real(_cp.fft.ifftn(phi_hat, axes=(0, 1, 2))).astype(_cp.float32)
                    del rhs, rhs_hat, denom, phi_hat
            _cp.cuda.Stream.null.synchronize()
            return phi_grids, perf_counter() - t0, "fft"
        except Exception:
            if solver_mode == "fft":
                raise

    t0 = perf_counter()
    rhs_all = mass_arr / rhs_scale
    phi_grids = _cp.asarray(init_phi_grids_g, dtype=_cp.float32).copy() if init_phi_grids_g is not None else _cp.zeros_like(rhs_all)
    phi_next = _cp.zeros_like(rhs_all)
    inv_h2 = np.float32(1.0 / max(spacing * spacing, 1.0e-30))
    omega = np.float32(0.8)
    n_sweeps = max(8, 8 * max(int(CEXT_HYBRID_BG_VCYCLES), 1))
    for bin_idx in range(int(rhs_all.shape[0])):
        lam = _cp.maximum(lambda_centers_g[bin_idx], _cp.float32(1.0e-8))
        lam_inv2 = _cp.float32(1.0) / (lam * lam)
        diag = lam_inv2 + _cp.float32(6.0) * inv_h2
        rhs = rhs_all[bin_idx]
        phi = phi_grids[bin_idx]
        phi_n = phi_next[bin_idx]
        for _ in range(n_sweeps):
            phi_n.fill(_cp.float32(0.0))
            nbr = (
                phi[:-2, 1:-1, 1:-1]
                + phi[2:, 1:-1, 1:-1]
                + phi[1:-1, :-2, 1:-1]
                + phi[1:-1, 2:, 1:-1]
                + phi[1:-1, 1:-1, :-2]
                + phi[1:-1, 1:-1, 2:]
            )
            jac = (rhs[1:-1, 1:-1, 1:-1] + np.float32(inv_h2) * nbr) / diag
            phi_n[1:-1, 1:-1, 1:-1] = (np.float32(1.0) - omega) * phi[1:-1, 1:-1, 1:-1] + omega * jac
            phi, phi_n = phi_n, phi
        phi_grids[bin_idx] = phi
    _cp.cuda.Stream.null.synchronize()
    return phi_grids, perf_counter() - t0, "jacobi"



_CEXT_LOCAL_CELL_MAX_LAMBDA_KERNEL = None
_CEXT_FAST_LOCAL_ONLY_SELF_KERNEL = None


def _get_cext_local_cell_max_lambda_kernel():
    global _CEXT_LOCAL_CELL_MAX_LAMBDA_KERNEL
    if _CEXT_LOCAL_CELL_MAX_LAMBDA_KERNEL is not None:
        return _CEXT_LOCAL_CELL_MAX_LAMBDA_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __device__ int atomicMaxFloatBits(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    int val_i = __float_as_int(val);
    while (__int_as_float(old) < val) {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, val_i);
        if (assumed == old) break;
    }
    return old;
}
extern "C" __global__ void cext_local_cell_max_lambda_kernel(
    const int* local_node_cell_flat,
    const int* node_seg_ids,
    const float* lambda_iv_gl,
    const unsigned char* active_source_mask,
    const int n_nodes,
    const int gl_order,
    float* cell_max_lambda
) {
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= n_nodes) return;
    int cell = local_node_cell_flat[node];
    if (cell < 0) return;
    int source_seg = node_seg_ids[node];
    if (active_source_mask[source_seg] == 0) return;
    int source_gl = node - source_seg * gl_order;
    if (source_gl < 0 || source_gl >= gl_order) return;
    float lam = lambda_iv_gl[source_seg * gl_order + source_gl];
    if (!(lam > 0.0f) || !isfinite(lam)) return;
    atomicMaxFloatBits(&cell_max_lambda[cell], lam);
}
'''
    _CEXT_LOCAL_CELL_MAX_LAMBDA_KERNEL = _cp.RawKernel(code, "cext_local_cell_max_lambda_kernel")
    return _CEXT_LOCAL_CELL_MAX_LAMBDA_KERNEL


def _get_cext_fast_local_only_self_kernel():
    global _CEXT_FAST_LOCAL_ONLY_SELF_KERNEL
    if _CEXT_FAST_LOCAL_ONLY_SELF_KERNEL is not None:
        return _CEXT_FAST_LOCAL_ONLY_SELF_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_fast_local_only_self_kernel(
    const float* gl_points,
    const float* segment_vectors,
    const float* radii,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    const unsigned char* active_source_mask,
    const int* cell_ptr,
    const int* cell_node_ids,
    const float* cell_max_lambda,
    const float origin_x,
    const float origin_y,
    const float origin_z,
    const float spacing,
    const int grid_n,
    const int local_rad_cells,
    const float near_radius_si,
    const float lambda_window,
    const int gl_order,
    const int nseg,
    const float diffusivity,
    float* c_ext_total,
    float* c_ext_cap
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nseg * gl_order;
    if (idx >= total) return;
    int target_seg = idx / gl_order;
    int target_gl = idx - target_seg * gl_order;
    int pidx = target_seg * gl_order + target_gl;
    float px = gl_points[3*pidx + 0];
    float py = gl_points[3*pidx + 1];
    float pz = gl_points[3*pidx + 2];
    int ix = (int)floorf((px - origin_x) / spacing);
    int iy = (int)floorf((py - origin_y) / spacing);
    int iz = (int)floorf((pz - origin_z) / spacing);
    if (ix < 0) ix = 0; if (ix >= grid_n) ix = grid_n - 1;
    if (iy < 0) iy = 0; if (iy >= grid_n) iy = grid_n - 1;
    if (iz < 0) iz = 0; if (iz >= grid_n) iz = grid_n - 1;
    float rcut2 = near_radius_si * near_radius_si;
    float target_radius = radii[target_seg];
    float total_val = 0.0f;
    float cap_val = 0.0f;
    const float four_pi = 12.566370614359172f;
    for (int dz = -local_rad_cells; dz <= local_rad_cells; ++dz) {
        int cz = iz + dz;
        if (cz < 0 || cz >= grid_n) continue;
        for (int dy = -local_rad_cells; dy <= local_rad_cells; ++dy) {
            int cy = iy + dy;
            if (cy < 0 || cy >= grid_n) continue;
            for (int dx = -local_rad_cells; dx <= local_rad_cells; ++dx) {
                int cx = ix + dx;
                if (cx < 0 || cx >= grid_n) continue;
                int cell = (cx * grid_n + cy) * grid_n + cz;
                float cell_lam = cell_max_lambda[cell];
                if (!(cell_lam > 0.0f) || !isfinite(cell_lam)) continue;
                float cell_cutoff = lambda_window * cell_lam;
                float center_x = origin_x + ((float)cx + 0.5f) * spacing;
                float center_y = origin_y + ((float)cy + 0.5f) * spacing;
                float center_z = origin_z + ((float)cz + 0.5f) * spacing;
                float ddx = fabsf(px - center_x) - 0.5f * spacing;
                float ddy = fabsf(py - center_y) - 0.5f * spacing;
                float ddz = fabsf(pz - center_z) - 0.5f * spacing;
                if (ddx < 0.0f) ddx = 0.0f;
                if (ddy < 0.0f) ddy = 0.0f;
                if (ddz < 0.0f) ddz = 0.0f;
                if ((ddx*ddx + ddy*ddy + ddz*ddz) > cell_cutoff * cell_cutoff) continue;
                int start = cell_ptr[cell];
                int end = cell_ptr[cell + 1];
                for (int ptr = start; ptr < end; ++ptr) {
                    int source_node = cell_node_ids[ptr];
                    int source_seg = source_node / gl_order;
                    if (source_seg == target_seg) continue;
                    if (active_source_mask[source_seg] == 0) continue;
                    int source_gl = source_node - source_seg * gl_order;
                    float lam = lambda_iv_gl[source_seg * gl_order + source_gl];
                    if (!(lam > 0.0f) || !isfinite(lam)) continue;
                    float source_cutoff = lambda_window * lam;
                    float sx = gl_points[3*source_node + 0];
                    float sy = gl_points[3*source_node + 1];
                    float sz = gl_points[3*source_node + 2];
                    float rx = px - sx;
                    float ry = py - sy;
                    float rz = pz - sz;
                    float dist2 = rx*rx + ry*ry + rz*rz;
                    float radius_sum = target_radius + radii[source_seg];
                    float rr2 = dist2 + radius_sum * radius_sum;
                    if (rr2 > rcut2 || rr2 > source_cutoff * source_cutoff) continue;
                    float rr = sqrtf(rr2);
                    if (!(rr > 0.0f) || !isfinite(rr)) continue;
                    float qw = q_weighted_gl[source_seg * gl_order + source_gl];
                    float kernel = expf(-rr / lam) / (four_pi * diffusivity * rr);
                    total_val += qw * kernel;
                    float o2_weight = mono2_weight_gl[source_seg * gl_order + source_gl]
                                      + dipole2_weight_gl[source_seg * gl_order + source_gl];
                    if (o2_weight != 0.0f) {
                        float vx = segment_vectors[source_seg * 3 + 0];
                        float vy = segment_vectors[source_seg * 3 + 1];
                        float vz = segment_vectors[source_seg * 3 + 2];
                        float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
                        if (vlen > 1.0e-30f) {
                            float tdotr = (vx * rx + vy * ry + vz * rz) / vlen;
                            float mu2 = (tdotr * tdotr) / (rr * rr);
                            if (mu2 > 1.0f) mu2 = 1.0f;
                            float inv_r = 1.0f / rr;
                            float inv_l = 1.0f / lam;
                            float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                                      + (1.0f - mu2) * inv_l * inv_l) * kernel;
                            total_val += o2_weight * pH;
                        }
                    }
                    float src_cap = seg_cap_gl[source_seg];
                    if (src_cap > cap_val) cap_val = src_cap;
                }
            }
        }
    }
    if (!isfinite(total_val) || total_val < 0.0f) total_val = 0.0f;
    c_ext_total[idx] = total_val;
    c_ext_cap[idx] = cap_val;
}
'''
    _CEXT_FAST_LOCAL_ONLY_SELF_KERNEL = _cp.RawKernel(code, "cext_fast_local_only_self_kernel")
    return _CEXT_FAST_LOCAL_ONLY_SELF_KERNEL


def _compute_cext_local_only_fast_self_gpu(context: dict, hybrid: dict, ext_state: dict, *, runtime_state: dict | None = None) -> tuple[object, object, float, float]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    hybrid_static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    state = runtime_state if isinstance(runtime_state, dict) else _ensure_cext_hybrid_bg_runtime_state(context, hybrid, ext_state)
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    if nseg <= 0 or gl_order <= 0 or float(hybrid.get("near_radius_si", 0.0)) <= 0.0:
        zeros = _cp.zeros((nseg, gl_order), dtype=_cp.float32)
        return zeros, zeros.copy(), 0.0, 0.0
    t_transfer = perf_counter()
    grid_cells = int(hybrid_static.get("grid_cells", int(hybrid["grid_n"]) ** 3))
    cell_max_lambda_g = state.get("local_cell_max_lambda_g")
    if cell_max_lambda_g is None or int(cell_max_lambda_g.size) != grid_cells:
        cell_max_lambda_g = _cp.zeros((grid_cells,), dtype=_cp.float32)
        state["local_cell_max_lambda_g"] = cell_max_lambda_g
    else:
        cell_max_lambda_g.fill(0.0)
    source_threads = 256
    n_nodes = int(np.asarray(hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32))).shape[0])
    source_blocks = (n_nodes + source_threads - 1) // source_threads
    max_kernel = _get_cext_local_cell_max_lambda_kernel()
    max_kernel(
        (source_blocks,),
        (source_threads,),
        (
            hybrid_static["local_node_cell_flat"],
            hybrid_static["node_seg_ids"],
            state["lambda_iv_gl"].ravel(),
            state["active_source_mask_g"],
            np.int32(n_nodes),
            np.int32(gl_order),
            cell_max_lambda_g,
        ),
    )
    c_ext_total_g = state["local_total_g"]
    c_ext_cap_g = state["local_cap_g"]
    _cp.cuda.Stream.null.synchronize()
    transfer_total = perf_counter() - t_transfer
    origin = np.asarray(hybrid["origin"], dtype=np.float32)
    threads = 128
    blocks = (nseg * gl_order + threads - 1) // threads
    kernel = _get_cext_fast_local_only_self_kernel()
    t_kernel = perf_counter()
    kernel(
        (blocks,),
        (threads,),
        (
            static["gl_points_si"].ravel(),
            static["segment_vectors"].ravel(),
            static["radii_si"],
            state["lambda_iv_gl"].ravel(),
            state["q_weighted_gl"].ravel(),
            state["mono2_weight_gl"].ravel(),
            state["dipole2_weight_gl"].ravel(),
            state["seg_cap_gl"],
            state["active_source_mask_g"],
            hybrid_static["local_cell_ptr"],
            hybrid_static["local_node_ids"],
            cell_max_lambda_g,
            np.float32(float(origin[0])),
            np.float32(float(origin[1])),
            np.float32(float(origin[2])),
            np.float32(float(hybrid["spacing"])),
            np.int32(int(hybrid["grid_n"])),
            np.int32(int(hybrid_static["local_rad_cells"])),
            np.float32(float(hybrid["near_radius_si"])),
            np.float32(float(CEXT_WINDOW_FACTOR)),
            np.int32(gl_order),
            np.int32(nseg),
            np.float32(float(context["diffusivity_si"])),
            c_ext_total_g.ravel(),
            c_ext_cap_g.ravel(),
        ),
    )
    _cp.cuda.Stream.null.synchronize()
    return c_ext_total_g, c_ext_cap_g, perf_counter() - t_kernel, transfer_total


def _sync_cext_local_only_lambda_radius(hybrid: dict, ext_state: dict, *, use_lambda_cutoff: bool) -> None:
    if not use_lambda_cutoff:
        return
    lambda_gl = np.asarray(ext_state.get("lambda_iv_gl", []), dtype=np.float32)
    finite = lambda_gl[np.isfinite(lambda_gl) & (lambda_gl > 0.0)]
    near_radius = float(CEXT_WINDOW_FACTOR) * float(np.max(finite)) if finite.size else 0.0
    old_near_radius = float(hybrid.get("near_radius_si", 0.0))
    hybrid["near_radius_si"] = float(near_radius)
    spacing = max(float(hybrid.get("spacing", 0.0)), 1.0e-30)
    hybrid["local_rad_cells"] = int(math.ceil(float(near_radius) / spacing)) if near_radius > 0.0 else 0
    static = hybrid.get("gpu_static")
    if isinstance(static, dict):
        static["local_rad_cells"] = int(hybrid["local_rad_cells"])
        has_local_nodes = int(getattr(static.get("local_node_ids"), "size", 0)) > 0
        if (near_radius > 0.0 and (old_near_radius <= 0.0 or not has_local_nodes)) or near_radius <= 0.0:
            hybrid["gpu_static"] = None


def _compute_cext_local_only_hybrid_gpu(context: dict, ext_state: dict, hybrid: dict, *, use_lambda_cutoff: bool, runtime_state: dict | None = None) -> tuple[np.ndarray, dict[str, float]]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    hybrid["local_only_lean"] = True
    hybrid["local_only_fast_self"] = bool(CEXT_LOCAL_ONLY_FAST_SELF)
    _sync_cext_local_only_lambda_radius(hybrid, ext_state, use_lambda_cutoff=use_lambda_cutoff)
    if not use_lambda_cutoff and float(hybrid.get("near_radius_si", 0.0)) <= 0.0:
        raise ValueError(
            "cext_hybrid_bg_mode='local_only' requires --cext-hybrid-bg-near-radius-mult > 0. "
            "Use local_only_nlambda for the default r <= N*lambda screened pairwise solve."
        )
    state = runtime_state if isinstance(runtime_state, dict) else _ensure_cext_hybrid_bg_runtime_state(context, hybrid, ext_state)
    _sync_cext_hybrid_bg_runtime_state(ext_state, state)
    local_corr_g, local_cap_g, t_local, t_local_transfer = _compute_cext_local_corr_gpu(context, hybrid, ext_state, runtime_state=state)
    out_g = _cp.asarray(local_corr_g, dtype=_cp.float32)
    global_cap = float(
        np.nanmax(np.asarray(ext_state.get("seg_cap_gl", np.zeros((0,), dtype=np.float32)), dtype=float))
    ) if np.asarray(ext_state.get("seg_cap_gl", ())).size else 0.0
    if global_cap > 0.0:
        _cp.minimum(out_g, _cp.float32(global_cap), out=out_g)
    _cp.maximum(out_g, _cp.float32(0.0), out=out_g)
    out = _cp.asnumpy(out_g).astype(np.float64, copy=False)
    return out, {
        "deposit_s": 0.0,
        "fft_s": 0.0,
        "sample_s": 0.0,
        "local_corr_s": float(t_local),
        "local_transfer_s": float(t_local_transfer),
        "bg_solver_mode": "local_only_nlambda" if use_lambda_cutoff else "local_only",
    }

def _get_cext_gpu_local_corr_kernel():
    global _CEXT_GPU_LOCAL_CORR_KERNEL
    if _CEXT_GPU_LOCAL_CORR_KERNEL is not None:
        return _CEXT_GPU_LOCAL_CORR_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_local_corr_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* radii_si,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* seg_cap_gl,
    const int* exclude_idx,
    const unsigned char* exclude_count,
    const unsigned char* active_source_mask,
    const int* cell_ptr,
    const int* cell_node_ids,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_n,
    int rad_cells,
    float near_radius_si,
    int gl_order,
    int exclude_width,
    int target_n,
    float diffusivity_si,
    float* out_total,
    float* out_cap
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_n * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];

    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float target_radius = radii_si[target_seg];

    int cx = (int)floorf((tx - grid_origin_x) / grid_spacing);
    int cy = (int)floorf((ty - grid_origin_y) / grid_spacing);
    int cz = (int)floorf((tz - grid_origin_z) / grid_spacing);
    int lo_x = cx - rad_cells;
    int lo_y = cy - rad_cells;
    int lo_z = cz - rad_cells;
    int hi_x = cx + rad_cells;
    int hi_y = cy + rad_cells;
    int hi_z = cz + rad_cells;
    if (lo_x < 0) lo_x = 0;
    if (lo_y < 0) lo_y = 0;
    if (lo_z < 0) lo_z = 0;
    if (hi_x >= grid_n) hi_x = grid_n - 1;
    if (hi_y >= grid_n) hi_y = grid_n - 1;
    if (hi_z >= grid_n) hi_z = grid_n - 1;

    float total = 0.0f;
    float cap_max = 0.0f;
    int excl_n = (int)exclude_count[target_seg];
    for (int ix = lo_x; ix <= hi_x; ++ix) {
        for (int iy = lo_y; iy <= hi_y; ++iy) {
            for (int iz = lo_z; iz <= hi_z; ++iz) {
                int cell_flat = (ix * grid_n + iy) * grid_n + iz;
                int row_start = cell_ptr[cell_flat];
                int row_end = cell_ptr[cell_flat + 1];
                for (int pos = row_start; pos < row_end; ++pos) {
                    int source_node_id = cell_node_ids[pos];
                    int source_seg = source_node_id / gl_order;
                    int source_node = source_node_id - source_seg * gl_order;
                    if (active_source_mask[source_seg] == 0) continue;
                    int excluded = 0;
                    for (int ei = 0; ei < excl_n; ++ei) {
                        if (exclude_idx[target_seg * exclude_width + ei] == source_seg) {
                            excluded = 1;
                            break;
                        }
                    }
                    if (excluded) continue;
                    float radius_sum = target_radius + radii_si[source_seg];
                    float seg_cap = seg_cap_gl[source_seg];
                    float source_lambda = lambda_iv_gl[source_node_id];
                    if (source_lambda < 1.0e-30f) source_lambda = 1.0e-30f;
                    float sx = gl_points_si[source_node_id * 3 + 0];
                    float sy = gl_points_si[source_node_id * 3 + 1];
                    float sz = gl_points_si[source_node_id * 3 + 2];
                    float dx = sx - tx;
                    float dy = sy - ty;
                    float dz = sz - tz;
                    float r = sqrtf(dx * dx + dy * dy + dz * dz + radius_sum * radius_sum);
                    if (r > near_radius_si) continue;
                    if (r < 1.0e-30f) r = 1.0e-30f;
                    float kernel = expf(-r / source_lambda) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                    total += q_weighted_gl[source_node_id] * kernel;
                    if (seg_cap > cap_max) cap_max = seg_cap;
                }
            }
        }
    }
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out_total[target_idx * gl_order + target_node] = total;
    out_cap[target_idx * gl_order + target_node] = cap_max;
}
'''
    _CEXT_GPU_LOCAL_CORR_KERNEL = _cp.RawKernel(code, "cext_local_corr_kernel")
    return _CEXT_GPU_LOCAL_CORR_KERNEL


def _get_cext_hybrid_bg_kernel():
    global _CEXT_HYBRID_BG_KERNEL
    if _CEXT_HYBRID_BG_KERNEL is not None:
        return _CEXT_HYBRID_BG_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_hybrid_bg_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* phi_grids,
    const float* mass_grids,
    const float* lambda_bins,
    int n_bins,
    int grid_n,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    float near_radius_si,
    int gl_order,
    int target_n,
    float diffusivity_si,
    float* out_bg
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_n * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];
    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];

    float gx = (tx - origin_x) / spacing - 0.5f;
    float gy = (ty - origin_y) / spacing - 0.5f;
    float gz = (tz - origin_z) / spacing - 0.5f;
    int i0 = (int)floorf(gx);
    int j0 = (int)floorf(gy);
    int k0 = (int)floorf(gz);
    float fx = gx - i0;
    float fy = gy - j0;
    float fz = gz - k0;
    if (i0 < 0) { i0 = 0; fx = 0.0f; }
    if (j0 < 0) { j0 = 0; fy = 0.0f; }
    if (k0 < 0) { k0 = 0; fz = 0.0f; }
    if (i0 >= grid_n - 1) { i0 = grid_n - 2; fx = 1.0f; }
    if (j0 >= grid_n - 1) { j0 = grid_n - 2; fy = 1.0f; }
    if (k0 >= grid_n - 1) { k0 = grid_n - 2; fz = 1.0f; }

    int rad_cells = (int)ceilf(near_radius_si / spacing);
    float total_bg = 0.0f;
    float near_bg = 0.0f;
    for (int bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
        int base = bin_idx * grid_n * grid_n * grid_n;
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;
        int idx000 = base + ((i0 * grid_n + j0) * grid_n + k0);
        int idx001 = base + ((i0 * grid_n + j0) * grid_n + k1);
        int idx010 = base + ((i0 * grid_n + j1) * grid_n + k0);
        int idx011 = base + ((i0 * grid_n + j1) * grid_n + k1);
        int idx100 = base + ((i1 * grid_n + j0) * grid_n + k0);
        int idx101 = base + ((i1 * grid_n + j0) * grid_n + k1);
        int idx110 = base + ((i1 * grid_n + j1) * grid_n + k0);
        int idx111 = base + ((i1 * grid_n + j1) * grid_n + k1);
        float c00 = phi_grids[idx000] * (1.0f - fx) + phi_grids[idx100] * fx;
        float c01 = phi_grids[idx001] * (1.0f - fx) + phi_grids[idx101] * fx;
        float c10 = phi_grids[idx010] * (1.0f - fx) + phi_grids[idx110] * fx;
        float c11 = phi_grids[idx011] * (1.0f - fx) + phi_grids[idx111] * fx;
        float c0 = c00 * (1.0f - fy) + c10 * fy;
        float c1 = c01 * (1.0f - fy) + c11 * fy;
        total_bg += c0 * (1.0f - fz) + c1 * fz;

        float lambda_bin = lambda_bins[bin_idx];
        if (lambda_bin < 1.0e-30f) lambda_bin = 1.0e-30f;
        int ilo = i0 - rad_cells;
        int jlo = j0 - rad_cells;
        int klo = k0 - rad_cells;
        int ihi = i0 + rad_cells + 1;
        int jhi = j0 + rad_cells + 1;
        int khi = k0 + rad_cells + 1;
        if (ilo < 0) ilo = 0;
        if (jlo < 0) jlo = 0;
        if (klo < 0) klo = 0;
        if (ihi >= grid_n) ihi = grid_n - 1;
        if (jhi >= grid_n) jhi = grid_n - 1;
        if (khi >= grid_n) khi = grid_n - 1;
        for (int ix = ilo; ix <= ihi; ++ix) {
            float cx = origin_x + (ix + 0.5f) * spacing;
            float dx = cx - tx;
            for (int iy = jlo; iy <= jhi; ++iy) {
                float cy = origin_y + (iy + 0.5f) * spacing;
                float dy = cy - ty;
                for (int iz = klo; iz <= khi; ++iz) {
                    float cz = origin_z + (iz + 0.5f) * spacing;
                    float dz = cz - tz;
                    float r = sqrtf(dx * dx + dy * dy + dz * dz);
                    if (r > near_radius_si || r < 1.0e-30f) continue;
                    float mass = mass_grids[base + ((ix * grid_n + iy) * grid_n + iz)];
                    if (!(mass != 0.0f)) continue;
                    near_bg += mass * expf(-r / lambda_bin) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                }
            }
        }
    }
    float total = total_bg - near_bg;
    if (!(total == total) || !isfinite(total)) total = 0.0f;
    out_bg[target_idx * gl_order + target_node] = total;
}
'''
    _CEXT_HYBRID_BG_KERNEL = _cp.RawKernel(code, "cext_hybrid_bg_kernel")
    return _CEXT_HYBRID_BG_KERNEL


def _compute_cext_local_corr_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    *,
    source_mask: np.ndarray | None = None,
    source_plan: dict | None = None,
    target_plan: dict | None = None,
    runtime_state: dict | None = None,
) -> tuple[object, object, float, float]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    state = runtime_state if isinstance(runtime_state, dict) else _ensure_cext_hybrid_bg_runtime_state(context, hybrid, ext_state)
    hybrid_static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    if (
        bool(hybrid.get("local_only_fast_self", False))
        and source_mask is None
        and source_plan is None
        and target_plan is None
    ):
        return _compute_cext_local_only_fast_self_gpu(context, hybrid, ext_state, runtime_state=state)
    kernel = _get_cext_gpu_local_corr_kernel()
    source_cfg = source_plan if isinstance(source_plan, dict) else _build_cext_hybrid_local_source_plan(hybrid, source_mask)
    active_count = int(source_cfg.get("active_count", nseg))
    target_cfg = target_plan if isinstance(target_plan, dict) else None
    target_count = int(target_cfg.get("target_count", nseg)) if target_cfg is not None else nseg
    if active_count <= 0 or target_count <= 0 or float(hybrid.get("near_radius_si", 0.0)) <= 0.0:
        zeros = _cp.zeros((target_count, gl_order), dtype=_cp.float32)
        return zeros, zeros.copy(), 0.0, 0.0

    t_transfer = perf_counter()
    c_ext_total_g = state["local_total_g"][:target_count]
    c_ext_cap_g = state["local_cap_g"][:target_count]
    if bool(source_cfg.get("uses_static", False)):
        cell_ptr_g = hybrid_static["local_cell_ptr"]
        cell_node_ids_g = hybrid_static["local_node_ids"]
    else:
        if "gpu_cell_ptr" not in source_cfg:
            source_cfg["gpu_cell_ptr"] = _cp.asarray(np.asarray(source_cfg.get("cell_ptr", np.zeros((1,), dtype=np.int32)), dtype=np.int32))
            source_cfg["gpu_node_ids"] = _cp.asarray(np.asarray(source_cfg.get("node_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32))
        cell_ptr_g = source_cfg["gpu_cell_ptr"]
        cell_node_ids_g = source_cfg["gpu_node_ids"]
    if target_cfg is None or bool(target_cfg.get("uses_static", False)):
        target_seg_ids_g = static["all_target_seg_ids"]
    else:
        if "gpu_target_seg_ids" not in target_cfg:
            target_cfg["gpu_target_seg_ids"] = _cp.asarray(np.asarray(target_cfg.get("target_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32))
        target_seg_ids_g = target_cfg["gpu_target_seg_ids"]
    _cp.cuda.Stream.null.synchronize()
    transfer_total = perf_counter() - t_transfer

    origin = np.asarray(hybrid["origin"], dtype=np.float32)
    threads = 128
    blocks = (target_count * gl_order + threads - 1) // threads
    t_kernel = perf_counter()
    kernel(
        (blocks,),
        (threads,),
        (
            target_seg_ids_g,
            static["gl_points_si"].ravel(),
            static["radii_si"],
            state["lambda_iv_gl"].ravel(),
            state["q_weighted_gl"].ravel(),
            state["seg_cap_gl"],
            static["exclude_idx"].ravel(),
            static["exclude_count"],
            state["active_source_mask_g"],
            cell_ptr_g,
            cell_node_ids_g,
            np.float32(float(origin[0])),
            np.float32(float(origin[1])),
            np.float32(float(origin[2])),
            np.float32(float(hybrid["spacing"])),
            np.int32(int(hybrid["grid_n"])),
            np.int32(int(hybrid_static["local_rad_cells"])),
            np.float32(float(hybrid["near_radius_si"])),
            np.int32(gl_order),
            np.int32(int(np.asarray(context["exclude_idx"]).shape[1])),
            np.int32(target_count),
            np.float32(float(context["diffusivity_si"])),
            c_ext_total_g.ravel(),
            c_ext_cap_g.ravel(),
        ),
    )
    _cp.cuda.Stream.null.synchronize()
    return c_ext_total_g, c_ext_cap_g, perf_counter() - t_kernel, transfer_total


def _get_cext_fft_moment_deposit_kernel():
    global _CEXT_FFT_MOMENT_DEPOSIT_KERNEL
    if _CEXT_FFT_MOMENT_DEPOSIT_KERNEL is not None:
        return _CEXT_FFT_MOMENT_DEPOSIT_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_fft_moment_deposit_kernel(
    const float* moment_weight_gl,
    const float* lambda_iv_gl,
    const int* node_seg_ids,
    const unsigned char* active_source_mask,
    const int* stencil_flat_idx,
    const float* stencil_weight,
    const float* lambda_bin_edges,
    int n_bins,
    int grid_cells,
    int stencil_n,
    int n_nodes,
    float* out_mass
) {
    int node_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_idx >= n_nodes) return;
    int seg_idx = node_seg_ids[node_idx];
    if (active_source_mask[seg_idx] == 0) return;
    float w0 = moment_weight_gl[node_idx];
    float lam = lambda_iv_gl[node_idx];
    if (!isfinite(w0) || !isfinite(lam) || w0 == 0.0f || lam <= 0.0f) return;
    int bin_idx = 0;
    for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
        if (lam > lambda_bin_edges[edge_idx]) {
            bin_idx = edge_idx;
        } else {
            break;
        }
    }
    int base = bin_idx * grid_cells;
    int stencil_base = node_idx * stencil_n;
    for (int slot = 0; slot < stencil_n; ++slot) {
        int flat = stencil_flat_idx[stencil_base + slot];
        if (flat < 0) continue;
        float sw = stencil_weight[stencil_base + slot];
        if (!(sw != 0.0f) || !isfinite(sw)) continue;
        atomicAdd(&out_mass[base + flat], w0 * sw);
    }
}
'''
    _CEXT_FFT_MOMENT_DEPOSIT_KERNEL = _cp.RawKernel(code, "cext_fft_moment_deposit_kernel")
    return _CEXT_FFT_MOMENT_DEPOSIT_KERNEL


def _get_cext_fft_moment_deposit_batch_kernel():
    global _CEXT_FFT_MOMENT_DEPOSIT_BATCH_KERNEL
    if _CEXT_FFT_MOMENT_DEPOSIT_BATCH_KERNEL is not None:
        return _CEXT_FFT_MOMENT_DEPOSIT_BATCH_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_fft_moment_deposit_batch_kernel(
    const float* o2_weight_gl,
    const float* node_tx,
    const float* node_ty,
    const float* node_tz,
    const float* lambda_iv_gl,
    const int* node_seg_ids,
    const unsigned char* active_source_mask,
    const int* stencil_flat_idx,
    const float* stencil_weight,
    const float* lambda_bin_edges,
    int moment_start,
    int moment_count,
    int n_bins,
    int grid_cells,
    int stencil_n,
    int n_nodes,
    float* out_mass
) {
    int node_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_idx >= n_nodes) return;
    int seg_idx = node_seg_ids[node_idx];
    if (active_source_mask[seg_idx] == 0) return;
    float w0 = o2_weight_gl[node_idx];
    float lam = lambda_iv_gl[node_idx];
    float tx = node_tx[node_idx];
    float ty = node_ty[node_idx];
    float tz = node_tz[node_idx];
    if (!isfinite(w0) || !isfinite(lam) || !isfinite(tx) || !isfinite(ty) || !isfinite(tz) || w0 == 0.0f || lam <= 0.0f) return;

    float coeffs[7];
    coeffs[0] = 1.0f;
    coeffs[1] = tx * tx;
    coeffs[2] = ty * ty;
    coeffs[3] = tz * tz;
    coeffs[4] = tx * ty;
    coeffs[5] = tx * tz;
    coeffs[6] = ty * tz;

    int bin_idx = 0;
    for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
        if (lam > lambda_bin_edges[edge_idx]) {
            bin_idx = edge_idx;
        } else {
            break;
        }
    }
    int stencil_base = node_idx * stencil_n;
    for (int slot = 0; slot < stencil_n; ++slot) {
        int flat = stencil_flat_idx[stencil_base + slot];
        if (flat < 0) continue;
        float sw = stencil_weight[stencil_base + slot];
        if (!(sw != 0.0f) || !isfinite(sw)) continue;
        for (int local_idx = 0; local_idx < moment_count; ++local_idx) {
            int moment_idx = moment_start + local_idx;
            if (moment_idx < 0 || moment_idx >= 7) continue;
            int out_base = (local_idx * n_bins + bin_idx) * grid_cells;
            atomicAdd(&out_mass[out_base + flat], w0 * coeffs[moment_idx] * sw);
        }
    }
}
'''
    _CEXT_FFT_MOMENT_DEPOSIT_BATCH_KERNEL = _cp.RawKernel(code, "cext_fft_moment_deposit_batch_kernel")
    return _CEXT_FFT_MOMENT_DEPOSIT_BATCH_KERNEL


def _cext_fft_deposit_moment_grid_gpu(
    context: dict,
    hybrid: dict,
    state: dict,
    moment_weight_g,
    edges: np.ndarray,
    *,
    synchronize: bool = True,
):
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    edge_arr = np.asarray(edges, dtype=np.float32)
    if "lambda_bin_edges" not in static or int(static["lambda_bin_edges"].size) != int(edge_arr.size):
        static["lambda_bin_edges"] = _cp.asarray(edge_arr)
    n_bins = max(int(np.asarray(edges).size - 1), 1)
    grid_shape = tuple(state["active_mass_grids_g"].shape)
    scratch = state.get("fft_o2_moment_mass_g")
    if scratch is None or tuple(scratch.shape) != grid_shape:
        scratch = _cp.zeros(grid_shape, dtype=_cp.float32)
        state["fft_o2_moment_mass_g"] = scratch
    else:
        scratch.fill(_cp.float32(0.0))
    kernel = _get_cext_fft_moment_deposit_kernel()
    n_nodes = int(moment_weight_g.size)
    if n_nodes <= 0:
        return scratch
    threads = 256
    blocks = (n_nodes + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            _cp.asarray(moment_weight_g, dtype=_cp.float32).ravel(),
            state["lambda_iv_gl"].ravel(),
            static["node_seg_ids"],
            state["active_source_mask_g"],
            static["stencil_flat_idx"].ravel(),
            static["stencil_weight"].ravel(),
            static["lambda_bin_edges"],
            np.int32(n_bins),
            np.int32(int(static["grid_cells"])),
            np.int32(int(static["stencil_n"])),
            np.int32(n_nodes),
            scratch.ravel(),
        ),
    )
    if synchronize:
        _cp.cuda.Stream.null.synchronize()
    return scratch


def _cext_fft_deposit_moment_batch_gpu(
    context: dict,
    hybrid: dict,
    state: dict,
    o2_node_g,
    tx_g,
    ty_g,
    tz_g,
    edges: np.ndarray,
    *,
    moment_start: int,
    moment_count: int,
    synchronize: bool = True,
):
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    edge_arr = np.asarray(edges, dtype=np.float32)
    if "lambda_bin_edges" not in static or int(static["lambda_bin_edges"].size) != int(edge_arr.size):
        static["lambda_bin_edges"] = _cp.asarray(edge_arr)
    n_bins = max(int(np.asarray(edges).size - 1), 1)
    base_shape = tuple(state["active_mass_grids_g"].shape)
    local_count = max(min(int(moment_count), 7 - int(moment_start)), 0)
    if local_count <= 0:
        return _cp.zeros((0,) + base_shape, dtype=_cp.float32)
    batch_shape = (local_count,) + base_shape
    scratch = state.get("fft_o2_moment_mass_batch_g")
    if scratch is None or tuple(scratch.shape) != batch_shape:
        scratch = _cp.zeros(batch_shape, dtype=_cp.float32)
        state["fft_o2_moment_mass_batch_g"] = scratch
    else:
        scratch.fill(_cp.float32(0.0))
    kernel = _get_cext_fft_moment_deposit_batch_kernel()
    n_nodes = int(_cp.asarray(o2_node_g).size)
    if n_nodes <= 0:
        return scratch
    threads = 256
    blocks = (n_nodes + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            _cp.asarray(o2_node_g, dtype=_cp.float32).ravel(),
            _cp.asarray(tx_g, dtype=_cp.float32).ravel(),
            _cp.asarray(ty_g, dtype=_cp.float32).ravel(),
            _cp.asarray(tz_g, dtype=_cp.float32).ravel(),
            state["lambda_iv_gl"].ravel(),
            static["node_seg_ids"],
            state["active_source_mask_g"],
            static["stencil_flat_idx"].ravel(),
            static["stencil_weight"].ravel(),
            static["lambda_bin_edges"],
            np.int32(int(moment_start)),
            np.int32(int(local_count)),
            np.int32(n_bins),
            np.int32(int(static["grid_cells"])),
            np.int32(int(static["stencil_n"])),
            np.int32(n_nodes),
            scratch.ravel(),
        ),
    )
    if synchronize:
        _cp.cuda.Stream.null.synchronize()
    return scratch


def _cext_fft_o2_term_correction_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    state: dict,
    *,
    target_plan: dict | None = None,
) -> tuple[object | None, float]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    if bool(CEXT_HYBRID_GPU_RUNTIME_WEIGHTS) and "mono2_weight_gl" in state and "dipole2_weight_gl" in state:
        o2_node = (state["mono2_weight_gl"].ravel() + state["dipole2_weight_gl"].ravel()).astype(_cp.float32, copy=False)
        if not bool(_cp.any(_cp.isfinite(o2_node) & (o2_node != _cp.float32(0.0))).item()):
            return None, 0.0
    else:
        mono = np.asarray(ext_state.get("mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32)
        dipole = np.asarray(ext_state.get("dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32)
        o2_weight = mono + dipole
        if not np.any(np.isfinite(o2_weight) & (o2_weight != 0.0)):
            return None, 0.0
        o2_node = _cp.asarray(o2_weight, dtype=_cp.float32).ravel()

    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gpu_static = _ensure_cext_gpu_static(context)
    grid_n = int(hybrid["grid_n"])
    spacing = float(hybrid["spacing"])
    diffusivity_si = float(context["diffusivity_si"])
    rhs_scale = np.float32(max((spacing ** 3) * diffusivity_si, 1.0e-30))
    edges, lambda_centers = _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    lambda_centers = np.asarray(lambda_centers, dtype=np.float32)
    static["lambda_bin_centers"] = _cp.asarray(lambda_centers, dtype=_cp.float32)

    kfreq = _cp.asarray(2.0 * np.pi * np.fft.fftfreq(grid_n, d=spacing), dtype=_cp.float32)
    kx = kfreq[:, None, None]
    ky = kfreq[None, :, None]
    kz = kfreq[None, None, :]
    symbols = (
        -static["k2"],
        kx * kx,
        ky * ky,
        kz * kz,
        _cp.float32(2.0) * kx * ky,
        _cp.float32(2.0) * kx * kz,
        _cp.float32(2.0) * ky * kz,
    )

    seg_vec = _cp.asarray(gpu_static["segment_vectors"], dtype=_cp.float32)
    seg_len = _cp.sqrt(_cp.sum(seg_vec * seg_vec, axis=1))
    seg_len = _cp.maximum(seg_len, _cp.float32(1.0e-30))
    t_hat = seg_vec / seg_len[:, None]
    node_seg_ids = static["node_seg_ids"]
    tx = t_hat[node_seg_ids, 0]
    ty = t_hat[node_seg_ids, 1]
    tz = t_hat[node_seg_ids, 2]
    moment_weights = (
        o2_node,
        o2_node * tx * tx,
        o2_node * ty * ty,
        o2_node * tz * tz,
        o2_node * tx * ty,
        o2_node * tx * tz,
        o2_node * ty * tz,
    )

    phi_corr = state.get("fft_o2_phi_corr_g")
    if phi_corr is None or tuple(phi_corr.shape) != tuple(state["active_mass_grids_g"].shape):
        phi_corr = _cp.zeros_like(state["active_mass_grids_g"])
        state["fft_o2_phi_corr_g"] = phi_corr
    else:
        phi_corr.fill(_cp.float32(0.0))

    lambda_centers_g = static["lambda_bin_centers"]
    n_bins = int(phi_corr.shape[0])
    t0 = perf_counter()
    try:
        batched_limit = int(os.environ.get("SVV_FFT_O2_BATCHED_MAX_BYTES", str(1024**3)))
    except ValueError:
        batched_limit = 3 * 1024**3
    use_batched = int(phi_corr.size) * 8 <= max(batched_limit, 0)
    use_fused_ifft = bool(CEXT_HYBRID_FFT_O2_FUSED_IFFT) and use_batched
    if use_fused_ifft:
        try:
            moment_batch = max(int(CEXT_HYBRID_FFT_O2_MOMENT_BATCH), 1)
            lam = _cp.maximum(lambda_centers_g, _cp.float32(1.0e-8)).reshape((n_bins, 1, 1, 1))
            denom_inv = _cp.reciprocal(static["k2"][None, :, :, :] + _cp.reciprocal(lam * lam))
            corr_hat_sum = _cp.zeros(phi_corr.shape, dtype=_cp.complex64)
            if moment_batch > 1:
                # Batch adjacent O(a^2) moment deposits so the stencil walk and
                # lambda-bin lookup are shared across multiple moment fields.
                for moment_start in range(0, len(symbols), moment_batch):
                    local_count = min(moment_batch, len(symbols) - moment_start)
                    moment_grid = _cext_fft_deposit_moment_batch_gpu(
                        context,
                        hybrid,
                        state,
                        o2_node,
                        tx,
                        ty,
                        tz,
                        np.asarray(edges, dtype=np.float32),
                        moment_start=moment_start,
                        moment_count=local_count,
                        synchronize=False,
                    )
                    rhs_hat = _cp.fft.fftn(moment_grid, axes=(2, 3, 4))
                    rhs_hat *= _cp.float32(1.0 / float(rhs_scale))
                    for local_idx in range(local_count):
                        symbol = symbols[moment_start + local_idx]
                        corr_hat_sum += rhs_hat[local_idx] * _cp.asarray(symbol, dtype=_cp.float32)[None, :, :, :] * denom_inv
                    del rhs_hat
            else:
                for moment_weight, symbol in zip(moment_weights, symbols):
                    moment_grid = _cext_fft_deposit_moment_grid_gpu(
                        context,
                        hybrid,
                        state,
                        moment_weight,
                        np.asarray(edges, dtype=np.float32),
                        synchronize=False,
                    )
                    rhs_hat = _cp.fft.fftn(moment_grid, axes=(1, 2, 3))
                    rhs_hat *= _cp.float32(1.0 / float(rhs_scale))
                    corr_hat_sum += rhs_hat * _cp.asarray(symbol, dtype=_cp.float32)[None, :, :, :] * denom_inv
                    del rhs_hat
            # The O(a^2) correction is linear in the seven second-order moment
            # fields, so summing their Fourier coefficients before the inverse
            # transform is algebraically equivalent to seven separate IFFTs.
            phi_corr += _cp.real(_cp.fft.ifftn(corr_hat_sum, axes=(1, 2, 3))).astype(_cp.float32)
            del lam, denom_inv, corr_hat_sum
        except Exception:
            if str(os.environ.get("SVV_CEXT_HYBRID_FFT_O2_FUSED_STRICT", "false")).strip().lower() in ("1", "true", "yes", "on"):
                raise
            use_fused_ifft = False
    if not use_fused_ifft:
        for moment_weight, symbol in zip(moment_weights, symbols):
            moment_grid = _cext_fft_deposit_moment_grid_gpu(context, hybrid, state, moment_weight, np.asarray(edges, dtype=np.float32))
            if use_batched:
                rhs_hat = _cp.fft.fftn(moment_grid, axes=(1, 2, 3))
                rhs_hat *= _cp.float32(1.0 / float(rhs_scale))
                for bin_idx in range(n_bins):
                    lam = _cp.maximum(lambda_centers_g[bin_idx], _cp.float32(1.0e-8))
                    denom = static["k2"] + _cp.reciprocal(lam * lam)
                    rhs_hat[bin_idx] *= symbol
                    rhs_hat[bin_idx] /= denom
                    del denom
                phi_corr += _cp.real(_cp.fft.ifftn(rhs_hat, axes=(1, 2, 3))).astype(_cp.float32)
                del rhs_hat
            else:
                for bin_idx in range(n_bins):
                    rhs = moment_grid[bin_idx] / rhs_scale
                    rhs_hat = _cp.fft.fftn(rhs, axes=(0, 1, 2))
                    lam = _cp.maximum(lambda_centers_g[bin_idx], _cp.float32(1.0e-8))
                    denom = static["k2"] + _cp.reciprocal(lam * lam)
                    corr_hat = symbol * rhs_hat / denom
                    phi_corr[bin_idx] += _cp.real(_cp.fft.ifftn(corr_hat, axes=(0, 1, 2))).astype(_cp.float32)
                    del rhs, rhs_hat, denom, corr_hat

    zero_mass = state.get("fft_o2_zero_mass_g")
    if zero_mass is None or tuple(zero_mass.shape) != tuple(phi_corr.shape):
        zero_mass = _cp.zeros_like(phi_corr)
        state["fft_o2_zero_mass_g"] = zero_mass
    corr_g, _ = _sample_cext_hybrid_bg_gpu(
        context,
        hybrid,
        phi_corr,
        zero_mass,
        target_plan=target_plan,
        runtime_state=state,
    )
    _cp.cuda.Stream.null.synchronize()
    return corr_g, perf_counter() - t0


def _get_cext_fft_discrete_self_kernel():
    global _CEXT_FFT_DISCRETE_SELF_KERNEL
    if _CEXT_FFT_DISCRETE_SELF_KERNEL is not None:
        return _CEXT_FFT_DISCRETE_SELF_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_fft_discrete_self_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* source_weight_gl,
    const float* lambda_iv_gl,
    const unsigned char* active_source_mask,
    const int* source_stencil_flat_idx,
    const float* source_stencil_weight,
    const float* lambda_bin_edges,
    const float* green_grids,
    int n_bins,
    int grid_n,
    int grid_cells,
    int source_stencil_n,
    int gl_order,
    int target_count,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    int assignment_mode,
    float* out_self
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_count * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];
    if (active_source_mask[target_seg] == 0) return;

    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];

    float gx = (tx - origin_x) / spacing - 0.5f;
    float gy = (ty - origin_y) / spacing - 0.5f;
    float gz = (tz - origin_z) / spacing - 0.5f;
    int ti[3], tj[3], tk[3];
    float wi[3], wj[3], wk[3];
    int target_stencil_n = 2;
    if (assignment_mode == 1) {
        int ci = (int)floorf(gx + 0.5f);
        int cj = (int)floorf(gy + 0.5f);
        int ck = (int)floorf(gz + 0.5f);
        target_stencil_n = 3;
        for (int slot = 0; slot < 3; ++slot) {
            int off = slot - 1;
            int ix = ci + off;
            int iy = cj + off;
            int iz = ck + off;
            float dx = fabsf(gx - (float)ix);
            float dy = fabsf(gy - (float)iy);
            float dz = fabsf(gz - (float)iz);
            ti[slot] = ix;
            tj[slot] = iy;
            tk[slot] = iz;
            wi[slot] = dx < 0.5f ? 0.75f - dx * dx : (dx < 1.5f ? 0.5f * (1.5f - dx) * (1.5f - dx) : 0.0f);
            wj[slot] = dy < 0.5f ? 0.75f - dy * dy : (dy < 1.5f ? 0.5f * (1.5f - dy) * (1.5f - dy) : 0.0f);
            wk[slot] = dz < 0.5f ? 0.75f - dz * dz : (dz < 1.5f ? 0.5f * (1.5f - dz) * (1.5f - dz) : 0.0f);
        }
    } else {
        int i0 = (int)floorf(gx);
        int j0 = (int)floorf(gy);
        int k0 = (int)floorf(gz);
        float fx = gx - i0;
        float fy = gy - j0;
        float fz = gz - k0;
        if (i0 < 0) { i0 = 0; fx = 0.0f; }
        if (j0 < 0) { j0 = 0; fy = 0.0f; }
        if (k0 < 0) { k0 = 0; fz = 0.0f; }
        if (i0 >= grid_n - 1) { i0 = grid_n - 2; fx = 1.0f; }
        if (j0 >= grid_n - 1) { j0 = grid_n - 2; fy = 1.0f; }
        if (k0 >= grid_n - 1) { k0 = grid_n - 2; fz = 1.0f; }
        ti[0] = i0; ti[1] = i0 + 1; ti[2] = -1;
        tj[0] = j0; tj[1] = j0 + 1; tj[2] = -1;
        tk[0] = k0; tk[1] = k0 + 1; tk[2] = -1;
        wi[0] = 1.0f - fx; wi[1] = fx; wi[2] = 0.0f;
        wj[0] = 1.0f - fy; wj[1] = fy; wj[2] = 0.0f;
        wk[0] = 1.0f - fz; wk[1] = fz; wk[2] = 0.0f;
    }

    float acc = 0.0f;
    for (int source_node = 0; source_node < gl_order; ++source_node) {
        int source_flat = target_seg * gl_order + source_node;
        float source_weight = source_weight_gl[source_flat];
        float lam = lambda_iv_gl[source_flat];
        if (!isfinite(source_weight) || source_weight == 0.0f || !isfinite(lam) || lam <= 0.0f) continue;
        int bin_idx = 0;
        for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
            if (lam > lambda_bin_edges[edge_idx]) {
                bin_idx = edge_idx;
            } else {
                break;
            }
        }
        int green_base = bin_idx * grid_cells;
        int stencil_base = source_flat * source_stencil_n;
        for (int source_slot = 0; source_slot < source_stencil_n; ++source_slot) {
            int source_cell = source_stencil_flat_idx[stencil_base + source_slot];
            if (source_cell < 0) continue;
            float source_stencil = source_stencil_weight[stencil_base + source_slot];
            if (!(source_stencil != 0.0f) || !isfinite(source_stencil)) continue;
            int sx = source_cell / (grid_n * grid_n);
            int rem = source_cell - sx * grid_n * grid_n;
            int sy = rem / grid_n;
            int sz = rem - sy * grid_n;
            float deposited = source_weight * source_stencil;
            for (int ax = 0; ax < target_stencil_n; ++ax) {
                if (ti[ax] < 0 || ti[ax] >= grid_n || !(wi[ax] != 0.0f) || !isfinite(wi[ax])) continue;
                int dx = ti[ax] - sx;
                if (dx < 0) dx += grid_n;
                float wx = wi[ax];
                for (int ay = 0; ay < target_stencil_n; ++ay) {
                    if (tj[ay] < 0 || tj[ay] >= grid_n || !(wj[ay] != 0.0f) || !isfinite(wj[ay])) continue;
                    int dy = tj[ay] - sy;
                    if (dy < 0) dy += grid_n;
                    float wxy = wx * wj[ay];
                    for (int az = 0; az < target_stencil_n; ++az) {
                        if (tk[az] < 0 || tk[az] >= grid_n || !(wk[az] != 0.0f) || !isfinite(wk[az])) continue;
                        int dz = tk[az] - sz;
                        if (dz < 0) dz += grid_n;
                        float target_weight = wxy * wk[az];
                        int green_idx = green_base + ((dx * grid_n + dy) * grid_n + dz);
                        acc += deposited * target_weight * green_grids[green_idx];
                    }
                }
            }
        }
    }
    if (isfinite(acc)) {
        out_self[flat] += acc;
    }
}
'''
    _CEXT_FFT_DISCRETE_SELF_KERNEL = _cp.RawKernel(code, "cext_fft_discrete_self_kernel")
    return _CEXT_FFT_DISCRETE_SELF_KERNEL


def _get_cext_fft_discrete_self_fused_o2_kernel():
    global _CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL
    if _CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL is not None:
        return _CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_fft_discrete_self_fused_o2_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* q_weighted_gl,
    const float* o2_weight_gl,
    const float* node_tx,
    const float* node_ty,
    const float* node_tz,
    const float* lambda_iv_gl,
    const unsigned char* active_source_mask,
    const int* source_stencil_flat_idx,
    const float* source_stencil_weight,
    const float* lambda_bin_edges,
    const float* base_response,
    const float* finite0,
    const float* finite1,
    const float* finite2,
    const float* finite3,
    const float* finite4,
    const float* finite5,
    const float* finite6,
    int n_bins,
    int grid_n,
    int grid_cells,
    int source_stencil_n,
    int gl_order,
    int target_count,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    int assignment_mode,
    float* out_self
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_count * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];
    if (active_source_mask[target_seg] == 0) return;

    float tx0 = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty0 = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz0 = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];

    float gx = (tx0 - origin_x) / spacing - 0.5f;
    float gy = (ty0 - origin_y) / spacing - 0.5f;
    float gz = (tz0 - origin_z) / spacing - 0.5f;
    int ti[3], tj[3], tk[3];
    float wi[3], wj[3], wk[3];
    int target_stencil_n = 2;
    if (assignment_mode == 1) {
        int ci = (int)floorf(gx + 0.5f);
        int cj = (int)floorf(gy + 0.5f);
        int ck = (int)floorf(gz + 0.5f);
        target_stencil_n = 3;
        for (int slot = 0; slot < 3; ++slot) {
            int off = slot - 1;
            int ix = ci + off;
            int iy = cj + off;
            int iz = ck + off;
            float dx = fabsf(gx - (float)ix);
            float dy = fabsf(gy - (float)iy);
            float dz = fabsf(gz - (float)iz);
            ti[slot] = ix;
            tj[slot] = iy;
            tk[slot] = iz;
            wi[slot] = dx < 0.5f ? 0.75f - dx * dx : (dx < 1.5f ? 0.5f * (1.5f - dx) * (1.5f - dx) : 0.0f);
            wj[slot] = dy < 0.5f ? 0.75f - dy * dy : (dy < 1.5f ? 0.5f * (1.5f - dy) * (1.5f - dy) : 0.0f);
            wk[slot] = dz < 0.5f ? 0.75f - dz * dz : (dz < 1.5f ? 0.5f * (1.5f - dz) * (1.5f - dz) : 0.0f);
        }
    } else {
        int i0 = (int)floorf(gx);
        int j0 = (int)floorf(gy);
        int k0 = (int)floorf(gz);
        float fx = gx - i0;
        float fy = gy - j0;
        float fz = gz - k0;
        if (i0 < 0) { i0 = 0; fx = 0.0f; }
        if (j0 < 0) { j0 = 0; fy = 0.0f; }
        if (k0 < 0) { k0 = 0; fz = 0.0f; }
        if (i0 >= grid_n - 1) { i0 = grid_n - 2; fx = 1.0f; }
        if (j0 >= grid_n - 1) { j0 = grid_n - 2; fy = 1.0f; }
        if (k0 >= grid_n - 1) { k0 = grid_n - 2; fz = 1.0f; }
        ti[0] = i0; ti[1] = i0 + 1; ti[2] = -1;
        tj[0] = j0; tj[1] = j0 + 1; tj[2] = -1;
        tk[0] = k0; tk[1] = k0 + 1; tk[2] = -1;
        wi[0] = 1.0f - fx; wi[1] = fx; wi[2] = 0.0f;
        wj[0] = 1.0f - fy; wj[1] = fy; wj[2] = 0.0f;
        wk[0] = 1.0f - fz; wk[1] = fz; wk[2] = 0.0f;
    }

    float acc = 0.0f;
    for (int source_node = 0; source_node < gl_order; ++source_node) {
        int source_flat = target_seg * gl_order + source_node;
        float q_weight = q_weighted_gl[source_flat];
        float o2_weight = o2_weight_gl[source_flat];
        bool use_q = isfinite(q_weight) && q_weight != 0.0f;
        bool use_o2 = isfinite(o2_weight) && o2_weight != 0.0f;
        float ux = node_tx[source_flat];
        float uy = node_ty[source_flat];
        float uz = node_tz[source_flat];
        use_o2 = use_o2 && isfinite(ux) && isfinite(uy) && isfinite(uz);
        if (!use_q && !use_o2) continue;
        float lam = lambda_iv_gl[source_flat];
        if (!isfinite(lam) || lam <= 0.0f) continue;
        int bin_idx = 0;
        for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
            if (lam > lambda_bin_edges[edge_idx]) {
                bin_idx = edge_idx;
            } else {
                break;
            }
        }
        int green_base = bin_idx * grid_cells;
        int stencil_base = source_flat * source_stencil_n;
        for (int source_slot = 0; source_slot < source_stencil_n; ++source_slot) {
            int source_cell = source_stencil_flat_idx[stencil_base + source_slot];
            if (source_cell < 0) continue;
            float source_stencil = source_stencil_weight[stencil_base + source_slot];
            if (!(source_stencil != 0.0f) || !isfinite(source_stencil)) continue;
            int sx = source_cell / (grid_n * grid_n);
            int rem = source_cell - sx * grid_n * grid_n;
            int sy = rem / grid_n;
            int sz = rem - sy * grid_n;
            for (int ax = 0; ax < target_stencil_n; ++ax) {
                if (ti[ax] < 0 || ti[ax] >= grid_n || !(wi[ax] != 0.0f) || !isfinite(wi[ax])) continue;
                int dx = ti[ax] - sx;
                if (dx < 0) dx += grid_n;
                float wx = wi[ax];
                for (int ay = 0; ay < target_stencil_n; ++ay) {
                    if (tj[ay] < 0 || tj[ay] >= grid_n || !(wj[ay] != 0.0f) || !isfinite(wj[ay])) continue;
                    int dy = tj[ay] - sy;
                    if (dy < 0) dy += grid_n;
                    float wxy = wx * wj[ay];
                    for (int az = 0; az < target_stencil_n; ++az) {
                        if (tk[az] < 0 || tk[az] >= grid_n || !(wk[az] != 0.0f) || !isfinite(wk[az])) continue;
                        int dz = tk[az] - sz;
                        if (dz < 0) dz += grid_n;
                        float target_weight = wxy * wk[az];
                        int green_idx = green_base + ((dx * grid_n + dy) * grid_n + dz);
                        float response = 0.0f;
                        if (use_q) {
                            response += q_weight * base_response[green_idx];
                        }
                        if (use_o2) {
                            response += o2_weight * (
                                finite0[green_idx]
                                + ux * ux * finite1[green_idx]
                                + uy * uy * finite2[green_idx]
                                + uz * uz * finite3[green_idx]
                                + ux * uy * finite4[green_idx]
                                + ux * uz * finite5[green_idx]
                                + uy * uz * finite6[green_idx]
                            );
                        }
                        acc += source_stencil * target_weight * response;
                    }
                }
            }
        }
    }
    if (isfinite(acc)) {
        out_self[flat] += acc;
    }
}
'''
    _CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL = _cp.RawKernel(code, "cext_fft_discrete_self_fused_o2_kernel")
    return _CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL


def _cext_fft_response_grids_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    state: dict,
    *,
    symbol_g=None,
    cache_name: str | None = None,
):
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    _, lambda_centers = _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    lambda_centers = np.asarray(lambda_centers, dtype=np.float32)
    static["lambda_bin_centers"] = _cp.asarray(lambda_centers, dtype=_cp.float32)
    grid_shape = tuple(state["active_mass_grids_g"].shape)
    key = (
        int(hybrid["grid_n"]),
        float(hybrid["spacing"]),
        float(context["diffusivity_si"]),
        tuple(float(x) for x in lambda_centers),
        bool(symbol_g is not None),
    )
    if cache_name:
        cached = state.get(cache_name)
        if cached is not None and tuple(cached.shape) == grid_shape and state.get(cache_name + "_key") == key:
            return cached
        response = _cp.empty(grid_shape, dtype=_cp.float32)
        state[cache_name] = response
        state[cache_name + "_key"] = key
    else:
        response = state.get("fft_symbol_response_g")
        if response is None or tuple(response.shape) != grid_shape:
            response = _cp.empty(grid_shape, dtype=_cp.float32)
            state["fft_symbol_response_g"] = response
    rhs_scale = _cp.float32(max((float(hybrid["spacing"]) ** 3) * float(context["diffusivity_si"]), 1.0e-30))
    lambda_centers_g = static["lambda_bin_centers"]
    if bool(CEXT_HYBRID_FFT_RESPONSE_BATCHED):
        try:
            lam = _cp.maximum(lambda_centers_g, _cp.float32(1.0e-8)).reshape((-1, 1, 1, 1))
            denom = static["k2"][None, :, :, :] + _cp.reciprocal(lam * lam)
            if symbol_g is None:
                response_hat = _cp.reciprocal(rhs_scale * denom)
            else:
                response_hat = _cp.asarray(symbol_g, dtype=_cp.float32)[None, :, :, :] / (rhs_scale * denom)
            response[...] = _cp.real(_cp.fft.ifftn(response_hat, axes=(1, 2, 3))).astype(_cp.float32)
            del lam, denom, response_hat
            _cp.cuda.Stream.null.synchronize()
            return response
        except Exception:
            if str(os.environ.get("SVV_CEXT_HYBRID_FFT_RESPONSE_BATCHED_STRICT", "false")).strip().lower() in ("1", "true", "yes", "on"):
                raise
    for bin_idx in range(int(response.shape[0])):
        lam = _cp.maximum(lambda_centers_g[bin_idx], _cp.float32(1.0e-8))
        denom = static["k2"] + _cp.reciprocal(lam * lam)
        if symbol_g is None:
            response_hat = _cp.reciprocal(rhs_scale * denom)
        else:
            response_hat = symbol_g / (rhs_scale * denom)
        response[bin_idx] = _cp.real(_cp.fft.ifftn(response_hat, axes=(0, 1, 2))).astype(_cp.float32)
        del denom, response_hat
    _cp.cuda.Stream.null.synchronize()
    return response


def _cext_fft_response_cache_name(state: dict, name: str, grid_shape: tuple[int, ...]) -> str | None:
    cached = state.get(name)
    if cached is not None and tuple(cached.shape) == tuple(grid_shape):
        return name
    bytes_needed = int(np.prod(np.asarray(grid_shape, dtype=np.int64))) * np.dtype(np.float32).itemsize
    try:
        env_budget = os.environ.get("SVV_FFT_RESPONSE_CACHE_MAX_BYTES")
        if env_budget is not None:
            budget = int(env_budget)
        elif _cp is not None:
            free_bytes, _total_bytes = _cp.cuda.Device().mem_info
            desired = bytes_needed * 8  # base response plus seven finite-radius moment responses.
            budget = int(min(max(512 * 1024**2, desired), max(0, int(0.5 * free_bytes))))
        else:
            budget = 512 * 1024**2
    except ValueError:
        budget = 512 * 1024**2
    if budget <= 0:
        return None
    used = int(state.get("_fft_response_cache_bytes", 0))
    if used + bytes_needed > budget:
        return None
    state["_fft_response_cache_bytes"] = used + bytes_needed
    return name


def _cext_fft_discrete_same_segment_contribution_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    state: dict,
    *,
    target_plan: dict | None = None,
    include_finite: bool = False,
) -> tuple[object | None, float]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gpu_static = _ensure_cext_gpu_static(context)
    edges, _ = _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    static["lambda_bin_edges"] = _cp.asarray(np.asarray(edges, dtype=np.float32))
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    target_cfg = target_plan if isinstance(target_plan, dict) else None
    target_count = int(target_cfg.get("target_count", nseg)) if target_cfg is not None else nseg
    if target_count <= 0:
        return None, 0.0
    if target_cfg is None or bool(target_cfg.get("uses_static", False)):
        target_seg_ids_g = static["all_target_seg_ids"]
    else:
        if "gpu_target_seg_ids" not in target_cfg:
            target_cfg["gpu_target_seg_ids"] = _cp.asarray(
                np.asarray(target_cfg.get("target_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
            )
        target_seg_ids_g = target_cfg["gpu_target_seg_ids"]

    out_g = state.get("fft_discrete_self_g")
    if out_g is None or tuple(out_g.shape) != (target_count, gl_order):
        out_g = _cp.zeros((target_count, gl_order), dtype=_cp.float32)
        state["fft_discrete_self_g"] = out_g
    else:
        out_g.fill(_cp.float32(0.0))

    target_sampling = str(CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING).strip().lower()
    if target_sampling in ("matched", "assignment", "source"):
        assignment_mode = 1 if str(hybrid.get("assignment", "tsc")).strip().lower() == "tsc" else 0
    elif target_sampling == "tsc":
        assignment_mode = 1
    else:
        assignment_mode = 0

    def accumulate(source_weight_g, response_g) -> None:
        kernel = _get_cext_fft_discrete_self_kernel()
        total = int(target_count * gl_order)
        threads = 128
        blocks = (total + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                target_seg_ids_g,
                gpu_static["gl_points_si"].ravel(),
                _cp.asarray(source_weight_g, dtype=_cp.float32).ravel(),
                state["lambda_iv_gl"].ravel(),
                state["active_source_mask_g"],
                static["stencil_flat_idx"].ravel(),
                static["stencil_weight"].ravel(),
                static["lambda_bin_edges"],
                _cp.asarray(response_g, dtype=_cp.float32).ravel(),
                np.int32(max(int(np.asarray(edges).size - 1), 1)),
                np.int32(int(hybrid["grid_n"])),
                np.int32(int(static["grid_cells"])),
                np.int32(int(static["stencil_n"])),
                np.int32(gl_order),
                np.int32(target_count),
                np.float32(float(hybrid["origin"][0])),
                np.float32(float(hybrid["origin"][1])),
                np.float32(float(hybrid["origin"][2])),
                np.float32(float(hybrid["spacing"])),
                np.int32(assignment_mode),
                out_g.ravel(),
            ),
        )

    t0 = perf_counter()
    base_response_g = _cext_fft_response_grids_gpu(
        context,
        hybrid,
        ext_state,
        state,
        cache_name="fft_base_response_g",
    )

    has_o2 = False
    o2_node = None
    symbols = ()
    moment_weights = ()
    tx = ty = tz = None
    if include_finite:
        if bool(CEXT_HYBRID_GPU_RUNTIME_WEIGHTS) and "mono2_weight_gl" in state and "dipole2_weight_gl" in state:
            o2_node = (state["mono2_weight_gl"].ravel() + state["dipole2_weight_gl"].ravel()).astype(_cp.float32, copy=False)
            has_o2 = bool(_cp.any(_cp.isfinite(o2_node) & (o2_node != _cp.float32(0.0))).item())
        else:
            mono = np.asarray(ext_state.get("mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32)
            dipole = np.asarray(ext_state.get("dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32)
            o2_weight = mono + dipole
            has_o2 = bool(np.any(np.isfinite(o2_weight) & (o2_weight != 0.0)))
            o2_node = _cp.asarray(o2_weight, dtype=_cp.float32).ravel()
        if has_o2:
            grid_n = int(hybrid["grid_n"])
            spacing = float(hybrid["spacing"])
            kfreq = _cp.asarray(2.0 * np.pi * np.fft.fftfreq(grid_n, d=spacing), dtype=_cp.float32)
            kx = kfreq[:, None, None]
            ky = kfreq[None, :, None]
            kz = kfreq[None, None, :]
            symbols = (
                -static["k2"],
                kx * kx,
                ky * ky,
                kz * kz,
                _cp.float32(2.0) * kx * ky,
                _cp.float32(2.0) * kx * kz,
                _cp.float32(2.0) * ky * kz,
            )
            seg_vec = _cp.asarray(gpu_static["segment_vectors"], dtype=_cp.float32)
            seg_len = _cp.sqrt(_cp.sum(seg_vec * seg_vec, axis=1))
            seg_len = _cp.maximum(seg_len, _cp.float32(1.0e-30))
            t_hat = seg_vec / seg_len[:, None]
            node_seg_ids = static["node_seg_ids"]
            tx = t_hat[node_seg_ids, 0]
            ty = t_hat[node_seg_ids, 1]
            tz = t_hat[node_seg_ids, 2]
            moment_weights = (
                o2_node,
                o2_node * tx * tx,
                o2_node * ty * ty,
                o2_node * tz * tz,
                o2_node * tx * ty,
                o2_node * tx * tz,
                o2_node * ty * tz,
            )

    fused_self_done = False
    if has_o2 and bool(CEXT_HYBRID_FFT_SELF_SUB_FUSED):
        try:
            grid_shape = tuple(state["active_mass_grids_g"].shape)
            finite_responses = []
            for moment_idx, symbol in enumerate(symbols):
                cache_name = _cext_fft_response_cache_name(state, f"fft_finite_self_response_g_{moment_idx}", grid_shape)
                if cache_name is None:
                    break
                response_g = _cext_fft_response_grids_gpu(
                    context,
                    hybrid,
                    ext_state,
                    state,
                    symbol_g=symbol,
                    cache_name=cache_name,
                )
                finite_responses.append(response_g)
            if len(finite_responses) == 7:
                kernel = _get_cext_fft_discrete_self_fused_o2_kernel()
                total = int(target_count * gl_order)
                threads = 128
                blocks = (total + threads - 1) // threads
                kernel(
                    (blocks,),
                    (threads,),
                    (
                        target_seg_ids_g,
                        gpu_static["gl_points_si"].ravel(),
                        state["q_weighted_gl"].ravel(),
                        o2_node.ravel(),
                        tx.ravel(),
                        ty.ravel(),
                        tz.ravel(),
                        state["lambda_iv_gl"].ravel(),
                        state["active_source_mask_g"],
                        static["stencil_flat_idx"].ravel(),
                        static["stencil_weight"].ravel(),
                        static["lambda_bin_edges"],
                        _cp.asarray(base_response_g, dtype=_cp.float32).ravel(),
                        _cp.asarray(finite_responses[0], dtype=_cp.float32).ravel(),
                        _cp.asarray(finite_responses[1], dtype=_cp.float32).ravel(),
                        _cp.asarray(finite_responses[2], dtype=_cp.float32).ravel(),
                        _cp.asarray(finite_responses[3], dtype=_cp.float32).ravel(),
                        _cp.asarray(finite_responses[4], dtype=_cp.float32).ravel(),
                        _cp.asarray(finite_responses[5], dtype=_cp.float32).ravel(),
                        _cp.asarray(finite_responses[6], dtype=_cp.float32).ravel(),
                        np.int32(max(int(np.asarray(edges).size - 1), 1)),
                        np.int32(int(hybrid["grid_n"])),
                        np.int32(int(static["grid_cells"])),
                        np.int32(int(static["stencil_n"])),
                        np.int32(gl_order),
                        np.int32(target_count),
                        np.float32(float(hybrid["origin"][0])),
                        np.float32(float(hybrid["origin"][1])),
                        np.float32(float(hybrid["origin"][2])),
                        np.float32(float(hybrid["spacing"])),
                        np.int32(assignment_mode),
                        out_g.ravel(),
                    ),
                )
                fused_self_done = True
            elif str(os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_FUSED_STRICT", "false")).strip().lower() in ("1", "true", "yes", "on"):
                raise RuntimeError("Not enough response-cache budget for fused same-segment subtraction.")
        except Exception:
            if str(os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_FUSED_STRICT", "false")).strip().lower() in ("1", "true", "yes", "on"):
                raise
            fused_self_done = False

    if not fused_self_done:
        accumulate(state["q_weighted_gl"], base_response_g)
        if has_o2:
            grid_shape = tuple(state["active_mass_grids_g"].shape)
            for moment_idx, (moment_weight, symbol) in enumerate(zip(moment_weights, symbols)):
                response_g = _cext_fft_response_grids_gpu(
                    context,
                    hybrid,
                    ext_state,
                    state,
                    symbol_g=symbol,
                    cache_name=_cext_fft_response_cache_name(state, f"fft_finite_self_response_g_{moment_idx}", grid_shape),
                )
                accumulate(moment_weight, response_g)

    _cp.cuda.Stream.null.synchronize()
    return out_g, perf_counter() - t0


def _sample_cext_hybrid_bg_gpu(
    context: dict,
    hybrid: dict,
    phi_grids_g,
    mass_grids_g,
    *,
    target_plan: dict | None = None,
    runtime_state: dict | None = None,
) -> tuple[object, float]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    state = runtime_state if isinstance(runtime_state, dict) else _ensure_cext_hybrid_bg_runtime_state(context, hybrid, {"c_ext_gl": np.zeros((int(np.asarray(context["midpoints_si"]).shape[0]), int(np.asarray(context["gl_points_si"]).shape[1])), dtype=np.float32)})
    if "lambda_bin_centers" not in static:
        centers = np.asarray(hybrid.get("lambda_bin_centers", np.asarray([1.0], dtype=np.float32)), dtype=np.float32)
        static["lambda_bin_centers"] = _cp.asarray(np.asarray(centers, dtype=np.float32))
    kernel = _get_cext_hybrid_bg_kernel()
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    target_cfg = target_plan if isinstance(target_plan, dict) else None
    target_count = int(target_cfg.get("target_count", nseg)) if target_cfg is not None else nseg
    if target_count <= 0:
        return _cp.zeros((0, gl_order), dtype=_cp.float32), 0.0
    if target_cfg is None or bool(target_cfg.get("uses_static", False)):
        target_seg_ids_g = static["all_target_seg_ids"]
    else:
        if "gpu_target_seg_ids" not in target_cfg:
            target_cfg["gpu_target_seg_ids"] = _cp.asarray(np.asarray(target_cfg.get("target_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32))
        target_seg_ids_g = target_cfg["gpu_target_seg_ids"]
    out_bg = state["bg_sample_g"][:target_count]
    threads = 128
    blocks = (target_count * gl_order + threads - 1) // threads
    t_kernel = perf_counter()
    kernel(
        (blocks,),
        (threads,),
        (
            target_seg_ids_g,
            _ensure_cext_gpu_static(context)["gl_points_si"].ravel(),
            _cp.asarray(phi_grids_g, dtype=_cp.float32).ravel(),
            _cp.asarray(mass_grids_g, dtype=_cp.float32).ravel(),
            static["lambda_bin_centers"],
            np.int32(int(static["lambda_bin_centers"].size)),
            np.int32(int(hybrid["grid_n"])),
            np.float32(float(hybrid["origin"][0])),
            np.float32(float(hybrid["origin"][1])),
            np.float32(float(hybrid["origin"][2])),
            np.float32(float(hybrid["spacing"])),
            np.float32(float(hybrid["near_radius_si"])),
            np.int32(gl_order),
            np.int32(target_count),
            np.float32(float(context["diffusivity_si"])),
            out_bg.ravel(),
        ),
    )
    _cp.cuda.Stream.null.synchronize()
    return out_bg, perf_counter() - t_kernel


def _compute_cext_hybrid_bg_gpu(
    context: dict,
    ext_state: dict,
    hybrid: dict,
    *,
    active_source_mask: np.ndarray | None = None,
    active_source_plan: dict | None = None,
    target_plan: dict | None = None,
    local_target_plan: dict | None = None,
    local_target_positions: np.ndarray | None = None,
    runtime_state: dict | None = None,
    frozen_mass_grids_g=None,
    frozen_phi_grids_g=None,
) -> tuple[np.ndarray, dict[str, float]]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    bg_mode = _resolve_cext_hybrid_bg_mode()
    if bg_mode in ("local_only_nlambda", "local_only"):
        return _compute_cext_local_only_hybrid_gpu(
            context,
            ext_state,
            hybrid,
            use_lambda_cutoff=(bg_mode == "local_only_nlambda"),
            runtime_state=runtime_state,
        )
    if bg_mode == "fft":
        hybrid["near_radius_si"] = 0.0
        hybrid["local_rad_cells"] = 0
    state = runtime_state if isinstance(runtime_state, dict) else _ensure_cext_hybrid_bg_runtime_state(context, hybrid, ext_state)
    _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    active_mass_grids_g, t_deposit = _cext_hybrid_deposit_sources_gpu(
        context,
        hybrid,
        ext_state,
        source_mask=active_source_mask,
        runtime_state=state,
        out_mass_grids_g=state["active_mass_grids_g"],
    )
    active_phi_grids_g, t_fft, bg_solver_mode = _cext_hybrid_solve_background_fft_gpu(
        context,
        hybrid,
        active_mass_grids_g,
        init_phi_grids_g=hybrid.get("active_phi_guess_g"),
    )
    hybrid["active_phi_guess_g"] = active_phi_grids_g
    sample_mass_grids_g = active_mass_grids_g
    sample_phi_grids_g = active_phi_grids_g
    if frozen_mass_grids_g is not None:
        if "total_mass_grids_g" not in state or tuple(state["total_mass_grids_g"].shape) != tuple(active_mass_grids_g.shape):
            state["total_mass_grids_g"] = _cp.empty_like(active_mass_grids_g)
        state["total_mass_grids_g"][...] = active_mass_grids_g
        state["total_mass_grids_g"] += frozen_mass_grids_g
        sample_mass_grids_g = state["total_mass_grids_g"]
    if frozen_phi_grids_g is not None:
        if "total_phi_grids_g" not in state or tuple(state["total_phi_grids_g"].shape) != tuple(active_phi_grids_g.shape):
            state["total_phi_grids_g"] = _cp.empty_like(active_phi_grids_g)
        state["total_phi_grids_g"][...] = active_phi_grids_g
        state["total_phi_grids_g"] += frozen_phi_grids_g
        sample_phi_grids_g = state["total_phi_grids_g"]
    bg_corr_g, t_sample = _sample_cext_hybrid_bg_gpu(
        context,
        hybrid,
        sample_phi_grids_g,
        sample_mass_grids_g,
        target_plan=target_plan,
        runtime_state=state,
    )
    total_g = _cp.asarray(bg_corr_g, dtype=_cp.float32).copy()
    if float(hybrid.get("near_radius_si", 0.0)) > 0.0:
        local_plan = local_target_plan if isinstance(local_target_plan, dict) else target_plan
        local_total_g, local_cap_g, t_local, t_transfer_local = _compute_cext_local_corr_gpu(
            context,
            hybrid,
            ext_state,
            source_mask=active_source_mask,
            source_plan=active_source_plan,
            target_plan=local_plan,
            runtime_state=state,
        )
    else:
        local_total_g = None
        local_cap_g = None
        t_local = 0.0
        t_transfer_local = 0.0
    if bool(CEXT_HYBRID_GPU_ITERATION_CACHE) and "seg_cap_gl" in state and int(state["seg_cap_gl"].size) > 0:
        global_cap = float(_cp.nanmax(state["seg_cap_gl"]).item())
    else:
        global_cap = float(np.nanmax(np.asarray(ext_state.get("seg_cap_gl", np.zeros((0,), dtype=np.float32)), dtype=float))) if np.asarray(ext_state.get("seg_cap_gl", ())).size else 0.0
    local_count = int(_cp.asarray(local_total_g).shape[0]) if local_total_g is not None else 0
    target_count = int(total_g.shape[0])
    local_report_count = local_count if float(hybrid.get("near_radius_si", 0.0)) > 0.0 else 0
    if local_count == target_count:
        total_g += _cp.asarray(local_total_g, dtype=_cp.float32)
    elif local_count > 0:
        if local_target_positions is None:
            raise ValueError("local_target_positions is required when hybrid local targets are a subset")
        local_pos_g = _cp.asarray(np.asarray(local_target_positions, dtype=np.int32))
        if int(local_pos_g.size) != local_count:
            raise ValueError("local_target_positions length does not match local target count")
        total_g[local_pos_g, :] += _cp.asarray(local_total_g, dtype=_cp.float32)
    if global_cap > 0.0:
        _cp.minimum(total_g, _cp.float32(global_cap), out=total_g)
    _cp.maximum(total_g, _cp.float32(0.0), out=total_g)
    t_o2 = 0.0
    t_self_sub = 0.0
    if bg_mode == "fft":
        if bool(CEXT_HYBRID_FFT_O2_CORRECTION):
            o2_corr_g, t_o2 = _cext_fft_o2_term_correction_gpu(
                context,
                hybrid,
                ext_state,
                state,
                target_plan=target_plan,
            )
            if o2_corr_g is not None:
                total_g += _cp.asarray(o2_corr_g, dtype=_cp.float32)
        if bool(CEXT_HYBRID_FFT_SELF_SUBTRACT):
            same_seg_g, t_self_sub = _cext_fft_discrete_same_segment_contribution_gpu(
                context,
                hybrid,
                ext_state,
                state,
                target_plan=target_plan,
                include_finite=bool(CEXT_HYBRID_FFT_O2_CORRECTION),
            )
            if same_seg_g is not None:
                total_g -= _cp.float32(float(CEXT_HYBRID_FFT_SELF_SUB_SCALE)) * _cp.asarray(same_seg_g, dtype=_cp.float32)
        if global_cap > 0.0:
            _cp.minimum(total_g, _cp.float32(global_cap), out=total_g)
        _cp.maximum(total_g, _cp.float32(0.0), out=total_g)
    t_download = perf_counter()
    c_ext_new = _cp.asnumpy(total_g)
    _cp.cuda.Stream.null.synchronize()
    timings = {
        "deposit_s": float(t_deposit),
        "fft_s": float(t_fft),
        "sample_s": float(t_sample),
        "fft_o2_terms_s": float(t_o2),
        "fft_discrete_self_subtract_s": float(t_self_sub),
        "local_corr_s": float(t_local),
        "transfer_s": float(t_transfer_local + (perf_counter() - t_download)),
        "bg_solver_mode": str(bg_solver_mode),
        "local_target_count": float(local_report_count),
    }
    return np.asarray(c_ext_new, dtype=np.float32), timings


def _part1by2_64(n: np.ndarray) -> np.ndarray:
    x = np.asarray(n, dtype=np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def _morton3d_codes(points: np.ndarray, origin: np.ndarray, side: float, *, bits: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size <= 0:
        return np.zeros((0,), dtype=np.uint64)
    bits = max(min(int(bits), 21), 1)
    scale = float((1 << bits) - 1)
    local = (pts - np.asarray(origin, dtype=np.float64)[None, :]) / max(float(side), 1.0e-30)
    local = np.clip(local, 0.0, 1.0)
    idx = np.floor(local * scale + 0.5).astype(np.uint64)
    return _part1by2_64(idx[:, 0]) | (_part1by2_64(idx[:, 1]) << np.uint64(1)) | (_part1by2_64(idx[:, 2]) << np.uint64(2))


def _treecode_child_offset(slot: int) -> np.ndarray:
    return np.asarray(
        [
            -0.5 if (slot & 0x4) == 0 else 0.5,
            -0.5 if (slot & 0x2) == 0 else 0.5,
            -0.5 if (slot & 0x1) == 0 else 0.5,
        ],
        dtype=np.float32,
    )


def _build_cext_treecode_context(context: dict) -> dict:
    cached = context.get("treecode_context")
    gl_points = np.asarray(context.get("gl_points_si", np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
    gl_order = int(gl_points.shape[1]) if gl_points.ndim >= 2 else 0
    nseg = int(gl_points.shape[0]) if gl_points.ndim >= 1 else 0
    leaf_nodes = max(int(CEXT_TREECODE_LEAF_NODES), 8)
    theta = max(float(CEXT_TREECODE_THETA), 1.0e-3)
    near_radius_mult = max(float(CEXT_TREECODE_NEAR_RADIUS_MULT), 1.0)
    lambda_bins = max(int(CEXT_TREECODE_LAMBDA_BINS), 1)
    order_mode = max(int(CEXT_TREECODE_ORDER), 0)
    if isinstance(cached, dict):
        if (
            int(cached.get("gl_order", -1)) == gl_order
            and int(cached.get("nseg", -1)) == nseg
            and int(cached.get("leaf_nodes", -1)) == leaf_nodes
            and int(cached.get("lambda_bins", -1)) == lambda_bins
            and float(cached.get("theta", -1.0)) == theta
            and float(cached.get("near_radius_mult", -1.0)) == near_radius_mult
            and int(cached.get("order_mode", -1)) == order_mode
        ):
            return cached
    gl_points_flat = np.asarray(gl_points.reshape(-1, 3), dtype=np.float32)
    n_nodes = int(gl_points_flat.shape[0])
    node_seg_ids = np.repeat(np.arange(nseg, dtype=np.int32), max(gl_order, 1)) if n_nodes > 0 else np.zeros((0,), dtype=np.int32)
    if n_nodes > 0:
        mins = np.min(gl_points_flat, axis=0)
        maxs = np.max(gl_points_flat, axis=0)
    else:
        mins = np.zeros((3,), dtype=np.float32)
        maxs = np.ones((3,), dtype=np.float32)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0e-8)
    pad = max(float(context.get("max_reach_si", 0.0)), 0.05 * span)
    side = float(span + 2.0 * pad)
    origin = np.asarray(center - 0.5 * side, dtype=np.float32)
    bits = int(min(21, max(8, math.ceil(math.log2(max((n_nodes / max(float(leaf_nodes), 1.0)) ** (1.0 / 3.0), 1.0))) + 3)))
    morton = _morton3d_codes(gl_points_flat, origin, side, bits=bits)
    order = np.argsort(morton, kind="stable")
    sorted_codes = np.asarray(morton[order], dtype=np.uint64)
    sorted_node_ids = np.asarray(order, dtype=np.int32)
    sorted_points = np.asarray(gl_points_flat[sorted_node_ids], dtype=np.float32)
    sorted_seg_ids = np.asarray(node_seg_ids[sorted_node_ids], dtype=np.int32)
    point_leaf_ids = np.full((n_nodes,), -1, dtype=np.int32)
    node_start: list[int] = []
    node_end: list[int] = []
    node_level: list[int] = []
    node_center: list[np.ndarray] = []
    node_half: list[float] = []
    node_parent: list[int] = []
    node_children: list[list[int]] = []
    node_is_leaf: list[bool] = []
    leaf_ids: list[int] = []
    queue: list[tuple[int, int, int, np.ndarray, float, int, int]] = [(0, n_nodes, 0, np.asarray(center, dtype=np.float32), 0.5 * side, -1, -1)]
    max_depth = 0
    while queue:
        lo, hi, level, node_ctr, half, parent_id, child_slot = queue.pop()
        node_id = len(node_start)
        node_start.append(int(lo))
        node_end.append(int(hi))
        node_level.append(int(level))
        node_center.append(np.asarray(node_ctr, dtype=np.float32))
        node_half.append(float(half))
        node_parent.append(int(parent_id))
        node_children.append([-1] * 8)
        count = int(hi - lo)
        is_leaf = count <= leaf_nodes or level >= bits or count <= 1
        node_is_leaf.append(bool(is_leaf))
        if parent_id >= 0 and child_slot >= 0:
            node_children[parent_id][child_slot] = node_id
        max_depth = max(max_depth, int(level))
        if is_leaf:
            leaf_ids.append(node_id)
            if hi > lo:
                point_leaf_ids[lo:hi] = np.int32(node_id)
            continue
        shift = max(3 * (bits - level - 1), 0)
        child_codes = np.asarray((sorted_codes[lo:hi] >> np.uint64(shift)) & np.uint64(0x7), dtype=np.int32)
        if child_codes.size <= 0:
            node_is_leaf[node_id] = True
            leaf_ids.append(node_id)
            continue
        split_idx = np.flatnonzero(np.diff(child_codes)) + 1
        bounds = np.concatenate(([0], split_idx, [child_codes.size]))
        child_half = float(half) * 0.5
        for bound_lo, bound_hi in zip(bounds[:-1], bounds[1:]):
            slot = int(child_codes[int(bound_lo)])
            child_ctr = np.asarray(node_ctr, dtype=np.float32) + np.float32(child_half) * _treecode_child_offset(slot)
            queue.append((int(lo + bound_lo), int(lo + bound_hi), int(level + 1), child_ctr, child_half, int(node_id), int(slot)))
    point_leaf_ids = np.asarray(point_leaf_ids, dtype=np.int32)
    node_level_arr = np.asarray(node_level, dtype=np.int16)
    node_is_leaf_arr = np.asarray(node_is_leaf, dtype=bool)
    upsweep_levels: list[np.ndarray] = []
    for level in range(int(max_depth) - 1, -1, -1):
        ids = np.flatnonzero((node_level_arr == level) & (~node_is_leaf_arr)).astype(np.int32, copy=False)
        if ids.size > 0:
            upsweep_levels.append(np.asarray(ids, dtype=np.int32))
    if upsweep_levels:
        upsweep_node_ids = np.concatenate(upsweep_levels).astype(np.int32, copy=False)
        upsweep_level_offsets = np.zeros((len(upsweep_levels) + 1,), dtype=np.int32)
        upsweep_level_offsets[1:] = np.cumsum(
            np.asarray([arr.size for arr in upsweep_levels], dtype=np.int64),
            dtype=np.int64,
        ).astype(np.int32)
    else:
        upsweep_node_ids = np.zeros((0,), dtype=np.int32)
        upsweep_level_offsets = np.zeros((1,), dtype=np.int32)
    treecode = {
        "gl_order": int(gl_order),
        "nseg": int(nseg),
        "leaf_nodes": int(leaf_nodes),
        "theta": float(theta),
        "lambda_bins": int(lambda_bins),
        "near_radius_mult": float(near_radius_mult),
        "order_mode": int(order_mode),
        "origin": np.asarray(origin, dtype=np.float32),
        "side": float(side),
        "center": np.asarray(center, dtype=np.float32),
        "morton_bits": int(bits),
        "sorted_node_ids": np.asarray(sorted_node_ids, dtype=np.int32),
        "sorted_points": np.asarray(sorted_points, dtype=np.float32),
        "sorted_seg_ids": np.asarray(sorted_seg_ids, dtype=np.int32),
        "point_leaf_ids": point_leaf_ids,
        "node_point_count": np.asarray(np.asarray(node_end, dtype=np.int32) - np.asarray(node_start, dtype=np.int32), dtype=np.int32),
        "node_start": np.asarray(node_start, dtype=np.int32),
        "node_end": np.asarray(node_end, dtype=np.int32),
        "node_level": node_level_arr,
        "node_center": np.asarray(node_center, dtype=np.float32),
        "node_half": np.asarray(node_half, dtype=np.float32),
        "node_parent": np.asarray(node_parent, dtype=np.int32),
        "node_children": np.asarray(node_children, dtype=np.int32),
        "node_is_leaf": node_is_leaf_arr,
        "leaf_ids": np.asarray(leaf_ids, dtype=np.int32),
        "upsweep_node_ids": upsweep_node_ids,
        "upsweep_level_offsets": upsweep_level_offsets,
        "node_count": int(len(node_start)),
        "leaf_count": int(len(leaf_ids)),
        "max_depth": int(max_depth),
        "runtime_state": None,
        "lambda_bin_edges": None,
        "lambda_bin_centers": None,
    }
    context["treecode_context"] = treecode
    return treecode


def _cext_treecode_init_lambda_bins(treecode: dict, ext_state: dict) -> tuple[np.ndarray, np.ndarray]:
    edges = treecode.get("lambda_bin_edges")
    centers = treecode.get("lambda_bin_centers")
    if edges is not None and centers is not None:
        return np.asarray(edges, dtype=np.float32), np.asarray(centers, dtype=np.float32)
    lambda_gl = np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32).reshape(-1)
    finite = lambda_gl[np.isfinite(lambda_gl) & (lambda_gl > 0.0)]
    if finite.size <= 0:
        finite = np.asarray([1.0e-6], dtype=np.float32)
    lam_min = max(float(np.min(finite)), 1.0e-8)
    lam_max = max(float(np.max(finite)), lam_min * (1.0 + 1.0e-6))
    n_bins = max(int(treecode.get("lambda_bins", CEXT_TREECODE_LAMBDA_BINS)), 1)
    if n_bins <= 1 or lam_max <= lam_min * (1.0 + 1.0e-6):
        centers = np.asarray([math.sqrt(lam_min * lam_max)], dtype=np.float32)
        edges = np.asarray([lam_min, lam_max], dtype=np.float32)
    else:
        edges = np.geomspace(lam_min, lam_max, n_bins + 1).astype(np.float32)
        centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
    treecode["lambda_bin_edges"] = np.asarray(edges, dtype=np.float32)
    treecode["lambda_bin_centers"] = np.asarray(centers, dtype=np.float32)
    return np.asarray(edges, dtype=np.float32), np.asarray(centers, dtype=np.float32)


def _ensure_cext_treecode_runtime_state(context: dict, treecode: dict, ext_state: dict) -> dict:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    state = treecode.get("runtime_state")
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    n_target_nodes = int(max(nseg * gl_order, 1))
    n_nodes = int(treecode["node_count"])
    n_points = int(np.asarray(treecode["sorted_node_ids"]).size)
    n_bins = max(int(treecode.get("lambda_bins", CEXT_TREECODE_LAMBDA_BINS)), 1)
    if isinstance(state, dict):
        if (
            int(state.get("n_target_nodes", -1)) == n_target_nodes
            and int(state.get("n_tree_nodes", -1)) == n_nodes
            and int(state.get("n_points", -1)) == n_points
            and int(state.get("n_bins", -1)) == n_bins
        ):
            return state
    static = _ensure_cext_gpu_static(context)
    state = {
        "n_target_nodes": int(n_target_nodes),
        "n_tree_nodes": int(n_nodes),
        "n_points": int(n_points),
        "n_bins": int(n_bins),
        "sorted_points": _cp.asarray(np.asarray(treecode["sorted_points"], dtype=np.float32)),
        "sorted_seg_ids": _cp.asarray(np.asarray(treecode["sorted_seg_ids"], dtype=np.int32)),
        "point_leaf_ids": _cp.asarray(np.asarray(treecode["point_leaf_ids"], dtype=np.int32)),
        "node_start": _cp.asarray(np.asarray(treecode["node_start"], dtype=np.int32)),
        "node_end": _cp.asarray(np.asarray(treecode["node_end"], dtype=np.int32)),
        "node_point_count": _cp.asarray(np.asarray(treecode["node_point_count"], dtype=np.int32)),
        "node_center": _cp.asarray(np.asarray(treecode["node_center"], dtype=np.float32)),
        "node_half": _cp.asarray(np.asarray(treecode["node_half"], dtype=np.float32)),
        "node_children": _cp.asarray(np.asarray(treecode["node_children"], dtype=np.int32)),
        "node_is_leaf": _cp.asarray(np.asarray(treecode["node_is_leaf"], dtype=np.uint8)),
        "upsweep_node_ids": _cp.asarray(np.asarray(treecode["upsweep_node_ids"], dtype=np.int32)),
        "upsweep_level_offsets": _cp.asarray(np.asarray(treecode["upsweep_level_offsets"], dtype=np.int32)),
        "all_target_seg_ids": static["all_target_seg_ids"],
        "sorted_lambda": _cp.zeros((max(n_points, 1),), dtype=_cp.float32),
        "sorted_q": _cp.zeros((max(n_points, 1),), dtype=_cp.float32),
        "active_seg_mask": _cp.ones((max(nseg, 1),), dtype=_cp.uint8),
        "sorted_point_active": _cp.ones((max(n_points, 1),), dtype=_cp.uint8),
        "bin_edges": _cp.zeros((max(n_bins + 1, 2),), dtype=_cp.float32),
        "node_total_mass": _cp.zeros((max(n_bins, 1), max(n_nodes, 1)), dtype=_cp.float32),
        "node_total_dipole": _cp.zeros((max(n_bins, 1), max(n_nodes, 1), 3), dtype=_cp.float32),
        "node_active_mass": _cp.zeros((max(n_bins, 1), max(n_nodes, 1)), dtype=_cp.float32),
        "node_active_dipole": _cp.zeros((max(n_bins, 1), max(n_nodes, 1), 3), dtype=_cp.float32),
        "node_active_count": _cp.zeros((max(n_nodes, 1),), dtype=_cp.int32),
        "lambda_centers": _cp.zeros((max(n_bins, 1),), dtype=_cp.float32),
        "out_total": _cp.zeros((max(nseg, 1), max(gl_order, 1)), dtype=_cp.float32),
        "metrics": _cp.zeros((3,), dtype=_cp.uint64),
    }
    treecode["runtime_state"] = state
    return state


def _sync_cext_treecode_runtime_state(treecode: dict, ext_state: dict, runtime_state: dict) -> np.ndarray:
    sorted_node_ids = np.asarray(treecode["sorted_node_ids"], dtype=np.int32)
    q_sorted = np.asarray(ext_state["q_weighted_gl"], dtype=np.float32).reshape(-1)[sorted_node_ids] if sorted_node_ids.size else np.zeros((0,), dtype=np.float32)
    lam_sorted = np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32).reshape(-1)[sorted_node_ids] if sorted_node_ids.size else np.zeros((0,), dtype=np.float32)
    if q_sorted.size:
        runtime_state["sorted_q"][: q_sorted.size].set(np.asarray(q_sorted, dtype=np.float32))
        runtime_state["sorted_lambda"][: lam_sorted.size].set(np.asarray(lam_sorted, dtype=np.float32))
    edges, centers = _cext_treecode_init_lambda_bins(treecode, ext_state)
    active_mask = np.asarray(ext_state.get("treecode_active_source_mask", np.ones((treecode["nseg"],), dtype=bool)), dtype=bool).reshape(-1)
    if active_mask.size != int(treecode["nseg"]):
        active_mask = np.ones((int(treecode["nseg"]),), dtype=bool)
    runtime_state["active_seg_mask"][: active_mask.size].set(np.asarray(active_mask, dtype=np.uint8))
    runtime_state["sorted_point_active"][:] = runtime_state["active_seg_mask"][runtime_state["sorted_seg_ids"]]
    runtime_state["bin_edges"][: edges.size].set(np.asarray(edges, dtype=np.float32))
    runtime_state["lambda_centers"][: centers.size].set(centers)
    runtime_state["metrics"].fill(np.uint64(0))
    return np.asarray(centers, dtype=np.float32)


def _get_cext_treecode_deposit_kernel():
    global _CEXT_TREECODE_DEPOSIT_KERNEL
    if _CEXT_TREECODE_DEPOSIT_KERNEL is not None:
        return _CEXT_TREECODE_DEPOSIT_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_treecode_deposit_kernel(
    const int* point_leaf_ids,
    const float* point_pos,
    const float* point_lambda,
    const float* point_q,
    const unsigned char* point_active,
    const float* node_center,
    const float* bin_edges,
    int n_bins,
    int n_nodes,
    int n_points,
    float* total_mass,
    float* total_dipole,
    float* active_mass,
    float* active_dipole,
    int* active_count
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;
    int leaf = point_leaf_ids[idx];
    float lam = point_lambda[idx];
    float q = point_q[idx];
    if (leaf < 0 || leaf >= n_nodes || !(q != 0.0f) || !isfinite(q) || !isfinite(lam) || lam <= 0.0f) return;
    int bin_idx = 0;
    for (int bi = 0; bi < n_bins - 1; ++bi) {
        if (lam > bin_edges[bi + 1]) bin_idx = bi + 1;
    }
    float rx = point_pos[idx * 3 + 0] - node_center[leaf * 3 + 0];
    float ry = point_pos[idx * 3 + 1] - node_center[leaf * 3 + 1];
    float rz = point_pos[idx * 3 + 2] - node_center[leaf * 3 + 2];
    int base = bin_idx * n_nodes + leaf;
    atomicAdd(total_mass + base, q);
    atomicAdd(total_dipole + base * 3 + 0, q * rx);
    atomicAdd(total_dipole + base * 3 + 1, q * ry);
    atomicAdd(total_dipole + base * 3 + 2, q * rz);
    if (point_active[idx] != 0) {
        atomicAdd(active_mass + base, q);
        atomicAdd(active_dipole + base * 3 + 0, q * rx);
        atomicAdd(active_dipole + base * 3 + 1, q * ry);
        atomicAdd(active_dipole + base * 3 + 2, q * rz);
        atomicAdd(active_count + leaf, 1);
    }
}
'''
    _CEXT_TREECODE_DEPOSIT_KERNEL = _cp.RawKernel(code, "cext_treecode_deposit_kernel")
    return _CEXT_TREECODE_DEPOSIT_KERNEL


def _get_cext_treecode_upsweep_kernel():
    global _CEXT_TREECODE_UPSWEEP_KERNEL
    if _CEXT_TREECODE_UPSWEEP_KERNEL is not None:
        return _CEXT_TREECODE_UPSWEEP_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_treecode_upsweep_kernel(
    const int* level_node_ids,
    const int* node_children,
    const float* node_center,
    int n_bins,
    int n_nodes_level,
    int n_nodes_total,
    float* total_mass,
    float* total_dipole,
    float* active_mass,
    float* active_dipole,
    int* active_count
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_nodes_level) return;
    int node = level_node_ids[idx];
    int count_sum = 0;
    for (int bi = 0; bi < n_bins; ++bi) {
        float tm = 0.0f, am = 0.0f;
        float tdx = 0.0f, tdy = 0.0f, tdz = 0.0f;
        float adx = 0.0f, ady = 0.0f, adz = 0.0f;
        for (int slot = 0; slot < 8; ++slot) {
            int child = node_children[node * 8 + slot];
            if (child < 0) continue;
            if (bi == 0) count_sum += active_count[child];
            int child_base = bi * n_nodes_total + child;
            float child_tm = total_mass[child_base];
            float child_am = active_mass[child_base];
            float dx = node_center[child * 3 + 0] - node_center[node * 3 + 0];
            float dy = node_center[child * 3 + 1] - node_center[node * 3 + 1];
            float dz = node_center[child * 3 + 2] - node_center[node * 3 + 2];
            tm += child_tm;
            am += child_am;
            tdx += total_dipole[child_base * 3 + 0] + child_tm * dx;
            tdy += total_dipole[child_base * 3 + 1] + child_tm * dy;
            tdz += total_dipole[child_base * 3 + 2] + child_tm * dz;
            adx += active_dipole[child_base * 3 + 0] + child_am * dx;
            ady += active_dipole[child_base * 3 + 1] + child_am * dy;
            adz += active_dipole[child_base * 3 + 2] + child_am * dz;
        }
        int base = bi * n_nodes_total + node;
        total_mass[base] = tm;
        active_mass[base] = am;
        total_dipole[base * 3 + 0] = tdx;
        total_dipole[base * 3 + 1] = tdy;
        total_dipole[base * 3 + 2] = tdz;
        active_dipole[base * 3 + 0] = adx;
        active_dipole[base * 3 + 1] = ady;
        active_dipole[base * 3 + 2] = adz;
    }
    active_count[node] = count_sum;
}
'''
    _CEXT_TREECODE_UPSWEEP_KERNEL = _cp.RawKernel(code, "cext_treecode_upsweep_kernel")
    return _CEXT_TREECODE_UPSWEEP_KERNEL


def _get_cext_treecode_kernel():
    global _CEXT_TREECODE_KERNEL
    if _CEXT_TREECODE_KERNEL is not None:
        return _CEXT_TREECODE_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_treecode_kernel(
    const int* target_seg_ids,
    const float* target_points_si,
    const float* radii_si,
    const int* exclude_idx,
    const unsigned char* exclude_count,
    const float* source_points_si,
    const int* source_seg_ids,
    const float* source_lambda,
    const float* source_q,
    const int* node_start,
    const int* node_end,
    const float* node_center,
    const float* node_half,
    const int* node_children,
    const unsigned char* node_is_leaf,
    const int* node_point_count,
    const int* node_active_count,
    const unsigned char* point_active,
    const float* node_total_mass,
    const float* node_total_dipole,
    const float* node_active_mass,
    const float* node_active_dipole,
    const float* lambda_centers,
    int n_bins,
    int gl_order,
    int exclude_width,
    int node_count,
    int target_count,
    int order_mode,
    float theta,
    float near_radius_si,
    float diffusivity_si,
    float window_factor,
    float global_cap,
    float* out_total,
    unsigned long long* metrics
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_count * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];
    float tx = target_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = target_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = target_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float target_radius = radii_si[target_seg];
    int excl_n = (int)exclude_count[target_seg];
    float total = 0.0f;
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        int node = stack[--top];
        if (node < 0 || node >= node_count) continue;
        float cx = node_center[node * 3 + 0];
        float cy = node_center[node * 3 + 1];
        float cz = node_center[node * 3 + 2];
        float dx = cx - tx;
        float dy = cy - ty;
        float dz = cz - tz;
        float dist2 = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(dist2 + 1.0e-24f);
        float node_r = 1.7320508075688772f * node_half[node];
        float sep = dist - node_r;
        if (sep < 1.0e-12f) sep = 1.0e-12f;
        int is_leaf = node_is_leaf[node] != 0;
        int total_count = node_point_count[node];
        int active_count = node_active_count[node];
        int inactive_count = total_count - active_count;
        if (sep > near_radius_si && inactive_count > 0) {
            for (int bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
                int base = bin_idx * node_count + node;
                float mass = node_total_mass[base] - node_active_mass[base];
                if (!(mass != 0.0f) || !isfinite(mass)) continue;
                float lam = lambda_centers[bin_idx];
                if (lam < 1.0e-30f) lam = 1.0e-30f;
                float g = expf(-dist / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * dist);
                float contrib = mass * g;
                if (order_mode > 0) {
                    float dotp =
                        (node_total_dipole[base * 3 + 0] - node_active_dipole[base * 3 + 0]) * dx
                        + (node_total_dipole[base * 3 + 1] - node_active_dipole[base * 3 + 1]) * dy
                        + (node_total_dipole[base * 3 + 2] - node_active_dipole[base * 3 + 2]) * dz;
                    contrib += g * (1.0f / lam + 1.0f / dist) * (dotp / dist);
                }
                total += contrib;
            }
            atomicAdd(metrics + 0, (unsigned long long)1);
        }
        if (active_count <= 0 && sep > near_radius_si) {
            continue;
        }
        if (!is_leaf && sep > near_radius_si && active_count > 0 && (node_r / sep) <= theta) {
            for (int bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
                int base = bin_idx * node_count + node;
                float mass = node_active_mass[base];
                if (!(mass != 0.0f) || !isfinite(mass)) continue;
                float lam = lambda_centers[bin_idx];
                if (lam < 1.0e-30f) lam = 1.0e-30f;
                float g = expf(-dist / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * dist);
                float contrib = mass * g;
                if (order_mode > 0) {
                    float dotp =
                        node_active_dipole[base * 3 + 0] * dx
                        + node_active_dipole[base * 3 + 1] * dy
                        + node_active_dipole[base * 3 + 2] * dz;
                    contrib += g * (1.0f / lam + 1.0f / dist) * (dotp / dist);
                }
                total += contrib;
            }
            atomicAdd(metrics + 0, (unsigned long long)1);
            continue;
        }
        if (is_leaf) {
            int need_all = sep <= near_radius_si;
            int start = node_start[node];
            int stop = node_end[node];
            for (int pos = start; pos < stop; ++pos) {
                if (!need_all && point_active[pos] == 0) continue;
                int source_seg = source_seg_ids[pos];
                int excluded = 0;
                for (int ei = 0; ei < excl_n; ++ei) {
                    if (exclude_idx[target_seg * exclude_width + ei] == source_seg) {
                        excluded = 1;
                        break;
                    }
                }
                if (excluded) continue;
                float radius_sum = target_radius + radii_si[source_seg];
                float sx = source_points_si[pos * 3 + 0];
                float sy = source_points_si[pos * 3 + 1];
                float sz = source_points_si[pos * 3 + 2];
                float sdx = sx - tx;
                float sdy = sy - ty;
                float sdz = sz - tz;
                float r = sqrtf(sdx * sdx + sdy * sdy + sdz * sdz + radius_sum * radius_sum);
                float lam = source_lambda[pos];
                if (lam < 1.0e-30f) lam = 1.0e-30f;
                if (r > window_factor * lam) continue;
                float q = source_q[pos];
                if (!(q != 0.0f) || !isfinite(q)) continue;
                total += q * expf(-r / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                atomicAdd(metrics + 2, (unsigned long long)1);
            }
            continue;
        }
        atomicAdd(metrics + 1, (unsigned long long)1);
        for (int slot = 0; slot < 8; ++slot) {
            int child = node_children[node * 8 + slot];
            if (child < 0) continue;
            if (node_active_count[child] <= 0) {
                float chx = node_center[child * 3 + 0];
                float chy = node_center[child * 3 + 1];
                float chz = node_center[child * 3 + 2];
                float cdx = chx - tx;
                float cdy = chy - ty;
                float cdz = chz - tz;
                float cdist = sqrtf(cdx * cdx + cdy * cdy + cdz * cdz + 1.0e-24f);
                float child_r = 1.7320508075688772f * node_half[child];
                float child_sep = cdist - child_r;
                if (child_sep > near_radius_si) continue;
            }
            if (top < 64) stack[top++] = child;
        }
    }
    if (global_cap > 0.0f) total = fminf(total, global_cap);
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out_total[target_idx * gl_order + target_node] = total;
}
'''
    _CEXT_TREECODE_KERNEL = _cp.RawKernel(code, "cext_treecode_kernel")
    return _CEXT_TREECODE_KERNEL


def _compute_cext_treecode_gpu(context: dict, treecode: dict, ext_state: dict) -> tuple[np.ndarray, dict[str, float | int]]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    t_aggregate = perf_counter()
    state = _ensure_cext_treecode_runtime_state(context, treecode, ext_state)
    centers = _sync_cext_treecode_runtime_state(treecode, ext_state, state)
    n_bins = int(max(centers.size, 1))
    state["node_total_mass"].fill(_cp.float32(0.0))
    state["node_total_dipole"].fill(_cp.float32(0.0))
    state["node_active_mass"].fill(_cp.float32(0.0))
    state["node_active_dipole"].fill(_cp.float32(0.0))
    state["node_active_count"].fill(np.int32(0))
    n_points = int(state["n_points"])
    if n_points > 0:
        threads = 256
        blocks = (n_points + threads - 1) // threads
        deposit_kernel = _get_cext_treecode_deposit_kernel()
        deposit_kernel(
            (blocks,),
            (threads,),
            (
                state["point_leaf_ids"],
                state["sorted_points"].ravel(),
                state["sorted_lambda"],
                state["sorted_q"],
                state["sorted_point_active"],
                state["node_center"].ravel(),
                state["bin_edges"],
                np.int32(n_bins),
                np.int32(int(treecode["node_count"])),
                np.int32(n_points),
                state["node_total_mass"].ravel(),
                state["node_total_dipole"].ravel(),
                state["node_active_mass"].ravel(),
                state["node_active_dipole"].ravel(),
                state["node_active_count"],
            ),
        )
        level_offsets = np.asarray(treecode["upsweep_level_offsets"], dtype=np.int32)
        upsweep_kernel = _get_cext_treecode_upsweep_kernel()
        for level_idx in range(max(level_offsets.size - 1, 0)):
            lo = int(level_offsets[level_idx])
            hi = int(level_offsets[level_idx + 1])
            if hi <= lo:
                continue
            n_level = hi - lo
            blocks_level = (n_level + threads - 1) // threads
            upsweep_kernel(
                (blocks_level,),
                (threads,),
                (
                    state["upsweep_node_ids"][lo:hi],
                    state["node_children"].ravel(),
                    state["node_center"].ravel(),
                    np.int32(n_bins),
                    np.int32(n_level),
                    np.int32(int(treecode["node_count"])),
                    state["node_total_mass"].ravel(),
                    state["node_total_dipole"].ravel(),
                    state["node_active_mass"].ravel(),
                    state["node_active_dipole"].ravel(),
                    state["node_active_count"],
                ),
            )
    _cp.cuda.Stream.null.synchronize()
    t_aggregate = perf_counter() - t_aggregate
    static = _ensure_cext_gpu_static(context)
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    target_count = int(nseg)
    out_total = state["out_total"][:target_count]
    out_total.fill(_cp.float32(0.0))
    threads = 128
    blocks = (target_count * gl_order + threads - 1) // threads
    kernel = _get_cext_treecode_kernel()
    global_cap = float(np.nanmax(np.asarray(ext_state.get("seg_cap_gl", np.zeros((0,), dtype=np.float32)), dtype=float))) if np.asarray(ext_state.get("seg_cap_gl", ())).size else 0.0
    t_traversal = perf_counter()
    kernel(
        (blocks,),
        (threads,),
        (
            state["all_target_seg_ids"],
            static["gl_points_si"].ravel(),
            static["radii_si"],
            static["exclude_idx"].ravel(),
            static["exclude_count"],
            state["sorted_points"].ravel(),
            state["sorted_seg_ids"],
            state["sorted_lambda"],
            state["sorted_q"],
            state["node_start"],
            state["node_end"],
            state["node_center"].ravel(),
            state["node_half"],
            state["node_children"].ravel(),
            state["node_is_leaf"],
            state["node_point_count"],
            state["node_active_count"],
            state["sorted_point_active"],
            state["node_total_mass"].ravel(),
            state["node_total_dipole"].ravel(),
            state["node_active_mass"].ravel(),
            state["node_active_dipole"].ravel(),
            state["lambda_centers"],
            np.int32(n_bins),
            np.int32(gl_order),
            np.int32(int(np.asarray(context["exclude_idx"]).shape[1])),
            np.int32(int(treecode["node_count"])),
            np.int32(target_count),
            np.int32(int(treecode.get("order_mode", 0))),
            np.float32(float(treecode.get("theta", CEXT_TREECODE_THETA))),
            np.float32(float(treecode.get("near_radius_si", 0.0))),
            np.float32(float(context["diffusivity_si"])),
            np.float32(float(CEXT_WINDOW_FACTOR)),
            np.float32(global_cap),
            out_total.ravel(),
            state["metrics"],
        ),
    )
    _cp.cuda.Stream.null.synchronize()
    t_traversal = perf_counter() - t_traversal
    t_download = perf_counter()
    c_ext_new = _cp.asnumpy(out_total)
    metrics = _cp.asnumpy(state["metrics"])
    _cp.cuda.Stream.null.synchronize()
    t_download = perf_counter() - t_download
    return np.asarray(c_ext_new, dtype=np.float32), {
        "aggregate_s": float(t_aggregate),
        "traversal_s": float(t_traversal),
        "transfer_s": float(t_download),
        "exact_s": 0.0,
        "accepted_nodes": int(metrics[0]) if metrics.size > 0 else 0,
        "opened_nodes": int(metrics[1]) if metrics.size > 1 else 0,
        "exact_pairs": int(metrics[2]) if metrics.size > 2 else 0,
    }


def _estimate_cext_tail_active_core(context: dict, ext_state: dict, mapped_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(ext_state["c_ext_gl"], dtype=np.float32)
    mapped = np.asarray(mapped_gl, dtype=np.float32)
    q_weighted = np.asarray(ext_state.get("q_weighted_gl", np.zeros_like(mapped)), dtype=np.float32)
    if current.shape != mapped.shape:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    delta_seg = np.max(np.abs(mapped - current), axis=1).astype(np.float32, copy=False)
    q_seg = np.max(np.abs(q_weighted), axis=1).astype(np.float32, copy=False) if q_weighted.ndim == 2 else np.zeros((current.shape[0],), dtype=np.float32)
    bound = np.maximum(delta_seg, np.sqrt(np.maximum(q_seg, np.float32(0.0)))).astype(np.float32, copy=False)
    scale = max(float(np.max(np.abs(np.asarray(mapped, dtype=float)))) if mapped.size else 0.0, float(VESS_CONC_FLOOR))
    thresh = max(float(CEXT_TAIL_CORE_ABS_THRESHOLD), float(CEXT_TAIL_CORE_REL_THRESHOLD) * scale)
    core = np.flatnonzero(bound > thresh).astype(np.int32, copy=False)
    if core.size > int(CEXT_TAIL_TRIGGER_ACTIVE_COUNT):
        order = np.argsort(bound[core], kind="stable")[::-1][: int(CEXT_TAIL_TRIGGER_ACTIVE_COUNT)]
        core = np.asarray(core[order], dtype=np.int32)
    return core, bound


def _cext_should_enter_tail_mode(residual_history: list[float], core_size: int, *, iter_idx: int) -> bool:
    if str(CEXT_TAIL_SOLVER).strip().lower() == "none":
        return False
    if int(iter_idx) < int(CEXT_TAIL_TRIGGER_START_ITER):
        return False
    if core_size <= 0 or core_size > int(CEXT_TAIL_TRIGGER_ACTIVE_COUNT):
        return False
    stall_iters = max(int(CEXT_TAIL_TRIGGER_STALL_ITERS), 2)
    if len(residual_history) < stall_iters:
        return False
    recent = np.asarray(residual_history[-stall_iters:], dtype=float)
    best = float(np.min(recent))
    first = float(recent[0])
    if np.isfinite(first) and np.isfinite(best) and best > float(CEXT_TAIL_TRIGGER_IMPROVEMENT_RATIO) * max(first, 1.0e-30):
        return True
    if np.any(recent < float(CEXT_TAIL_REBOUND_ARM_REL)) and recent[-1] > float(CEXT_TAIL_REBOUND_REL):
        return True
    return False


def _evaluate_cext_treecode_map(
    context: dict,
    treecode: dict,
    ext_state_base: dict,
    *,
    c_ext_candidate: np.ndarray,
    inlet_concentration: float,
    vmax: float,
    km: float,
    chb_max: np.ndarray,
    fluid_mode: str,
    frozen_backend: str,
) -> tuple[np.ndarray, dict, str]:
    ext_trial = {
        "c_ext_gl": np.asarray(c_ext_candidate, dtype=np.float32).copy(),
        "c_iv_gl": np.asarray(ext_state_base["c_iv_gl"], dtype=np.float32).copy(),
        "cin_seg": np.asarray(ext_state_base["cin_seg"], dtype=np.float32).copy(),
        "cout_seg": np.asarray(ext_state_base["cout_seg"], dtype=np.float32).copy(),
        "vmax": float(vmax),
        "km": float(km),
        "window_factor": float(CEXT_WINDOW_FACTOR),
    }
    cin_seg, cout_seg, c_iv_gl, frozen_backend, _ = _run_topdown_ext_frozen_step(
        context,
        ext_trial,
        inlet_concentration=float(inlet_concentration),
        vmax=float(vmax),
        km=float(km),
        chb_max=np.asarray(chb_max, dtype=np.float32),
        fluid_mode=fluid_mode,
        frozen_backend=frozen_backend,
    )
    ext_trial["cin_seg"] = np.asarray(cin_seg, dtype=np.float32)
    ext_trial["cout_seg"] = np.asarray(cout_seg, dtype=np.float32)
    ext_trial["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    _build_cext_iteration_cache(context, ext_trial)
    mapped_gl, _ = _compute_cext_treecode_gpu(context, treecode, ext_trial)
    return np.asarray(mapped_gl, dtype=np.float32), ext_trial, frozen_backend


def _solve_cext_treecode_active_core_nk(
    context: dict,
    treecode: dict,
    ext_state: dict,
    *,
    core_seg_ids: np.ndarray,
    inlet_concentration: float,
    vmax: float,
    km: float,
    chb_max: np.ndarray,
    fluid_mode: str,
    frozen_backend: str,
) -> tuple[np.ndarray, dict]:
    core_ids = np.asarray(core_seg_ids, dtype=np.int32).reshape(-1)
    gl_order = int(np.asarray(ext_state["c_ext_gl"]).shape[1])
    x = np.asarray(ext_state["c_ext_gl"][core_ids], dtype=np.float32).reshape(-1).copy()
    base_full = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).copy()
    gmres_iters_total = 0
    nonlinear_iters = 0
    success = False
    last_rel = float("inf")

    def build_full(active_flat: np.ndarray) -> np.ndarray:
        full = np.asarray(base_full, dtype=np.float32).copy()
        full[core_ids] = np.asarray(active_flat, dtype=np.float32).reshape((-1, gl_order))
        return full

    def residual(active_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, str]:
        full = build_full(active_flat)
        mapped_gl, trial_state, backend_out = _evaluate_cext_treecode_map(
            context,
            treecode,
            ext_state,
            c_ext_candidate=full,
            inlet_concentration=inlet_concentration,
            vmax=vmax,
            km=km,
            chb_max=chb_max,
            fluid_mode=fluid_mode,
            frozen_backend=frozen_backend,
        )
        return (
            np.asarray(mapped_gl[core_ids], dtype=np.float32).reshape(-1) - np.asarray(active_flat, dtype=np.float32).reshape(-1),
            mapped_gl,
            trial_state,
            backend_out,
        )

    mapped_last = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).copy()
    state_last = ext_state
    backend_last = frozen_backend
    if not _HAVE_SCIPY_SPARSE or _splinalg is None:
        for _ in range(max(int(CEXT_TAIL_MAX_NONLINEAR_ITERS), 1)):
            nonlinear_iters += 1
            res, mapped_last, state_last, backend_last = residual(x)
            last_rel = float(np.max(np.abs(res))) / max(float(np.max(np.abs(mapped_last[core_ids]))) if core_ids.size else 0.0, float(VESS_CONC_FLOOR))
            x = np.asarray(x + np.float32(0.5) * res, dtype=np.float32)
            if last_rel <= max(float(CEXT_VESS_COUPLING_REL_TOL), 1.0e-3):
                success = True
                break
    else:
        for _ in range(max(int(CEXT_TAIL_MAX_NONLINEAR_ITERS), 1)):
            nonlinear_iters += 1
            res, mapped_last, state_last, backend_last = residual(x)
            res_norm = float(np.max(np.abs(res))) if res.size else 0.0
            last_rel = res_norm / max(float(np.max(np.abs(mapped_last[core_ids]))) if core_ids.size else 0.0, float(VESS_CONC_FLOOR))
            if last_rel <= max(float(CEXT_VESS_COUPLING_REL_TOL), 1.0e-3):
                success = True
                break
            eps_base = 1.0e-4

            def mv(v: np.ndarray) -> np.ndarray:
                v_arr = np.asarray(v, dtype=np.float32).reshape(-1)
                scale = float(np.linalg.norm(v_arr))
                if scale <= 1.0e-30:
                    return -v_arr
                eps = np.float32(eps_base / scale)
                res_p, _, _, _ = residual(np.asarray(x + eps * v_arr, dtype=np.float32))
                return np.asarray((res_p - res) / eps, dtype=np.float32)

            linop = _splinalg.LinearOperator((x.size, x.size), matvec=mv, dtype=np.float32)
            gmres_counter = {"n": 0}

            def gmres_cb(_rk=None):
                gmres_counter["n"] += 1

            step, info = _splinalg.gmres(
                linop,
                -np.asarray(res, dtype=np.float32),
                restart=max(int(CEXT_TAIL_GMRES_RESTART), 4),
                maxiter=max(int(CEXT_TAIL_GMRES_MAXITER), 8),
                callback=gmres_cb,
            )
            gmres_iters_total += int(gmres_counter["n"])
            if info != 0 or not np.all(np.isfinite(step)):
                x = np.asarray(x + np.float32(0.5) * res, dtype=np.float32)
                continue
            alpha = 1.0
            accepted = False
            for _ls in range(8):
                trial_x = np.asarray(x + np.float32(alpha) * np.asarray(step, dtype=np.float32), dtype=np.float32)
                res_trial, mapped_trial, state_trial, backend_trial = residual(trial_x)
                trial_norm = float(np.max(np.abs(res_trial))) if res_trial.size else 0.0
                if trial_norm < res_norm:
                    x = np.asarray(trial_x, dtype=np.float32)
                    mapped_last = np.asarray(mapped_trial, dtype=np.float32)
                    state_last = state_trial
                    backend_last = backend_trial
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                x = np.asarray(x + np.float32(0.5) * res, dtype=np.float32)
        if not success:
            res, mapped_last, state_last, backend_last = residual(x)
            last_rel = float(np.max(np.abs(res))) / max(float(np.max(np.abs(mapped_last[core_ids]))) if core_ids.size else 0.0, float(VESS_CONC_FLOOR))
            success = last_rel <= max(float(CEXT_VESS_COUPLING_REL_TOL), 1.0e-3)
    full_out = np.asarray(base_full, dtype=np.float32).copy()
    full_out[core_ids] = np.asarray(x, dtype=np.float32).reshape((-1, gl_order))
    return full_out, {
        "success": bool(success),
        "rel_residual": float(last_rel),
        "gmres_iters": int(gmres_iters_total),
        "nonlinear_iters": int(nonlinear_iters),
        "mapped_gl": np.asarray(mapped_last, dtype=np.float32),
        "state": state_last,
        "frozen_backend": backend_last,
    }


def _rebuild_cext_hybrid_frozen_cache(
    context: dict,
    ext_state: dict,
    hybrid: dict,
    frozen_source_mask: np.ndarray,
    *,
    runtime_state: dict | None = None,
) -> tuple[object, object, dict[str, float]]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    state = runtime_state if isinstance(runtime_state, dict) else _ensure_cext_hybrid_bg_runtime_state(context, hybrid, ext_state)
    mass_grids_g, t_deposit = _cext_hybrid_deposit_sources_gpu(
        context,
        hybrid,
        ext_state,
        source_mask=frozen_source_mask,
        runtime_state=state,
    )
    phi_grids_g, t_fft, bg_solver_mode = _cext_hybrid_solve_background_fft_gpu(
        context,
        hybrid,
        mass_grids_g,
        init_phi_grids_g=hybrid.get("frozen_phi_guess_g"),
    )
    hybrid["frozen_phi_guess_g"] = phi_grids_g
    return mass_grids_g, phi_grids_g, {
        "deposit_s": float(t_deposit),
        "fft_s": float(t_fft),
        "bg_solver_mode": str(bg_solver_mode),
    }


def _validate_cext_gpu_batch(
    context: dict,
    ext_state: dict,
    target_seg_ids: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    gpu_out: np.ndarray,
) -> None:
    validate_n = min(int(CEXT_GPU_VALIDATE_SEGMENTS), int(target_seg_ids.size))
    if validate_n <= 0:
        return
    if row_ptr.size <= 1:
        return
    cpu_row_end = int(row_ptr[validate_n])
    cpu_out = _compute_cext_batch_cpu(
        context,
        ext_state,
        np.asarray(target_seg_ids[:validate_n], dtype=np.int32),
        np.asarray(row_ptr[: validate_n + 1], dtype=np.int32),
        np.asarray(col_idx[:cpu_row_end], dtype=np.int32),
    )
    gpu_slice = np.asarray(gpu_out[:validate_n], dtype=float)
    finite = np.isfinite(cpu_out) & np.isfinite(gpu_slice)
    max_abs = float(np.max(np.abs(np.asarray(cpu_out, dtype=float)[finite] - gpu_slice[finite]))) if np.any(finite) else float("nan")
    rel_l2 = _relative_l2(np.asarray(cpu_out, dtype=float)[finite], gpu_slice[finite]) if np.any(finite) else float("nan")
    print(
        "  Cext GPU validate: "
        f"segments={validate_n} max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}"
    )


def _default_cext_timing_details(*, backend: str = "none") -> dict[str, float | str | int | bool]:
    return {
        "backend": backend,
        "t_cext_total_s": 0.0,
        "t_cext_hct_setup_s": 0.0,
        "t_cext_context_build_s": 0.0,
        "t_cext_solver_setup_s": 0.0,
        "t_cext_query_s": 0.0,
        "t_cext_kernel_s": 0.0,
        "t_cext_gpu_transfer_s": 0.0,
        "t_cext_frozen_gpu_transfer_s": 0.0,
        "t_cext_init_s": 0.0,
        "t_cext_init_kernel_s": 0.0,
        "t_cext_init_gpu_transfer_s": 0.0,
        "t_cext_init_frozen_gpu_transfer_s": 0.0,
        "t_cext_init_frozen_kernel_s": 0.0,
        "t_cext_init_cache_s": 0.0,
        "t_cext_init_hybrid_call_s": 0.0,
        "t_cext_final_frozen_gpu_transfer_s": 0.0,
        "t_cext_final_frozen_kernel_s": 0.0,
        "t_cext_final_cache_s": 0.0,
        "t_cext_final_source_state_s": 0.0,
        "t_cext_hybrid_deposit_s": 0.0,
        "t_cext_hybrid_fft_s": 0.0,
        "t_cext_hybrid_o2_terms_s": 0.0,
        "t_cext_hybrid_self_subtract_s": 0.0,
        "t_cext_hybrid_local_corr_s": 0.0,
        "t_cext_hybrid_sample_s": 0.0,
        "t_cext_treecode_build_s": 0.0,
        "t_cext_treecode_aggregate_s": 0.0,
        "t_cext_treecode_traversal_s": 0.0,
        "t_cext_treecode_exact_s": 0.0,
        "cext_hybrid_bg_solver": "",
        "cext_treecode_theta": 0.0,
        "cext_treecode_order": 0,
        "cext_treecode_leaf_nodes": 0,
        "cext_treecode_lambda_bins": 0,
        "cext_treecode_near_radius_si": 0.0,
        "cext_treecode_node_count": 0,
        "cext_treecode_leaf_count": 0,
        "cext_treecode_max_depth": 0,
        "cext_treecode_accepted_nodes": 0,
        "cext_treecode_opened_nodes": 0,
        "cext_treecode_exact_pairs": 0,
        "cext_init_mode": str(CEXT_INIT_MODE).lower(),
        "cext_init_performed": False,
        "cext_outer_iterations_completed": 0,
        "cext_total_iterations_effective": 0,
        "cext_accel_mode": str(CEXT_VESS_COUPLING_ACCEL).lower(),
        "cext_accel_step_last": "picard",
        "cext_accel_rejections": 0,
        "cext_accel_restarts": 0,
        "cext_omega_last": float(CEXT_VESS_COUPLING_OMEGA),
        "cext_rel_residual_last": 0.0,
        "cext_max_delta_last": 0.0,
        "cext_best_max_delta": 0.0,
        "cext_best_seen_stop": False,
        "cext_step_rejections": 0,
        "cext_candidate_batches": 0,
        "cext_frozen_backend": "none",
        "cext_active_source_count": 0,
        "cext_frozen_source_count": 0,
        "cext_active_target_count": 0,
        "cext_frozen_target_count": 0,
        "cext_active_component_count": 0,
        "cext_largest_component_size": 0,
        "cext_active_freeze_events": 0,
        "cext_active_refreshes": 0,
        "cext_target_freeze_events": 0,
        "cext_target_reactivations": 0,
        "cext_hybrid_bg_grid": 0,
        "cext_hybrid_lambda_bins": 0,
        "cext_hybrid_lambda_bin_policy": "",
        "cext_hybrid_lambda_bin_edges_hash": "",
        "cext_hybrid_near_radius_si": 0.0,
        "cext_tail_mode_entered": False,
        "cext_tail_solver": "none",
        "cext_tail_core_count": 0,
        "cext_tail_nonlinear_iters": 0,
        "cext_tail_gmres_iters": 0,
    }


def _maybe_trace_cext_iteration(
    iteration: int,
    ext_state: dict,
    *,
    solver: str,
    context: dict | None = None,
    backend: str | None = None,
    max_delta: float | None = None,
    rel_residual: float | None = None,
) -> None:
    callback = CEXT_TRACE_CALLBACK
    if callback is None:
        return
    iteration_i = int(iteration)
    trace_iterations = CEXT_TRACE_ITERATIONS
    if trace_iterations is not None and iteration_i not in trace_iterations:
        return
    if context is not None and "c_ext_gl" in ext_state and "c_wall_gl" in ext_state:
        _build_cext_iteration_cache(context, ext_state)
    c_ext_gl = ext_state.get("c_ext_gl")
    if c_ext_gl is None:
        return
    c_ext_arr = np.asarray(c_ext_gl, dtype=np.float32).copy()

    def _trace_array(name: str) -> np.ndarray:
        value = ext_state.get(name)
        if value is None:
            return np.full_like(c_ext_arr, np.nan, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != c_ext_arr.shape:
            return np.full_like(c_ext_arr, np.nan, dtype=np.float32)
        return arr.copy()

    tissue_cx = np.empty((0,), dtype=np.float32)
    if bool(CEXT_TRACE_TISSUE_ENABLED) and context is not None:
        points = CEXT_TRACE_TISSUE_POINTS
        if points is not None:
            points_arr = np.asarray(points, dtype=float)
            if points_arr.size:
                cext_state = _snapshot_cext_source_state(
                    context,
                    ext_state,
                    solver=str(solver),
                    backend=str(backend or ext_state.get("backend", "")),
                )
                starts_cm = np.asarray(context["flow_starts_si"], dtype=float) / CM_TO_M
                ends_cm = np.asarray(context["flow_ends_si"], dtype=float) / CM_TO_M
                radii_cm = np.asarray(context["radii_si"], dtype=float) / CM_TO_M
                mask, values = compute_tissue_samples_greens_from_cext_state(
                    points_arr,
                    starts_cm,
                    ends_cm,
                    radii_cm,
                    cext_state,
                    tissue_cache=CEXT_TRACE_TISSUE_CACHE,
                )
                tissue_cx = np.asarray(values[np.asarray(mask, dtype=bool)], dtype=np.float32).reshape(-1)

    callback(
        iteration=iteration_i,
        c_ext_gl=c_ext_arr,
        c_iv_gl=_trace_array("c_iv_gl"),
        c_bulk_gl=_trace_array("c_bulk_gl"),
        c_wall_gl=_trace_array("c_wall_gl"),
        lambda_iv_gl=_trace_array("lambda_iv_gl"),
        k_if_gl=_trace_array("k_if_gl"),
        q_line_gl=_trace_array("q_line_gl"),
        tissue_cx=tissue_cx,
        solver=str(solver),
        fluid=str(ACTIVE_FLUID),
        max_delta=None if max_delta is None else float(max_delta),
        rel_residual=None if rel_residual is None else float(rel_residual),
    )


def _cext_relaxation_bounds() -> tuple[float, float]:
    omega_min = max(float(CEXT_VESS_COUPLING_OMEGA_MIN), 0.0)
    omega_max = max(float(CEXT_VESS_COUPLING_OMEGA_MAX), omega_min)
    return omega_min, omega_max


def _clip_cext_omega(value: float, *, fallback: float | None = None) -> float:
    omega_min, omega_max = _cext_relaxation_bounds()
    candidate = float(value)
    if not np.isfinite(candidate):
        candidate = float(CEXT_VESS_COUPLING_OMEGA if fallback is None else fallback)
    return float(np.clip(candidate, omega_min, omega_max))


def _cext_residual_metrics(
    current: np.ndarray,
    mapped: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    residual = np.asarray(mapped, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    abs_inf = float(np.nanmax(np.abs(np.asarray(residual, dtype=float)))) if residual.size else 0.0
    scale = float(np.nanmax(np.abs(np.asarray(mapped, dtype=float)))) if mapped.size else float(VESS_CONC_FLOOR)
    scale = max(scale, float(VESS_CONC_FLOOR))
    rel_inf = abs_inf / scale if scale > 0.0 else 0.0
    return np.asarray(residual, dtype=np.float32), abs_inf, float(rel_inf)


def _cext_source_state_delta_metrics(
    prev_c_ext_gl: np.ndarray | None,
    prev_c_iv_gl: np.ndarray | None,
    curr_c_ext_gl: np.ndarray,
    curr_c_iv_gl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    curr_c_ext = np.asarray(curr_c_ext_gl, dtype=np.float32)
    curr_c_iv = np.asarray(curr_c_iv_gl, dtype=np.float32)
    nseg = int(curr_c_ext.shape[0])
    if (
        prev_c_ext_gl is None
        or prev_c_iv_gl is None
        or np.asarray(prev_c_ext_gl).shape != curr_c_ext.shape
        or np.asarray(prev_c_iv_gl).shape != curr_c_iv.shape
    ):
        return np.full((nseg,), np.inf, dtype=np.float32), np.full((nseg,), np.inf, dtype=np.float32)
    prev_c_ext = np.asarray(prev_c_ext_gl, dtype=np.float32)
    prev_c_iv = np.asarray(prev_c_iv_gl, dtype=np.float32)
    abs_delta = np.maximum(
        np.max(np.abs(curr_c_ext - prev_c_ext), axis=1),
        np.max(np.abs(curr_c_iv - prev_c_iv), axis=1),
    ).astype(np.float32, copy=False)
    scale = np.maximum.reduce(
        [
            np.max(np.abs(curr_c_ext), axis=1),
            np.max(np.abs(prev_c_ext), axis=1),
            np.max(np.abs(curr_c_iv), axis=1),
            np.max(np.abs(prev_c_iv), axis=1),
            np.full((nseg,), float(VESS_CONC_FLOOR), dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    rel_delta = np.divide(
        abs_delta,
        np.maximum(scale, np.float32(VESS_CONC_FLOOR)),
        out=np.full_like(abs_delta, np.inf, dtype=np.float32),
        where=np.maximum(scale, np.float32(VESS_CONC_FLOOR)) > 0.0,
    )
    return abs_delta, rel_delta


def _cext_active_source_cell_stats(
    context: dict,
    source_mask: np.ndarray,
    source_abs_delta: np.ndarray,
    source_rel_delta: np.ndarray,
) -> dict:
    n_cells = int(context.get("gpu_direct_n_cells", 0))
    source_mask_arr = np.asarray(source_mask, dtype=bool).reshape(-1)
    home_flat = np.asarray(context.get("gpu_direct_home_cell_flat", np.zeros((source_mask_arr.size,), dtype=np.int64)), dtype=np.int64)
    active_ids = np.flatnonzero(source_mask_arr).astype(np.int32, copy=False)
    cell_counts = np.zeros((max(n_cells, 0),), dtype=np.int32)
    cell_abs_max = np.zeros((max(n_cells, 0),), dtype=np.float32)
    cell_rel_max = np.zeros((max(n_cells, 0),), dtype=np.float32)
    max_active_reach = 0.0
    if active_ids.size > 0 and n_cells > 0:
        active_flat = np.asarray(home_flat[active_ids], dtype=np.int64)
        counts = np.bincount(active_flat, minlength=n_cells)
        cell_counts = np.asarray(counts, dtype=np.int32)
        np.maximum.at(cell_abs_max, active_flat, np.asarray(source_abs_delta[active_ids], dtype=np.float32))
        np.maximum.at(cell_rel_max, active_flat, np.asarray(source_rel_delta[active_ids], dtype=np.float32))
        reach_si = np.asarray(context.get("reach_si", np.zeros((source_mask_arr.size,), dtype=np.float32)), dtype=np.float32)
        max_active_reach = float(np.max(np.asarray(reach_si[active_ids], dtype=np.float32))) if active_ids.size else 0.0
    return {
        "active_ids": active_ids,
        "cell_counts": cell_counts,
        "cell_abs_max": cell_abs_max,
        "cell_rel_max": cell_rel_max,
        "max_active_reach": float(max_active_reach),
    }


def _cext_target_query_cells(
    context: dict,
    target_seg_ids: np.ndarray,
    *,
    max_active_reach: float,
    neighbor_pad: int,
) -> list[np.ndarray]:
    target_ids = np.asarray(target_seg_ids, dtype=np.int32).reshape(-1)
    if target_ids.size == 0:
        return []
    cell_size = float(context.get("cell_size", 0.0))
    dims = np.asarray(context.get("gpu_direct_cell_dims", np.zeros((3,), dtype=np.int32)), dtype=np.int32)
    origin = np.asarray(context.get("gpu_direct_cell_origin", np.zeros((3,), dtype=np.int32)), dtype=np.int32)
    midpoints = np.asarray(context.get("midpoints_si", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    reach_si = np.asarray(context.get("reach_si", np.zeros((midpoints.shape[0],), dtype=np.float32)), dtype=np.float32)
    if target_ids.size == 0 or cell_size <= 0.0 or dims.size < 3 or int(np.prod(dims)) <= 0:
        return [np.zeros((0,), dtype=np.int32) for _ in range(target_ids.size)]
    reach_scale = max(float(CEXT_APPROX_WINDOW_SCALE), 0.0)
    out: list[np.ndarray] = []
    pad = max(int(neighbor_pad), 0)
    dim_x = int(dims[0])
    dim_y = int(dims[1])
    dim_z = int(dims[2])
    for target_seg in target_ids:
        mid = np.asarray(midpoints[int(target_seg)], dtype=float)
        query_reach = reach_scale * float(reach_si[int(target_seg)] + max_active_reach)
        lo = np.floor((mid - query_reach) / cell_size).astype(np.int64) - origin.astype(np.int64) - pad
        hi = np.floor((mid + query_reach) / cell_size).astype(np.int64) - origin.astype(np.int64) + pad
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray([dim_x - 1, dim_y - 1, dim_z - 1], dtype=np.int64))
        cells: list[int] = []
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                base = dim_x * iy
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    flat = ix + dim_x * (iy + dim_y * iz)
                    cells.append(int(flat))
        out.append(np.asarray(cells, dtype=np.int32))
    return out


def _cext_target_neighbor_metrics(
    context: dict,
    target_seg_ids: np.ndarray,
    cell_stats: dict,
    *,
    neighbor_pad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    target_ids = np.asarray(target_seg_ids, dtype=np.int32).reshape(-1)
    if target_ids.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty.astype(np.int32), empty, empty, []
    query_cells = _cext_target_query_cells(
        context,
        target_ids,
        max_active_reach=float(cell_stats.get("max_active_reach", 0.0)),
        neighbor_pad=neighbor_pad,
    )
    cell_counts = np.asarray(cell_stats.get("cell_counts", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    cell_abs_max = np.asarray(cell_stats.get("cell_abs_max", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    cell_rel_max = np.asarray(cell_stats.get("cell_rel_max", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    neighbor_counts = np.zeros((target_ids.size,), dtype=np.int32)
    neighbor_abs = np.zeros((target_ids.size,), dtype=np.float32)
    neighbor_rel = np.zeros((target_ids.size,), dtype=np.float32)
    for idx, cells in enumerate(query_cells):
        if cells.size <= 0:
            continue
        valid_cells = cells[(cells >= 0) & (cells < cell_counts.size)]
        if valid_cells.size <= 0:
            continue
        neighbor_counts[idx] = int(np.sum(cell_counts[valid_cells], dtype=np.int64))
        neighbor_abs[idx] = float(np.max(cell_abs_max[valid_cells])) if valid_cells.size else 0.0
        neighbor_rel[idx] = float(np.max(cell_rel_max[valid_cells])) if valid_cells.size else 0.0
    return neighbor_counts, neighbor_abs, neighbor_rel, query_cells


def _cext_hybrid_local_target_ids(
    context: dict,
    hybrid: dict,
    active_target_ids: np.ndarray,
    active_source_mask: np.ndarray,
) -> np.ndarray:
    target_ids = np.asarray(active_target_ids, dtype=np.int32).reshape(-1)
    if target_ids.size <= 0:
        return np.zeros((0,), dtype=np.int32)
    node_seg_ids = np.asarray(hybrid.get("node_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    node_cell_flat = np.asarray(hybrid.get("local_node_cell_flat", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    if node_seg_ids.size <= 0 or node_cell_flat.size != node_seg_ids.size:
        return target_ids
    source_mask = np.asarray(active_source_mask, dtype=bool).reshape(-1)
    if source_mask.size <= 0 or int(np.count_nonzero(source_mask)) >= int(source_mask.size):
        return target_ids
    grid_n = int(hybrid.get("grid_n", 0))
    grid_cells = int(grid_n * grid_n * grid_n)
    if grid_n <= 0 or grid_cells <= 0:
        return target_ids
    active_nodes = source_mask[np.clip(node_seg_ids, 0, max(source_mask.size - 1, 0))]
    active_cells = node_cell_flat[np.asarray(active_nodes, dtype=bool)]
    active_cells = active_cells[(active_cells >= 0) & (active_cells < grid_cells)]
    if active_cells.size <= 0:
        return np.zeros((0,), dtype=np.int32)
    active_grid = np.zeros((grid_cells,), dtype=bool)
    active_grid[np.asarray(active_cells, dtype=np.int64)] = True
    active_grid = active_grid.reshape((grid_n, grid_n, grid_n))
    rad_cells = int(max(int(hybrid.get("local_rad_cells", 0)), 0))
    if rad_cells > 0:
        if _HAVE_SCIPY_NDIMAGE and _scipy_ndimage is not None:
            structure = np.ones((2 * rad_cells + 1, 2 * rad_cells + 1, 2 * rad_cells + 1), dtype=bool)
            active_grid = _scipy_ndimage.binary_dilation(active_grid, structure=structure, border_value=False)
        else:
            expanded = np.zeros_like(active_grid, dtype=bool)
            active_idx = np.argwhere(active_grid)
            for ix, iy, iz in active_idx:
                lo_x = max(int(ix) - rad_cells, 0)
                hi_x = min(int(ix) + rad_cells + 1, grid_n)
                lo_y = max(int(iy) - rad_cells, 0)
                hi_y = min(int(iy) + rad_cells + 1, grid_n)
                lo_z = max(int(iz) - rad_cells, 0)
                hi_z = min(int(iz) + rad_cells + 1, grid_n)
                expanded[lo_x:hi_x, lo_y:hi_y, lo_z:hi_z] = True
            active_grid = expanded
    gl_order = int(np.asarray(context.get("gl_points_si", np.zeros((0, 0, 3), dtype=np.float32))).shape[1])
    if gl_order <= 0:
        return target_ids
    target_node_ids = (target_ids[:, None].astype(np.int64) * np.int64(gl_order) + np.arange(gl_order, dtype=np.int64)[None, :]).reshape(-1)
    valid_nodes = (target_node_ids >= 0) & (target_node_ids < node_cell_flat.size)
    target_node_cells = np.full((target_node_ids.size,), -1, dtype=np.int32)
    target_node_cells[valid_nodes] = node_cell_flat[target_node_ids[valid_nodes]]
    target_active = np.zeros((target_node_ids.size,), dtype=bool)
    valid_cells = (target_node_cells >= 0) & (target_node_cells < grid_cells)
    if np.any(valid_cells):
        target_active[valid_cells] = active_grid.reshape(-1)[target_node_cells[valid_cells]]
    target_active = target_active.reshape((target_ids.size, gl_order))
    return np.asarray(target_ids[np.any(target_active, axis=1)], dtype=np.int32)


def _cext_build_active_components_from_cells(
    target_seg_ids: np.ndarray,
    query_cells: list[np.ndarray],
    cell_stats: dict,
) -> tuple[list[np.ndarray], int]:
    target_ids = np.asarray(target_seg_ids, dtype=np.int32).reshape(-1)
    ntarget = int(target_ids.size)
    if ntarget <= 0:
        return [], 0
    if ntarget == 1:
        return [np.asarray(target_ids, dtype=np.int32)], 1
    cell_counts = np.asarray(cell_stats.get("cell_counts", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    parent = np.arange(ntarget, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    cell_to_targets: dict[int, int] = {}
    for local_idx, cells in enumerate(query_cells):
        if cells.size <= 0:
            continue
        valid_cells = cells[(cells >= 0) & (cells < cell_counts.size)]
        for cell in np.unique(valid_cells):
            if int(cell_counts[int(cell)]) <= 0:
                continue
            prev = cell_to_targets.get(int(cell))
            if prev is None:
                cell_to_targets[int(cell)] = int(local_idx)
            else:
                union(int(prev), int(local_idx))

    groups: dict[int, list[int]] = defaultdict(list)
    for local_idx in range(ntarget):
        groups[find(local_idx)].append(local_idx)
    components = [np.asarray(target_ids[np.asarray(local_ids, dtype=np.int32)], dtype=np.int32) for local_ids in groups.values()]
    components.sort(key=lambda arr: int(arr.size), reverse=True)
    largest = int(components[0].size) if components else 0
    return components, largest


def _cext_component_source_ids_from_cells(
    context: dict,
    active_source_mask: np.ndarray,
    component_cells: Sequence[np.ndarray],
) -> np.ndarray:
    cell_ptr = np.asarray(context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)), dtype=np.int32)
    cell_seg_ids = np.asarray(context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    active_mask = np.asarray(active_source_mask, dtype=bool).reshape(-1)
    if cell_ptr.size <= 1 or cell_seg_ids.size <= 0 or not np.any(active_mask):
        return np.zeros((0,), dtype=np.int32)
    source_ids: list[np.ndarray] = []
    for cells in component_cells:
        valid_cells = np.asarray(cells, dtype=np.int32)
        valid_cells = valid_cells[(valid_cells >= 0) & (valid_cells + 1 < cell_ptr.size)]
        if valid_cells.size <= 0:
            continue
        for cell in np.unique(valid_cells):
            start = int(cell_ptr[int(cell)])
            stop = int(cell_ptr[int(cell) + 1])
            if stop <= start:
                continue
            segs = np.asarray(cell_seg_ids[start:stop], dtype=np.int32)
            if segs.size > 0:
                source_ids.append(segs[np.asarray(active_mask[segs], dtype=bool)])
    if not source_ids:
        return np.zeros((0,), dtype=np.int32)
    return np.unique(np.concatenate(source_ids).astype(np.int32, copy=False))


def _cext_adaptive_aitken_omega(
    prev_residual: np.ndarray | None,
    curr_residual: np.ndarray,
    prev_omega: float,
) -> float:
    omega = _clip_cext_omega(prev_omega)
    if prev_residual is None or np.asarray(prev_residual).shape != np.asarray(curr_residual).shape:
        return omega
    delta = np.asarray(curr_residual, dtype=np.float32) - np.asarray(prev_residual, dtype=np.float32)
    denom = float(np.dot(delta, delta))
    if not np.isfinite(denom) or denom <= 1.0e-30:
        return omega
    numer = float(np.dot(np.asarray(prev_residual, dtype=np.float32), delta))
    return _clip_cext_omega(-omega * numer / denom, fallback=omega)


def _cext_project_candidate(candidate: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(candidate, dtype=np.float32)
    if arr.size == 0:
        return np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return None
    return np.maximum(arr, np.float32(0.0)).astype(np.float32, copy=False)


def _cext_apply_trust_region(
    current: np.ndarray,
    candidate: np.ndarray,
    mapped: np.ndarray,
) -> np.ndarray | None:
    projected = _cext_project_candidate(candidate)
    if projected is None:
        return None
    trust_abs = max(float(CEXT_VESS_COUPLING_TRUST_ABS), 0.0)
    trust_rel = max(float(CEXT_VESS_COUPLING_TRUST_REL), 0.0)
    if trust_abs <= 0.0 and trust_rel <= 0.0:
        return projected
    current_arr = np.asarray(current, dtype=np.float32)
    mapped_arr = np.asarray(mapped, dtype=np.float32)
    step = np.asarray(projected, dtype=np.float32) - current_arr
    scale = np.maximum(
        np.maximum(np.abs(current_arr), np.abs(mapped_arr)),
        np.float32(VESS_CONC_FLOOR),
    )
    cap = np.float32(trust_abs) + np.float32(trust_rel) * scale
    bounded = current_arr + np.clip(step, -cap, cap).astype(np.float32, copy=False)
    return _cext_project_candidate(bounded)


def _cext_should_enable_anderson(
    *,
    iter_idx: int,
    stable_iters: int,
    prev_abs_residual: float,
    curr_abs_residual: float,
    omega_fallback: float,
) -> bool:
    if iter_idx < max(int(CEXT_VESS_COUPLING_ANDERSON_START), 1):
        return False
    if stable_iters < max(int(CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS), 0):
        return False
    if not np.isfinite(prev_abs_residual) or prev_abs_residual <= 0.0:
        return False
    ratio = float(curr_abs_residual) / max(float(prev_abs_residual), 1.0e-30)
    if ratio > max(float(CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO), 0.0):
        return False
    omega_min, _ = _cext_relaxation_bounds()
    if float(omega_fallback) <= omega_min * max(float(CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR), 1.0):
        return False
    return True


def _cext_apply_component_acceleration(
    current_flat: np.ndarray,
    mapped_flat: np.ndarray,
    state: dict | None,
    *,
    accel_mode: str,
) -> tuple[np.ndarray, dict, dict]:
    current_arr = np.asarray(current_flat, dtype=np.float32).reshape(-1)
    mapped_arr = np.asarray(mapped_flat, dtype=np.float32).reshape(-1)
    if state is None:
        state = {}
    residual_flat, max_delta_last, rel_residual_last = _cext_residual_metrics(current_arr, mapped_arr)
    prev_abs_residual = float(state.get("prev_abs_residual", float("inf")))
    prev_residual_flat = state.get("prev_residual_flat")
    if prev_residual_flat is not None:
        prev_residual_flat = np.asarray(prev_residual_flat, dtype=np.float32)
        if prev_residual_flat.shape != residual_flat.shape:
            prev_residual_flat = None
    residual_ratio = (
        float(max_delta_last) / max(float(prev_abs_residual), 1.0e-30)
        if np.isfinite(prev_abs_residual)
        else 0.0
    )

    stable_iters = int(state.get("stable_iters", 0))
    anderson_g_history = [np.asarray(arr, dtype=np.float32).copy() for arr in state.get("anderson_g_history", [])]
    anderson_r_history = [np.asarray(arr, dtype=np.float32).copy() for arr in state.get("anderson_r_history", [])]
    accel_rejections = int(state.get("accel_rejections", 0))
    accel_restarts = int(state.get("accel_restarts", 0))
    anderson_depth_used = int(state.get("anderson_depth_used", 0))
    omega_last_prev = float(state.get("omega_last", _clip_cext_omega(CEXT_VESS_COUPLING_OMEGA)))

    if np.isfinite(prev_abs_residual) and max_delta_last > max(float(CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR), 1.0) * prev_abs_residual:
        anderson_g_history.clear()
        anderson_r_history.clear()
        accel_restarts += 1
        stable_iters = 0
    elif np.isfinite(prev_abs_residual) and residual_ratio <= max(float(CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO), 0.0):
        stable_iters += 1
    elif np.isfinite(prev_abs_residual):
        stable_iters = 0

    omega_fallback = _clip_cext_omega(CEXT_VESS_COUPLING_OMEGA)
    fallback_step = "picard"
    if accel_mode in ("aitken", "anderson"):
        omega_fallback = _cext_adaptive_aitken_omega(prev_residual_flat, residual_flat, omega_last_prev)
        fallback_step = "aitken"
    fallback_candidate = _cext_project_candidate(current_arr + np.float32(omega_fallback) * residual_flat)
    if fallback_candidate is None:
        fallback_candidate = _cext_project_candidate(mapped_arr)
        omega_fallback = _clip_cext_omega(1.0, fallback=omega_fallback)
        fallback_step = "picard"
    if fallback_candidate is None:
        fallback_candidate = np.asarray(mapped_arr, dtype=np.float32)
    trusted_fallback = _cext_apply_trust_region(current_arr, fallback_candidate, mapped_arr)
    if trusted_fallback is not None:
        fallback_candidate = np.asarray(trusted_fallback, dtype=np.float32)

    candidate_flat = np.asarray(fallback_candidate, dtype=np.float32)
    omega_last = float(omega_fallback)
    accel_step_last = fallback_step

    g_hist_entry = np.asarray(mapped_arr, dtype=np.float32).copy()
    r_hist_entry = np.asarray(residual_flat, dtype=np.float32).copy()
    anderson_g_history.append(g_hist_entry)
    anderson_r_history.append(r_hist_entry)
    max_hist = max(int(CEXT_VESS_COUPLING_ANDERSON_DEPTH), 1)
    if len(anderson_g_history) > max_hist:
        anderson_g_history.pop(0)
        anderson_r_history.pop(0)

    iter_local = int(state.get("iter_count", 0)) + 1
    anderson_allowed = (
        accel_mode == "anderson"
        and _cext_should_enable_anderson(
            iter_idx=iter_local,
            stable_iters=stable_iters,
            prev_abs_residual=prev_abs_residual,
            curr_abs_residual=max_delta_last,
            omega_fallback=omega_fallback,
        )
    )
    if anderson_allowed:
        accel_candidate, predicted_abs, depth_used, coeff_l1 = _cext_anderson_candidate(anderson_g_history, anderson_r_history)
        anderson_depth_used = max(anderson_depth_used, int(depth_used))
        accept_factor = max(float(CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR), 0.0)
        if (
            accel_candidate is not None
            and np.isfinite(predicted_abs)
            and predicted_abs <= max_delta_last * accept_factor
            and np.isfinite(coeff_l1)
            and coeff_l1 <= 20.0
        ):
            accel_candidate = _cext_project_candidate(accel_candidate)
            if accel_candidate is not None:
                step_base = float(np.nanmax(np.abs(np.asarray(fallback_candidate, dtype=float) - np.asarray(current_arr, dtype=float)))) if current_arr.size else 0.0
                step_accel = float(np.nanmax(np.abs(np.asarray(accel_candidate, dtype=float) - np.asarray(current_arr, dtype=float)))) if current_arr.size else 0.0
                step_factor = max(float(CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR), 1.0)
                lam = 1.0
                if step_base > 0.0 and step_accel > step_factor * step_base:
                    lam = min(1.0, (step_factor * step_base) / max(step_accel, 1.0e-30))
                blended = np.asarray(fallback_candidate, dtype=np.float32) + np.float32(lam) * (np.asarray(accel_candidate, dtype=np.float32) - np.asarray(fallback_candidate, dtype=np.float32))
                projected_blend = _cext_apply_trust_region(current_arr, blended, mapped_arr)
                if projected_blend is not None:
                    candidate_flat = np.asarray(projected_blend, dtype=np.float32)
                    accel_step_last = "anderson" if lam >= 0.999 else "anderson_blend"
        else:
            accel_rejections += 1
            anderson_g_history.clear()
            anderson_r_history.clear()
            accel_restarts += 1
            stable_iters = 0

    state_out = {
        "prev_residual_flat": np.asarray(residual_flat, dtype=np.float32).copy(),
        "prev_abs_residual": float(max_delta_last),
        "stable_iters": int(stable_iters),
        "anderson_g_history": anderson_g_history,
        "anderson_r_history": anderson_r_history,
        "accel_rejections": int(accel_rejections),
        "accel_restarts": int(accel_restarts),
        "anderson_depth_used": int(anderson_depth_used),
        "omega_last": float(omega_last),
        "iter_count": int(iter_local),
    }
    metrics = {
        "max_delta_last": float(max_delta_last),
        "rel_residual_last": float(rel_residual_last),
        "omega_last": float(omega_last),
        "accel_step_last": accel_step_last,
        "accel_rejections": int(accel_rejections),
        "accel_restarts": int(accel_restarts),
        "anderson_depth_used": int(anderson_depth_used),
        "stable_iters": int(stable_iters),
    }
    return candidate_flat, state_out, metrics


def _cext_anderson_candidate(
    g_history: list[np.ndarray],
    r_history: list[np.ndarray],
) -> tuple[np.ndarray | None, float, int, float]:
    depth = max(int(CEXT_VESS_COUPLING_ANDERSON_DEPTH), 0)
    if depth <= 1 or len(g_history) < 2 or len(r_history) < 2:
        return None, float("inf"), 0, float("inf")
    g_used = list(g_history[-depth:])
    r_used = list(r_history[-depth:])
    p = min(len(g_used), len(r_used))
    if p < 2:
        return None, float("inf"), 0, float("inf")
    g_used = g_used[-p:]
    r_used = r_used[-p:]

    gram = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        ri = np.asarray(r_used[i], dtype=np.float32)
        for j in range(i, p):
            val = float(np.dot(ri, np.asarray(r_used[j], dtype=np.float32)))
            gram[i, j] = val
            gram[j, i] = val
    gram.flat[:: p + 1] += max(float(CEXT_VESS_COUPLING_ANDERSON_REG), 0.0)
    ones = np.ones((p, 1), dtype=np.float64)
    system = np.block(
        [
            [gram, ones],
            [ones.T, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    rhs = np.zeros((p + 1,), dtype=np.float64)
    rhs[-1] = 1.0
    try:
        sol = np.linalg.solve(system, rhs)
        alpha = np.asarray(sol[:p], dtype=np.float64)
    except np.linalg.LinAlgError:
        return None, float("inf"), 0, float("inf")
    if not np.all(np.isfinite(alpha)):
        return None, float("inf"), 0, float("inf")

    candidate = np.zeros_like(np.asarray(g_used[-1], dtype=np.float32), dtype=np.float32)
    predicted = np.zeros_like(np.asarray(r_used[-1], dtype=np.float32), dtype=np.float32)
    for coeff, g_arr, r_arr in zip(alpha, g_used, r_used):
        coeff32 = np.float32(coeff)
        candidate += coeff32 * np.asarray(g_arr, dtype=np.float32)
        predicted += coeff32 * np.asarray(r_arr, dtype=np.float32)
    predicted_abs = float(np.nanmax(np.abs(np.asarray(predicted, dtype=float)))) if predicted.size else 0.0
    coeff_l1 = float(np.sum(np.abs(alpha)))
    return candidate, predicted_abs, int(p), coeff_l1


def _run_topdown_ext_frozen_step(
    context: dict,
    ext_state: dict,
    *,
    inlet_concentration: float,
    vmax: float,
    km: float,
    chb_max: np.ndarray,
    fluid_mode: str,
    frozen_backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    global _LAST_CEXT_FROZEN_STEP_TIMINGS
    t_step_total = perf_counter()
    frozen_transfer = 0.0
    backend_local = str(frozen_backend)
    closure_mode = str(LUMEN_WALL_CLOSURE or "wellmixed").strip().lower()
    if closure_mode not in ("wellmixed", "graetz"):
        raise ValueError("--lumen-wall-closure must be 'wellmixed' or 'graetz'.")
    if backend_local == "gpu":
        try:
            if closure_mode == "graetz":
                cin_seg, cout_seg, c_iv_gl, c_wall_gl, frozen_timings = _solve_topdown_ext_frozen_graetz_gpu(
                    context,
                    np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
                    diffusivity_si=float(context["diffusivity_si"]),
                    vmax=float(vmax),
                    km=float(km),
                    inlet_concentration=float(inlet_concentration),
                    chb_max=np.asarray(chb_max, dtype=np.float32),
                    fluid_mode=fluid_mode,
                )
            else:
                cin_seg, cout_seg, c_iv_gl, frozen_timings = _solve_topdown_ext_frozen_gpu(
                    context,
                    np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
                    diffusivity_si=float(context["diffusivity_si"]),
                    vmax=float(vmax),
                    km=float(km),
                    inlet_concentration=float(inlet_concentration),
                    chb_max=np.asarray(chb_max, dtype=np.float32),
                    fluid_mode=fluid_mode,
                )
                c_wall_gl = np.asarray(c_iv_gl, dtype=np.float32)
            ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
            ext_state["c_bulk_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
            ext_state["c_wall_gl"] = np.asarray(c_wall_gl, dtype=np.float32)
            frozen_transfer = float(frozen_timings.get("upload_s", 0.0)) + float(frozen_timings.get("download_s", 0.0))
            _LAST_CEXT_FROZEN_STEP_TIMINGS = {
                "backend": backend_local,
                "lumen_wall_closure": closure_mode,
                "upload_s": float(frozen_timings.get("upload_s", 0.0)),
                "kernel_s": float(frozen_timings.get("kernel_s", 0.0)),
                "download_s": float(frozen_timings.get("download_s", 0.0)),
                "transfer_s": float(frozen_transfer),
                "total_s": float(perf_counter() - t_step_total),
            }
            for key, value in frozen_timings.items():
                if str(key).startswith("graetz_"):
                    _LAST_CEXT_FROZEN_STEP_TIMINGS[str(key)] = value
            return cin_seg, cout_seg, c_iv_gl, backend_local, frozen_transfer
        except Exception:
            if str(CEXT_FROZEN_ACCEL_MODE).lower() == "gpu":
                raise
            backend_local = "cpu"
    if closure_mode == "graetz":
        raise RuntimeError("Graetz lumen closure currently requires the GPU frozen topdown backend in TissueSim_cube_expanded.py.")
    t_cpu = perf_counter()
    if _HAVE_NUMBA and CONC_USE_NUMBA:
        cin_seg, cout_seg, c_iv_gl = _solve_topdown_ext_frozen_numba(
            np.asarray(context["level_order"], dtype=np.int32),
            np.asarray(context["level_offsets"], dtype=np.int32),
            np.asarray(context["parents"], dtype=np.int32),
            np.asarray(context["flows_si"], dtype=np.float32),
            np.asarray(context["radii_si"], dtype=np.float32),
            np.asarray(context["lengths_si"], dtype=np.float32),
            np.asarray(context["gl_t"], dtype=np.float32),
            np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
            float(context["diffusivity_si"]),
            float(vmax),
            float(km),
            float(inlet_concentration),
            np.asarray(chb_max, dtype=np.float32),
            1 if fluid_mode == "blood" else 0,
        )
    else:
        cin_seg, cout_seg, c_iv_gl = _solve_topdown_ext_frozen_python(
            np.asarray(context["level_order"], dtype=np.int32),
            np.asarray(context["level_offsets"], dtype=np.int32),
            np.asarray(context["parents"], dtype=np.int32),
            np.asarray(context["flows_si"], dtype=np.float32),
            np.asarray(context["radii_si"], dtype=np.float32),
            np.asarray(context["lengths_si"], dtype=np.float32),
            np.asarray(context["gl_t"], dtype=np.float32),
            np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
            float(context["diffusivity_si"]),
            float(vmax),
            float(km),
            float(inlet_concentration),
            np.asarray(chb_max, dtype=np.float32),
            fluid_mode == "blood",
        )
    _LAST_CEXT_FROZEN_STEP_TIMINGS = {
        "backend": backend_local,
        "lumen_wall_closure": closure_mode,
        "upload_s": 0.0,
        "kernel_s": float(perf_counter() - t_cpu),
        "download_s": 0.0,
        "transfer_s": 0.0,
        "total_s": float(perf_counter() - t_step_total),
    }
    ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    ext_state["c_bulk_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    ext_state["c_wall_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    return cin_seg, cout_seg, c_iv_gl, backend_local, frozen_transfer


def _compute_cext_image_step(
    context: dict,
    ext_state: dict,
    *,
    backend: str,
    source_plan: dict | None = None,
    target_plan: dict | None = None,
    rebuild_cache: bool = True,
) -> tuple[np.ndarray, str, float, float]:
    if rebuild_cache:
        _build_cext_iteration_cache(context, ext_state)
    backend_local = str(backend)
    kernel_total = 0.0
    transfer_total = 0.0
    if backend_local == "gpu":
        if _context_uses_cext_gpu_direct(context):
            try:
                c_ext_total, c_ext_cap, kernel_total, transfer_total = _compute_cext_direct_gpu(
                    context,
                    ext_state,
                    source_plan=source_plan,
                    target_plan=target_plan,
                )
                c_ext_new = np.asarray(c_ext_total, dtype=np.float32)
                cap_arr = np.asarray(c_ext_cap, dtype=np.float32)
                positive_cap = cap_arr > 0.0
                if np.any(positive_cap):
                    c_ext_new = np.where(positive_cap, np.minimum(c_ext_new, cap_arr), c_ext_new).astype(np.float32, copy=False)
                c_ext_new = np.maximum(c_ext_new, np.float32(0.0)).astype(np.float32, copy=False)
                return c_ext_new, backend_local, kernel_total, transfer_total
            except Exception:
                if str(CEXT_ACCEL_MODE).lower() == "gpu":
                    raise
                backend_local = "cpu"
                context["candidate_query_mode"] = "kdtree" if context.get("candidate_kdtree") is not None else "grid"
        batches, _ = _ensure_cext_candidate_batches(context)
        batches = _coerce_candidate_batches_for_gpu(context)
        t_transfer = perf_counter()
        lambda_iv_g = _cp.asarray(np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32))
        q_weighted_g = _cp.asarray(np.asarray(ext_state["q_weighted_gl"], dtype=np.float32))
        mono2_weight_g = _cp.asarray(np.asarray(ext_state.get("mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32))
        dipole2_weight_g = _cp.asarray(np.asarray(ext_state.get("dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])), dtype=np.float32))
        seg_cap_g = _cp.asarray(np.asarray(ext_state["seg_cap_gl"], dtype=np.float32))
        c_ext_new_g = _cp.zeros(np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape, dtype=_cp.float32)
        _cp.cuda.Stream.null.synchronize()
        transfer_total += perf_counter() - t_transfer

        batch_idx = 0
        gpu_failed = False
        while batch_idx < len(batches):
            batch = batches[batch_idx]
            if int(np.asarray(batch["target_seg_ids"]).size) <= 0:
                batch_idx += 1
                continue
            try:
                t_batch_kernel = perf_counter()
                _compute_cext_batch_gpu_into(
                    context,
                    batch,
                    lambda_iv_g,
                    q_weighted_g,
                    mono2_weight_g,
                    dipole2_weight_g,
                    seg_cap_g,
                    c_ext_new_g,
                )
                _cp.cuda.Stream.null.synchronize()
                kernel_total += perf_counter() - t_batch_kernel
                if int(CEXT_GPU_VALIDATE_SEGMENTS) > 0:
                    t_validate_transfer = perf_counter()
                    gpu_out = _cp.asnumpy(c_ext_new_g[batch["target_start"]: batch["target_stop"]])
                    transfer_total += perf_counter() - t_validate_transfer
                    _validate_cext_gpu_batch(
                        context,
                        ext_state,
                        np.asarray(batch["target_seg_ids"], dtype=np.int32),
                        np.asarray(batch["row_ptr"], dtype=np.int32),
                        np.asarray(batch["col_idx"], dtype=np.int32),
                        gpu_out,
                    )
                batch_idx += 1
            except Exception as exc:
                if _cp is None or not isinstance(exc, _cp.cuda.memory.OutOfMemoryError):
                    if str(CEXT_ACCEL_MODE).lower() == "gpu":
                        raise
                    backend_local = "cpu"
                    gpu_failed = True
                    break
                if int(np.asarray(batch["target_seg_ids"]).size) <= 1:
                    if str(CEXT_ACCEL_MODE).lower() == "gpu":
                        raise
                    backend_local = "cpu"
                    gpu_failed = True
                    break
                split_batches = _split_cext_candidate_batch(context, batch)
                batches[batch_idx: batch_idx + 1] = split_batches
                context["candidate_batches"] = batches
                _coerce_candidate_batches_for_gpu(context)
        if backend_local == "gpu" and not gpu_failed:
            t_download = perf_counter()
            c_ext_new = _cp.asnumpy(c_ext_new_g)
            transfer_total += perf_counter() - t_download
            context["candidate_batches"] = batches
            return c_ext_new, backend_local, kernel_total, transfer_total
        c_ext_new = np.zeros_like(ext_state["c_ext_gl"])
        for batch in batches:
            if int(np.asarray(batch["target_seg_ids"]).size) <= 0:
                continue
            t_batch_kernel = perf_counter()
            out_batch = _compute_cext_batch_cpu(
                context,
                ext_state,
                np.asarray(batch["target_seg_ids"], dtype=np.int32),
                np.asarray(batch["row_ptr"], dtype=np.int32),
                np.asarray(batch["col_idx"], dtype=np.int32),
            )
            kernel_total += perf_counter() - t_batch_kernel
            c_ext_new[int(batch["target_start"]): int(batch["target_stop"])] = out_batch
        context["candidate_batches"] = batches
        return c_ext_new, backend_local, kernel_total, transfer_total

    batches, _ = _ensure_cext_candidate_batches(context)
    c_ext_new = np.zeros_like(ext_state["c_ext_gl"])
    for batch in batches:
        if int(np.asarray(batch["target_seg_ids"]).size) <= 0:
            continue
        t_batch_kernel = perf_counter()
        out_batch = _compute_cext_batch_cpu(
            context,
            ext_state,
            np.asarray(batch["target_seg_ids"], dtype=np.int32),
            np.asarray(batch["row_ptr"], dtype=np.int32),
            np.asarray(batch["col_idx"], dtype=np.int32),
        )
        kernel_total += perf_counter() - t_batch_kernel
        c_ext_new[int(batch["target_start"]): int(batch["target_stop"])] = out_batch
    return c_ext_new, backend_local, kernel_total, transfer_total


def _solve_channel_concentrations_topdown_ext(
    tree: Tree,
    flows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    fluid: str,
) -> Tuple[np.ndarray, np.ndarray]:
    global _LAST_CONCENTRATION_TIMINGS
    if float(CEXT_WINDOW_FACTOR) <= 0.0:
        cin, cout = _solve_channel_concentrations_topdown(
            tree,
            flows,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
        _LAST_CONCENTRATION_TIMINGS = _default_cext_timing_details(backend="disabled")
        return cin, cout

    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty

    t_total = perf_counter()
    t_stage = perf_counter()
    fluid_mode = (fluid or getattr(getattr(tree, "parameters", None), "fluid", None) or ACTIVE_FLUID).lower()
    hct_context = _hematocrit_context_for_tree(tree)
    radii_arr = np.asarray(hct_context["radii"], dtype=float)
    flows_arr = np.asarray(flows, dtype=float)
    chb_max = np.zeros_like(radii_arr)
    hct_source = "none"
    if fluid_mode == "blood":
        cached_hct = _get_tree_hematocrit_cache(
            tree,
            nseg,
            model=HEMATOCRIT_MODEL,
            flows=flows_arr,
        )
        if cached_hct is not None:
            HD, HT = cached_hct
            hct_source = "cache"
        else:
            HD, HT = compute_tree_hematocrit(
                tree,
                hd_root=HD_DISCHARGE,
                flows=flows_arr,
                model=HEMATOCRIT_MODEL,
                order=np.asarray(hct_context["order"], dtype=np.int64),
            )
            _store_tree_hematocrit_cache(
                tree,
                HD,
                HT,
                model=HEMATOCRIT_MODEL,
                flows=flows_arr,
                fixed_flow_bc=False,
            )
            hct_source = "computed"
        chb_max = np.asarray(HT, dtype=float) * float(O2_CAP_PER_HCT)
    hct_setup_s = perf_counter() - t_stage

    t_stage = perf_counter()
    context = _build_cext_geometry_context(
        tree,
        flows_arr,
        starts,
        ends,
        radii,
        lengths,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=vmax,
        km=km,
    )
    ext_state = {
        "c_ext_gl": np.zeros((nseg, int(GL_ORDER_CEXT)), dtype=np.float32),
        "c_iv_gl": np.full((nseg, int(GL_ORDER_CEXT)), float(inlet_concentration), dtype=np.float32),
        "c_bulk_gl": np.full((nseg, int(GL_ORDER_CEXT)), float(inlet_concentration), dtype=np.float32),
        "c_wall_gl": np.full((nseg, int(GL_ORDER_CEXT)), float(inlet_concentration), dtype=np.float32),
        "cin_seg": np.full((nseg,), float(inlet_concentration), dtype=np.float32),
        "cout_seg": np.full((nseg,), float(inlet_concentration), dtype=np.float32),
        "vmax": float(vmax),
        "km": float(km),
        "window_factor": float(CEXT_WINDOW_FACTOR),
    }

    backend = _resolve_cext_accel_mode()
    frozen_backend = _resolve_cext_frozen_accel_mode()
    if backend == "gpu" and _cp is not None and bool(context.get("gpu_direct_available")):
        context["candidate_query_mode"] = "gpu_direct"
        context["candidate_build_time_s"] = 0.0
        query_total = 0.0
    else:
        _, candidate_query_time = _ensure_cext_candidate_batches(context)
        query_total = float(candidate_query_time)
    kernel_total = 0.0
    transfer_total = 0.0
    frozen_transfer_total = 0.0
    init_elapsed = 0.0
    init_kernel_total = 0.0
    init_transfer_total = 0.0
    init_frozen_transfer_total = 0.0
    init_frozen_kernel_total = 0.0
    init_cache_total = 0.0
    init_hybrid_call_total = 0.0
    iteration_cache_total = 0.0
    final_frozen_transfer_total = 0.0
    final_frozen_kernel_total = 0.0
    final_cache_total = 0.0
    final_source_state_total = 0.0
    init_performed = False
    max_delta_last = 0.0
    rel_residual_last = 0.0
    omega_last = _clip_cext_omega(CEXT_VESS_COUPLING_OMEGA)
    accel_step_last = "picard"
    accel_rejections = 0
    accel_restarts = 0
    anderson_depth_used = 0
    completed_iters = 0
    chunk_targets = _cext_initial_chunk_targets()
    init_mode = str(CEXT_INIT_MODE or "zero").strip().lower()
    accel_mode = str(CEXT_VESS_COUPLING_ACCEL or "none").strip().lower()
    if accel_mode not in ("none", "aitken", "anderson"):
        raise ValueError("--cext-vess-coupling-accel must be 'none', 'aitken', or 'anderson'.")
    anderson_g_history: list[np.ndarray] = []
    anderson_r_history: list[np.ndarray] = []
    prev_residual_flat: np.ndarray | None = None
    prev_abs_residual = float("inf")
    stable_iters = 0
    active_set_enabled = bool(CEXT_ACTIVE_SET_ENABLE) and backend == "gpu" and _context_uses_cext_gpu_direct(context)
    target_active_set_enabled = bool(CEXT_TARGET_ACTIVE_SET_ENABLE) and active_set_enabled
    active_source_mask = np.ones((nseg,), dtype=bool)
    active_source_stable_counts = np.zeros((nseg,), dtype=np.int16)
    frozen_source_total_gl = np.zeros((nseg, int(GL_ORDER_CEXT)), dtype=np.float32)
    frozen_source_cap_gl = np.zeros((nseg, int(GL_ORDER_CEXT)), dtype=np.float32)
    active_source_plan: dict | None = None
    frozen_source_plan: dict | None = None
    active_freeze_events = 0
    active_refreshes = 0
    last_active_refresh_iter = 0
    active_source_count = int(nseg)
    frozen_source_count = 0
    active_target_mask = np.ones((nseg,), dtype=bool)
    active_target_stable_counts = np.zeros((nseg,), dtype=np.int16)
    active_target_count = int(nseg)
    frozen_target_count = 0
    target_freeze_events = 0
    target_reactivations = 0
    active_component_count = 1
    largest_component_size = int(nseg)
    component_states: dict[tuple[int, ...], dict] = {}
    prev_source_c_ext_gl: np.ndarray | None = None
    prev_source_c_iv_gl: np.ndarray | None = None
    min_active_sources = max(
        1,
        int(CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT),
        int(math.ceil(max(float(CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION), 0.0) * float(nseg))),
    )
    min_active_targets = max(1, int(CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT))
    component_split_threshold = max(8 * min_active_targets, 32768)
    if init_mode not in ("zero", "decoupled_greens"):
        raise ValueError("--cext-init-mode must be 'zero' or 'decoupled_greens'.")

    if SOLVER_TIMING_DETAILS:
        print(
            "  Concentration topdown_ext: "
            f"nseg={nseg} fluid={fluid_mode} backend={backend} frozen={frozen_backend} gl_order={GL_ORDER_CEXT} "
            f"hct={hct_source} accel={accel_mode} active_set={'on' if active_set_enabled else 'off'} "
            f"wall={LUMEN_WALL_CLOSURE} o2_terms={FINITE_RADIUS_O2_TERMS} "
            f"cext_lambda={_normalize_cext_lambda_source()} "
            f"graetz_max_fp_iters={GRAETZ_MAX_FP_ITERS} tail_buffer_patch=enabled "
            f"candidates={context.get('candidate_query_mode', 'grid')} "
            f"initial_chunk_targets={chunk_targets}"
        )

    if init_mode == "decoupled_greens":
        t_init = perf_counter()
        cin_seg, cout_seg, c_iv_gl, frozen_backend, init_frozen_transfer = _run_topdown_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=float(inlet_concentration),
            vmax=float(vmax),
            km=float(km),
            chb_max=np.asarray(chb_max, dtype=np.float32),
            fluid_mode=fluid_mode,
            frozen_backend=frozen_backend,
        )
        ext_state["cin_seg"] = cin_seg
        ext_state["cout_seg"] = cout_seg
        ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
        c_ext_new, backend, init_kernel, init_transfer = _compute_cext_image_step(
            context,
            ext_state,
            backend=backend,
        )
        ext_state["c_ext_gl"] = np.asarray(c_ext_new, dtype=np.float32)
        init_elapsed = perf_counter() - t_init
        init_kernel_total = float(init_kernel)
        init_transfer_total = float(init_transfer)
        init_frozen_transfer_total = float(init_frozen_transfer)
        init_performed = True
        if SOLVER_TIMING_DETAILS:
            print(
                "    Cext init predictor: "
                f"max_cext={float(np.nanmax(np.asarray(ext_state['c_ext_gl'], dtype=float))) if ext_state['c_ext_gl'].size else 0.0:.3e}"
            )
        _maybe_trace_cext_iteration(0, ext_state, solver="topdown_ext_image", context=context, backend=backend)

    if not init_performed:
        _maybe_trace_cext_iteration(0, ext_state, solver="topdown_ext_image", context=context, backend=backend)

    for iter_idx in range(1, int(CEXT_VESS_COUPLING_MAX_ITER) + 1):
        cin_seg, cout_seg, c_iv_gl, frozen_backend, frozen_transfer = _run_topdown_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=float(inlet_concentration),
            vmax=float(vmax),
            km=float(km),
            chb_max=np.asarray(chb_max, dtype=np.float32),
            fluid_mode=fluid_mode,
            frozen_backend=frozen_backend,
        )
        frozen_transfer_total += float(frozen_transfer)
        ext_state["cin_seg"] = cin_seg
        ext_state["cout_seg"] = cout_seg
        ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
        _build_cext_iteration_cache(context, ext_state)
        source_c_ext_snapshot = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).copy()
        source_c_iv_snapshot = np.asarray(ext_state["c_iv_gl"], dtype=np.float32).copy()
        current_c_ext_gl = np.asarray(ext_state["c_ext_gl"], dtype=np.float32)

        if active_set_enabled:
            source_abs_delta, source_rel_delta = _cext_source_state_delta_metrics(
                prev_source_c_ext_gl,
                prev_source_c_iv_gl,
                source_c_ext_snapshot,
                source_c_iv_snapshot,
            )
            active_mask_now = np.asarray(active_source_mask, dtype=bool)
            stable_local = np.asarray(
                (source_abs_delta <= float(CEXT_ACTIVE_SET_ABS_TOL)) | (source_rel_delta <= float(CEXT_ACTIVE_SET_REL_TOL)),
                dtype=bool,
            )
            active_source_stable_counts[active_mask_now & stable_local] += np.int16(1)
            active_source_stable_counts[active_mask_now & ~stable_local] = np.int16(0)

            freeze_now = np.zeros((nseg,), dtype=bool)
            if iter_idx >= max(int(CEXT_ACTIVE_SET_START), 1):
                eligible = active_mask_now & (active_source_stable_counts >= max(int(CEXT_ACTIVE_SET_STABLE_ITERS), 1))
                max_freeze = max(int(np.count_nonzero(active_mask_now)) - int(min_active_sources), 0)
                eligible_ids = np.flatnonzero(eligible)
                if max_freeze > 0 and eligible_ids.size > 0:
                    if eligible_ids.size > max_freeze:
                        rank_key = np.asarray(source_rel_delta[eligible_ids], dtype=np.float32)
                        keep_order = np.argsort(rank_key, kind="stable")[:max_freeze]
                        eligible_ids = np.asarray(eligible_ids[keep_order], dtype=np.int64)
                    freeze_now[np.asarray(eligible_ids, dtype=np.int64)] = True
                    active_source_mask[freeze_now] = False
                    active_source_stable_counts[freeze_now] = np.int16(0)
                    active_source_count = int(np.count_nonzero(active_source_mask))
                    frozen_source_count = int(nseg - active_source_count)
                    active_source_plan = None if active_source_count >= nseg else _build_cext_gpu_direct_source_plan(context, active_source_mask)
                    frozen_source_plan = None if frozen_source_count <= 0 else _build_cext_gpu_direct_source_plan(context, ~active_source_mask)
                    active_freeze_events += int(np.count_nonzero(freeze_now))
                    component_states.clear()

            refresh_due = False
            refresh_period = max(int(CEXT_ACTIVE_SET_REFRESH_PERIOD), 0)
            if frozen_source_count > 0:
                if refresh_period > 0 and (iter_idx - last_active_refresh_iter) >= refresh_period:
                    refresh_due = True
                if np.any(~active_source_mask) and (np.any(freeze_now) or refresh_due):
                    if frozen_source_plan is None:
                        frozen_source_plan = _build_cext_gpu_direct_source_plan(context, ~active_source_mask)
                    frozen_total_step, frozen_cap_step, frozen_kernel_step, frozen_transfer_step = _compute_cext_direct_gpu(
                        context,
                        ext_state,
                        source_plan=frozen_source_plan,
                    )
                    frozen_source_total_gl = np.asarray(frozen_total_step, dtype=np.float32)
                    frozen_source_cap_gl = np.asarray(frozen_cap_step, dtype=np.float32)
                    kernel_total += float(frozen_kernel_step)
                    transfer_total += float(frozen_transfer_step)
                    active_refreshes += 1
                    last_active_refresh_iter = int(iter_idx)
            else:
                frozen_source_total_gl.fill(np.float32(0.0))
                frozen_source_cap_gl.fill(np.float32(0.0))
                active_source_count = int(np.count_nonzero(active_source_mask))
                frozen_source_count = int(nseg - active_source_count)
                active_source_plan = None if active_source_count >= nseg else active_source_plan
                frozen_source_plan = None

            source_cell_stats = _cext_active_source_cell_stats(
                context,
                active_source_mask,
                source_abs_delta,
                source_rel_delta,
            )

            if target_active_set_enabled and frozen_target_count > 0 and (np.any(freeze_now) or refresh_due):
                frozen_target_ids = np.flatnonzero(~active_target_mask).astype(np.int32, copy=False)
                if frozen_target_ids.size > 0:
                    neighbor_counts_frozen, neighbor_abs_frozen, neighbor_rel_frozen, _ = _cext_target_neighbor_metrics(
                        context,
                        frozen_target_ids,
                        source_cell_stats,
                        neighbor_pad=int(CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD),
                    )
                    reactivate = (neighbor_counts_frozen > 0) & (
                        (neighbor_abs_frozen > float(CEXT_TARGET_ACTIVE_SET_ABS_TOL))
                        | (neighbor_rel_frozen > float(CEXT_TARGET_ACTIVE_SET_REL_TOL))
                    )
                    if np.any(reactivate):
                        reactivate_ids = np.asarray(frozen_target_ids[np.asarray(reactivate, dtype=bool)], dtype=np.int32)
                        active_target_mask[reactivate_ids] = True
                        active_target_stable_counts[reactivate_ids] = np.int16(0)
                        target_reactivations += int(reactivate_ids.size)
                        component_states.clear()

            active_target_count = int(np.count_nonzero(active_target_mask))
            frozen_target_count = int(nseg - active_target_count)
            active_target_ids = np.flatnonzero(active_target_mask).astype(np.int32, copy=False)
            active_component_count = 0
            largest_component_size = 0
            target_abs_delta_field = np.zeros((nseg,), dtype=np.float32)
            target_rel_delta_field = np.zeros((nseg,), dtype=np.float32)
            candidate_c_ext_gl = np.asarray(current_c_ext_gl, dtype=np.float32).copy()
            next_component_states: dict[tuple[int, ...], dict] = {}
            max_delta_last = 0.0
            rel_residual_last = 0.0
            accel_step_last = "picard"
            omega_last = _clip_cext_omega(CEXT_VESS_COUPLING_OMEGA)
            stable_iters = 0

            component_lists: list[np.ndarray] = []
            query_cells_map: dict[int, np.ndarray] = {}
            if active_target_ids.size > 0:
                if int(active_target_ids.size) > int(component_split_threshold):
                    component_lists = [np.asarray(active_target_ids, dtype=np.int32)]
                    active_component_count = 1
                    largest_component_size = int(active_target_ids.size)
                else:
                    _, _, _, query_cells_all = _cext_target_neighbor_metrics(
                        context,
                        active_target_ids,
                        source_cell_stats,
                        neighbor_pad=int(CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD),
                    )
                    query_cells_map = {
                        int(seg_id): np.asarray(cells, dtype=np.int32)
                        for seg_id, cells in zip(np.asarray(active_target_ids, dtype=np.int32), query_cells_all)
                    }
                    component_lists, largest_component_size = _cext_build_active_components_from_cells(
                        active_target_ids,
                        query_cells_all,
                        source_cell_stats,
                    )
                    active_component_count = int(len(component_lists))

            for component_ids in component_lists:
                comp_targets = np.asarray(component_ids, dtype=np.int32)
                if comp_targets.size <= 0:
                    continue
                if comp_targets.size == active_target_ids.size and not query_cells_map:
                    comp_source_plan = active_source_plan
                else:
                    component_cells = [query_cells_map.get(int(seg), np.zeros((0,), dtype=np.int32)) for seg in comp_targets]
                    comp_source_ids = _cext_component_source_ids_from_cells(context, active_source_mask, component_cells)
                    if comp_source_ids.size >= active_source_count:
                        comp_source_plan = active_source_plan
                    else:
                        comp_source_plan = _build_cext_gpu_direct_source_plan_from_ids(context, comp_source_ids)
                comp_target_plan = _build_cext_gpu_direct_target_plan(context, comp_targets)
                active_total_step, active_cap_step, kernel_step, transfer_step = _compute_cext_direct_gpu(
                    context,
                    ext_state,
                    source_plan=comp_source_plan,
                    target_plan=comp_target_plan,
                )
                kernel_total += float(kernel_step)
                transfer_total += float(transfer_step)

                active_total_arr = np.asarray(active_total_step, dtype=np.float32)
                active_cap_arr = np.asarray(active_cap_step, dtype=np.float32)
                total_arr = active_total_arr + np.asarray(frozen_source_total_gl, dtype=np.float32)
                cap_arr = np.maximum(active_cap_arr, np.asarray(frozen_source_cap_gl, dtype=np.float32))
                positive_cap = cap_arr > 0.0
                mapped_component_gl = np.where(positive_cap, np.minimum(total_arr, cap_arr), total_arr).astype(np.float32, copy=False)
                mapped_component_gl = np.maximum(mapped_component_gl, np.float32(0.0)).astype(np.float32, copy=False)
                mapped_component = np.asarray(mapped_component_gl[comp_targets], dtype=np.float32)
                current_component = np.asarray(candidate_c_ext_gl[comp_targets], dtype=np.float32)
                comp_key = tuple(int(v) for v in np.asarray(comp_targets, dtype=np.int32).tolist())
                comp_candidate_flat, comp_state, comp_metrics = _cext_apply_component_acceleration(
                    current_component.reshape(-1),
                    mapped_component.reshape(-1),
                    component_states.get(comp_key),
                    accel_mode=accel_mode,
                )
                comp_candidate = np.asarray(comp_candidate_flat.reshape(current_component.shape), dtype=np.float32)
                candidate_c_ext_gl[comp_targets] = comp_candidate
                next_component_states[comp_key] = comp_state

                comp_abs = np.max(np.abs(np.asarray(mapped_component, dtype=np.float32) - np.asarray(current_component, dtype=np.float32)), axis=1)
                comp_scale = np.maximum.reduce(
                    [
                        np.max(np.abs(np.asarray(mapped_component, dtype=np.float32)), axis=1),
                        np.max(np.abs(np.asarray(current_component, dtype=np.float32)), axis=1),
                        np.full((comp_targets.size,), float(VESS_CONC_FLOOR), dtype=np.float32),
                    ]
                )
                comp_rel = np.divide(
                    np.asarray(comp_abs, dtype=np.float32),
                    np.maximum(comp_scale, np.float32(VESS_CONC_FLOOR)),
                    out=np.zeros_like(np.asarray(comp_abs, dtype=np.float32)),
                    where=np.maximum(comp_scale, np.float32(VESS_CONC_FLOOR)) > 0.0,
                )
                target_abs_delta_field[comp_targets] = np.asarray(comp_abs, dtype=np.float32)
                target_rel_delta_field[comp_targets] = np.asarray(comp_rel, dtype=np.float32)
                if float(comp_metrics["max_delta_last"]) >= float(max_delta_last):
                    max_delta_last = float(comp_metrics["max_delta_last"])
                    rel_residual_last = float(comp_metrics["rel_residual_last"])
                    omega_last = float(comp_metrics["omega_last"])
                    accel_step_last = str(comp_metrics["accel_step_last"])
                    stable_iters = int(comp_metrics["stable_iters"])

            component_states = next_component_states
            accel_rejections = int(sum(int(state.get("accel_rejections", 0)) for state in component_states.values()))
            accel_restarts = int(sum(int(state.get("accel_restarts", 0)) for state in component_states.values()))
            anderson_depth_used = int(max([int(state.get("anderson_depth_used", 0)) for state in component_states.values()] or [0]))

            if target_active_set_enabled and active_target_ids.size > 0:
                stable_target_local = (
                    (target_abs_delta_field[active_target_ids] <= float(CEXT_TARGET_ACTIVE_SET_ABS_TOL))
                    | (target_rel_delta_field[active_target_ids] <= float(CEXT_TARGET_ACTIVE_SET_REL_TOL))
                )
                active_target_stable_counts[active_target_ids[np.asarray(stable_target_local, dtype=bool)]] += np.int16(1)
                active_target_stable_counts[active_target_ids[~np.asarray(stable_target_local, dtype=bool)]] = np.int16(0)

                if iter_idx >= max(int(CEXT_TARGET_ACTIVE_SET_START), 1):
                    eligible_target_ids = np.asarray(
                        active_target_ids[
                            active_target_stable_counts[active_target_ids] >= max(int(CEXT_TARGET_ACTIVE_SET_STABLE_ITERS), 1)
                        ],
                        dtype=np.int32,
                    )
                    max_target_freeze = max(int(np.count_nonzero(active_target_mask)) - int(min_active_targets), 0)
                    if max_target_freeze > 0 and eligible_target_ids.size > 0:
                        neighbor_counts_t, neighbor_abs_t, neighbor_rel_t, _ = _cext_target_neighbor_metrics(
                            context,
                            eligible_target_ids,
                            source_cell_stats,
                            neighbor_pad=int(CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD),
                        )
                        can_freeze = (neighbor_counts_t <= 0) | (
                            (neighbor_abs_t <= float(CEXT_TARGET_ACTIVE_SET_ABS_TOL))
                            & (neighbor_rel_t <= float(CEXT_TARGET_ACTIVE_SET_REL_TOL))
                        )
                        freeze_target_ids = np.asarray(eligible_target_ids[np.asarray(can_freeze, dtype=bool)], dtype=np.int32)
                        if freeze_target_ids.size > max_target_freeze:
                            rank_key = np.asarray(target_rel_delta_field[freeze_target_ids], dtype=np.float32)
                            keep_order = np.argsort(rank_key, kind="stable")[:max_target_freeze]
                            freeze_target_ids = np.asarray(freeze_target_ids[keep_order], dtype=np.int32)
                        if freeze_target_ids.size > 0:
                            active_target_mask[freeze_target_ids] = False
                            active_target_stable_counts[freeze_target_ids] = np.int16(0)
                            target_freeze_events += int(freeze_target_ids.size)
                            component_states.clear()

            ext_state["c_ext_gl"] = np.asarray(candidate_c_ext_gl, dtype=np.float32)
            active_target_count = int(np.count_nonzero(active_target_mask))
            frozen_target_count = int(nseg - active_target_count)
            prev_source_c_ext_gl = source_c_ext_snapshot
            prev_source_c_iv_gl = source_c_iv_snapshot
            completed_iters = iter_idx
            prev_residual_flat = None
            prev_abs_residual = float("inf")
        else:
            c_ext_new, backend, kernel_step, transfer_step = _compute_cext_image_step(
                context,
                ext_state,
                backend=backend,
                rebuild_cache=False,
            )
            kernel_total += float(kernel_step)
            transfer_total += float(transfer_step)
            candidate_flat, accel_state, accel_metrics = _cext_apply_component_acceleration(
                current_c_ext_gl.reshape(-1),
                np.asarray(c_ext_new, dtype=np.float32).reshape(-1),
                {
                    "prev_residual_flat": prev_residual_flat,
                    "prev_abs_residual": prev_abs_residual,
                    "stable_iters": stable_iters,
                    "anderson_g_history": anderson_g_history,
                    "anderson_r_history": anderson_r_history,
                    "accel_rejections": accel_rejections,
                    "accel_restarts": accel_restarts,
                    "anderson_depth_used": anderson_depth_used,
                    "omega_last": omega_last,
                    "iter_count": completed_iters,
                },
                accel_mode=accel_mode,
            )
            ext_state["c_ext_gl"] = np.asarray(candidate_flat.reshape(np.asarray(ext_state["c_ext_gl"]).shape), dtype=np.float32)
            completed_iters = iter_idx
            prev_source_c_ext_gl = source_c_ext_snapshot
            prev_source_c_iv_gl = source_c_iv_snapshot
            prev_residual_flat = np.asarray(accel_state["prev_residual_flat"], dtype=np.float32).copy()
            prev_abs_residual = float(accel_state["prev_abs_residual"])
            stable_iters = int(accel_state["stable_iters"])
            anderson_g_history = [np.asarray(arr, dtype=np.float32).copy() for arr in accel_state["anderson_g_history"]]
            anderson_r_history = [np.asarray(arr, dtype=np.float32).copy() for arr in accel_state["anderson_r_history"]]
            accel_rejections = int(accel_metrics["accel_rejections"])
            accel_restarts = int(accel_metrics["accel_restarts"])
            anderson_depth_used = int(accel_metrics["anderson_depth_used"])
            omega_last = float(accel_metrics["omega_last"])
            accel_step_last = str(accel_metrics["accel_step_last"])
            max_delta_last = float(accel_metrics["max_delta_last"])
            rel_residual_last = float(accel_metrics["rel_residual_last"])
            active_target_count = int(nseg)
            frozen_target_count = 0
            active_component_count = 1
            largest_component_size = int(nseg)
        if SOLVER_TIMING_DETAILS:
            print(
                f"    Cext iter {iter_idx}/{int(CEXT_VESS_COUPLING_MAX_ITER)}: "
                f"max_delta={max_delta_last:.3e} rel={rel_residual_last:.3e} "
                f"step={accel_step_last} omega={omega_last:.3f} stable={stable_iters} "
                f"active_src={active_source_count} frozen_src={frozen_source_count} "
                f"active_tgt={active_target_count} frozen_tgt={frozen_target_count} "
                f"comps={active_component_count} largest={largest_component_size}"
            )
        _maybe_trace_cext_iteration(
            iter_idx,
            ext_state,
            solver="topdown_ext_image",
            context=context,
            backend=backend,
            max_delta=max_delta_last,
            rel_residual=rel_residual_last,
        )
        if max_delta_last < float(CEXT_VESS_COUPLING_TOL):
            break
        if float(CEXT_VESS_COUPLING_REL_TOL) > 0.0 and rel_residual_last < float(CEXT_VESS_COUPLING_REL_TOL):
            break

    # Recompute intravascular quadrature values against the final relaxed Cext
    # field, then cache q_line = k_if * (Civ - Cext) for the final tissue solve.
    cin_seg, cout_seg, c_iv_gl, frozen_backend, final_frozen_transfer = _run_topdown_ext_frozen_step(
        context,
        ext_state,
        inlet_concentration=float(inlet_concentration),
        vmax=float(vmax),
        km=float(km),
        chb_max=np.asarray(chb_max, dtype=np.float32),
        fluid_mode=fluid_mode,
        frozen_backend=frozen_backend,
    )
    frozen_transfer_total += float(final_frozen_transfer)
    ext_state["cin_seg"] = np.asarray(cin_seg, dtype=np.float32)
    ext_state["cout_seg"] = np.asarray(cout_seg, dtype=np.float32)
    ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    _build_cext_iteration_cache(context, ext_state)
    _set_last_cext_source_state(
        context,
        ext_state,
        solver="topdown_ext",
        backend=backend,
    )

    total_elapsed = perf_counter() - t_total
    _LAST_CONCENTRATION_TIMINGS = {
        "backend": backend,
        "t_cext_total_s": float(total_elapsed),
        "t_cext_query_s": float(query_total),
        "t_cext_kernel_s": float(kernel_total),
        "t_cext_gpu_transfer_s": float(transfer_total),
        "t_cext_frozen_gpu_transfer_s": float(frozen_transfer_total),
        "t_cext_init_s": float(init_elapsed),
        "t_cext_init_kernel_s": float(init_kernel_total),
        "t_cext_init_gpu_transfer_s": float(init_transfer_total),
        "t_cext_init_frozen_gpu_transfer_s": float(init_frozen_transfer_total),
        "cext_init_mode": init_mode,
        "cext_init_performed": bool(init_performed),
        "cext_outer_iterations_completed": int(completed_iters),
        "cext_total_iterations_effective": int(completed_iters + (1 if init_performed else 0)),
        "cext_accel_mode": accel_mode,
        "cext_accel_step_last": accel_step_last,
        "cext_accel_rejections": int(accel_rejections),
        "cext_accel_restarts": int(accel_restarts),
        "cext_anderson_depth_used": int(anderson_depth_used),
        "cext_omega_last": float(omega_last),
        "cext_rel_residual_last": float(rel_residual_last),
        "cext_max_delta_last": float(max_delta_last),
        "cext_candidate_batches": int(len(context.get("candidate_batches", ()))),
        "cext_candidate_query_mode": str(context.get("candidate_query_mode", "grid")),
        "cext_frozen_backend": str(frozen_backend),
        "cext_active_source_count": int(active_source_count),
        "cext_frozen_source_count": int(frozen_source_count),
        "cext_active_target_count": int(active_target_count),
        "cext_frozen_target_count": int(frozen_target_count),
        "cext_active_component_count": int(active_component_count),
        "cext_largest_component_size": int(largest_component_size),
        "cext_active_freeze_events": int(active_freeze_events),
        "cext_active_refreshes": int(active_refreshes),
        "cext_target_freeze_events": int(target_freeze_events),
        "cext_target_reactivations": int(target_reactivations),
    }
    return np.asarray(ext_state["cin_seg"], dtype=float), np.asarray(ext_state["cout_seg"], dtype=float)


def _solve_channel_concentrations_topdown_ext_treecode(
    tree: Tree,
    flows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    fluid: str,
) -> Tuple[np.ndarray, np.ndarray]:
    global _LAST_CONCENTRATION_TIMINGS
    if float(CEXT_WINDOW_FACTOR) <= 0.0:
        cin, cout = _solve_channel_concentrations_topdown(
            tree,
            flows,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
        _LAST_CONCENTRATION_TIMINGS = _default_cext_timing_details(backend="disabled")
        return cin, cout
    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty
    if _cp is None:
        return _solve_channel_concentrations_topdown_ext(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )

    t_total = perf_counter()
    fluid_mode = (fluid or getattr(getattr(tree, "parameters", None), "fluid", None) or ACTIVE_FLUID).lower()
    hct_context = _hematocrit_context_for_tree(tree)
    radii_arr = np.asarray(hct_context["radii"], dtype=float)
    flows_arr = np.asarray(flows, dtype=float)
    chb_max = np.zeros_like(radii_arr)
    hct_source = "none"
    if fluid_mode == "blood":
        cached_hct = _get_tree_hematocrit_cache(
            tree,
            nseg,
            model=HEMATOCRIT_MODEL,
            flows=flows_arr,
        )
        if cached_hct is not None:
            HD, HT = cached_hct
            hct_source = "cache"
        else:
            HD, HT = compute_tree_hematocrit(
                tree,
                hd_root=HD_DISCHARGE,
                flows=flows_arr,
                model=HEMATOCRIT_MODEL,
                order=np.asarray(hct_context["order"], dtype=np.int64),
            )
            _store_tree_hematocrit_cache(
                tree,
                HD,
                HT,
                model=HEMATOCRIT_MODEL,
                flows=flows_arr,
                fixed_flow_bc=False,
            )
            hct_source = "computed"
        chb_max = np.asarray(HT, dtype=float) * float(O2_CAP_PER_HCT)
    hct_setup_s = perf_counter() - t_stage

    t_stage = perf_counter()
    context = _build_cext_geometry_context(
        tree,
        flows_arr,
        starts,
        ends,
        radii,
        lengths,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=vmax,
        km=km,
    )
    backend = _resolve_cext_accel_mode()
    frozen_backend = _resolve_cext_frozen_accel_mode()
    if backend != "gpu":
        return _solve_channel_concentrations_topdown_ext(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    context["candidate_query_mode"] = "treecode"
    context["candidate_build_time_s"] = 0.0
    t_build = perf_counter()
    treecode = _build_cext_treecode_context(context)
    leaf_scale = float(treecode["side"]) / float(max(1 << max(int(treecode["max_depth"]), 0), 1))
    treecode["near_radius_si"] = float(max(float(CEXT_TREECODE_NEAR_RADIUS_MULT), 1.0) * max(leaf_scale, 1.0e-9))
    treecode_build_s = perf_counter() - t_build

    cache_key = _cext_state_cache_key(
        fluid_mode=fluid_mode,
        inlet_concentration=float(inlet_concentration),
        diffusivity_si=float(diffusivity * CM2_TO_M2),
        vmax=float(vmax),
        km=float(km),
    )
    ext_state = _initialize_cext_state(
        tree,
        cache_key=cache_key,
        nseg=nseg,
        inlet_concentration=float(inlet_concentration),
        vmax=float(vmax),
        km=float(km),
    )
    accel_mode = str(CEXT_VESS_COUPLING_ACCEL or "none").strip().lower()
    if accel_mode not in ("none", "aitken", "anderson"):
        raise ValueError("--cext-vess-coupling-accel must be 'none', 'aitken', or 'anderson'.")
    init_mode = str(CEXT_INIT_MODE or "zero").strip().lower()
    if init_mode not in ("zero", "decoupled_greens"):
        raise ValueError("--cext-init-mode must be 'zero' or 'decoupled_greens'.")

    aggregate_total = 0.0
    traversal_total = 0.0
    transfer_total = 0.0
    frozen_transfer_total = 0.0
    completed_iters = 0
    max_delta_last = 0.0
    rel_residual_last = 0.0
    omega_last = _clip_cext_omega(CEXT_VESS_COUPLING_OMEGA)
    accel_step_last = "picard"
    accel_rejections = 0
    accel_restarts = 0
    anderson_depth_used = 0
    prev_residual_flat: np.ndarray | None = None
    prev_abs_residual = float("inf")
    stable_iters = 0
    anderson_g_history: list[np.ndarray] = []
    anderson_r_history: list[np.ndarray] = []
    residual_history: list[float] = []
    tail_mode_entered = False
    tail_core_count = 0
    tail_nonlinear_iters = 0
    tail_gmres_iters = 0
    accepted_nodes_last = 0
    opened_nodes_last = 0
    exact_pairs_last = 0
    active_source_mask = np.ones((nseg,), dtype=bool)
    active_source_stable_counts = np.zeros((nseg,), dtype=np.int16)
    prev_source_c_ext_gl: np.ndarray | None = None
    prev_source_c_iv_gl: np.ndarray | None = None
    ext_state["treecode_active_source_mask"] = np.asarray(active_source_mask, dtype=bool)

    if SOLVER_TIMING_DETAILS:
        print(
            "  Concentration topdown_ext_treecode: "
            f"nseg={nseg} fluid={fluid_mode} backend={backend} frozen={frozen_backend} gl_order={GL_ORDER_CEXT} "
            f"hct={hct_source} accel={accel_mode} theta={float(treecode['theta']):.3f} "
            f"cext_lambda={_normalize_cext_lambda_source()} "
            f"leaf={int(treecode['leaf_nodes'])} bins={int(treecode['lambda_bins'])} "
            f"near={float(treecode['near_radius_si']):.3e} nodes={int(treecode['node_count'])} "
            f"leaves={int(treecode['leaf_count'])} depth={int(treecode['max_depth'])}"
        )

    if init_mode == "decoupled_greens":
        cin_seg, cout_seg, c_iv_gl, frozen_backend, frozen_transfer = _run_topdown_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=float(inlet_concentration),
            vmax=float(vmax),
            km=float(km),
            chb_max=np.asarray(chb_max, dtype=np.float32),
            fluid_mode=fluid_mode,
            frozen_backend=frozen_backend,
        )
        frozen_transfer_total += float(frozen_transfer)
        ext_state["cin_seg"] = cin_seg
        ext_state["cout_seg"] = cout_seg
        ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
        ext_state["treecode_active_source_mask"] = np.asarray(active_source_mask, dtype=bool)
        _build_cext_iteration_cache(context, ext_state)
        c_ext_new, tc_metrics = _compute_cext_treecode_gpu(context, treecode, ext_state)
        aggregate_total += float(tc_metrics.get("aggregate_s", 0.0))
        traversal_total += float(tc_metrics.get("traversal_s", 0.0))
        transfer_total += float(tc_metrics.get("transfer_s", 0.0))
        ext_state["c_ext_gl"] = np.asarray(c_ext_new, dtype=np.float32)
        accepted_nodes_last = int(tc_metrics.get("accepted_nodes", 0))
        opened_nodes_last = int(tc_metrics.get("opened_nodes", 0))
        exact_pairs_last = int(tc_metrics.get("exact_pairs", 0))
        if SOLVER_TIMING_DETAILS:
            print(
                "    Cext init predictor: "
                f"max_cext={float(np.nanmax(np.asarray(ext_state['c_ext_gl'], dtype=float))) if ext_state['c_ext_gl'].size else 0.0:.3e}"
            )
        _maybe_trace_cext_iteration(0, ext_state, solver="topdown_ext_treecode", context=context, backend=backend)

    if not init_performed:
        _maybe_trace_cext_iteration(0, ext_state, solver="topdown_ext_treecode", context=context, backend=backend)

    for iter_idx in range(1, int(CEXT_VESS_COUPLING_MAX_ITER) + 1):
        cin_seg, cout_seg, c_iv_gl, frozen_backend, frozen_transfer = _run_topdown_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=float(inlet_concentration),
            vmax=float(vmax),
            km=float(km),
            chb_max=np.asarray(chb_max, dtype=np.float32),
            fluid_mode=fluid_mode,
            frozen_backend=frozen_backend,
        )
        frozen_transfer_total += float(frozen_transfer)
        ext_state["cin_seg"] = cin_seg
        ext_state["cout_seg"] = cout_seg
        ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
        ext_state["treecode_active_source_mask"] = np.asarray(active_source_mask, dtype=bool)
        _build_cext_iteration_cache(context, ext_state)
        current_c_ext_gl = np.asarray(ext_state["c_ext_gl"], dtype=np.float32)
        c_ext_new, tc_metrics = _compute_cext_treecode_gpu(context, treecode, ext_state)
        aggregate_total += float(tc_metrics.get("aggregate_s", 0.0))
        traversal_total += float(tc_metrics.get("traversal_s", 0.0))
        transfer_total += float(tc_metrics.get("transfer_s", 0.0))
        accepted_nodes_last = int(tc_metrics.get("accepted_nodes", 0))
        opened_nodes_last = int(tc_metrics.get("opened_nodes", 0))
        exact_pairs_last = int(tc_metrics.get("exact_pairs", 0))

        candidate_flat, accel_state, accel_metrics = _cext_apply_component_acceleration(
            current_c_ext_gl.reshape(-1),
            np.asarray(c_ext_new, dtype=np.float32).reshape(-1),
            {
                "prev_residual_flat": prev_residual_flat,
                "prev_abs_residual": prev_abs_residual,
                "stable_iters": stable_iters,
                "anderson_g_history": anderson_g_history,
                "anderson_r_history": anderson_r_history,
                "accel_rejections": accel_rejections,
                "accel_restarts": accel_restarts,
                "anderson_depth_used": anderson_depth_used,
                "omega_last": omega_last,
                "iter_count": completed_iters,
            },
            accel_mode=accel_mode,
        )
        ext_state["c_ext_gl"] = np.asarray(candidate_flat.reshape(current_c_ext_gl.shape), dtype=np.float32)
        completed_iters = iter_idx
        prev_residual_flat = np.asarray(accel_state["prev_residual_flat"], dtype=np.float32).copy()
        prev_abs_residual = float(accel_state["prev_abs_residual"])
        stable_iters = int(accel_state["stable_iters"])
        anderson_g_history = [np.asarray(arr, dtype=np.float32).copy() for arr in accel_state["anderson_g_history"]]
        anderson_r_history = [np.asarray(arr, dtype=np.float32).copy() for arr in accel_state["anderson_r_history"]]
        accel_rejections = int(accel_metrics["accel_rejections"])
        accel_restarts = int(accel_metrics["accel_restarts"])
        anderson_depth_used = int(accel_metrics["anderson_depth_used"])
        omega_last = float(accel_metrics["omega_last"])
        accel_step_last = str(accel_metrics["accel_step_last"])
        max_delta_last = float(accel_metrics["max_delta_last"])
        rel_residual_last = float(accel_metrics["rel_residual_last"])
        residual_history.append(float(rel_residual_last))

        source_abs_delta, source_rel_delta = _cext_source_state_delta_metrics(
            prev_source_c_ext_gl,
            prev_source_c_iv_gl,
            np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
            np.asarray(ext_state["c_iv_gl"], dtype=np.float32),
        )
        prev_source_c_ext_gl = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).copy()
        prev_source_c_iv_gl = np.asarray(ext_state["c_iv_gl"], dtype=np.float32).copy()
        q_weighted = np.asarray(ext_state.get("q_weighted_gl", np.zeros_like(ext_state["c_ext_gl"])), dtype=np.float32)
        q_seg = np.max(np.abs(q_weighted), axis=1).astype(np.float32, copy=False) if q_weighted.ndim == 2 else np.zeros((nseg,), dtype=np.float32)
        q_scale = max(float(np.max(q_seg)) if q_seg.size else 0.0, float(VESS_CONC_FLOOR))
        q_rel = np.divide(
            q_seg,
            np.float32(q_scale),
            out=np.zeros_like(q_seg, dtype=np.float32),
            where=q_scale > 0.0,
        )
        stable_now = (
            np.asarray(source_abs_delta, dtype=np.float32) <= float(CEXT_ACTIVE_SET_ABS_TOL)
        ) & (
            np.asarray(source_rel_delta, dtype=np.float32) <= float(CEXT_ACTIVE_SET_REL_TOL)
        )
        active_source_stable_counts[active_source_mask & stable_now] = np.minimum(
            np.asarray(active_source_stable_counts[active_source_mask & stable_now], dtype=np.int32) + 1,
            np.int16(np.iinfo(np.int16).max),
        ).astype(np.int16, copy=False)
        active_source_stable_counts[active_source_mask & (~stable_now)] = np.int16(0)
        reactivate_mask = (~active_source_mask) & (
            (np.asarray(source_abs_delta, dtype=np.float32) > float(CEXT_ACTIVE_SET_ABS_TOL) * 2.0)
            | (np.asarray(source_rel_delta, dtype=np.float32) > float(CEXT_ACTIVE_SET_REL_TOL) * 2.0)
            | (q_rel > float(CEXT_TREECODE_FREEZE_QREL_TOL) * 1.5)
        )
        if np.any(reactivate_mask):
            active_source_mask[reactivate_mask] = True
            active_source_stable_counts[reactivate_mask] = np.int16(0)
        if bool(CEXT_ACTIVE_SET_ENABLE) and iter_idx >= int(CEXT_ACTIVE_SET_START):
            min_active_sources = max(1, int(CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT))
            freeze_candidates = active_source_mask & (
                active_source_stable_counts >= int(CEXT_ACTIVE_SET_STABLE_ITERS)
            ) & (
                q_rel <= float(CEXT_TREECODE_FREEZE_QREL_TOL)
            )
            active_count_now = int(np.count_nonzero(active_source_mask))
            if np.any(freeze_candidates) and active_count_now > min_active_sources:
                freeze_ids = np.flatnonzero(freeze_candidates).astype(np.int32, copy=False)
                max_freeze = max(active_count_now - min_active_sources, 0)
                if freeze_ids.size > max_freeze:
                    order = np.lexsort((
                        np.asarray(source_abs_delta[freeze_ids], dtype=np.float32),
                        np.asarray(source_rel_delta[freeze_ids], dtype=np.float32),
                        np.asarray(q_rel[freeze_ids], dtype=np.float32),
                    ))
                    freeze_ids = np.asarray(freeze_ids[order[:max_freeze]], dtype=np.int32)
                if freeze_ids.size > 0:
                    active_source_mask[freeze_ids] = False
                    active_source_stable_counts[freeze_ids] = np.int16(0)
        ext_state["treecode_active_source_mask"] = np.asarray(active_source_mask, dtype=bool)

        core_ids, _ = _estimate_cext_tail_active_core(context, ext_state, np.asarray(c_ext_new, dtype=np.float32))
        tail_core_count = int(core_ids.size)
        if (not tail_mode_entered) and _cext_should_enter_tail_mode(residual_history, tail_core_count, iter_idx=iter_idx):
            tail_mode_entered = True
            if SOLVER_TIMING_DETAILS:
                print(f"    Cext tail mode: solver={CEXT_TAIL_SOLVER} core={tail_core_count}")
            tail_full, tail_metrics = _solve_cext_treecode_active_core_nk(
                context,
                treecode,
                ext_state,
                core_seg_ids=core_ids,
                inlet_concentration=float(inlet_concentration),
                vmax=float(vmax),
                km=float(km),
                chb_max=np.asarray(chb_max, dtype=np.float32),
                fluid_mode=fluid_mode,
                frozen_backend=frozen_backend,
            )
            tail_nonlinear_iters = int(tail_metrics.get("nonlinear_iters", 0))
            tail_gmres_iters = int(tail_metrics.get("gmres_iters", 0))
            ext_state["c_ext_gl"] = np.asarray(tail_full, dtype=np.float32)
            if isinstance(tail_metrics.get("state"), dict):
                tail_state = dict(tail_metrics["state"])
                for key in (
                    "cin_seg", "cout_seg", "c_iv_gl", "c_bulk_gl", "c_wall_gl",
                    "lambda_iv_gl", "k_if_gl", "q_line_gl", "q_weighted_gl",
                    "mono2_weight_gl", "dipole2_weight_gl", "seg_cap_gl",
                ):
                    if key in tail_state:
                        ext_state[key] = np.asarray(tail_state[key], dtype=np.float32) if key.endswith("_gl") or key.endswith("_seg") else tail_state[key]
            frozen_backend = str(tail_metrics.get("frozen_backend", frozen_backend))
            rel_residual_last = float(tail_metrics.get("rel_residual", rel_residual_last))
            max_delta_last = float(rel_residual_last)
            accel_step_last = str(CEXT_TAIL_SOLVER)
            if bool(tail_metrics.get("success", False)):
                if SOLVER_TIMING_DETAILS:
                    print(
                        f"    Cext tail solved: rel={rel_residual_last:.3e} "
                        f"gmres={tail_gmres_iters} nonlinear={tail_nonlinear_iters}"
                    )
                break

        if SOLVER_TIMING_DETAILS:
            print(
                f"    Cext iter {iter_idx}/{int(CEXT_VESS_COUPLING_MAX_ITER)}: "
                f"max_delta={max_delta_last:.3e} rel={rel_residual_last:.3e} "
                f"step={accel_step_last} omega={omega_last:.3f} stable={stable_iters} "
                f"active_src={int(np.count_nonzero(active_source_mask))} "
                f"frozen_src={int(nseg - np.count_nonzero(active_source_mask))} active_tgt={nseg} frozen_tgt=0 "
                f"accepted={accepted_nodes_last} opened={opened_nodes_last} exact_pairs={exact_pairs_last}"
            )
        if max_delta_last < float(CEXT_VESS_COUPLING_TOL):
            break
        if float(CEXT_VESS_COUPLING_REL_TOL) > 0.0 and rel_residual_last < float(CEXT_VESS_COUPLING_REL_TOL):
            break

    total_elapsed = perf_counter() - t_total
    _LAST_CONCENTRATION_TIMINGS = _default_cext_timing_details(backend=backend)
    _LAST_CONCENTRATION_TIMINGS.update(
        {
            "backend": backend,
            "t_cext_total_s": float(total_elapsed),
            "t_cext_kernel_s": float(traversal_total),
            "t_cext_gpu_transfer_s": float(transfer_total),
            "t_cext_frozen_gpu_transfer_s": float(frozen_transfer_total),
            "t_cext_treecode_build_s": float(treecode_build_s),
            "t_cext_treecode_aggregate_s": float(aggregate_total),
            "t_cext_treecode_traversal_s": float(traversal_total),
            "t_cext_treecode_exact_s": 0.0,
            "cext_treecode_theta": float(treecode["theta"]),
            "cext_treecode_order": int(treecode["order_mode"]),
            "cext_treecode_leaf_nodes": int(treecode["leaf_nodes"]),
            "cext_treecode_lambda_bins": int(treecode["lambda_bins"]),
            "cext_treecode_near_radius_si": float(treecode["near_radius_si"]),
            "cext_treecode_node_count": int(treecode["node_count"]),
            "cext_treecode_leaf_count": int(treecode["leaf_count"]),
            "cext_treecode_max_depth": int(treecode["max_depth"]),
            "cext_treecode_accepted_nodes": int(accepted_nodes_last),
            "cext_treecode_opened_nodes": int(opened_nodes_last),
            "cext_treecode_exact_pairs": int(exact_pairs_last),
            "cext_init_mode": init_mode,
            "cext_outer_iterations_completed": int(completed_iters),
            "cext_total_iterations_effective": int(completed_iters + (1 if init_mode == "decoupled_greens" else 0)),
            "cext_accel_mode": accel_mode,
            "cext_accel_step_last": accel_step_last,
            "cext_accel_rejections": int(accel_rejections),
            "cext_accel_restarts": int(accel_restarts),
            "cext_anderson_depth_used": int(anderson_depth_used),
            "cext_omega_last": float(omega_last),
            "cext_rel_residual_last": float(rel_residual_last),
            "cext_max_delta_last": float(max_delta_last),
            "cext_candidate_batches": 0,
            "cext_candidate_query_mode": "treecode",
            "cext_frozen_backend": str(frozen_backend),
            "cext_active_source_count": int(nseg),
            "cext_frozen_source_count": 0,
            "cext_active_target_count": int(nseg),
            "cext_frozen_target_count": 0,
            "cext_tail_mode_entered": bool(tail_mode_entered),
            "cext_tail_solver": str(CEXT_TAIL_SOLVER if tail_mode_entered else "none"),
            "cext_tail_core_count": int(tail_core_count),
            "cext_tail_nonlinear_iters": int(tail_nonlinear_iters),
            "cext_tail_gmres_iters": int(tail_gmres_iters),
        }
    )
    return np.asarray(ext_state["cin_seg"], dtype=float), np.asarray(ext_state["cout_seg"], dtype=float)


def _solve_channel_concentrations_topdown_ext_hybrid_bg(
    tree: Tree,
    flows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    fluid: str,
) -> Tuple[np.ndarray, np.ndarray]:
    global _LAST_CONCENTRATION_TIMINGS
    if float(CEXT_WINDOW_FACTOR) <= 0.0:
        cin, cout = _solve_channel_concentrations_topdown(
            tree,
            flows,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
        _LAST_CONCENTRATION_TIMINGS = _default_cext_timing_details(backend="disabled")
        return cin, cout

    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty

    if _cp is None:
        return _solve_channel_concentrations_topdown_ext(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )

    t_total = perf_counter()
    t_stage = t_total
    fluid_mode = (fluid or getattr(getattr(tree, "parameters", None), "fluid", None) or ACTIVE_FLUID).lower()
    hct_context = _hematocrit_context_for_tree(tree)
    radii_arr = np.asarray(hct_context["radii"], dtype=float)
    flows_arr = np.asarray(flows, dtype=float)
    chb_max = np.zeros_like(radii_arr)
    hct_source = "none"
    if fluid_mode == "blood":
        cached_hct = _get_tree_hematocrit_cache(
            tree,
            nseg,
            model=HEMATOCRIT_MODEL,
            flows=flows_arr,
        )
        if cached_hct is not None:
            HD, HT = cached_hct
            hct_source = "cache"
        else:
            HD, HT = compute_tree_hematocrit(
                tree,
                hd_root=HD_DISCHARGE,
                flows=flows_arr,
                model=HEMATOCRIT_MODEL,
                order=np.asarray(hct_context["order"], dtype=np.int64),
            )
            _store_tree_hematocrit_cache(
                tree,
                HD,
                HT,
                model=HEMATOCRIT_MODEL,
                flows=flows_arr,
                fixed_flow_bc=False,
            )
            hct_source = "computed"
        chb_max = np.asarray(HT, dtype=float) * float(O2_CAP_PER_HCT)
    hct_setup_s = perf_counter() - t_stage

    t_stage = perf_counter()
    context = _build_cext_geometry_context(
        tree,
        flows_arr,
        starts,
        ends,
        radii,
        lengths,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=vmax,
        km=km,
        build_candidate_index=False,
    )
    context_build_s = perf_counter() - t_stage

    t_stage = perf_counter()
    backend = _resolve_cext_accel_mode()
    frozen_backend = _resolve_cext_frozen_accel_mode()
    if backend != "gpu":
        return _solve_channel_concentrations_topdown_ext(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    context["candidate_query_mode"] = "hybrid_bg"
    context["candidate_build_time_s"] = 0.0
    hybrid = _ensure_cext_hybrid_bg_context(context)
    cache_key = _cext_state_cache_key(
        fluid_mode=fluid_mode,
        inlet_concentration=float(inlet_concentration),
        diffusivity_si=float(diffusivity * CM2_TO_M2),
        vmax=float(vmax),
        km=float(km),
    )
    ext_state = _initialize_cext_state(
        tree,
        cache_key=cache_key,
        nseg=nseg,
        inlet_concentration=float(inlet_concentration),
        vmax=float(vmax),
        km=float(km),
    )

    query_total = 0.0
    kernel_total = 0.0
    transfer_total = 0.0
    frozen_transfer_total = 0.0
    hybrid_deposit_total = 0.0
    hybrid_fft_total = 0.0
    hybrid_o2_total = 0.0
    hybrid_self_sub_total = 0.0
    hybrid_local_total = 0.0
    hybrid_sample_total = 0.0
    hybrid_bg_solver_mode = ""
    init_elapsed = 0.0
    init_kernel_total = 0.0
    init_transfer_total = 0.0
    init_frozen_transfer_total = 0.0
    init_frozen_kernel_total = 0.0
    init_cache_total = 0.0
    init_hybrid_call_total = 0.0
    iteration_cache_total = 0.0
    final_frozen_transfer_total = 0.0
    final_frozen_kernel_total = 0.0
    final_cache_total = 0.0
    final_source_state_total = 0.0
    init_performed = False
    completed_iters = 0
    max_delta_last = 0.0
    rel_residual_last = 0.0
    omega_last = _clip_cext_omega(CEXT_VESS_COUPLING_OMEGA)
    accel_step_last = "picard"
    accel_rejections = 0
    accel_restarts = 0
    anderson_depth_used = 0
    init_mode = str(CEXT_INIT_MODE or "zero").strip().lower()
    accel_mode = str(CEXT_VESS_COUPLING_ACCEL or "none").strip().lower()
    if accel_mode not in ("none", "aitken", "anderson"):
        raise ValueError("--cext-vess-coupling-accel must be 'none', 'aitken', or 'anderson'.")
    if init_mode not in ("zero", "decoupled_greens"):
        raise ValueError("--cext-init-mode must be 'zero' or 'decoupled_greens'.")

    prev_residual_flat: np.ndarray | None = None
    prev_abs_residual = float("inf")
    stable_iters = 0
    anderson_g_history: list[np.ndarray] = []
    anderson_r_history: list[np.ndarray] = []
    best_c_ext_gl: np.ndarray | None = None
    best_max_delta = float("inf")
    best_rel_residual = float("inf")
    best_iter = 0
    best_stall_iters = 0
    best_seen_stop = False
    omega_floor, _omega_ceiling = _cext_relaxation_bounds()
    last_eval_c_ext_gl: np.ndarray | None = None
    last_eval_mapped_gl: np.ndarray | None = None
    last_eval_max_delta = float("inf")
    last_eval_omega = float(omega_last)
    step_rejections = 0

    active_set_enabled = bool(CEXT_ACTIVE_SET_ENABLE) and bool(CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING)
    if _resolve_cext_hybrid_bg_mode() == "fft" and bool(CEXT_HYBRID_FFT_QUANTILE_BINS):
        active_set_enabled = False
    # The target-freezing path was designed for direct solves. In the hybrid
    # solver it adds large CPU neighbor scans but rarely freezes anything,
    # while local-target pruning already shrinks the expensive GPU correction.
    target_active_set_enabled = False
    # Earlier freezing was faster only when the map was already close. On large
    # hybrid runs it changed the operator too early and caused residual spikes.
    hybrid_active_set_start = max(int(CEXT_ACTIVE_SET_START), 6)
    hybrid_active_set_stable_iters = max(int(CEXT_ACTIVE_SET_STABLE_ITERS), 3)
    hybrid_freeze_resid_gate = max(
        10.0 * float(CEXT_VESS_COUPLING_TOL),
        float(CEXT_ACTIVE_SET_ABS_TOL),
    )
    active_source_mask = np.ones((nseg,), dtype=bool)
    active_source_stable_counts = np.zeros((nseg,), dtype=np.int16)
    active_freeze_events = 0
    active_refreshes = 0
    last_active_refresh_iter = 0
    active_source_count = int(nseg)
    frozen_source_count = 0
    active_source_plan: dict | None = None
    prev_source_c_ext_gl: np.ndarray | None = None
    prev_source_c_iv_gl: np.ndarray | None = None
    min_active_sources = max(
        1,
        int(CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT),
        int(math.ceil(max(float(CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION), 0.0) * float(nseg))),
    )
    frozen_mass_grids_g = None
    frozen_phi_grids_g = None
    active_target_mask = np.ones((nseg,), dtype=bool)
    active_target_stable_counts = np.zeros((nseg,), dtype=np.int16)
    active_target_count = int(nseg)
    frozen_target_count = 0
    target_freeze_events = 0
    target_reactivations = 0
    active_component_count = 1
    largest_component_size = int(nseg)
    min_active_targets = max(1, int(CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT))
    active_target_plan: dict | None = None
    target_mask_changed = False
    runtime_state = _ensure_cext_hybrid_bg_runtime_state(context, hybrid, ext_state)
    gpu_iteration_cache_enabled = (
        bool(CEXT_HYBRID_GPU_ITERATION_CACHE)
        and str(_resolve_cext_hybrid_bg_mode()).strip().lower() == "fft"
        and not bool(active_set_enabled)
    )
    all_target_ids = np.arange(nseg, dtype=np.int32)
    setup_s = perf_counter() - t_stage

    if SOLVER_TIMING_DETAILS:
        print(
            "  Concentration topdown_ext_hybrid_bg: "
            f"nseg={nseg} fluid={fluid_mode} backend={backend} frozen={frozen_backend} gl_order={GL_ORDER_CEXT} "
            f"hct={hct_source} accel={accel_mode} active_set={'on' if active_set_enabled else 'off'} "
            f"wall={LUMEN_WALL_CLOSURE} o2_terms={FINITE_RADIUS_O2_TERMS} "
            f"cext_lambda={_normalize_cext_lambda_source()} "
            f"graetz_max_fp_iters={GRAETZ_MAX_FP_ITERS} tail_buffer_patch=enabled "
            f"grid={int(hybrid['grid_n'])} bins={int(hybrid['lambda_bins'])} "
            f"bg_mode={_resolve_cext_hybrid_bg_mode()} "
            f"bg_solver={_resolve_cext_hybrid_bg_solver_mode()} "
            f"near={float(hybrid['near_radius_si']):.3e}"
        )

    if init_mode == "decoupled_greens":
        t_init = perf_counter()
        cin_seg, cout_seg, c_iv_gl, frozen_backend, init_frozen_transfer = _run_topdown_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=float(inlet_concentration),
            vmax=float(vmax),
            km=float(km),
            chb_max=np.asarray(chb_max, dtype=np.float32),
            fluid_mode=fluid_mode,
            frozen_backend=frozen_backend,
        )
        init_frozen_timings = dict(_LAST_CEXT_FROZEN_STEP_TIMINGS)
        ext_state["cin_seg"] = cin_seg
        ext_state["cout_seg"] = cout_seg
        ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
        t_cache = perf_counter()
        if gpu_iteration_cache_enabled:
            _build_cext_iteration_cache_gpu(context, ext_state, runtime_state)
        else:
            _build_cext_iteration_cache(context, ext_state)
        init_cache_total += perf_counter() - t_cache
        _cext_hybrid_init_lambda_bins(hybrid, ext_state)
        if not gpu_iteration_cache_enabled:
            _sync_cext_hybrid_bg_runtime_state(ext_state, runtime_state)
        t_hybrid = perf_counter()
        c_ext_new, hybrid_timings = _compute_cext_hybrid_bg_gpu(
            context,
            ext_state,
            hybrid,
            active_source_plan=active_source_plan,
            target_plan=active_target_plan,
            runtime_state=runtime_state,
        )
        init_hybrid_call_total += perf_counter() - t_hybrid
        ext_state["c_ext_gl"] = np.asarray(c_ext_new, dtype=np.float32)
        runtime_state["c_ext_gl"].set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))
        hybrid_bg_solver_mode = str(hybrid_timings.get("bg_solver_mode", hybrid_bg_solver_mode or _resolve_cext_hybrid_bg_solver_mode()))
        init_elapsed = perf_counter() - t_init
        init_kernel_total = (
            float(hybrid_timings.get("deposit_s", 0.0))
            + float(hybrid_timings.get("fft_s", 0.0))
            + float(hybrid_timings.get("fft_o2_terms_s", 0.0))
            + float(hybrid_timings.get("fft_discrete_self_subtract_s", 0.0))
            + float(hybrid_timings.get("local_corr_s", 0.0))
            + float(hybrid_timings.get("sample_s", 0.0))
        )
        init_transfer_total = float(hybrid_timings.get("transfer_s", 0.0))
        init_frozen_transfer_total = float(init_frozen_transfer)
        init_frozen_kernel_total = float(init_frozen_timings.get("kernel_s", 0.0))
        init_performed = True
        if SOLVER_TIMING_DETAILS:
            print(
                "    Cext init predictor: "
                f"max_cext={float(np.nanmax(np.asarray(ext_state['c_ext_gl'], dtype=float))) if ext_state['c_ext_gl'].size else 0.0:.3e}"
            )
        _maybe_trace_cext_iteration(0, ext_state, solver="topdown_ext_hybrid_bg", context=context, backend=backend)

    if not init_performed:
        _maybe_trace_cext_iteration(0, ext_state, solver="topdown_ext_hybrid_bg", context=context, backend=backend)

    for iter_idx in range(1, int(CEXT_VESS_COUPLING_MAX_ITER) + 1):
        cin_seg, cout_seg, c_iv_gl, frozen_backend, frozen_transfer = _run_topdown_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=float(inlet_concentration),
            vmax=float(vmax),
            km=float(km),
            chb_max=np.asarray(chb_max, dtype=np.float32),
            fluid_mode=fluid_mode,
            frozen_backend=frozen_backend,
        )
        frozen_transfer_total += float(frozen_transfer)
        ext_state["cin_seg"] = cin_seg
        ext_state["cout_seg"] = cout_seg
        ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
        t_iter_cache = perf_counter()
        if gpu_iteration_cache_enabled:
            _build_cext_iteration_cache_gpu(context, ext_state, runtime_state)
        else:
            _build_cext_iteration_cache(context, ext_state)
        iteration_cache_total += perf_counter() - t_iter_cache
        _cext_hybrid_init_lambda_bins(hybrid, ext_state)
        source_c_ext_snapshot = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).copy()
        source_c_iv_snapshot = np.asarray(ext_state["c_iv_gl"], dtype=np.float32).copy()
        current_c_ext_gl = np.asarray(ext_state["c_ext_gl"], dtype=np.float32)
        if not gpu_iteration_cache_enabled:
            _sync_cext_hybrid_bg_runtime_state(ext_state, runtime_state)

        refresh_due = False
        source_abs_delta = np.zeros((nseg,), dtype=np.float32)
        source_rel_delta = np.zeros((nseg,), dtype=np.float32)
        if active_set_enabled:
            source_abs_delta, source_rel_delta = _cext_source_state_delta_metrics(
                prev_source_c_ext_gl,
                prev_source_c_iv_gl,
                source_c_ext_snapshot,
                source_c_iv_snapshot,
            )
            active_mask_now = np.asarray(active_source_mask, dtype=bool)
            stable_local = np.asarray(
                (source_abs_delta <= float(CEXT_ACTIVE_SET_ABS_TOL)) | (source_rel_delta <= float(CEXT_ACTIVE_SET_REL_TOL)),
                dtype=bool,
            )
            active_source_stable_counts[active_mask_now & stable_local] += np.int16(1)
            active_source_stable_counts[active_mask_now & ~stable_local] = np.int16(0)

            freeze_now = np.zeros((nseg,), dtype=bool)
            global_freeze_ready = (
                np.isfinite(prev_abs_residual)
                and float(prev_abs_residual) <= float(hybrid_freeze_resid_gate)
            )
            if iter_idx >= int(hybrid_active_set_start) and global_freeze_ready:
                eligible = active_mask_now & (active_source_stable_counts >= int(hybrid_active_set_stable_iters))
                max_freeze = max(int(np.count_nonzero(active_mask_now)) - int(min_active_sources), 0)
                eligible_ids = np.flatnonzero(eligible)
                if max_freeze > 0 and eligible_ids.size > 0:
                    if eligible_ids.size > max_freeze:
                        rank_key = np.asarray(source_rel_delta[eligible_ids], dtype=np.float32)
                        keep_order = np.argsort(rank_key, kind="stable")[:max_freeze]
                        eligible_ids = np.asarray(eligible_ids[keep_order], dtype=np.int64)
                    freeze_now[np.asarray(eligible_ids, dtype=np.int64)] = True
                    active_source_mask[freeze_now] = False
                    active_source_stable_counts[freeze_now] = np.int16(0)
                    active_freeze_events += int(np.count_nonzero(freeze_now))
                    active_source_plan = None
            active_source_count = int(np.count_nonzero(active_source_mask))
            frozen_source_count = int(nseg - active_source_count)
            refresh_period = max(int(CEXT_ACTIVE_SET_REFRESH_PERIOD), 0)
            if frozen_source_count > 0 and refresh_period > 0 and (iter_idx - last_active_refresh_iter) >= refresh_period:
                refresh_due = True
            if frozen_source_count > 0 and (np.any(freeze_now) or refresh_due):
                frozen_mass_grids_g, frozen_phi_grids_g, frozen_timings = _rebuild_cext_hybrid_frozen_cache(
                    context,
                    ext_state,
                    hybrid,
                    ~active_source_mask,
                    runtime_state=runtime_state,
                )
                hybrid_deposit_total += float(frozen_timings.get("deposit_s", 0.0))
                hybrid_fft_total += float(frozen_timings.get("fft_s", 0.0))
                hybrid_bg_solver_mode = str(frozen_timings.get("bg_solver_mode", hybrid_bg_solver_mode or _resolve_cext_hybrid_bg_solver_mode()))
                active_refreshes += 1
                last_active_refresh_iter = int(iter_idx)
            elif frozen_source_count <= 0:
                frozen_mass_grids_g = None
                frozen_phi_grids_g = None
                active_source_plan = None
            if active_source_count < nseg and active_source_plan is None:
                active_source_plan = _build_cext_hybrid_local_source_plan(hybrid, active_source_mask)
        else:
            active_source_count = int(nseg)
            frozen_source_count = 0
            active_source_plan = None
            frozen_mass_grids_g = None
            frozen_phi_grids_g = None

        source_cell_stats = _cext_active_source_cell_stats(
            context,
            active_source_mask,
            source_abs_delta,
            source_rel_delta,
        ) if target_active_set_enabled else {"cell_count": int(context.get("gpu_direct_n_cells", 0))}

        if target_active_set_enabled and frozen_target_count > 0:
            frozen_target_ids = np.flatnonzero(~active_target_mask).astype(np.int32, copy=False)
            if frozen_target_ids.size > 0:
                neighbor_counts_frozen, neighbor_abs_frozen, neighbor_rel_frozen, _ = _cext_target_neighbor_metrics(
                    context,
                    frozen_target_ids,
                    source_cell_stats,
                    neighbor_pad=int(CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD),
                )
                reactivate = (neighbor_counts_frozen > 0) & (
                    (neighbor_abs_frozen > float(CEXT_TARGET_ACTIVE_SET_ABS_TOL))
                    | (neighbor_rel_frozen > float(CEXT_TARGET_ACTIVE_SET_REL_TOL))
                )
                if np.any(reactivate):
                    reactivate_ids = np.asarray(frozen_target_ids[np.asarray(reactivate, dtype=bool)], dtype=np.int32)
                    active_target_mask[reactivate_ids] = True
                    active_target_stable_counts[reactivate_ids] = np.int16(0)
                    target_reactivations += int(reactivate_ids.size)
                    active_target_plan = None
                    target_mask_changed = True

        if target_active_set_enabled:
            active_target_count = int(np.count_nonzero(active_target_mask))
            frozen_target_count = int(nseg - active_target_count)
            active_target_ids = np.flatnonzero(active_target_mask).astype(np.int32, copy=False)
        else:
            active_target_count = int(nseg)
            frozen_target_count = 0
            active_target_ids = all_target_ids
        active_component_count = 1 if active_target_count > 0 else 0
        largest_component_size = int(active_target_count)
        if active_target_ids.size <= 0:
            max_delta_last = 0.0
            rel_residual_last = 0.0
            accel_step_last = "picard"
            omega_last = _clip_cext_omega(CEXT_VESS_COUPLING_OMEGA)
            stable_iters = 0
            completed_iters = iter_idx
            prev_source_c_ext_gl = source_c_ext_snapshot
            prev_source_c_iv_gl = source_c_iv_snapshot
            prev_residual_flat = None
            prev_abs_residual = float("inf")
            anderson_g_history = []
            anderson_r_history = []
            break

        if active_target_plan is None:
            active_target_plan = None if active_target_count >= nseg else _build_cext_gpu_direct_target_plan(context, active_target_ids)

        if target_mask_changed:
            prev_residual_flat = None
            prev_abs_residual = float("inf")
            stable_iters = 0
            anderson_g_history = []
            anderson_r_history = []
            accel_rejections = 0
            accel_restarts = 0
            anderson_depth_used = 0
            target_mask_changed = False

        local_target_ids = active_target_ids
        local_target_positions = None
        local_target_plan = active_target_plan
        if active_set_enabled and active_source_count < nseg and active_target_ids.size > 0:
            local_target_ids = _cext_hybrid_local_target_ids(
                context,
                hybrid,
                active_target_ids,
                active_source_mask,
            )
            if local_target_ids.size != active_target_ids.size:
                local_target_positions = np.searchsorted(active_target_ids, local_target_ids).astype(np.int32, copy=False)
                local_target_plan = _build_cext_gpu_direct_target_plan(context, local_target_ids)

        c_ext_new, hybrid_timings = _compute_cext_hybrid_bg_gpu(
            context,
            ext_state,
            hybrid,
            active_source_mask=active_source_mask if active_set_enabled else None,
            active_source_plan=active_source_plan,
            target_plan=active_target_plan,
            local_target_plan=local_target_plan,
            local_target_positions=local_target_positions,
            runtime_state=runtime_state,
            frozen_mass_grids_g=frozen_mass_grids_g,
            frozen_phi_grids_g=frozen_phi_grids_g,
        )
        hybrid_deposit_total += float(hybrid_timings.get("deposit_s", 0.0))
        hybrid_fft_total += float(hybrid_timings.get("fft_s", 0.0))
        hybrid_o2_total += float(hybrid_timings.get("fft_o2_terms_s", 0.0))
        hybrid_self_sub_total += float(hybrid_timings.get("fft_discrete_self_subtract_s", 0.0))
        hybrid_local_total += float(hybrid_timings.get("local_corr_s", 0.0))
        hybrid_sample_total += float(hybrid_timings.get("sample_s", 0.0))
        hybrid_bg_solver_mode = str(hybrid_timings.get("bg_solver_mode", hybrid_bg_solver_mode or _resolve_cext_hybrid_bg_solver_mode()))
        transfer_total += float(hybrid_timings.get("transfer_s", 0.0))
        kernel_total += (
            float(hybrid_timings.get("fft_s", 0.0))
            + float(hybrid_timings.get("fft_o2_terms_s", 0.0))
            + float(hybrid_timings.get("fft_discrete_self_subtract_s", 0.0))
            + float(hybrid_timings.get("local_corr_s", 0.0))
            + float(hybrid_timings.get("sample_s", 0.0))
        )

        current_target_gl = np.asarray(current_c_ext_gl if active_target_count >= nseg else current_c_ext_gl[active_target_ids], dtype=np.float32)
        accel_state_in = {
            "prev_residual_flat": prev_residual_flat,
            "prev_abs_residual": prev_abs_residual,
            "stable_iters": stable_iters,
            "anderson_g_history": anderson_g_history,
            "anderson_r_history": anderson_r_history,
            "accel_rejections": accel_rejections,
            "accel_restarts": accel_restarts,
            "anderson_depth_used": anderson_depth_used,
            "omega_last": omega_last,
            "iter_count": completed_iters,
        }
        candidate_flat, accel_state, accel_metrics = _cext_apply_component_acceleration(
            current_target_gl.reshape(-1),
            np.asarray(c_ext_new, dtype=np.float32).reshape(-1),
            accel_state_in,
            accel_mode=accel_mode,
        )
        if active_target_count >= nseg:
            candidate_c_ext_gl = np.asarray(candidate_flat.reshape(current_target_gl.shape), dtype=np.float32)
            mapped_c_ext_gl = np.asarray(c_ext_new, dtype=np.float32).reshape(current_target_gl.shape).copy()
        else:
            candidate_c_ext_gl = np.asarray(current_c_ext_gl, dtype=np.float32).copy()
            candidate_c_ext_gl[active_target_ids] = np.asarray(candidate_flat.reshape(current_target_gl.shape), dtype=np.float32)
            mapped_c_ext_gl = np.asarray(current_c_ext_gl, dtype=np.float32).copy()
            mapped_c_ext_gl[active_target_ids] = np.asarray(c_ext_new, dtype=np.float32).reshape(current_target_gl.shape)

        comp_abs = np.max(np.abs(np.asarray(c_ext_new, dtype=np.float32) - current_target_gl), axis=1)
        comp_scale = np.maximum.reduce(
            [
                np.max(np.abs(np.asarray(c_ext_new, dtype=np.float32)), axis=1),
                np.max(np.abs(current_target_gl), axis=1),
                np.full((active_target_ids.size,), float(VESS_CONC_FLOOR), dtype=np.float32),
            ]
        )
        comp_rel = np.divide(
            np.asarray(comp_abs, dtype=np.float32),
            np.maximum(comp_scale, np.float32(VESS_CONC_FLOOR)),
            out=np.zeros_like(np.asarray(comp_abs, dtype=np.float32)),
            where=np.maximum(comp_scale, np.float32(VESS_CONC_FLOOR)) > 0.0,
        )

        if target_active_set_enabled and active_target_ids.size > 0:
            target_abs_delta_field = np.zeros((nseg,), dtype=np.float32)
            target_rel_delta_field = np.zeros((nseg,), dtype=np.float32)
            target_abs_delta_field[active_target_ids] = np.asarray(comp_abs, dtype=np.float32)
            target_rel_delta_field[active_target_ids] = np.asarray(comp_rel, dtype=np.float32)
            stable_target_local = (
                (target_abs_delta_field[active_target_ids] <= float(CEXT_TARGET_ACTIVE_SET_ABS_TOL))
                | (target_rel_delta_field[active_target_ids] <= float(CEXT_TARGET_ACTIVE_SET_REL_TOL))
            )
            active_target_stable_counts[active_target_ids[np.asarray(stable_target_local, dtype=bool)]] += np.int16(1)
            active_target_stable_counts[active_target_ids[~np.asarray(stable_target_local, dtype=bool)]] = np.int16(0)

            if iter_idx >= max(int(CEXT_TARGET_ACTIVE_SET_START), 1):
                eligible_target_ids = np.asarray(
                    active_target_ids[
                        active_target_stable_counts[active_target_ids] >= max(int(CEXT_TARGET_ACTIVE_SET_STABLE_ITERS), 1)
                    ],
                    dtype=np.int32,
                )
                max_target_freeze = max(int(np.count_nonzero(active_target_mask)) - int(min_active_targets), 0)
                if max_target_freeze > 0 and eligible_target_ids.size > 0:
                    neighbor_counts_t, neighbor_abs_t, neighbor_rel_t, _ = _cext_target_neighbor_metrics(
                        context,
                        eligible_target_ids,
                        source_cell_stats,
                        neighbor_pad=int(CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD),
                    )
                    can_freeze = (neighbor_counts_t <= 0) | (
                        (neighbor_abs_t <= float(CEXT_TARGET_ACTIVE_SET_ABS_TOL))
                        & (neighbor_rel_t <= float(CEXT_TARGET_ACTIVE_SET_REL_TOL))
                    )
                    freeze_target_ids = np.asarray(eligible_target_ids[np.asarray(can_freeze, dtype=bool)], dtype=np.int32)
                    if freeze_target_ids.size > max_target_freeze:
                        rank_key = np.asarray(target_rel_delta_field[freeze_target_ids], dtype=np.float32)
                        keep_order = np.argsort(rank_key, kind="stable")[:max_target_freeze]
                        freeze_target_ids = np.asarray(freeze_target_ids[keep_order], dtype=np.int32)
                    if freeze_target_ids.size > 0:
                        active_target_mask[freeze_target_ids] = False
                        active_target_stable_counts[freeze_target_ids] = np.int16(0)
                        target_freeze_events += int(freeze_target_ids.size)
                        active_target_plan = None
                        target_mask_changed = True

        ext_state["c_ext_gl"] = np.asarray(candidate_c_ext_gl, dtype=np.float32)
        runtime_state["c_ext_gl"].set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))
        completed_iters = iter_idx
        prev_source_c_ext_gl = source_c_ext_snapshot
        prev_source_c_iv_gl = source_c_iv_snapshot
        prev_residual_val = accel_state["prev_residual_flat"]
        prev_residual_flat = None if prev_residual_val is None else np.asarray(prev_residual_val, dtype=np.float32).copy()
        prev_abs_residual = float(accel_state["prev_abs_residual"])
        stable_iters = int(accel_state["stable_iters"])
        anderson_g_history = [np.asarray(arr, dtype=np.float32).copy() for arr in accel_state["anderson_g_history"]]
        anderson_r_history = [np.asarray(arr, dtype=np.float32).copy() for arr in accel_state["anderson_r_history"]]
        accel_rejections = int(accel_metrics["accel_rejections"])
        accel_restarts = int(accel_metrics["accel_restarts"])
        anderson_depth_used = int(accel_metrics["anderson_depth_used"])
        omega_last = float(accel_metrics["omega_last"])
        accel_step_last = str(accel_metrics["accel_step_last"])
        max_delta_last = float(accel_metrics["max_delta_last"])
        rel_residual_last = float(accel_metrics["rel_residual_last"])
        step_rejected = False
        if (
            last_eval_c_ext_gl is not None
            and last_eval_mapped_gl is not None
            and np.isfinite(last_eval_max_delta)
            and max_delta_last
            > max(float(last_eval_max_delta) * max(float(CEXT_VESS_COUPLING_STEP_REJECT_FACTOR), 1.0), float(last_eval_max_delta) + 0.05 * float(CEXT_VESS_COUPLING_TOL))
        ):
            retry_omega = max(
                float(omega_floor),
                min(float(last_eval_omega), float(last_eval_omega) * max(float(CEXT_VESS_COUPLING_STEP_RETRY_FACTOR), 0.0)),
            )
            retry_candidate = np.asarray(last_eval_c_ext_gl, dtype=np.float32) + np.float32(retry_omega) * (
                np.asarray(last_eval_mapped_gl, dtype=np.float32) - np.asarray(last_eval_c_ext_gl, dtype=np.float32)
            )
            trusted_retry = _cext_apply_trust_region(
                np.asarray(last_eval_c_ext_gl, dtype=np.float32).reshape(-1),
                np.asarray(retry_candidate, dtype=np.float32).reshape(-1),
                np.asarray(last_eval_mapped_gl, dtype=np.float32).reshape(-1),
            )
            if trusted_retry is not None:
                retry_candidate = np.asarray(trusted_retry, dtype=np.float32).reshape(np.asarray(last_eval_c_ext_gl).shape)
            candidate_c_ext_gl = np.asarray(retry_candidate, dtype=np.float32)
            step_rejected = True
            step_rejections += 1
            accel_rejections += 1
            accel_restarts += 1
            prev_residual_flat = None
            prev_abs_residual = float("inf")
            stable_iters = 0
            anderson_g_history = []
            anderson_r_history = []
            omega_last = float(retry_omega)
            last_eval_omega = float(retry_omega)
            accel_step_last = "reject_retry"
            if SOLVER_TIMING_DETAILS:
                print(
                    "    Cext residual rejected: "
                    f"bad_max_delta={max_delta_last:.3e} prior_max_delta={last_eval_max_delta:.3e} "
                    f"retry_omega={retry_omega:.3f}"
                )
        if not step_rejected:
            last_eval_c_ext_gl = np.asarray(current_c_ext_gl, dtype=np.float32).copy()
            last_eval_mapped_gl = np.asarray(mapped_c_ext_gl, dtype=np.float32).copy()
            last_eval_max_delta = float(max_delta_last)
            last_eval_omega = max(float(omega_last), float(omega_floor))
        else:
            ext_state["c_ext_gl"] = np.asarray(candidate_c_ext_gl, dtype=np.float32)
            runtime_state["c_ext_gl"].set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))
        active_target_count = int(np.count_nonzero(active_target_mask))
        frozen_target_count = int(nseg - active_target_count)
        active_component_count = 1 if active_target_count > 0 else 0
        largest_component_size = int(active_target_count)
        if (
            active_set_enabled
            and frozen_source_count > 0
            and max_delta_last > max(4.0 * float(hybrid_freeze_resid_gate), 2.0 * float(hybrid_freeze_resid_gate + 1e-30))
        ):
            active_source_mask[:] = True
            active_source_stable_counts[:] = np.int16(0)
            active_source_count = int(nseg)
            frozen_source_count = 0
            frozen_mass_grids_g = None
            frozen_phi_grids_g = None
            active_source_plan = None
            prev_residual_flat = None
            prev_abs_residual = float("inf")
            stable_iters = 0
            anderson_g_history = []
            anderson_r_history = []
            accel_step_last = "freeze_rollback"
            if SOLVER_TIMING_DETAILS:
                print(
                    "    Cext active source freeze rolled back: "
                    f"max_delta={max_delta_last:.3e} gate={hybrid_freeze_resid_gate:.3e}"
                )

        improved_best = bool(
            (not step_rejected)
            and np.isfinite(max_delta_last)
            and (
                best_c_ext_gl is None
                or max_delta_last < best_max_delta * (1.0 - 1.0e-4)
                or max_delta_last < best_max_delta - max(0.01 * float(CEXT_VESS_COUPLING_TOL), 1.0e-12)
            )
        )
        if improved_best:
            # max_delta is the residual of current_c_ext_gl -> F(current_c_ext_gl),
            # so rollback must preserve current_c_ext_gl, not the relaxed candidate.
            best_c_ext_gl = np.asarray(current_c_ext_gl, dtype=np.float32).copy()
            best_max_delta = float(max_delta_last)
            best_rel_residual = float(rel_residual_last)
            best_iter = int(iter_idx)
            best_stall_iters = 0
        elif best_c_ext_gl is not None and np.isfinite(best_max_delta):
            if max_delta_last >= best_max_delta or omega_last <= float(omega_floor) * 1.001:
                best_stall_iters += 1
            else:
                best_stall_iters = max(best_stall_iters - 1, 0)

        stop_on_best_seen = False
        if best_c_ext_gl is not None and int(CEXT_VESS_COUPLING_BEST_STALL_ITERS) > 0:
            revert_factor = max(float(CEXT_VESS_COUPLING_BEST_REVERT_FACTOR), 1.0)
            stalled_long_enough = best_stall_iters >= max(int(CEXT_VESS_COUPLING_BEST_STALL_ITERS), 1)
            worse_than_best = max_delta_last >= best_max_delta * revert_factor
            floor_stalled = omega_last <= float(omega_floor) * 1.001 and max_delta_last >= best_max_delta
            if stalled_long_enough and (worse_than_best or floor_stalled):
                # The nonlinear map has stopped improving and relaxation has
                # collapsed or the residual is walking uphill. Keep the best
                # fixed-point iterate seen so far instead of spending the rest
                # of the run moving away from it.
                ext_state["c_ext_gl"] = np.asarray(best_c_ext_gl, dtype=np.float32).copy()
                runtime_state["c_ext_gl"].set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))
                max_delta_last = float(best_max_delta)
                rel_residual_last = float(best_rel_residual)
                accel_step_last = "best_seen_stop"
                best_seen_stop = True
                stop_on_best_seen = True
                prev_residual_flat = None
                prev_abs_residual = float("inf")
                stable_iters = 0
                anderson_g_history = []
                anderson_r_history = []
                if SOLVER_TIMING_DETAILS:
                    print(
                        "    Cext best-seen stop: "
                        f"reverted_to_iter={best_iter} best_max_delta={best_max_delta:.3e} "
                        f"stall_iters={best_stall_iters} omega_floor={float(omega_floor):.3f}"
                    )

        _maybe_trace_cext_iteration(
            iter_idx,
            ext_state,
            solver="topdown_ext_hybrid_bg",
            context=context,
            backend=backend,
            max_delta=max_delta_last,
            rel_residual=rel_residual_last,
        )

        if SOLVER_TIMING_DETAILS:
            print(
                f"    Cext iter {iter_idx}/{int(CEXT_VESS_COUPLING_MAX_ITER)}: "
                f"max_delta={max_delta_last:.3e} rel={rel_residual_last:.3e} "
                f"step={accel_step_last} omega={omega_last:.3f} stable={stable_iters} "
                f"active_src={active_source_count} frozen_src={frozen_source_count} "
                f"active_tgt={active_target_count} frozen_tgt={frozen_target_count} "
                f"local_tgt={int(hybrid_timings.get('local_target_count', active_target_count))}"
            )
        if stop_on_best_seen:
            break
        if max_delta_last < float(CEXT_VESS_COUPLING_TOL):
            break
        if float(CEXT_VESS_COUPLING_REL_TOL) > 0.0 and rel_residual_last < float(CEXT_VESS_COUPLING_REL_TOL):
            break

    # Recompute intravascular quadrature values against the final relaxed Cext
    # field, then cache q_line = k_if * (Civ - Cext) for the final tissue solve.
    cin_seg, cout_seg, c_iv_gl, frozen_backend, final_frozen_transfer = _run_topdown_ext_frozen_step(
        context,
        ext_state,
        inlet_concentration=float(inlet_concentration),
        vmax=float(vmax),
        km=float(km),
        chb_max=np.asarray(chb_max, dtype=np.float32),
        fluid_mode=fluid_mode,
        frozen_backend=frozen_backend,
    )
    final_frozen_timings = dict(_LAST_CEXT_FROZEN_STEP_TIMINGS)
    frozen_transfer_total += float(final_frozen_transfer)
    final_frozen_transfer_total = float(final_frozen_transfer)
    final_frozen_kernel_total = float(final_frozen_timings.get("kernel_s", 0.0))
    ext_state["cin_seg"] = np.asarray(cin_seg, dtype=np.float32)
    ext_state["cout_seg"] = np.asarray(cout_seg, dtype=np.float32)
    ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    t_cache = perf_counter()
    _build_cext_iteration_cache(context, ext_state)
    final_cache_total = perf_counter() - t_cache
    t_source_state = perf_counter()
    _set_last_cext_source_state(
        context,
        ext_state,
        solver="topdown_ext_hybrid_bg",
        backend=backend,
    )
    final_source_state_total = perf_counter() - t_source_state

    total_elapsed = perf_counter() - t_total
    _LAST_CONCENTRATION_TIMINGS = {
        "backend": backend,
        "t_cext_total_s": float(total_elapsed),
        "t_cext_hct_setup_s": float(hct_setup_s),
        "t_cext_context_build_s": float(context_build_s),
        "t_cext_solver_setup_s": float(setup_s),
        "t_cext_query_s": float(query_total),
        "t_cext_kernel_s": float(kernel_total),
        "t_cext_gpu_transfer_s": float(transfer_total),
        "t_cext_frozen_gpu_transfer_s": float(frozen_transfer_total),
        "t_cext_init_s": float(init_elapsed),
        "t_cext_init_kernel_s": float(init_kernel_total),
        "t_cext_init_gpu_transfer_s": float(init_transfer_total),
        "t_cext_init_frozen_gpu_transfer_s": float(init_frozen_transfer_total),
        "t_cext_init_frozen_kernel_s": float(init_frozen_kernel_total),
        "t_cext_init_cache_s": float(init_cache_total),
        "t_cext_init_hybrid_call_s": float(init_hybrid_call_total),
        "t_cext_iteration_cache_s": float(iteration_cache_total),
        "t_cext_final_frozen_gpu_transfer_s": float(final_frozen_transfer_total),
        "t_cext_final_frozen_kernel_s": float(final_frozen_kernel_total),
        "t_cext_final_cache_s": float(final_cache_total),
        "t_cext_final_source_state_s": float(final_source_state_total),
        "t_cext_hybrid_deposit_s": float(hybrid_deposit_total),
        "t_cext_hybrid_fft_s": float(hybrid_fft_total),
        "t_cext_hybrid_o2_terms_s": float(hybrid_o2_total),
        "t_cext_hybrid_self_subtract_s": float(hybrid_self_sub_total),
        "t_cext_hybrid_local_corr_s": float(hybrid_local_total),
        "t_cext_hybrid_sample_s": float(hybrid_sample_total),
        "cext_hybrid_bg_solver": str(hybrid_bg_solver_mode),
        "cext_init_mode": init_mode,
        "cext_init_performed": bool(init_performed),
        "cext_outer_iterations_completed": int(completed_iters),
        "cext_total_iterations_effective": int(completed_iters + (1 if init_performed else 0)),
        "cext_accel_mode": accel_mode,
        "cext_accel_step_last": accel_step_last,
        "cext_accel_rejections": int(accel_rejections),
        "cext_accel_restarts": int(accel_restarts),
        "cext_anderson_depth_used": int(anderson_depth_used),
        "cext_omega_last": float(omega_last),
        "cext_rel_residual_last": float(rel_residual_last),
        "cext_max_delta_last": float(max_delta_last),
        "cext_best_max_delta": float(best_max_delta if np.isfinite(best_max_delta) else max_delta_last),
        "cext_best_seen_stop": bool(best_seen_stop),
        "cext_step_rejections": int(step_rejections),
        "cext_candidate_batches": 0,
        "cext_candidate_query_mode": "hybrid_bg",
        "cext_frozen_backend": str(frozen_backend),
        "cext_active_source_count": int(active_source_count),
        "cext_frozen_source_count": int(frozen_source_count),
        "cext_active_target_count": int(active_target_count),
        "cext_frozen_target_count": int(frozen_target_count),
        "cext_active_component_count": int(active_component_count),
        "cext_largest_component_size": int(largest_component_size),
        "cext_active_freeze_events": int(active_freeze_events),
        "cext_active_refreshes": int(active_refreshes),
        "cext_target_freeze_events": 0,
        "cext_target_reactivations": 0,
        "cext_hybrid_bg_grid": int(hybrid["grid_n"]),
        "cext_hybrid_lambda_bins": int(hybrid["lambda_bins"]),
        "cext_hybrid_lambda_bin_policy": str(hybrid.get("lambda_bin_policy", "")),
        "cext_hybrid_lambda_bin_edges_hash": str(hybrid.get("lambda_bin_edges_hash", "")),
        "cext_hybrid_near_radius_si": float(hybrid["near_radius_si"]),
    }
    _store_tree_cext_state_cache(tree, cache_key, ext_state)
    return np.asarray(ext_state["cin_seg"], dtype=float), np.asarray(ext_state["cout_seg"], dtype=float)


def _segment_decay_factor(
    flow: float,
    radius: float,
    length: float,
    diffusivity: float,
    cin_plasma: float | None = None,
    Chb_max: float | None = None,
) -> float:
    if length <= 0.0 or radius <= 0.0:
        return 1.0
    area = np.pi * radius * radius
    velocity = flow / area if area > 0.0 else 0.0
    if velocity <= 0.0:
        return 1.0

    if cin_plasma is not None and Chb_max is not None:
        B = buffer_factor_B(cin_plasma, Chb_max)
        velocity_eff = velocity * max(B, 1e-6)
    else:
        velocity_eff = velocity

    permeation_rate = _segment_permeation_rate(radius, diffusivity)
    if diffusivity <= 0.0:
        lam = -permeation_rate / velocity_eff
    else:
        discriminant = max(velocity_eff * velocity_eff + 4.0 * diffusivity * permeation_rate, 0.0)
        lam = (velocity_eff - np.sqrt(discriminant)) / (2.0 * diffusivity)
    exponent = np.clip(lam * length, -150.0, 50.0)
    return float(np.exp(exponent))


def normalize_tree_inlet_flow(tree: Tree, target_uL_per_min: Optional[float]) -> Optional[np.ndarray]:
    if target_uL_per_min is None or not np.isfinite(target_uL_per_min):
        return None

    seg_count = int(getattr(tree, "segment_count", 0))
    if seg_count <= 0:
        return None

    flows = np.asarray(tree.data[:seg_count, 22], dtype=float)
    root_flow = float(flows[0]) if flows.size else 0.0
    if not np.isfinite(root_flow) or root_flow == 0.0:
        return None

    target_cms = float(target_uL_per_min) * 1e-3 / 60.0
    scale = target_cms / root_flow
    if not np.isfinite(scale) or scale == 0.0:
        return None

    original = flows.copy()
    flows *= scale
    return original


def build_tissue_cache_from_tree(
    tree: Tree,
    sample_points: np.ndarray,
) -> dict:
    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        return _prepare_tissue_geometry(sample_points, np.empty((0, 3)), np.empty((0, 3)), np.empty((0,)), max_nearby=0)
    # Avoid copying the full (nseg x 31) table; TreeData is already floating-point.
    data = np.asarray(tree.data[:nseg])
    starts = data[:, 0:3]
    ends = data[:, 3:6]
    radii = data[:, 21]
    max_nearby = min(NEAREST_TISSUE_VESSELS, len(starts))
    if _resolve_tissue_accel_mode() == "gpu":
        context = _build_tissue_geometry_context(starts, ends, radii)
        return {
            "gpu": True,
            "context": context,
            "max_nearby": int(max_nearby),
            "n_points": int(sample_points.shape[0]),
        }
    if TISSUE_STREAMING_ENABLED and int(sample_points.shape[0]) >= int(TISSUE_STREAMING_MIN_POINTS):
        context = _build_tissue_geometry_context(starts, ends, radii)
        return {
            "streaming": True,
            "context": context,
            "max_nearby": int(max_nearby),
            "n_points": int(sample_points.shape[0]),
        }
    return _prepare_tissue_geometry(sample_points, starts, ends, radii, max_nearby=max_nearby)


def _compute_tissue_samples_greens_streaming(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cin: np.ndarray,
    flows: np.ndarray,
    *,
    diffusivity: float,
    vmax: float,
    km: float,
    window_factor: float,
    inlet_concentration: float | None,
    tissue_cache: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    max_nearby = min(NEAREST_TISSUE_VESSELS, len(starts))
    if tissue_cache is not None and tissue_cache.get("streaming"):
        context = tissue_cache["context"]
        max_nearby = int(tissue_cache.get("max_nearby", max_nearby))
    else:
        context = _build_tissue_geometry_context(starts, ends, radii)

    starts_si = context["starts_si"]
    radii_si = context["radii_si"]
    segment_vectors = context["segment_vectors"]
    seg_len_sq = context["seg_len_sq"]
    seg_len = context["seg_len"]
    valid = context["valid_mask"]

    if starts_si.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    cin = cin[valid]
    flows_si = flows[valid] * CM3_TO_M3
    diffusivity_si = diffusivity * CM2_TO_M2

    seg_len = np.sqrt(seg_len_sq)
    cin_pos = np.maximum(np.nan_to_num(cin, nan=0.0), 0.0)
    flows_si = np.nan_to_num(flows_si, nan=0.0)
    radii_si = np.maximum(np.nan_to_num(radii_si, nan=0.0), 0.0)

    denom = np.maximum(km + cin_pos, 1e-30)
    k1 = vmax / denom
    lam_edge = np.sqrt(diffusivity_si / np.maximum(k1, 1e-30))
    phi_edge = radii_si / np.maximum(lam_edge, 1e-30)
    ratio_edge = _k_ratio(phi_edge)
    flow_mag = np.maximum(np.abs(flows_si), 1e-30)
    alpha_edge = (2.0 * np.pi * radii_si / flow_mag) * (diffusivity_si / np.maximum(lam_edge, 1e-30)) * ratio_edge
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet()
    lam_ref = float(np.sqrt(diffusivity_si / max(vmax / max(km + float(inlet_concentration), 1e-30), 1e-30)))
    flow_sign = np.sign(flows_si)
    gl_nodes, gl_weights = _get_gl_nodes_weights(GL_ORDER)

    influence_radius = np.asarray(window_factor * lam_edge, dtype=context.get("cache_float", np.float32))
    result = np.zeros(points.shape[0], dtype=float)
    keep_mask_all = np.zeros(points.shape[0], dtype=bool)
    chunk_size = _streaming_tissue_chunk_size(max_nearby)
    n_chunks = int(math.ceil(points.shape[0] / max(chunk_size, 1)))
    print(
        f"  Streaming tissue solve: points={points.shape[0]} max_nearby={max_nearby} "
        f"chunk_points={chunk_size} chunks={n_chunks} prune_by_window={TISSUE_STREAMING_PRUNE_BY_WINDOW} "
        f"chunk_workers={TISSUE_STREAMING_CHUNK_WORKERS}"
    )

    used_numba = bool(_HAVE_NUMBA and TISSUE_USE_NUMBA and _K0_LUT.size)
    _ensure_tissue_context_kdtree(context)
    state = {
        "context": context,
        "points": points,
        "max_nearby": max_nearby,
        "influence_radius": influence_radius,
        "prune_by_window": bool(TISSUE_STREAMING_PRUNE_BY_WINDOW),
        "used_numba": used_numba,
        "starts_si": starts_si,
        "segment_vectors": segment_vectors,
        "seg_len": seg_len,
        "radii_si": radii_si,
        "cin_pos": cin_pos,
        "alpha_edge": alpha_edge,
        "flow_sign": flow_sign,
        "diffusivity_si": diffusivity_si,
        "km": km,
        "vmax": vmax,
        "window_factor": window_factor,
        "lam_ref": lam_ref,
        "gl_nodes": gl_nodes,
        "gl_weights": gl_weights,
    }
    tasks = [
        (chunk_i, start_idx, min(start_idx + chunk_size, len(points)), state)
        for chunk_i, start_idx in enumerate(range(0, len(points), chunk_size), start=1)
    ]

    old_numba_threads = None
    if used_numba and set_num_threads is not None and get_num_threads is not None:
        try:
            old_numba_threads = int(get_num_threads())
            set_num_threads(max(int(TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER), 1))
            print(
                f"  Streaming tissue: numba_threads_per_chunk="
                f"{max(int(TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER), 1)} "
                f"(was {old_numba_threads})"
            )
        except Exception:
            old_numba_threads = None

    completed = 0
    total_kept = 0
    total_before_candidates = 0
    total_after_candidates = 0
    workers = max(int(TISSUE_STREAMING_CHUNK_WORKERS), 1)
    try:
        if workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {executor.submit(_compute_streaming_tissue_chunk, task): task[0] for task in tasks}
                for fut in as_completed(future_map):
                    chunk_i, start_idx, end_idx, keep_mask, out, kept, before_c, after_c = fut.result()
                    result[start_idx:end_idx] = out
                    keep_mask_all[start_idx:end_idx] = keep_mask
                    completed += 1
                    total_kept += kept
                    total_before_candidates += before_c
                    total_after_candidates += after_c
                    if TISSUE_STREAMING_LOG_EVERY_CHUNKS and (
                        completed == 1 or completed == n_chunks or completed % int(TISSUE_STREAMING_LOG_EVERY_CHUNKS) == 0
                    ):
                        reduction = 100.0 * (1.0 - total_after_candidates / max(total_before_candidates, 1))
                        print(
                            f"    Streaming tissue chunks done {completed}/{n_chunks}: "
                            f"rows_done~{completed * chunk_size} kept={total_kept} "
                            f"candidate_reduction={reduction:.1f}%"
                        )
        else:
            for task in tasks:
                chunk_i, start_idx, end_idx, keep_mask, out, kept, before_c, after_c = _compute_streaming_tissue_chunk(task)
                result[start_idx:end_idx] = out
                keep_mask_all[start_idx:end_idx] = keep_mask
                completed += 1
                total_kept += kept
                total_before_candidates += before_c
                total_after_candidates += after_c
                if TISSUE_STREAMING_LOG_EVERY_CHUNKS and (
                    completed == 1 or completed == n_chunks or completed % int(TISSUE_STREAMING_LOG_EVERY_CHUNKS) == 0
                ):
                    reduction = 100.0 * (1.0 - total_after_candidates / max(total_before_candidates, 1))
                    print(
                        f"    Streaming tissue chunks done {completed}/{n_chunks}: "
                        f"rows_done={end_idx} kept={total_kept} candidate_reduction={reduction:.1f}%"
                    )
    finally:
        if old_numba_threads is not None and set_num_threads is not None:
            try:
                set_num_threads(old_numba_threads)
            except Exception:
                pass

    return keep_mask_all, result


_TISSUE_GPU_KERNEL = None
_CEXT_TISSUE_GPU_KERNEL = None
_TISSUE_CONTEXT_KDTREE_LOCK = threading.Lock()


def _get_tissue_gpu_kernel():
    global _TISSUE_GPU_KERNEL
    if _TISSUE_GPU_KERNEL is not None:
        return _TISSUE_GPU_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __device__ float interp_lut(float x, const float* xs, const float* ys, int n) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    float x0 = xs[lo];
    float x1 = xs[lo + 1];
    float y0 = ys[lo];
    float y1 = ys[lo + 1];
    if (x1 == x0) return y0;
    float t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

extern "C" __global__ void tissue_greens_kernel(
    const float* points_si,
    const float* starts_si,
    const float* segment_vectors,
    const float* seg_len,
    const float* radii_si,
    const int* nearest_idx,
    const float* proj_raw,
    const float* d_center,
    const unsigned char* keep_mask,
    const float* cin_pos,
    const float* alpha_edge,
    const float* flow_sign,
    float diffusivity_si,
    float km,
    float vmax,
    float window_factor,
    float lam_ref,
    const float* gl_nodes,
    const float* gl_weights,
    int gl_order,
    const float* xs_lut,
    const float* k0_lut,
    const float* ratio_lut,
    int lut_n,
    int n_points,
    int keep_k,
    float* out
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_points) return;
    if (!keep_mask[row]) {
        out[row] = 0.0f;
        return;
    }

    float px = points_si[row * 3 + 0];
    float py = points_si[row * 3 + 1];
    float pz = points_si[row * 3 + 2];
    float cap_max = 1.0e-6f;
    float total = 0.0f;

    for (int local_i = 0; local_i < keep_k; ++local_i) {
        int flat = row * keep_k + local_i;
        int seg_i = nearest_idx[flat];
        if (seg_i < 0) continue;
        float L = seg_len[seg_i];
        if (L <= 0.0f) continue;

        float proj = proj_raw[flat];
        if (flow_sign[seg_i] < 0.0f) proj = 1.0f - proj;
        proj = fminf(1.0f, fmaxf(0.0f, proj));
        float s_star = proj * L;

        float Cc_star = cin_pos[seg_i] * expf(-alpha_edge[seg_i] * s_star);
        float denom_gate = km + fmaxf(Cc_star, 1.0e-12f);
        denom_gate = fmaxf(denom_gate, 1.0e-30f);
        float lam_gate = sqrtf(diffusivity_si / fmaxf(vmax / denom_gate, 1.0e-30f));
        if (d_center[flat] > window_factor * lam_gate) continue;
        if (Cc_star > cap_max) cap_max = Cc_star;

        float halfW = window_factor * lam_ref;
        float s0 = fmaxf(0.0f, s_star - halfW);
        float s1 = fminf(L, s_star + halfW);
        if (s1 <= s0 + 1.0e-15f) continue;

        float mid = 0.5f * (s0 + s1);
        float half = 0.5f * (s1 - s0);
        float total_local = 0.0f;

        float sx0 = starts_si[seg_i * 3 + 0];
        float sx1 = starts_si[seg_i * 3 + 1];
        float sx2 = starts_si[seg_i * 3 + 2];
        float vx0 = segment_vectors[seg_i * 3 + 0];
        float vx1 = segment_vectors[seg_i * 3 + 1];
        float vx2 = segment_vectors[seg_i * 3 + 2];
        float radius = radii_si[seg_i];
        float cin_seg = cin_pos[seg_i];
        float alpha = alpha_edge[seg_i];

        for (int g = 0; g < gl_order; ++g) {
            float s = mid + half * gl_nodes[g];
            float t = s / L;
            float xs0 = sx0 + t * vx0;
            float xs1 = sx1 + t * vx1;
            float xs2 = sx2 + t * vx2;
            float r0 = px - xs0;
            float r1 = py - xs1;
            float r2 = pz - xs2;
            float r = sqrtf(r0 * r0 + r1 * r1 + r2 * r2);
            if (r <= 1.0e-12f) continue;

            float cc_s = cin_seg * expf(-alpha * s);
            float denom_loc = km + fmaxf(cc_s, 1.0e-12f);
            denom_loc = fmaxf(denom_loc, 1.0e-30f);
            float lam_loc = sqrtf(diffusivity_si / fmaxf(vmax / denom_loc, 1.0e-30f));
            float phi_loc = fmaxf(radius / fmaxf(lam_loc, 1.0e-30f), 1.0e-12f);
            float r_over_lam = r / fmaxf(lam_loc, 1.0e-30f);
            float k0_num = interp_lut(r_over_lam, xs_lut, k0_lut, lut_n);
            float k0_den = fmaxf(interp_lut(phi_loc, xs_lut, k0_lut, lut_n), 1.0e-30f);
            float Ci_R = cc_s * (k0_num / k0_den);

            float denom_corr = km + fminf(3.0f * Ci_R, cc_s);
            denom_corr = fmaxf(denom_corr, 1.0e-30f);
            float lam_corr = sqrtf(diffusivity_si / fmaxf(vmax / denom_corr, 1.0e-30f));
            float phi = fmaxf(radius / fmaxf(lam_corr, 1.0e-30f), 1.0e-12f);
            float ratio = interp_lut(phi, xs_lut, ratio_lut, lut_n);
            float wall_factor = (diffusivity_si / fmaxf(lam_corr, 1.0e-30f)) * ratio;
            float q_s = (2.0f * 3.14159265358979323846f * radius) * wall_factor * cc_s;
            float kernel = expf(-r / fmaxf(lam_corr, 1.0e-30f)) / (4.0f * 3.14159265358979323846f * r);
            float integrand = q_s * (kernel / diffusivity_si);
            total_local += gl_weights[g] * integrand;
        }
        total += half * total_local;
    }
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out[row] = fminf(total, cap_max);
}
'''
    _TISSUE_GPU_KERNEL = _cp.RawKernel(code, "tissue_greens_kernel")
    return _TISSUE_GPU_KERNEL


def _compute_tissue_samples_greens_gpu(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cin: np.ndarray,
    flows: np.ndarray,
    *,
    diffusivity: float,
    vmax: float,
    km: float,
    window_factor: float,
    inlet_concentration: float | None,
    tissue_cache: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    global TISSUE_ACCEL_MODE, TISSUE_STREAMING_ENABLED, _LAST_TISSUE_TIMINGS
    _LAST_TISSUE_TIMINGS = {}
    if _resolve_tissue_accel_mode(require_gpu=True) != "gpu" or _cp is None:
        raise RuntimeError("Tissue GPU mode could not be initialized.")
    if points.size == 0 or starts.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    t_total = perf_counter()
    max_nearby = min(NEAREST_TISSUE_VESSELS, len(starts))
    if tissue_cache is not None and tissue_cache.get("gpu"):
        context = tissue_cache["context"]
        max_nearby = int(tissue_cache.get("max_nearby", max_nearby))
    else:
        context = _build_tissue_geometry_context(starts, ends, radii)

    starts_si = context["starts_si"]
    radii_si = context["radii_si"]
    segment_vectors = context["segment_vectors"]
    seg_len_sq = context["seg_len_sq"]
    seg_len = context["seg_len"]
    valid = context["valid_mask"]
    nseg = int(starts_si.shape[0])
    if nseg == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    candidate_k = min(nseg, max(max_nearby, max_nearby * TISSUE_KDTREE_CANDIDATE_MULT))
    keep_k = min(max_nearby, candidate_k, nseg)
    tree = _ensure_tissue_context_kdtree(context)
    if tree is None:
        raise RuntimeError("Tissue GPU mode requires scipy cKDTree for CPU candidate query.")

    t0 = perf_counter()
    cin_valid = np.asarray(cin[valid], dtype=np.float32)
    flows_si = np.asarray(flows[valid] * CM3_TO_M3, dtype=np.float32)
    diffusivity_si = float(diffusivity * CM2_TO_M2)
    cin_pos = np.maximum(np.nan_to_num(cin_valid, nan=0.0), 0.0).astype(np.float32, copy=False)
    flows_si = np.nan_to_num(flows_si, nan=0.0).astype(np.float32, copy=False)
    radii_si_np = np.maximum(np.nan_to_num(np.asarray(radii_si, dtype=np.float32), nan=0.0), 0.0)

    denom = np.maximum(float(km) + cin_pos, 1e-30).astype(np.float32, copy=False)
    k1 = np.asarray(float(vmax) / denom, dtype=np.float32)
    lam_edge = np.sqrt(np.asarray(diffusivity_si / np.maximum(k1, 1e-30), dtype=np.float32))
    phi_edge = radii_si_np / np.maximum(lam_edge, 1e-30)
    ratio_edge = _k_ratio(phi_edge).astype(np.float32, copy=False)
    flow_mag = np.maximum(np.abs(flows_si), 1e-30)
    alpha_edge = ((2.0 * np.pi * radii_si_np / flow_mag) * (diffusivity_si / np.maximum(lam_edge, 1e-30)) * ratio_edge).astype(np.float32, copy=False)
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet()
    lam_ref = float(np.sqrt(diffusivity_si / max(vmax / max(km + float(inlet_concentration), 1e-30), 1e-30)))
    flow_sign = np.sign(flows_si).astype(np.float32, copy=False)
    gl_nodes, gl_weights = _get_gl_nodes_weights(GL_ORDER)
    t_setup_cpu = perf_counter() - t0

    t0 = perf_counter()
    starts_g = _cp.asarray(np.asarray(starts_si, dtype=np.float32))
    segment_vectors_g = _cp.asarray(np.asarray(segment_vectors, dtype=np.float32))
    seg_len_sq_g = _cp.asarray(np.asarray(seg_len_sq, dtype=np.float32))
    seg_len_g = _cp.asarray(np.asarray(seg_len, dtype=np.float32))
    radii_g = _cp.asarray(radii_si_np)
    cin_pos_g = _cp.asarray(cin_pos)
    alpha_edge_g = _cp.asarray(alpha_edge)
    flow_sign_g = _cp.asarray(flow_sign)
    gl_nodes_g = _cp.asarray(np.asarray(gl_nodes, dtype=np.float32))
    gl_weights_g = _cp.asarray(np.asarray(gl_weights, dtype=np.float32))
    xs_lut_g = _cp.asarray(np.asarray(_KRATIO_XS, dtype=np.float32))
    k0_lut_g = _cp.asarray(np.asarray(_K0_LUT, dtype=np.float32))
    ratio_lut_g = _cp.asarray(np.asarray(_KRATIO_YS, dtype=np.float32))
    _cp.cuda.Stream.null.synchronize()
    t_upload_static = perf_counter() - t0

    kernel = _get_tissue_gpu_kernel()
    keep_mask_all = np.zeros(points.shape[0], dtype=bool)
    result = np.zeros(points.shape[0], dtype=np.float32)
    points_si_all = np.asarray(points * CM_TO_M, dtype=np.float32)

    chunk_points = max(int(TISSUE_GPU_CHUNK_POINTS), int(TISSUE_GPU_MIN_CHUNK_POINTS))
    min_chunk = max(int(TISSUE_GPU_MIN_CHUNK_POINTS), 1)
    start_idx = 0
    chunks_done = 0
    t_query = 0.0
    t_transfer = 0.0
    t_refine = 0.0
    t_greens = 0.0
    t_download = 0.0

    while start_idx < points.shape[0]:
        end_idx = min(start_idx + chunk_points, points.shape[0])
        chunk = points_si_all[start_idx:end_idx]
        try:
            t0 = perf_counter()
            _, cand = _ckdtree_query(tree, chunk, k=candidate_k)
            t_query += perf_counter() - t0
            if cand.ndim == 1:
                cand = cand[:, None]

            t0 = perf_counter()
            points_g = _cp.asarray(chunk, dtype=_cp.float32)
            cand_g = _cp.asarray(cand, dtype=_cp.int32)
            _cp.cuda.Stream.null.synchronize()
            t_transfer += perf_counter() - t0

            t0 = perf_counter()
            starts_c = starts_g[cand_g]
            seg_c = segment_vectors_g[cand_g]
            seg_len_sq_c = seg_len_sq_g[cand_g]
            diff = points_g[:, None, :] - starts_c
            proj = _cp.sum(diff * seg_c, axis=2) / seg_len_sq_c
            proj_clip = _cp.clip(proj, 0.0, 1.0)
            closest = starts_c + proj_clip[:, :, None] * seg_c
            delta = points_g[:, None, :] - closest
            dist_sq = _cp.sum(delta * delta, axis=2)
            sel = _cp.argpartition(dist_sq, keep_k - 1, axis=1)[:, :keep_k]
            nearest_g = _cp.take_along_axis(cand_g, sel, axis=1).astype(_cp.int32, copy=False)
            proj_g = _cp.take_along_axis(proj, sel, axis=1).astype(_cp.float32, copy=False)
            d_center_g = _cp.sqrt(_cp.take_along_axis(dist_sq, sel, axis=1)).astype(_cp.float32, copy=False)
            radius_local = _cp.minimum(radii_g[nearest_g], seg_len_g[nearest_g])
            inside_any = _cp.any((proj_g >= 0.0) & (proj_g <= 1.0) & (d_center_g <= radius_local), axis=1)
            keep_g = (~inside_any).astype(_cp.uint8, copy=False)
            _cp.cuda.Stream.null.synchronize()
            t_refine += perf_counter() - t0

            t0 = perf_counter()
            out_g = _cp.zeros((chunk.shape[0],), dtype=_cp.float32)
            threads = 128
            blocks = (int(chunk.shape[0]) + threads - 1) // threads
            kernel(
                (blocks,),
                (threads,),
                (
                    points_g.ravel(),
                    starts_g.ravel(),
                    segment_vectors_g.ravel(),
                    seg_len_g,
                    radii_g,
                    nearest_g.ravel(),
                    proj_g.ravel(),
                    d_center_g.ravel(),
                    keep_g,
                    cin_pos_g,
                    alpha_edge_g,
                    flow_sign_g,
                    np.float32(diffusivity_si),
                    np.float32(km),
                    np.float32(vmax),
                    np.float32(window_factor),
                    np.float32(lam_ref),
                    gl_nodes_g,
                    gl_weights_g,
                    np.int32(GL_ORDER),
                    xs_lut_g,
                    k0_lut_g,
                    ratio_lut_g,
                    np.int32(_KRATIO_XS.size),
                    np.int32(chunk.shape[0]),
                    np.int32(keep_k),
                    out_g,
                ),
            )
            _cp.cuda.Stream.null.synchronize()
            t_greens += perf_counter() - t0

            t0 = perf_counter()
            keep_mask_all[start_idx:end_idx] = _cp.asnumpy(keep_g).astype(bool)
            result[start_idx:end_idx] = _cp.asnumpy(out_g)
            t_download += perf_counter() - t0
            chunks_done += 1
            start_idx = end_idx
        except _cp.cuda.memory.OutOfMemoryError:
            _cp.get_default_memory_pool().free_all_blocks()
            if chunk_points <= min_chunk:
                raise
            chunk_points = max(min_chunk, chunk_points // 2)
            print(f"WARNING: GPU OOM in tissue chunk; retrying with --tissue-gpu-chunk-points={chunk_points}")

    if SOLVER_TIMING_DETAILS:
        print(
            "  Tissue GPU solve: "
            f"points={points.shape[0]} active_points={int(np.count_nonzero(keep_mask_all))} "
            f"candidate_k={candidate_k} keep_k={keep_k} chunks={chunks_done} chunk_points={chunk_points} "
            f"setup_cpu={_fmt_seconds(t_setup_cpu)} upload_static={_fmt_seconds(t_upload_static)} "
            f"query={_fmt_seconds(t_query)} transfer={_fmt_seconds(t_transfer)} "
            f"refine={_fmt_seconds(t_refine)} greens={_fmt_seconds(t_greens)} "
            f"download={_fmt_seconds(t_download)} total={_fmt_seconds(perf_counter() - t_total)}"
        )

    total_elapsed = perf_counter() - t_total
    geometry_elapsed = float(t_setup_cpu + t_upload_static + t_query + t_transfer + t_refine + t_download)
    _LAST_TISSUE_TIMINGS = {
        "backend": "gpu",
        "t_tissue_geometry_s": geometry_elapsed,
        "t_tissue_oxygen_s": float(t_greens),
        "t_tissue_total_s": float(total_elapsed),
        "t_tissue_kdtree_query_s": float(t_query),
        "t_tissue_gpu_refine_s": float(t_refine),
        "t_tissue_gpu_transfer_s": float(t_transfer + t_download + t_upload_static),
    }

    validate_n = min(int(TISSUE_GPU_VALIDATE_POINTS), int(points.shape[0]))
    if validate_n > 0:
        old_accel = TISSUE_ACCEL_MODE
        old_streaming = TISSUE_STREAMING_ENABLED
        try:
            TISSUE_ACCEL_MODE = "cpu"
            TISSUE_STREAMING_ENABLED = False
            cpu_mask, cpu_vals = compute_tissue_samples_greens(
                points[:validate_n],
                starts,
                ends,
                radii,
                cin,
                flows,
                diffusivity=diffusivity,
                vmax=vmax,
                km=km,
                window_factor=window_factor,
                inlet_concentration=inlet_concentration,
                tissue_cache=None,
            )
        finally:
            TISSUE_ACCEL_MODE = old_accel
            TISSUE_STREAMING_ENABLED = old_streaming
        gpu_mask = keep_mask_all[:validate_n]
        gpu_vals = result[:validate_n].astype(float)
        mask_disagree = int(np.count_nonzero(gpu_mask != cpu_mask))
        finite = np.isfinite(cpu_vals) & np.isfinite(gpu_vals)
        max_abs = float(np.max(np.abs(gpu_vals[finite] - cpu_vals[finite]))) if np.any(finite) else float("nan")
        rel_l2 = _relative_l2(gpu_vals[finite], cpu_vals[finite]) if np.any(finite) else float("nan")
        if CONC_MAX_FOR_NORMALIZATION and np.isfinite(CONC_MAX_FOR_NORMALIZATION):
            cpu_frac = float(np.mean((cpu_vals[cpu_mask] / CONC_MAX_FOR_NORMALIZATION) >= 0.01)) if np.any(cpu_mask) else float("nan")
            gpu_frac = float(np.mean((gpu_vals[gpu_mask] / CONC_MAX_FOR_NORMALIZATION) >= 0.01)) if np.any(gpu_mask) else float("nan")
            frac_delta = gpu_frac - cpu_frac if np.isfinite(cpu_frac) and np.isfinite(gpu_frac) else float("nan")
        else:
            cpu_frac = gpu_frac = frac_delta = float("nan")
        print(
            "  Tissue GPU validation: "
            f"points={validate_n} mask_disagree={mask_disagree} "
            f"max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} "
            f"FracAbove1pct_cpu={cpu_frac:.6g} FracAbove1pct_gpu={gpu_frac:.6g} "
            f"FracAbove1pct_delta={frac_delta:.3e}"
        )

    return keep_mask_all, result.astype(float, copy=False)


def _get_cext_tissue_gpu_kernel():
    global _CEXT_TISSUE_GPU_KERNEL
    if _CEXT_TISSUE_GPU_KERNEL is not None:
        return _CEXT_TISSUE_GPU_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_tissue_greens_kernel(
    const float* points_si,
    const int* nearest_idx,
    const unsigned char* keep_mask,
    const float* gl_points_si,
    const float* segment_vectors,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    float diffusivity_si,
    float window_factor,
    int n_points,
    int keep_k,
    int gl_order,
    float* out
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_points) return;
    if (!keep_mask[row]) {
        out[row] = 0.0f;
        return;
    }

    float px = points_si[row * 3 + 0];
    float py = points_si[row * 3 + 1];
    float pz = points_si[row * 3 + 2];
    float total = 0.0f;
    float cap_max = 0.0f;

    for (int local_i = 0; local_i < keep_k; ++local_i) {
        int seg_i = nearest_idx[row * keep_k + local_i];
        if (seg_i < 0) continue;
        int contributed = 0;
        for (int g = 0; g < gl_order; ++g) {
            int node_idx = seg_i * gl_order + g;
            float lam = lambda_iv_gl[node_idx];
            if (!(lam > 0.0f) || !isfinite(lam)) continue;
            float sx = gl_points_si[node_idx * 3 + 0];
            float sy = gl_points_si[node_idx * 3 + 1];
            float sz = gl_points_si[node_idx * 3 + 2];
            float dx = px - sx;
            float dy = py - sy;
            float dz = pz - sz;
            float r = sqrtf(dx * dx + dy * dy + dz * dz);
            if (r <= 1.0e-12f || r > window_factor * lam) continue;
            float q = q_weighted_gl[node_idx];
            float o2_weight = mono2_weight_gl[node_idx] + dipole2_weight_gl[node_idx];
            if (!isfinite(q) || !isfinite(o2_weight) || (q == 0.0f && o2_weight == 0.0f)) continue;
            float kernel = expf(-r / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
            total += q * kernel;
            if (o2_weight != 0.0f) {
                float vx = segment_vectors[seg_i * 3 + 0];
                float vy = segment_vectors[seg_i * 3 + 1];
                float vz = segment_vectors[seg_i * 3 + 2];
                float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
                if (vlen > 1.0e-30f) {
                    float tdotr = (vx * dx + vy * dy + vz * dz) / vlen;
                    float mu2 = (tdotr * tdotr) / (r * r);
                    if (mu2 > 1.0f) mu2 = 1.0f;
                    float inv_r = 1.0f / r;
                    float inv_l = 1.0f / lam;
                    float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                              + (1.0f - mu2) * inv_l * inv_l) * kernel;
                    total += o2_weight * pH;
                }
            }
            contributed = 1;
        }
        if (contributed) {
            float cap = seg_cap_gl[seg_i];
            if (cap > cap_max) cap_max = cap;
        }
    }

    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    if (cap_max > 0.0f && total > cap_max) total = cap_max;
    out[row] = total;
}
'''
    _CEXT_TISSUE_GPU_KERNEL = _cp.RawKernel(code, "cext_tissue_greens_kernel")
    return _CEXT_TISSUE_GPU_KERNEL


def _get_cext_tissue_cell_gpu_kernel():
    global _CEXT_TISSUE_CELL_GPU_KERNEL
    if _CEXT_TISSUE_CELL_GPU_KERNEL is not None:
        return _CEXT_TISSUE_CELL_GPU_KERNEL
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    code = r'''
extern "C" __global__ void cext_tissue_cell_greens_kernel(
    const float* points_si,
    const int* cell_ptr,
    const int* cell_node_ids,
    const float* starts_si,
    const float* segment_vectors,
    const float* seg_len_sq,
    const float* seg_len,
    const float* radii_si,
    const float* gl_points_si,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    float diffusivity_si,
    float window_factor,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    int grid_n,
    int rad_cells,
    int n_points,
    int gl_order,
    unsigned char* keep_mask,
    float* out
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_points) return;

    float px = points_si[row * 3 + 0];
    float py = points_si[row * 3 + 1];
    float pz = points_si[row * 3 + 2];

    int cx = (int)floorf((px - origin_x) / spacing);
    int cy = (int)floorf((py - origin_y) / spacing);
    int cz = (int)floorf((pz - origin_z) / spacing);
    int lo_x = cx - rad_cells;
    int lo_y = cy - rad_cells;
    int lo_z = cz - rad_cells;
    int hi_x = cx + rad_cells;
    int hi_y = cy + rad_cells;
    int hi_z = cz + rad_cells;
    if (lo_x < 0) lo_x = 0;
    if (lo_y < 0) lo_y = 0;
    if (lo_z < 0) lo_z = 0;
    if (hi_x >= grid_n) hi_x = grid_n - 1;
    if (hi_y >= grid_n) hi_y = grid_n - 1;
    if (hi_z >= grid_n) hi_z = grid_n - 1;

    float total = 0.0f;
    float cap_max = 0.0f;
    int inside_any = 0;

    if (lo_x <= hi_x && lo_y <= hi_y && lo_z <= hi_z) {
        for (int ix = lo_x; ix <= hi_x; ++ix) {
            for (int iy = lo_y; iy <= hi_y; ++iy) {
                for (int iz = lo_z; iz <= hi_z; ++iz) {
                    int cell_flat = (ix * grid_n + iy) * grid_n + iz;
                    int row_start = cell_ptr[cell_flat];
                    int row_end = cell_ptr[cell_flat + 1];
                    for (int pos = row_start; pos < row_end; ++pos) {
                        int node_idx = cell_node_ids[pos];
                        if (node_idx < 0) continue;
                        int seg_i = node_idx / gl_order;

                        if (!inside_any) {
                            float sx0 = starts_si[seg_i * 3 + 0];
                            float sx1 = starts_si[seg_i * 3 + 1];
                            float sx2 = starts_si[seg_i * 3 + 2];
                            float vx0 = segment_vectors[seg_i * 3 + 0];
                            float vx1 = segment_vectors[seg_i * 3 + 1];
                            float vx2 = segment_vectors[seg_i * 3 + 2];
                            float len2 = fmaxf(seg_len_sq[seg_i], 1.0e-30f);
                            float wx0 = px - sx0;
                            float wx1 = py - sx1;
                            float wx2 = pz - sx2;
                            float proj = (wx0 * vx0 + wx1 * vx1 + wx2 * vx2) / len2;
                            proj = fminf(1.0f, fmaxf(0.0f, proj));
                            float cx0 = sx0 + proj * vx0;
                            float cx1 = sx1 + proj * vx1;
                            float cx2 = sx2 + proj * vx2;
                            float dxs = px - cx0;
                            float dys = py - cx1;
                            float dzs = pz - cx2;
                            float dseg = sqrtf(dxs * dxs + dys * dys + dzs * dzs);
                            float rlim = fminf(radii_si[seg_i], seg_len[seg_i]);
                            if (dseg <= rlim) inside_any = 1;
                        }

                        float lam = lambda_iv_gl[node_idx];
                        if (!(lam > 0.0f) || !isfinite(lam)) continue;
                        float sx = gl_points_si[node_idx * 3 + 0];
                        float sy = gl_points_si[node_idx * 3 + 1];
                        float sz = gl_points_si[node_idx * 3 + 2];
                        float dx = px - sx;
                        float dy = py - sy;
                        float dz = pz - sz;
                        float r = sqrtf(dx * dx + dy * dy + dz * dz);
                        if (r <= 1.0e-12f || r > window_factor * lam) continue;
                        float q = q_weighted_gl[node_idx];
                        float o2_weight = mono2_weight_gl[node_idx] + dipole2_weight_gl[node_idx];
                        if (!isfinite(q) || !isfinite(o2_weight) || (q == 0.0f && o2_weight == 0.0f)) continue;
                        float kernel = expf(-r / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                        total += q * kernel;
                        if (o2_weight != 0.0f) {
                            float vx0 = segment_vectors[seg_i * 3 + 0];
                            float vx1 = segment_vectors[seg_i * 3 + 1];
                            float vx2 = segment_vectors[seg_i * 3 + 2];
                            float vlen = sqrtf(vx0 * vx0 + vx1 * vx1 + vx2 * vx2);
                            if (vlen > 1.0e-30f) {
                                float tdotr = (vx0 * dx + vx1 * dy + vx2 * dz) / vlen;
                                float mu2 = (tdotr * tdotr) / (r * r);
                                if (mu2 > 1.0f) mu2 = 1.0f;
                                float inv_r = 1.0f / r;
                                float inv_l = 1.0f / lam;
                                float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                                          + (1.0f - mu2) * inv_l * inv_l) * kernel;
                                total += o2_weight * pH;
                            }
                        }
                        float cap = seg_cap_gl[seg_i];
                        if (cap > cap_max) cap_max = cap;
                    }
                }
            }
        }
    }

    if (inside_any) {
        keep_mask[row] = 0;
        out[row] = 0.0f;
        return;
    }
    keep_mask[row] = 1;
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    if (cap_max > 0.0f && total > cap_max) total = cap_max;
    out[row] = total;
}
'''
    _CEXT_TISSUE_CELL_GPU_KERNEL = _cp.RawKernel(code, "cext_tissue_cell_greens_kernel")
    return _CEXT_TISSUE_CELL_GPU_KERNEL


def _build_cext_tissue_cell_list_gpu(
    gl_points_valid: np.ndarray,
    lambda_valid: np.ndarray,
    points_si_all: np.ndarray,
    *,
    window_factor: float,
    radii_si: np.ndarray,
) -> tuple[dict, dict[str, float | int]]:
    if _cp is None:
        raise RuntimeError("CuPy is not available.")
    t0 = perf_counter()
    gl_flat = np.asarray(gl_points_valid, dtype=np.float32).reshape(-1, 3)
    lambda_flat = np.asarray(lambda_valid, dtype=np.float32).reshape(-1)
    n_nodes = int(gl_flat.shape[0])
    if n_nodes <= 0:
        raise ValueError("Cannot build an empty Cext tissue cell list.")
    finite_lam = lambda_flat[np.isfinite(lambda_flat) & (lambda_flat > 0.0)]
    max_window = float(window_factor) * float(np.max(finite_lam)) if finite_lam.size else 0.0
    max_radius = float(np.nanmax(np.asarray(radii_si, dtype=np.float32))) if np.asarray(radii_si).size else 0.0
    search_radius = max(max_window, max_radius, 1.0e-12)
    mins = np.min(gl_flat, axis=0)
    maxs = np.max(gl_flat, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0e-9)
    side = span + 2.0 * search_radius
    origin = np.asarray(center - 0.5 * side, dtype=np.float32)

    min_grid = max(int(TISSUE_CEXT_CELL_MIN_GRID), 1)
    max_grid = max(min(int(TISSUE_CEXT_CELL_MAX_GRID), 512), min_grid)
    target_occ = max(int(TISSUE_CEXT_CELL_TARGET_OCCUPANCY), 1)
    occ_grid = int(math.ceil((float(n_nodes) / float(target_occ)) ** (1.0 / 3.0)))
    start_grid = max(min_grid, min(max_grid, max(occ_grid // 2, min_grid)))
    stop_grid = max(min(max_grid, max(occ_grid * 2, start_grid)), start_grid)
    max_rad_cells = max(int(TISSUE_CEXT_CELL_MAX_RAD_CELLS), 1)
    empty_cell_weight = 0.05
    best_grid = max(min_grid, min(max_grid, occ_grid))
    best_rad = max(1, int(math.ceil(search_radius / max(float(side) / float(best_grid), 1.0e-30))) + 1)
    best_score = float("inf")
    for grid_candidate in range(start_grid, stop_grid + 1):
        spacing_candidate = float(side) / float(grid_candidate)
        rad_candidate = max(1, int(math.ceil(search_radius / max(spacing_candidate, 1.0e-30))) + 1)
        if rad_candidate > max_rad_cells:
            continue
        avg_occ_candidate = float(n_nodes) / max(float(grid_candidate) ** 3, 1.0)
        cells_visited = float((2 * rad_candidate + 1) ** 3)
        score = cells_visited * (avg_occ_candidate + empty_cell_weight)
        if score < best_score:
            best_score = float(score)
            best_grid = int(grid_candidate)
            best_rad = int(rad_candidate)
    if not np.isfinite(best_score):
        rad_cap_grid = int(math.floor(float(max_rad_cells) * side / max(search_radius, 1.0e-30)))
        best_grid = max(min_grid, min(max_grid, max(rad_cap_grid, min_grid)))
        best_rad = max(1, int(math.ceil(search_radius / max(float(side) / float(best_grid), 1.0e-30))) + 1)
    grid_n = int(best_grid)
    spacing = float(side) / float(grid_n)
    rad_cells = int(best_rad)

    coords = np.floor((gl_flat - origin[None, :]) / np.float32(spacing)).astype(np.int32)
    coords = np.clip(coords, 0, grid_n - 1)
    flat = (
        (coords[:, 0].astype(np.int64) * np.int64(grid_n) + coords[:, 1].astype(np.int64))
        * np.int64(grid_n)
        + coords[:, 2].astype(np.int64)
    )
    order = np.argsort(flat, kind="stable").astype(np.int32, copy=False)
    sorted_flat = np.asarray(flat[order], dtype=np.int64)
    grid_cells = int(grid_n * grid_n * grid_n)
    counts = np.bincount(sorted_flat, minlength=grid_cells)
    cell_ptr = np.zeros((grid_cells + 1,), dtype=np.int32)
    cell_ptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64).astype(np.int32)
    build_s = perf_counter() - t0

    t0 = perf_counter()
    cell = {
        "cell_ptr_g": _cp.asarray(cell_ptr),
        "cell_node_ids_g": _cp.asarray(order),
        "origin": np.asarray(origin, dtype=np.float32),
        "spacing": float(spacing),
        "grid_n": int(grid_n),
        "rad_cells": int(rad_cells),
        "max_window": float(max_window),
        "search_radius": float(search_radius),
        "n_nodes": int(n_nodes),
    }
    _cp.cuda.Stream.null.synchronize()
    upload_s = perf_counter() - t0
    return cell, {
        "build_s": float(build_s),
        "upload_s": float(upload_s),
        "grid_n": int(grid_n),
        "rad_cells": int(rad_cells),
        "avg_occupancy": float(n_nodes) / max(float(grid_cells), 1.0),
        "estimated_cells_per_point": float((2 * int(rad_cells) + 1) ** 3),
        "estimated_nodes_per_point": float((2 * int(rad_cells) + 1) ** 3) * (float(n_nodes) / max(float(grid_cells), 1.0)),
        "max_window": float(max_window),
        "search_radius": float(search_radius),
    }


def _compute_tissue_samples_greens_from_cext_state_gpu(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cext_state: dict,
    *,
    tissue_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    global _LAST_TISSUE_TIMINGS
    _LAST_TISSUE_TIMINGS = {}
    if _cp is None or _resolve_tissue_accel_mode(require_gpu=True) != "gpu":
        raise RuntimeError("Cext tissue GPU mode could not be initialized.")
    if points.size == 0 or starts.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    t_total = perf_counter()
    t0 = perf_counter()
    max_nearby = min(NEAREST_TISSUE_VESSELS, len(starts))
    if tissue_cache is not None and tissue_cache.get("gpu"):
        context = tissue_cache["context"]
        max_nearby = int(tissue_cache.get("max_nearby", max_nearby))
    elif tissue_cache is not None and tissue_cache.get("streaming") and isinstance(tissue_cache.get("context"), dict):
        context = tissue_cache["context"]
        max_nearby = int(tissue_cache.get("max_nearby", max_nearby))
    else:
        context = _build_tissue_geometry_context(starts, ends, radii)

    starts_si = context["starts_si"]
    radii_si = context["radii_si"]
    segment_vectors = context["segment_vectors"]
    seg_len_sq = context["seg_len_sq"]
    seg_len = context["seg_len"]
    valid = np.asarray(context["valid_mask"], dtype=bool)
    nseg = int(starts_si.shape[0])
    if nseg == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    candidate_k = min(nseg, max(max_nearby, max_nearby * TISSUE_KDTREE_CANDIDATE_MULT))
    keep_k = min(max_nearby, candidate_k, nseg)
    # If the KDTree candidate request is already dense, querying/sorting most
    # segments on CPU is slower than doing the exact all-segment refinement on
    # GPU and avoids transferring the large candidate-index matrix.
    use_all_segment_refine = candidate_k >= nseg or (candidate_k / max(float(nseg), 1.0)) >= 0.5
    tree = None
    valid_source_ids = np.flatnonzero(valid).astype(np.int64, copy=False)
    gl_points_all = np.asarray(cext_state["gl_points_si"], dtype=np.float32)
    lambda_all = np.asarray(cext_state["lambda_iv_gl"], dtype=np.float32)
    q_weighted_all = np.asarray(cext_state["q_weighted_gl"], dtype=np.float32)
    mono2_weight_all = np.asarray(cext_state.get("mono2_weight_gl", np.zeros_like(q_weighted_all)), dtype=np.float32)
    dipole2_weight_all = np.asarray(cext_state.get("dipole2_weight_gl", np.zeros_like(q_weighted_all)), dtype=np.float32)
    seg_cap_all = np.asarray(cext_state["seg_cap_gl"], dtype=np.float32)
    gl_points_valid = np.asarray(gl_points_all[valid_source_ids], dtype=np.float32)
    lambda_valid = np.asarray(lambda_all[valid_source_ids], dtype=np.float32)
    q_weighted_valid = np.asarray(q_weighted_all[valid_source_ids], dtype=np.float32)
    mono2_weight_valid = np.asarray(mono2_weight_all[valid_source_ids], dtype=np.float32)
    dipole2_weight_valid = np.asarray(dipole2_weight_all[valid_source_ids], dtype=np.float32)
    seg_cap_valid = np.asarray(seg_cap_all[valid_source_ids], dtype=np.float32)
    gl_order = int(gl_points_valid.shape[1]) if gl_points_valid.ndim >= 3 else 0
    if gl_order <= 0:
        return np.ones((points.shape[0],), dtype=bool), np.zeros((points.shape[0],), dtype=float)

    diffusivity_si = float(cext_state["diffusivity_si"])
    window_factor = float(cext_state.get("window_factor", CEXT_WINDOW_FACTOR))
    t_setup_cpu = perf_counter() - t0

    t0 = perf_counter()
    starts_g = _cp.asarray(np.asarray(starts_si, dtype=np.float32))
    segment_vectors_g = _cp.asarray(np.asarray(segment_vectors, dtype=np.float32))
    seg_len_sq_g = _cp.asarray(np.asarray(seg_len_sq, dtype=np.float32))
    seg_len_g = _cp.asarray(np.asarray(seg_len, dtype=np.float32))
    radii_g = _cp.asarray(np.asarray(radii_si, dtype=np.float32))
    gl_points_g = _cp.asarray(gl_points_valid)
    lambda_g = _cp.asarray(lambda_valid)
    q_weighted_g = _cp.asarray(q_weighted_valid)
    mono2_weight_g = _cp.asarray(mono2_weight_valid)
    dipole2_weight_g = _cp.asarray(dipole2_weight_valid)
    seg_cap_g = _cp.asarray(seg_cap_valid)
    _cp.cuda.Stream.null.synchronize()
    t_upload_static = perf_counter() - t0

    keep_mask_all = np.zeros(points.shape[0], dtype=bool)
    result = np.zeros(points.shape[0], dtype=np.float32)
    points_si_all = np.asarray(points * CM_TO_M, dtype=np.float32)

    chunk_points = max(int(TISSUE_GPU_CHUNK_POINTS), int(TISSUE_GPU_MIN_CHUNK_POINTS))
    min_chunk = max(int(TISSUE_GPU_MIN_CHUNK_POINTS), 1)
    start_idx = 0
    chunks_done = 0
    t_query = 0.0
    t_transfer = 0.0
    t_refine = 0.0
    t_greens = 0.0
    t_download = 0.0

    validate_n = min(int(TISSUE_GPU_VALIDATE_POINTS), int(points.shape[0]))
    if bool(TISSUE_CEXT_CELL_LIST_ENABLE) and validate_n <= 0:
        try:
            cell, cell_metrics = _build_cext_tissue_cell_list_gpu(
                gl_points_valid,
                lambda_valid,
                points_si_all,
                window_factor=window_factor,
                radii_si=np.asarray(radii_si, dtype=np.float32),
            )
            kernel_cell = _get_cext_tissue_cell_gpu_kernel()
            start_idx = 0
            chunks_done = 0
            while start_idx < points.shape[0]:
                end_idx = min(start_idx + chunk_points, points.shape[0])
                chunk = points_si_all[start_idx:end_idx]
                try:
                    t0 = perf_counter()
                    points_g = _cp.asarray(chunk, dtype=_cp.float32)
                    keep_g = _cp.empty((chunk.shape[0],), dtype=_cp.uint8)
                    out_g = _cp.zeros((chunk.shape[0],), dtype=_cp.float32)
                    _cp.cuda.Stream.null.synchronize()
                    t_transfer += perf_counter() - t0

                    t0 = perf_counter()
                    threads = 128
                    blocks = (int(chunk.shape[0]) + threads - 1) // threads
                    origin = np.asarray(cell["origin"], dtype=np.float32)
                    kernel_cell(
                        (blocks,),
                        (threads,),
                        (
                            points_g.ravel(),
                            cell["cell_ptr_g"],
                            cell["cell_node_ids_g"],
                            starts_g.ravel(),
                            segment_vectors_g.ravel(),
                            seg_len_sq_g,
                            seg_len_g,
                            radii_g,
                            gl_points_g.ravel(),
                            lambda_g.ravel(),
                            q_weighted_g.ravel(),
                            mono2_weight_g.ravel(),
                            dipole2_weight_g.ravel(),
                            seg_cap_g,
                            np.float32(diffusivity_si),
                            np.float32(window_factor),
                            np.float32(float(origin[0])),
                            np.float32(float(origin[1])),
                            np.float32(float(origin[2])),
                            np.float32(float(cell["spacing"])),
                            np.int32(int(cell["grid_n"])),
                            np.int32(int(cell["rad_cells"])),
                            np.int32(chunk.shape[0]),
                            np.int32(gl_order),
                            keep_g,
                            out_g,
                        ),
                    )
                    _cp.cuda.Stream.null.synchronize()
                    t_greens += perf_counter() - t0

                    t0 = perf_counter()
                    keep_mask_all[start_idx:end_idx] = _cp.asnumpy(keep_g).astype(bool)
                    result[start_idx:end_idx] = _cp.asnumpy(out_g)
                    t_download += perf_counter() - t0
                    chunks_done += 1
                    start_idx = end_idx
                except _cp.cuda.memory.OutOfMemoryError:
                    _cp.get_default_memory_pool().free_all_blocks()
                    if chunk_points <= min_chunk:
                        raise
                    chunk_points = max(min_chunk, chunk_points // 2)
                    print(f"WARNING: GPU OOM in Cext tissue cell-list chunk; retrying with --tissue-gpu-chunk-points={chunk_points}")

            if SOLVER_TIMING_DETAILS:
                print(
                    "  Tissue GPU Cext solve: "
                    f"points={points.shape[0]} active_points={int(np.count_nonzero(keep_mask_all))} "
                    f"candidate_mode=cell_list cell_grid={int(cell_metrics['grid_n'])} "
                    f"rad_cells={int(cell_metrics['rad_cells'])} avg_cell_occ={float(cell_metrics['avg_occupancy']):.3f} "
                    f"est_nodes_per_point={float(cell_metrics['estimated_nodes_per_point']):.1f} "
                    f"chunks={chunks_done} chunk_points={chunk_points} "
                    f"setup_cpu={_fmt_seconds(t_setup_cpu)} upload_static={_fmt_seconds(t_upload_static)} "
                    f"cell_build={_fmt_seconds(cell_metrics['build_s'])} cell_upload={_fmt_seconds(cell_metrics['upload_s'])} "
                    f"query={_fmt_seconds(0.0)} transfer={_fmt_seconds(t_transfer)} "
                    f"refine={_fmt_seconds(0.0)} greens={_fmt_seconds(t_greens)} "
                    f"download={_fmt_seconds(t_download)} total={_fmt_seconds(perf_counter() - t_total)}"
                )

            total_elapsed = perf_counter() - t_total
            geometry_elapsed = float(
                t_setup_cpu
                + t_upload_static
                + float(cell_metrics["build_s"])
                + float(cell_metrics["upload_s"])
                + t_transfer
                + t_download
            )
            _LAST_TISSUE_TIMINGS = {
                "backend": "cext_gpu_cell_list",
                "source_mode": "cext_converged_q_flux",
                "cache_mode": "gpu_cell_list",
                "t_tissue_geometry_s": geometry_elapsed,
                "t_tissue_oxygen_s": float(t_greens),
                "t_tissue_total_s": float(total_elapsed),
                "t_tissue_kdtree_query_s": 0.0,
                "t_tissue_gpu_refine_s": 0.0,
                "t_tissue_gpu_transfer_s": float(t_transfer + t_download + t_upload_static + float(cell_metrics["upload_s"])),
            }
            return keep_mask_all, result.astype(float, copy=False)
        except Exception:
            print("WARNING: Cext tissue cell-list GPU path failed; falling back to KDTree/refine path.")
            traceback.print_exc()
            _cp.get_default_memory_pool().free_all_blocks()
            keep_mask_all[:] = False
            result[:] = np.float32(0.0)
            chunks_done = 0
            t_query = 0.0
            t_transfer = 0.0
            t_refine = 0.0
            t_greens = 0.0
            t_download = 0.0

    kernel = _get_cext_tissue_gpu_kernel()
    if not use_all_segment_refine:
        tree = _ensure_tissue_context_kdtree(context)
        if tree is None:
            raise RuntimeError("Cext tissue GPU mode requires scipy cKDTree for CPU candidate query.")

    while start_idx < points.shape[0]:
        end_idx = min(start_idx + chunk_points, points.shape[0])
        chunk = points_si_all[start_idx:end_idx]
        try:
            cand = None
            if not use_all_segment_refine:
                t0 = perf_counter()
                _, cand = _ckdtree_query(tree, chunk, k=candidate_k)
                t_query += perf_counter() - t0
                if cand.ndim == 1:
                    cand = cand[:, None]

            t0 = perf_counter()
            points_g = _cp.asarray(chunk, dtype=_cp.float32)
            if use_all_segment_refine:
                cand_g = None
            else:
                cand_g = _cp.asarray(cand, dtype=_cp.int32)
            _cp.cuda.Stream.null.synchronize()
            t_transfer += perf_counter() - t0

            t0 = perf_counter()
            if use_all_segment_refine:
                starts_c = starts_g[None, :, :]
                seg_c = segment_vectors_g[None, :, :]
                seg_len_sq_c = seg_len_sq_g[None, :]
            else:
                starts_c = starts_g[cand_g]
                seg_c = segment_vectors_g[cand_g]
                seg_len_sq_c = seg_len_sq_g[cand_g]
            diff = points_g[:, None, :] - starts_c
            proj = _cp.sum(diff * seg_c, axis=2) / seg_len_sq_c
            proj_clip = _cp.clip(proj, 0.0, 1.0)
            closest = starts_c + proj_clip[:, :, None] * seg_c
            delta = points_g[:, None, :] - closest
            dist_sq = _cp.sum(delta * delta, axis=2)
            sel = _cp.argpartition(dist_sq, keep_k - 1, axis=1)[:, :keep_k]
            if use_all_segment_refine:
                nearest_g = sel.astype(_cp.int32, copy=False)
            else:
                nearest_g = _cp.take_along_axis(cand_g, sel, axis=1).astype(_cp.int32, copy=False)
            proj_g = _cp.take_along_axis(proj, sel, axis=1).astype(_cp.float32, copy=False)
            d_center_g = _cp.sqrt(_cp.take_along_axis(dist_sq, sel, axis=1)).astype(_cp.float32, copy=False)
            radius_local = _cp.minimum(radii_g[nearest_g], seg_len_g[nearest_g])
            inside_any = _cp.any((proj_g >= 0.0) & (proj_g <= 1.0) & (d_center_g <= radius_local), axis=1)
            keep_g = (~inside_any).astype(_cp.uint8, copy=False)
            _cp.cuda.Stream.null.synchronize()
            t_refine += perf_counter() - t0

            t0 = perf_counter()
            out_g = _cp.zeros((chunk.shape[0],), dtype=_cp.float32)
            threads = 128
            blocks = (int(chunk.shape[0]) + threads - 1) // threads
            kernel(
                (blocks,),
                (threads,),
                (
                    points_g.ravel(),
                    nearest_g.ravel(),
                    keep_g,
                    gl_points_g.ravel(),
                    segment_vectors_g.ravel(),
                    lambda_g.ravel(),
                    q_weighted_g.ravel(),
                    mono2_weight_g.ravel(),
                    dipole2_weight_g.ravel(),
                    seg_cap_g,
                    np.float32(diffusivity_si),
                    np.float32(window_factor),
                    np.int32(chunk.shape[0]),
                    np.int32(keep_k),
                    np.int32(gl_order),
                    out_g,
                ),
            )
            _cp.cuda.Stream.null.synchronize()
            t_greens += perf_counter() - t0

            t0 = perf_counter()
            keep_mask_all[start_idx:end_idx] = _cp.asnumpy(keep_g).astype(bool)
            result[start_idx:end_idx] = _cp.asnumpy(out_g)
            t_download += perf_counter() - t0
            chunks_done += 1
            start_idx = end_idx
        except _cp.cuda.memory.OutOfMemoryError:
            _cp.get_default_memory_pool().free_all_blocks()
            if chunk_points <= min_chunk:
                raise
            chunk_points = max(min_chunk, chunk_points // 2)
            print(f"WARNING: GPU OOM in Cext tissue chunk; retrying with --tissue-gpu-chunk-points={chunk_points}")

    if SOLVER_TIMING_DETAILS:
        print(
            "  Tissue GPU Cext solve: "
            f"points={points.shape[0]} active_points={int(np.count_nonzero(keep_mask_all))} "
            f"candidate_k={candidate_k} keep_k={keep_k} candidate_mode={'all_gpu' if use_all_segment_refine else 'kdtree'} "
            f"chunks={chunks_done} chunk_points={chunk_points} "
            f"setup_cpu={_fmt_seconds(t_setup_cpu)} upload_static={_fmt_seconds(t_upload_static)} "
            f"query={_fmt_seconds(t_query)} transfer={_fmt_seconds(t_transfer)} "
            f"refine={_fmt_seconds(t_refine)} greens={_fmt_seconds(t_greens)} "
            f"download={_fmt_seconds(t_download)} total={_fmt_seconds(perf_counter() - t_total)}"
        )

    total_elapsed = perf_counter() - t_total
    geometry_elapsed = float(t_setup_cpu + t_upload_static + t_query + t_transfer + t_refine + t_download)
    _LAST_TISSUE_TIMINGS = {
        "backend": "cext_gpu",
        "source_mode": "cext_converged_q_flux",
        "cache_mode": "gpu_streaming",
        "t_tissue_geometry_s": geometry_elapsed,
        "t_tissue_oxygen_s": float(t_greens),
        "t_tissue_total_s": float(total_elapsed),
        "t_tissue_kdtree_query_s": float(t_query),
        "t_tissue_gpu_refine_s": float(t_refine),
        "t_tissue_gpu_transfer_s": float(t_transfer + t_download + t_upload_static),
    }

    validate_n = min(int(TISSUE_GPU_VALIDATE_POINTS), int(points.shape[0]))
    if validate_n > 0:
        old_accel = TISSUE_ACCEL_MODE
        old_streaming = TISSUE_STREAMING_ENABLED
        try:
            TISSUE_ACCEL_MODE = "cpu"
            TISSUE_STREAMING_ENABLED = False
            cpu_mask, cpu_vals = compute_tissue_samples_greens_from_cext_state(
                points[:validate_n],
                starts,
                ends,
                radii,
                cext_state,
                tissue_cache=None,
            )
        finally:
            TISSUE_ACCEL_MODE = old_accel
            TISSUE_STREAMING_ENABLED = old_streaming
        gpu_mask = keep_mask_all[:validate_n]
        gpu_vals = result[:validate_n].astype(float)
        mask_disagree = int(np.count_nonzero(gpu_mask != cpu_mask))
        finite = np.isfinite(cpu_vals) & np.isfinite(gpu_vals)
        max_abs = float(np.max(np.abs(gpu_vals[finite] - cpu_vals[finite]))) if np.any(finite) else float("nan")
        rel_l2 = _relative_l2(gpu_vals[finite], cpu_vals[finite]) if np.any(finite) else float("nan")
        print(
            "  Cext tissue GPU validation: "
            f"points={validate_n} mask_disagree={mask_disagree} "
            f"max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}"
        )

    return keep_mask_all, result.astype(float, copy=False)


def _process_cext_tissue_chunk(start_idx: int, end_idx: int, data: dict) -> tuple[int, np.ndarray]:
    points_si = data["points_si"]
    nearest_idx = data["nearest_idx"]
    keep_mask = data["keep_mask"]
    valid_source_ids = data["valid_source_ids"]
    gl_points_si = data["gl_points_si"]
    segment_vectors = data.get("segment_vectors")
    lambda_iv_gl = data["lambda_iv_gl"]
    q_weighted_gl = data["q_weighted_gl"]
    mono2_weight_gl = data.get("mono2_weight_gl")
    dipole2_weight_gl = data.get("dipole2_weight_gl")
    seg_cap_gl = data["seg_cap_gl"]
    diffusivity_si = float(data["diffusivity_si"])
    window_factor = float(data["window_factor"])

    chunk = points_si[start_idx:end_idx]
    out = np.zeros((len(chunk),), dtype=float)
    if chunk.size == 0:
        return start_idx, out

    idx_local = nearest_idx[start_idx:end_idx]
    gl_order = int(gl_points_si.shape[1]) if gl_points_si.ndim >= 3 else 0
    for row, point in enumerate(chunk):
        if not keep_mask[start_idx + row]:
            continue
        total = 0.0
        cap_max = 0.0
        for compact_seg in idx_local[row]:
            if compact_seg < 0 or compact_seg >= valid_source_ids.size:
                continue
            source_seg = int(valid_source_ids[int(compact_seg)])
            if source_seg < 0 or source_seg >= gl_points_si.shape[0]:
                continue
            seg_contributed = False
            for source_node in range(gl_order):
                source_lambda = max(float(lambda_iv_gl[source_seg, source_node]), 1e-30)
                source_point = gl_points_si[source_seg, source_node]
                r = float(np.linalg.norm(point - source_point))
                if r <= 1e-12 or r > window_factor * source_lambda:
                    continue
                kernel = math.exp(-r / source_lambda) / (4.0 * math.pi * diffusivity_si * r)
                q = float(q_weighted_gl[source_seg, source_node])
                o2_weight = 0.0
                if mono2_weight_gl is not None:
                    o2_weight += float(mono2_weight_gl[source_seg, source_node])
                if dipole2_weight_gl is not None:
                    o2_weight += float(dipole2_weight_gl[source_seg, source_node])
                total += q * kernel
                if o2_weight != 0.0 and segment_vectors is not None:
                    vec = np.asarray(segment_vectors[source_seg], dtype=float)
                    vlen = float(np.linalg.norm(vec))
                    if vlen > 1e-30:
                        dr = np.asarray(point - source_point, dtype=float)
                        tdotr = float(np.dot(vec, dr) / vlen)
                        mu2 = min((tdotr * tdotr) / max(r * r, 1e-30), 1.0)
                        inv_r = 1.0 / r
                        inv_l = 1.0 / source_lambda
                        pH = ((1.0 - 3.0 * mu2) * (inv_r * inv_r + inv_l * inv_r)
                              + (1.0 - mu2) * inv_l * inv_l) * kernel
                        total += o2_weight * pH
                seg_contributed = True
            if seg_contributed:
                cap_max = max(cap_max, float(seg_cap_gl[source_seg]))
        if total < 0.0 or not np.isfinite(total):
            total = 0.0
        if cap_max > 0.0:
            total = min(total, cap_max)
        out[row] = total
    return start_idx, out


def _compute_cext_streaming_tissue_chunk(task: tuple[int, int, int, dict]) -> tuple[int, int, int, np.ndarray, np.ndarray, int, int]:
    chunk_i, start_idx, end_idx, state = task
    chunk_cache = _prepare_tissue_geometry_from_context(
        state["points"][start_idx:end_idx],
        state["context"],
        max_nearby=int(state["max_nearby"]),
    )
    before_candidates = int(np.count_nonzero(chunk_cache["nearest_idx"] >= 0))
    if state["prune_by_window"]:
        chunk_cache = _compact_tissue_cache_by_influence(chunk_cache, state["influence_radius"])
    after_candidates = int(np.count_nonzero(chunk_cache["nearest_idx"] >= 0))
    worker_data = {
        "points_si": chunk_cache["points_si"],
        "nearest_idx": chunk_cache["nearest_idx"],
        "keep_mask": chunk_cache["keep_mask"],
        "valid_source_ids": state["valid_source_ids"],
        "gl_points_si": state["gl_points_si"],
        "lambda_iv_gl": state["lambda_iv_gl"],
        "q_weighted_gl": state["q_weighted_gl"],
        "mono2_weight_gl": state["mono2_weight_gl"],
        "dipole2_weight_gl": state["dipole2_weight_gl"],
        "segment_vectors": state["segment_vectors"],
        "seg_cap_gl": state["seg_cap_gl"],
        "diffusivity_si": state["diffusivity_si"],
        "window_factor": state["window_factor"],
    }
    _, out = _process_cext_tissue_chunk(0, end_idx - start_idx, worker_data)
    keep_mask = np.asarray(chunk_cache["keep_mask"], dtype=bool)
    return chunk_i, start_idx, end_idx, keep_mask, np.asarray(out, dtype=float), before_candidates, after_candidates


def compute_tissue_samples_greens_from_cext_state(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cext_state: dict,
    *,
    tissue_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate tissue oxygen from converged Cext source fluxes."""
    global _LAST_TISSUE_TIMINGS
    _LAST_TISSUE_TIMINGS = {}
    if points.size == 0 or starts.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    t_total = perf_counter()
    t0 = perf_counter()
    gl_points_si = np.asarray(cext_state["gl_points_si"], dtype=np.float32)
    lambda_iv_gl = np.asarray(cext_state["lambda_iv_gl"], dtype=np.float32)
    q_weighted_gl = np.asarray(cext_state["q_weighted_gl"], dtype=np.float32)
    mono2_weight_gl = np.asarray(cext_state.get("mono2_weight_gl", np.zeros_like(q_weighted_gl)), dtype=np.float32)
    dipole2_weight_gl = np.asarray(cext_state.get("dipole2_weight_gl", np.zeros_like(q_weighted_gl)), dtype=np.float32)
    segment_vectors_state = np.asarray(
        cext_state.get("segment_vectors", np.zeros((gl_points_si.shape[0], 3), dtype=np.float32)),
        dtype=np.float32,
    )
    seg_cap_gl = np.asarray(cext_state["seg_cap_gl"], dtype=np.float32)
    diffusivity_si = float(cext_state["diffusivity_si"])
    window_factor = float(cext_state.get("window_factor", CEXT_WINDOW_FACTOR))
    max_nearby = min(NEAREST_TISSUE_VESSELS, len(starts))
    result = np.zeros(points.shape[0], dtype=float)
    validation = validate_cext_tissue_flux_consistency(cext_state)

    if _resolve_tissue_accel_mode() == "gpu" and _cp is not None:
        keep_mask, gpu_result = _compute_tissue_samples_greens_from_cext_state_gpu(
            points,
            starts,
            ends,
            radii,
            cext_state,
            tissue_cache=tissue_cache,
        )
        if SOLVER_TIMING_DETAILS:
            print(
                "    Cext tissue diagnostics: "
                f"{_diagnostic_stats('c_iv_gl', np.asarray(cext_state['c_iv_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('c_ext_gl', np.asarray(cext_state['c_ext_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('c_iv_minus_c_ext', np.asarray(cext_state['c_iv_gl'], dtype=np.float32) - np.asarray(cext_state['c_ext_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('q_line_gl', np.asarray(cext_state['q_line_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('tissue', gpu_result)} "
                f"flux_check_ok={validation['ok']} flux_rel_l2={float(validation['rel_l2']):.3e}"
            )
        _LAST_TISSUE_TIMINGS.update(
            {
                "cext_flux_check_rel_l2": float(validation["rel_l2"]),
                "cext_flux_check_max_abs": float(validation["max_abs"]),
            }
        )
        return keep_mask, gpu_result

    use_streaming = (
        (tissue_cache is not None and (tissue_cache.get("streaming") or tissue_cache.get("gpu")))
        or (tissue_cache is None and int(points.shape[0]) >= int(TISSUE_STREAMING_MIN_POINTS))
    )
    if use_streaming:
        if tissue_cache is not None and isinstance(tissue_cache.get("context"), dict):
            context = tissue_cache["context"]
            max_nearby = int(tissue_cache.get("max_nearby", max_nearby))
        else:
            context = _build_tissue_geometry_context(starts, ends, radii)
        _ensure_tissue_context_kdtree(context)
        valid_source_ids = np.flatnonzero(np.asarray(context["valid_mask"], dtype=bool)).astype(np.int64, copy=False)
        lambda_valid = np.asarray(lambda_iv_gl[valid_source_ids], dtype=np.float32) if valid_source_ids.size else np.empty((0, 0), dtype=np.float32)
        influence_radius = np.asarray(window_factor * np.max(lambda_valid, axis=1), dtype=context.get("cache_float", np.float32)) if lambda_valid.size else np.empty((0,), dtype=context.get("cache_float", np.float32))
        chunk_size = _streaming_tissue_chunk_size(max_nearby)
        tasks = [
            (chunk_i, start_idx, min(start_idx + chunk_size, len(points)), None)
            for chunk_i, start_idx in enumerate(range(0, len(points), chunk_size), start=1)
        ]
        state = {
            "points": points,
            "context": context,
            "max_nearby": max_nearby,
            "valid_source_ids": valid_source_ids,
            "gl_points_si": gl_points_si,
            "lambda_iv_gl": lambda_iv_gl,
            "q_weighted_gl": q_weighted_gl,
            "mono2_weight_gl": mono2_weight_gl,
            "dipole2_weight_gl": dipole2_weight_gl,
            "segment_vectors": segment_vectors_state,
            "seg_cap_gl": seg_cap_gl,
            "diffusivity_si": diffusivity_si,
            "window_factor": window_factor,
            "influence_radius": influence_radius,
            "prune_by_window": bool(TISSUE_STREAMING_PRUNE_BY_WINDOW),
        }
        tasks = [(chunk_i, start_idx, end_idx, state) for chunk_i, start_idx, end_idx, _ in tasks]
        keep_mask = np.zeros((points.shape[0],), dtype=bool)
        total_before_candidates = 0
        total_after_candidates = 0
        t_setup = perf_counter() - t0
        t0 = perf_counter()
        workers = max(int(TISSUE_STREAMING_CHUNK_WORKERS), 1)
        if workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {executor.submit(_compute_cext_streaming_tissue_chunk, task): task[0] for task in tasks}
                for fut in as_completed(future_map):
                    _, start_idx, end_idx, chunk_keep, out, before_c, after_c = fut.result()
                    keep_mask[start_idx:end_idx] = chunk_keep
                    result[start_idx:end_idx] = out
                    total_before_candidates += int(before_c)
                    total_after_candidates += int(after_c)
        else:
            for task in tasks:
                _, start_idx, end_idx, chunk_keep, out, before_c, after_c = _compute_cext_streaming_tissue_chunk(task)
                keep_mask[start_idx:end_idx] = chunk_keep
                result[start_idx:end_idx] = out
                total_before_candidates += int(before_c)
                total_after_candidates += int(after_c)
        t_kernel = perf_counter() - t0
        cache_mode = "streaming"
        keep_k = max_nearby
        candidate_slots = int(total_after_candidates)
    else:
        if tissue_cache is None:
            tissue_cache = _prepare_tissue_geometry(
                points,
                starts,
                ends,
                radii,
                max_nearby=max_nearby,
            )
        points_si = np.asarray(tissue_cache["points_si"], dtype=float)
        nearest_idx = np.asarray(tissue_cache["nearest_idx"], dtype=np.int64)
        keep_mask = np.asarray(tissue_cache["keep_mask"], dtype=bool).copy()
        valid_source_ids = np.flatnonzero(np.asarray(tissue_cache["valid_mask"], dtype=bool)).astype(np.int64, copy=False)
        t_setup = perf_counter() - t0
        worker_data = {
            "points_si": points_si,
            "nearest_idx": nearest_idx,
            "keep_mask": keep_mask,
            "valid_source_ids": valid_source_ids,
            "gl_points_si": gl_points_si,
            "lambda_iv_gl": lambda_iv_gl,
            "q_weighted_gl": q_weighted_gl,
            "mono2_weight_gl": mono2_weight_gl,
            "dipole2_weight_gl": dipole2_weight_gl,
            "segment_vectors": segment_vectors_state,
            "seg_cap_gl": seg_cap_gl,
            "diffusivity_si": diffusivity_si,
            "window_factor": window_factor,
        }
        ranges = [(i, min(i + DISTANCE_CHUNK_SIZE, len(points))) for i in range(0, len(points), DISTANCE_CHUNK_SIZE)]
        t0 = perf_counter()
        if TISSUE_PARALLEL_WORKERS > 1 and len(ranges) > 1:
            with ThreadPoolExecutor(max_workers=TISSUE_PARALLEL_WORKERS) as executor:
                for start_idx, out in executor.map(
                    lambda r: _process_cext_tissue_chunk(*r, worker_data),
                    ranges,
                ):
                    result[start_idx: start_idx + len(out)] = out
        else:
            for r in ranges:
                start_idx, out = _process_cext_tissue_chunk(*r, worker_data)
                result[start_idx: start_idx + len(out)] = out
        t_kernel = perf_counter() - t0
        cache_mode = "dense"
        keep_k = int(nearest_idx.shape[1]) if nearest_idx.ndim == 2 else 0
        candidate_slots = int(nearest_idx.shape[0] * nearest_idx.shape[1]) if nearest_idx.ndim == 2 else 0

    if SOLVER_TIMING_DETAILS:
        print(
            "  Tissue Greens solve: source_mode=cext_converged_q_flux "
            f"solver={cext_state.get('solver', 'unknown')} points={points.shape[0]} "
            f"cache={cache_mode} active_points={int(np.count_nonzero(keep_mask))} keep_k={keep_k} "
            f"candidate_slots={candidate_slots} "
            f"gl_order_cext={gl_points_si.shape[1] if gl_points_si.ndim >= 3 else 0} "
            f"setup={_fmt_seconds(t_setup)} kernel={_fmt_seconds(t_kernel)} "
            f"total={_fmt_seconds(perf_counter() - t_total)}"
        )
        print(
            "    Cext tissue diagnostics: "
            f"{_diagnostic_stats('c_iv_gl', np.asarray(cext_state['c_iv_gl'], dtype=np.float32))} "
            f"{_diagnostic_stats('c_ext_gl', np.asarray(cext_state['c_ext_gl'], dtype=np.float32))} "
            f"{_diagnostic_stats('c_iv_minus_c_ext', np.asarray(cext_state['c_iv_gl'], dtype=np.float32) - np.asarray(cext_state['c_ext_gl'], dtype=np.float32))} "
            f"{_diagnostic_stats('q_line_gl', np.asarray(cext_state['q_line_gl'], dtype=np.float32))} "
            f"{_diagnostic_stats('tissue', result)} "
            f"flux_check_ok={validation['ok']} flux_rel_l2={float(validation['rel_l2']):.3e}"
        )

    _LAST_TISSUE_TIMINGS = {
        "backend": "cext_cpu",
        "source_mode": "cext_converged_q_flux",
        "cache_mode": cache_mode,
        "t_tissue_geometry_s": 0.0,
        "t_tissue_oxygen_s": float(t_kernel),
        "t_tissue_total_s": float(perf_counter() - t_total),
        "cext_flux_check_rel_l2": float(validation["rel_l2"]),
        "cext_flux_check_max_abs": float(validation["max_abs"]),
    }
    return keep_mask, result


@profile
def compute_tissue_samples_greens(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cin: np.ndarray,
    flows: np.ndarray,
    *,
    diffusivity: float = SOLUTE_DIFFUSIVITY,
    vmax: float = VMAX_MM,
    km: float = K_M_MM,
    window_factor: float = WINDOW_FACTOR,
    inlet_concentration: float | None = None,
    tissue_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    global _LAST_TISSUE_TIMINGS
    _LAST_TISSUE_TIMINGS = {}
    if points.size == 0 or starts.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    if _resolve_tissue_accel_mode() == "gpu":
        return _compute_tissue_samples_greens_gpu(
            points,
            starts,
            ends,
            radii,
            cin,
            flows,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            window_factor=window_factor,
            inlet_concentration=inlet_concentration,
            tissue_cache=tissue_cache,
        )

    if (
        TISSUE_STREAMING_ENABLED
        and (
            (tissue_cache is not None and tissue_cache.get("streaming"))
            or (tissue_cache is None and int(points.shape[0]) >= int(TISSUE_STREAMING_MIN_POINTS))
        )
    ):
        return _compute_tissue_samples_greens_streaming(
            points,
            starts,
            ends,
            radii,
            cin,
            flows,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            window_factor=window_factor,
            inlet_concentration=inlet_concentration,
            tissue_cache=tissue_cache,
        )

    max_nearby = min(NEAREST_TISSUE_VESSELS, len(starts))
    if tissue_cache is None:
        tissue_cache = _prepare_tissue_geometry(
            points,
            starts,
            ends,
            radii,
            max_nearby=max_nearby,
        )

    t_total = perf_counter()
    t0 = perf_counter()
    points_si = tissue_cache["points_si"]
    starts_si = tissue_cache["starts_si"]
    radii_si = tissue_cache["radii_si"]
    segment_vectors = tissue_cache["segment_vectors"]
    seg_len_sq = tissue_cache["seg_len_sq"]
    seg_len = tissue_cache["seg_len"]
    nearest_idx = tissue_cache["nearest_idx"]
    proj_raw = tissue_cache["proj_raw"]
    d_center = tissue_cache["d_center"]
    keep_mask = tissue_cache["keep_mask"].copy()
    valid = tissue_cache["valid_mask"]

    cin = cin[valid]
    flows_si = flows[valid] * CM3_TO_M3
    diffusivity_si = diffusivity * CM2_TO_M2
    if starts_si.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    seg_len = np.sqrt(seg_len_sq)
    cin_pos = np.maximum(np.nan_to_num(cin, nan=0.0), 0.0)
    flows_si = np.nan_to_num(flows_si, nan=0.0)
    radii_si = np.maximum(np.nan_to_num(radii_si, nan=0.0), 0.0)

    denom = np.maximum(km + cin_pos, 1e-30)
    k1 = vmax / denom
    lam_edge = np.sqrt(diffusivity_si / np.maximum(k1, 1e-30))
    phi_edge = radii_si / np.maximum(lam_edge, 1e-30)
    ratio_edge = _k_ratio(phi_edge)
    flow_mag = np.maximum(np.abs(flows_si), 1e-30)
    alpha_edge = (2.0 * np.pi * radii_si / flow_mag) * (diffusivity_si / np.maximum(lam_edge, 1e-30)) * ratio_edge
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet()
    lam_ref = float(np.sqrt(diffusivity_si / max(vmax / max(km + float(inlet_concentration), 1e-30), 1e-30)))

    flow_sign = np.sign(flows_si)
    result = np.zeros(points.shape[0], dtype=float)
    gl_nodes, gl_weights = _get_gl_nodes_weights(GL_ORDER)
    t_setup = perf_counter() - t0
    worker_data = {
        "points_si": points_si,
        "starts_si": starts_si,
        "segment_vectors": segment_vectors,
        "seg_len": seg_len,
        "radii_si": radii_si,
        "nearest_idx": nearest_idx,
        "proj_raw": proj_raw,
        "d_center": d_center,
        "keep_mask": keep_mask,
        "cin_pos": cin_pos,
        "alpha_edge": alpha_edge,
        "flow_sign": flow_sign,
        "diffusivity_si": diffusivity_si,
        "km": km,
        "vmax": vmax,
        "window_factor": window_factor,
        "lam_ref": lam_ref,
        "gl_nodes": gl_nodes,
        "gl_weights": gl_weights,
    }

    ranges = [(i, min(i + DISTANCE_CHUNK_SIZE, len(points))) for i in range(0, len(points), DISTANCE_CHUNK_SIZE)]
    used_numba = False
    t_kernel = 0.0
    if _HAVE_NUMBA and TISSUE_USE_NUMBA and _K0_LUT.size:
        try:
            t0 = perf_counter()
            result = _tissue_kernel_numba(
                points_si,
                starts_si,
                segment_vectors,
                seg_len,
                radii_si,
                nearest_idx,
                proj_raw,
                d_center,
                keep_mask,
                cin_pos,
                alpha_edge,
                flow_sign,
                float(diffusivity_si),
                float(km),
                float(vmax),
                float(window_factor),
                float(lam_ref),
                gl_nodes,
                gl_weights,
                _KRATIO_XS,
                _K0_LUT,
            )
            t_kernel = perf_counter() - t0
            used_numba = True
        except Exception:
            print("WARNING: Numba tissue kernel failed; falling back to non-numba path.")
            traceback.print_exc()
            used_numba = False
    if not used_numba and TISSUE_PARALLEL_WORKERS > 1 and len(ranges) > 1:
        t0 = perf_counter()
        with ThreadPoolExecutor(max_workers=TISSUE_PARALLEL_WORKERS) as executor:
            for start_idx, out in executor.map(
                lambda r: _process_tissue_chunk(*r, worker_data),
                ranges,
            ):
                result[start_idx: start_idx + len(out)] = out
        t_kernel = perf_counter() - t0
    elif not used_numba:
        t0 = perf_counter()
        for r in ranges:
            start_idx, out = _process_tissue_chunk(*r, worker_data)
            result[start_idx: start_idx + len(out)] = out
        t_kernel = perf_counter() - t0

    if SOLVER_TIMING_DETAILS:
        active_points = int(np.count_nonzero(keep_mask))
        candidate_slots = int(nearest_idx.shape[0] * nearest_idx.shape[1]) if nearest_idx.ndim == 2 else 0
        print(
            "  Tissue Greens solve: "
            f"points={points.shape[0]} active_points={active_points} keep_k={nearest_idx.shape[1] if nearest_idx.ndim == 2 else 0} "
            f"candidate_slots={candidate_slots} gl_order={GL_ORDER} numba={used_numba} "
            f"setup={_fmt_seconds(t_setup)} kernel={_fmt_seconds(t_kernel)} "
            f"total={_fmt_seconds(perf_counter() - t_total)}"
        )

    _LAST_TISSUE_TIMINGS = {
        "backend": "cpu",
        "t_tissue_geometry_s": 0.0,
        "t_tissue_oxygen_s": float(t_kernel),
        "t_tissue_total_s": float(perf_counter() - t_total),
    }

    return keep_mask, result


def estimate_bulk_tissue_concentration(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    vessel_sources: np.ndarray,
    *,
    extravascular_concentration: float = EXTRAVASCULAR_CONCENTRATION,
    # decay_length: float = TISSUE_DECAY_LENGTH,
) -> float:
    if points.size == 0 or starts.size == 0:
        return float("nan")
    decay_length = max(decay_length, 1e-9)
    vessel_sources = np.nan_to_num(vessel_sources, nan=extravascular_concentration)

    segment_vectors = ends - starts
    seg_len_sq = np.sum(segment_vectors**2, axis=1)
    valid = seg_len_sq > 1e-10
    starts = starts[valid]
    ends = ends[valid]
    radii = radii[valid]
    vessel_sources = vessel_sources[valid]
    segment_vectors = segment_vectors[valid]
    seg_len_sq = seg_len_sq[valid]

    collected = []
    for idx in range(0, len(points), DISTANCE_CHUNK_SIZE):
        chunk = points[idx: idx + DISTANCE_CHUNK_SIZE]
        if chunk.size == 0:
            continue
        diff = chunk[:, None, :] - starts[None, :, :]
        proj = np.sum(diff * segment_vectors[None, :, :], axis=2) / seg_len_sq[None, :]
        proj = np.clip(proj, 0.0, 1.0)
        closest = starts[None, :, :] + proj[:, :, None] * segment_vectors[None, :, :]
        distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
        idx_min = np.argmin(distances, axis=1)
        d_center = distances[np.arange(len(chunk)), idx_min]
        radius_local = radii[idx_min]
        outside = d_center > radius_local
        if not np.any(outside):
            continue
        d_wall = np.maximum(d_center - radius_local, 0.0)
        supplied = vessel_sources[idx_min]
        weight = np.clip(1.0 - d_wall / decay_length, 0.0, 1.0)
        concentration = extravascular_concentration + (supplied - extravascular_concentration) * weight
        collected.append(concentration[outside])

    if not collected:
        return float("nan")
    values = np.concatenate(collected)
    return float(np.mean(values)) if values.size else float("nan")


def _compute_tissue_samples_linear(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    vessel_sources: np.ndarray,
    *,
    extravascular_concentration: float = EXTRAVASCULAR_CONCENTRATION,
    # decay_length: float = TISSUE_DECAY_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0 or starts.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    decay_length = max(decay_length, 1e-9)
    vessel_sources = np.nan_to_num(vessel_sources, nan=extravascular_concentration)
    segment_vectors = ends - starts
    seg_len_sq = np.sum(segment_vectors**2, axis=1)
    valid = seg_len_sq > 1e-10
    starts = starts[valid]
    ends = ends[valid]
    radii = radii[valid]
    vessel_sources = vessel_sources[valid]
    segment_vectors = segment_vectors[valid]
    seg_len_sq = seg_len_sq[valid]

    result = np.empty(points.shape[0], dtype=float)
    keep_mask = np.ones(points.shape[0], dtype=bool)
    max_nearby = min(NEAREST_TISSUE_VESSELS, len(starts))
    for idx in range(0, len(points), DISTANCE_CHUNK_SIZE):
        chunk = points[idx: idx + DISTANCE_CHUNK_SIZE]
        if chunk.size == 0:
            continue
        diff = chunk[:, None, :] - starts[None, :, :]
        proj_raw = np.sum(diff * segment_vectors[None, :, :], axis=2) / seg_len_sq[None, :]
        proj = np.clip(proj_raw, 0.0, 1.0)
        closest = starts[None, :, :] + proj[:, :, None] * segment_vectors[None, :, :]
        distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
        nearest_idx = np.argpartition(distances, kth=max_nearby - 1, axis=1)[:, :max_nearby]
        d_center = np.take_along_axis(distances, nearest_idx, axis=1)
        proj_sel = np.take_along_axis(proj_raw, nearest_idx, axis=1)
        radius_local = np.minimum(
            np.take_along_axis(radii[None, :], nearest_idx, axis=1),
            np.sqrt(np.take_along_axis(seg_len_sq[None, :], nearest_idx, axis=1)),
        )
        inside_any = np.any((proj_sel >= 0.0) & (proj_sel <= 1.0) & (d_center <= radius_local), axis=1)
        keep_mask[idx: idx + len(chunk)] = ~inside_any
        d_wall = np.maximum(d_center - radius_local, 0.0)
        supplied = np.take_along_axis(vessel_sources[None, :], nearest_idx, axis=1)
        weight = np.clip(1.0 - d_wall / decay_length, 0.0, 1.0)
        conc_candidates = extravascular_concentration + (supplied - extravascular_concentration) * weight
        result[idx: idx + len(chunk)] = np.max(conc_candidates, axis=1)

    return keep_mask, result


def compute_concentration_profiles(
    tree,
    *,
    inlet_concentration: float | None = None,
    extravascular_concentration: float = EXTRAVASCULAR_CONCENTRATION,
    diffusivity: float = SOLUTE_DIFFUSIVITY,
    fluid: str | None = None,
    concentration_solver: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _ = extravascular_concentration
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet(fluid)
    q_inlet_cm3_s = float(QIN_TARGET) * 1e-3 / 60.0
    _, _, _, _, _, cin, cout, _ = solve_tree_greens(
        tree,
        q_inlet_cm3_s,
        fluid=fluid or ACTIVE_FLUID,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=VMAX_MM,
        km=K_M_MM,
        omega=OMEGA,
        concentration_solver=concentration_solver,
    )
    return cin, cout


def compute_concentration_metrics(
    tree,
    sample_points: np.ndarray,
    *,
    inlet_concentration: float | None = None,
    extravascular_concentration: float = EXTRAVASCULAR_CONCENTRATION,
    diffusivity: float = SOLUTE_DIFFUSIVITY,
    # tissue_lengthscale: float = TISSUE_DECAY_LENGTH,
    fluid: str | None = None,
    concentration_solver: str | None = None,
) -> tuple[float, float, Dict[str, float]]:
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet(fluid)
    q_inlet_cm3_s = float(QIN_TARGET) * 1e-3 / 60.0
    starts, ends, radii, lengths, flows, cin, cout, _ = solve_tree_greens(
        tree,
        q_inlet_cm3_s,
        fluid=fluid or ACTIVE_FLUID,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=VMAX_MM,
        km=K_M_MM,
        omega=OMEGA,
        concentration_solver=concentration_solver,
    )

    vessel_map = getattr(tree, "vessel_map", {}) or {}
    terminals = [idx for idx in range(cout.size) if not vessel_map.get(idx, {}).get("downstream")] if vessel_map else []
    if not terminals:
        data = np.asarray(tree.data[: cout.size], dtype=float)
        terminals = list(np.where(np.isnan(data[:, 15]) & np.isnan(data[:, 16]))[0])
    target_concs = cout[terminals] if terminals else cout
    finite = target_concs[np.isfinite(target_concs)]
    c_lq = float(np.percentile(finite, 25.0)) if finite.size else float("nan")

    mask, tissue_conc = compute_tissue_samples_greens(
        sample_points,
        starts,
        ends,
        radii,
        cin,
        flows,
        diffusivity=SOLUTE_DIFFUSIVITY,
        vmax=VMAX_MM,
        km=K_M_MM,
        window_factor=WINDOW_FACTOR,
        inlet_concentration=inlet_concentration,
    )
    tissue_vals = tissue_conc[mask]
    tissue_avg = float(np.nanmean(tissue_vals)) if tissue_vals.size else float("nan")

    if np.isfinite(CONC_MAX_FOR_NORMALIZATION) and CONC_MAX_FOR_NORMALIZATION != 0.0:
        ratio_lq = c_lq / CONC_MAX_FOR_NORMALIZATION if np.isfinite(c_lq) else float("nan")
        ratio_tiss = tissue_avg / CONC_MAX_FOR_NORMALIZATION if np.isfinite(tissue_avg) else float("nan")
        fractions = {}
        if tissue_vals.size:
            normalized = tissue_vals / CONC_MAX_FOR_NORMALIZATION
            for threshold, key in [
                (0.50, "FracAbove50pct"),
                (0.25, "FracAbove25pct"),
                (0.10, "FracAbove10pct"),
                (0.05, "FracAbove5pct"),
                (0.01, "FracAbove1pct"),
            ]:
                fractions[key] = float(np.mean(normalized >= threshold))
        else:
            fractions = {k: float("nan") for k in ["FracAbove50pct", "FracAbove25pct", "FracAbove10pct", "FracAbove5pct", "FracAbove1pct"]}
    else:
        ratio_lq = float("nan")
        ratio_tiss = float("nan")
        fractions = {k: float("nan") for k in ["FracAbove50pct", "FracAbove25pct", "FracAbove10pct", "FracAbove5pct", "FracAbove1pct"]}
    return ratio_lq, ratio_tiss, fractions

_GL5_NODES = np.array([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459], dtype=float)
_GL5_WEIGHTS = np.array([0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850], dtype=float)
_GL9_NODES = np.array(
    [
        -0.9681602395076261,
        -0.8360311073266358,
        -0.6133714327005904,
        -0.3242534234038089,
        0.0,
        0.3242534234038089,
        0.6133714327005904,
        0.8360311073266358,
        0.9681602395076261,
    ],
    dtype=float,
)
_GL9_WEIGHTS = np.array(
    [
        0.0812743883615744,
        0.1806481606948574,
        0.2606106964029354,
        0.3123470770400029,
        0.3302393550012598,
        0.3123470770400029,
        0.2606106964029354,
        0.1806481606948574,
        0.0812743883615744,
    ],
    dtype=float,
)


def _get_gl_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order <= 0:
        raise ValueError(f"Unsupported GL order: {order}. Use a positive integer.")
    if order == 5:
        return _GL5_NODES, _GL5_WEIGHTS
    if order == 9:
        return _GL9_NODES, _GL9_WEIGHTS
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(order)
    return nodes.astype(float), weights.astype(float)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _build_vessel_polydata(
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    scalars_start: np.ndarray,
    scalars_end: np.ndarray,
    *,
    resolution: int = PLOT_LINE_RESOLUTION,
) -> pv.PolyData:
    points = []
    scalars = []
    radii_samples = []
    lines = []
    offset = 0
    for idx, (start, end) in enumerate(zip(starts, ends)):
        pts = np.linspace(start, end, num=max(resolution, 2), endpoint=True)
        vals = np.linspace(scalars_start[idx], scalars_end[idx], num=pts.shape[0])
        points.append(pts)
        scalars.append(vals)
        radii_samples.append(np.full(pts.shape[0], radii[idx]))
        lines.append(np.concatenate(([pts.shape[0]], np.arange(offset, offset + pts.shape[0], dtype=int))))
        offset += pts.shape[0]
    if not points:
        return pv.PolyData()
    mesh = pv.PolyData(np.vstack(points), lines=np.concatenate(lines))
    mesh["scalar"] = np.concatenate(scalars)
    mesh["radius"] = np.concatenate(radii_samples)
    return mesh


def _plot_flow(
    domain: Domain,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    flows: np.ndarray,
) -> None:
    if starts.size == 0:
        return
    q_abs = np.abs(flows)
    q_abs_ul_min = q_abs * 60000.0
    vessel_lines = _build_vessel_polydata(
        starts,
        ends,
        radii,
        q_abs_ul_min,
        q_abs_ul_min,
        resolution=PLOT_LINE_RESOLUTION,
    )
    if vessel_lines.n_points == 0:
        return
    vessel_mesh = vessel_lines.tube(
        radius=0.0,
        scalars="radius",
        absolute=True,
        n_sides=PLOT_TUBE_SIDES,
        capping=True,
    )
    clim = [float(np.nanmin(q_abs_ul_min)), float(np.nanmax(q_abs_ul_min))]
    plotter = pv.Plotter(window_size=PLOT_WINDOW_SIZE)
    plotter.add_mesh(
        vessel_mesh,
        scalars="scalar",
        cmap=_plot_cmap(),
        clim=clim,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Flow (uL/min)"},
    )
    _add_domain_outline(plotter, domain)
    plotter.camera.Zoom(PLOT_ZOOM)
    _show_plotter(plotter)


def _plot_concentration(
    domain: Domain,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cin: np.ndarray,
    cout: np.ndarray,
    tissue_points: np.ndarray,
    tissue_values: np.ndarray,
    *,
    show_points: bool,
    inlet_concentration: float,
) -> None:
    if starts.size == 0:
        return
    vessel_lines = _build_vessel_polydata(
        starts,
        ends,
        radii,
        cin,
        cout,
        resolution=PLOT_LINE_RESOLUTION,
    )
    if vessel_lines.n_points == 0:
        return
    vessel_mesh = vessel_lines.tube(
        radius=0.0,
        scalars="radius",
        absolute=True,
        n_sides=PLOT_TUBE_SIDES,
        capping=True,
    )
    clim = [min(float(np.nanmin(cout)), EXTRAVASCULAR_CONCENTRATION), float(inlet_concentration)]
    plotter = pv.Plotter(window_size=PLOT_WINDOW_SIZE)
    plotter.add_mesh(vessel_mesh, scalars="scalar", cmap=_plot_cmap(), clim=clim, show_scalar_bar=True)
    if show_points and tissue_points.size:
        plotter.add_mesh(
            pv.PolyData(tissue_points),
            scalars=tissue_values,
            cmap=_plot_cmap(),
            point_size=8,
            render_points_as_spheres=True,
            clim=clim,
            show_scalar_bar=False,
        )
    _add_domain_outline(plotter, domain)
    plotter.camera.Zoom(PLOT_ZOOM)
    _show_plotter(plotter)


def _plot_viability_points(
    domain: Domain,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cin: np.ndarray,
    cout: np.ndarray,
    tissue_points: np.ndarray,
    tissue_values: np.ndarray,
    *,
    inlet_concentration: float,
) -> None:
    if tissue_points.size == 0 or tissue_values.size == 0:
        return
    thresh = 0.01 * float(inlet_concentration)
    vessel_mesh = None
    if starts.size != 0:
        alive_start = (cin >= thresh).astype(float)
        alive_end = (cout >= thresh).astype(float)
        vessel_lines = _build_vessel_polydata(
            starts,
            ends,
            radii,
            alive_start,
            alive_end,
            resolution=PLOT_LINE_RESOLUTION,
        )
        if vessel_lines.n_points:
            vessel_mesh = vessel_lines.tube(
                radius=0.0,
                scalars="radius",
                absolute=True,
                n_sides=PLOT_TUBE_SIDES,
                capping=True,
            )
    alive = tissue_values >= thresh
    frac_alive = float(np.count_nonzero(alive) / max(tissue_values.size, 1))
    print(f"Viable (>= 1% inlet): {frac_alive * 100.0:.2f}%")

    plotter = pv.Plotter(window_size=PLOT_WINDOW_SIZE)
    if vessel_mesh is not None and vessel_mesh.n_points:
        plotter.add_mesh(
            vessel_mesh,
            scalars="scalar",
            cmap=["#FF0000", "#048700"],
            clim=[0.0, 1.0],
            show_scalar_bar=False,
        )
    plotter.add_mesh(
        pv.PolyData(tissue_points),
        scalars=alive.astype(float),
        cmap=["#FF0000", "#048700"],
        clim=[0.0, 1.0],
        point_size=PLOT_POINT_SIZE,
        render_points_as_spheres=True,
        show_scalar_bar=False,
        opacity=POINT_OPACITY,
    )
    _add_domain_outline(plotter, domain)
    plotter.camera.Zoom(PLOT_ZOOM)
    plotter.add_text("Viability (red=dead, green=alive)", position="upper_left", font_size=12)
    _show_plotter(plotter)


def _plot_mass_balance(history: Dict[str, List[float]]) -> None:
    if not history.get("iter"):
        return
    import matplotlib.pyplot as plt

    it = np.array(history["iter"], dtype=float)
    m_in = np.array(history["M_in"], dtype=float)
    m_out = np.array(history["M_out"], dtype=float)
    m_drop = np.array(history["M_drop"], dtype=float)
    mb_res = np.array(history["MB_resid"], dtype=float)

    plt.figure()
    plt.plot(it, m_in - m_out, label="M_in - M_out")
    plt.plot(it, m_drop, label="sum(Q*(Cin-Cout))", linestyle="--")
    plt.plot(it, mb_res, label="residual", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("mol/s")
    plt.title("Mass balance convergence")
    plt.legend()
    plt.grid(True)
    plt.show()


def _dnc_histogram_fractions_um(
    dnc_values_um: np.ndarray,
    *,
    bin_max_um: float = 5000.0,
    bin_width_um: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    dnc_values_um = np.asarray(dnc_values_um, dtype=float).reshape(-1)
    bin_max_um = float(bin_max_um)
    bin_width_um = float(bin_width_um)
    if not np.isfinite(bin_max_um) or bin_max_um <= 0.0:
        raise ValueError("bin_max_um must be positive and finite")
    if not np.isfinite(bin_width_um) or bin_width_um <= 0.0:
        raise ValueError("bin_width_um must be positive and finite")

    edges_regular = np.arange(0.0, bin_max_um + bin_width_um, bin_width_um, dtype=float)
    if edges_regular.size < 2 or not np.isclose(edges_regular[-1], bin_max_um):
        edges_regular = np.linspace(0.0, bin_max_um, int(round(bin_max_um / bin_width_um)) + 1, dtype=float)

    finite = dnc_values_um[np.isfinite(dnc_values_um)]
    if finite.size == 0:
        edges_plot = np.concatenate([edges_regular, np.array([bin_max_um + bin_width_um], dtype=float)])
        return edges_plot, np.zeros(edges_regular.size - 1 + 1, dtype=float)

    regular = finite[finite <= bin_max_um]
    overflow = finite[finite > bin_max_um]
    counts_regular, _ = np.histogram(regular, bins=edges_regular) if regular.size else (np.zeros(edges_regular.size - 1, dtype=int), edges_regular)
    overflow_count = int(overflow.size)
    total = int(finite.size)
    fractions = np.concatenate([counts_regular.astype(float), np.array([float(overflow_count)], dtype=float)]) / float(total)

    edges_plot = np.concatenate([edges_regular, np.array([bin_max_um + bin_width_um], dtype=float)])
    return edges_plot, fractions


def _fit_truncnorm_mu_sigma_lower0(values_um: np.ndarray) -> tuple[float, float]:
    values_um = np.asarray(values_um, dtype=float).reshape(-1)
    values_um = values_um[np.isfinite(values_um)]
    if values_um.size < 2:
        return float("nan"), float("nan")
    values_um = values_um[values_um >= 0.0]
    if values_um.size < 2:
        return float("nan"), float("nan")

    x = values_um
    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)

    def _log_survival(z: float) -> float:
        # log(1 - Phi(z)) with some basic stabilization
        p = 0.5 * (1.0 + math.erf(z / sqrt2))
        p = min(max(p, 0.0), 1.0)
        s = 1.0 - p
        if s <= 0.0:
            return -1e300
        return math.log(s)

    def _nll(mu: float, log_sigma: float) -> float:
        sigma = math.exp(log_sigma)
        if not (np.isfinite(sigma) and sigma > 0.0):
            return float("inf")
        z = (x - mu) / sigma
        # log pdf of truncated normal on [0, inf):
        # log(phi(z)) - log(sigma) - log(1 - Phi((0-mu)/sigma))
        log_phi = -0.5 * z * z - math.log(sqrt2pi)
        logZ = _log_survival((0.0 - mu) / sigma)
        if not np.isfinite(logZ):
            return float("inf")
        ll = np.sum(log_phi) - x.size * math.log(sigma) - x.size * logZ
        return float(-ll)

    mu0 = float(np.mean(x))
    sigma0 = float(np.std(x, ddof=0))
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        sigma0 = 1.0
    log_sigma0 = math.log(sigma0)

    best_mu = mu0
    best_log_sigma = log_sigma0
    best = _nll(best_mu, best_log_sigma)

    mu_span = max(6.0 * sigma0, 200.0)
    log_span = math.log(4.0)
    for _ in range(3):
        mu_grid = np.linspace(best_mu - mu_span / 2.0, best_mu + mu_span / 2.0, 61, dtype=float)
        log_grid = np.linspace(best_log_sigma - log_span / 2.0, best_log_sigma + log_span / 2.0, 61, dtype=float)
        for mu in mu_grid:
            for ls in log_grid:
                val = _nll(float(mu), float(ls))
                if val < best:
                    best = val
                    best_mu = float(mu)
                    best_log_sigma = float(ls)
        mu_span *= 0.35
        log_span *= 0.35

    return float(best_mu), float(math.exp(best_log_sigma))


def _init_violin_points_csv(path: Path, *, target_counts: Sequence[int], n_rows: int) -> None:
    headers = [str(int(t)) for t in target_counts]
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        empty_row = ["" for _ in headers]
        for _ in range(int(max(n_rows, 0))):
            writer.writerow(empty_row)


def _update_violin_points_csv(
    path: Path,
    *,
    target_counts: Sequence[int],
    col_idx: int,
    values_um: np.ndarray,
) -> None:
    target_counts = list(target_counts)
    if col_idx < 0 or col_idx >= len(target_counts):
        raise IndexError("col_idx out of range for target_counts")

    values = np.asarray(values_um, dtype=float).reshape(-1)

    with path.open("r", newline="") as csvfile:
        rows = list(csv.reader(csvfile))
    if not rows:
        raise ValueError("violin CSV is empty")

    expected_header = [str(int(t)) for t in target_counts]
    if rows[0] != expected_header:
        raise ValueError(f"violin CSV header mismatch: expected {expected_header}, got {rows[0]}")

    n_rows = len(rows) - 1
    if values.size != n_rows:
        raise ValueError(f"values_um length ({values.size}) does not match CSV rows ({n_rows})")

    for i in range(n_rows):
        v = values[i]
        rows[i + 1][col_idx] = "" if not np.isfinite(v) else f"{float(v):.8g}"

    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def _plot_dnc_gaussians(
    *,
    mu_sigma: Sequence[tuple[float, float]],
    target_terminals: Sequence[int],
    title: str,
) -> None:
    if not mu_sigma or not target_terminals:
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    targets = np.asarray(list(target_terminals), dtype=float)
    if targets.size == 0:
        return

    cmapcolorscheme = "jet"
    try:
        cmap = mpl.colormaps.get_cmap(cmapcolorscheme)
    except Exception:  # pragma: no cover
        cmap = mpl.cm.get_cmap(cmapcolorscheme)
    vmin = float(np.nanmin(targets))
    vmax = float(np.nanmax(targets))
    if GAUSSIAN_COLORMAP_LOG10 and np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0.0 and vmax > vmin:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    x = np.linspace(0.0, float(GAUSSIAN_PLOT_XMAX_UM), int(max(GAUSSIAN_PLOT_NPTS, 2)), dtype=float)
    x_log_min = float(GAUSSIAN_LOG_XMIN_UM)
    if not np.isfinite(x_log_min) or x_log_min <= 0.0:
        x_log_min = 0.1
    x_pos = np.linspace(x_log_min, float(GAUSSIAN_PLOT_XMAX_UM), int(max(GAUSSIAN_PLOT_NPTS, 2)), dtype=float)
    y_logx = np.log10(x_pos)

    fig, (ax_lin, ax_log) = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True)
    for t, (mu, sigma) in zip(targets, mu_sigma):
        mu = float(mu)
        sigma = float(sigma)
        if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0):
            continue
        color = cmap(norm(float(t)))
        y = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax_lin.plot(
            x,
            y,
            color=color,
            alpha=HISTOGRAM_ALPHA,
            linewidth=2.0,
        )
        y_pos = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x_pos - mu) / sigma) ** 2)
        # Plot the same Gaussian values, just against log10(x) (no Jacobian / renormalization).
        y_log = y_pos
        ax_log.plot(
            y_logx,
            y_log,
            color=color,
            alpha=HISTOGRAM_ALPHA,
            linewidth=2.0,
        )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if isinstance(norm, mpl.colors.LogNorm):
        cbar = fig.colorbar(sm, ax=[ax_lin, ax_log], label="N_vessels (target_terminals, log scale)")
        try:
            import matplotlib.ticker as mticker

            cbar.formatter = mticker.LogFormatterMathtext(base=10)
            cbar.update_ticks()
        except Exception:  # pragma: no cover
            pass
    else:
        fig.colorbar(sm, ax=[ax_lin, ax_log], label="N_vessels (target_terminals)")

    ax_lin.set_xlim(0.0, float(GAUSSIAN_PLOT_XMAX_UM))
    ax_lin.set_ylim(bottom=0.0)
    ax_lin.set_xlabel("DNCW (Âµm)")
    ax_lin.set_ylabel("Probability density (1/Âµm)")
    ax_lin.grid(True, alpha=0.3)

    ax_log.set_xlim(float(np.log10(x_log_min)), float(np.log10(GAUSSIAN_PLOT_XMAX_UM)))
    ax_log.set_ylim(bottom=0.0)
    ax_log.set_xlabel("log10(DNCW)")
    ax_log.set_ylabel("Probability density (same as linear; 1/Âµm)")
    ax_log.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.show()


def plot_checker_tree(
    tree: Tree,
    sample_points: np.ndarray,
    *,
    inlet_flow_cm3_s: Optional[float] = None,
    fluid: str | None = None,
) -> None:
    inlet_concentration = get_concentration_inlet(fluid)
    q_inlet = inlet_flow_cm3_s
    if q_inlet is None or not np.isfinite(q_inlet):
        q_inlet = float(QIN_TARGET) * 1e-3 / 60.0
    starts, ends, radii, lengths, flows, cin, cout, _ = solve_tree_greens(
        tree,
        q_inlet,
        fluid=fluid or ACTIVE_FLUID,
        inlet_concentration=inlet_concentration,
    )
    mask, tissue_conc = compute_tissue_samples_greens(
        sample_points,
        starts,
        ends,
        radii,
        cin,
        flows,
        diffusivity=SOLUTE_DIFFUSIVITY,
        vmax=VMAX_MM,
        km=K_M_MM,
        window_factor=WINDOW_FACTOR,
        inlet_concentration=inlet_concentration,
    )
    tissue_pts = sample_points[mask]
    tissue_vals = tissue_conc[mask]
    domain = getattr(tree, "domain", None)
    _plot_concentration(
        domain,
        starts,
        ends,
        radii,
        cin,
        cout,
        tissue_pts,
        tissue_vals,
        show_points=True,
        inlet_concentration=inlet_concentration,
    )


def _load_validation_points(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
    if data.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    names = {name.strip().lower(): name for name in data.dtype.names}
    def _pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            key = cand.lower()
            if key in names:
                return names[key]
        return None

    x_key = _pick("x", "x_m")
    y_key = _pick("y", "y_m")
    z_key = _pick("z", "z_m")
    a_key = _pick("a")
    if not (x_key and y_key and z_key and a_key):
        raise ValueError("CSV must include columns for x,y,z and A (or x_m,y_m,z_m).")

    pts = np.column_stack([data[x_key], data[y_key], data[z_key]])
    vals = np.asarray(data[a_key], dtype=float)
    return pts, vals


def _plot_validation_scatter(x_ref: np.ndarray, y_pred: np.ndarray) -> None:
    if x_ref.size == 0 or y_pred.size == 0:
        return
    import matplotlib.pyplot as plt

    fig, (ax_scatter, ax_bar) = plt.subplots(
        ncols=2,
        figsize=(9, 4),
        gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True,
    )
    ax_scatter.scatter(x_ref, y_pred, s=8, alpha=0.7)
    mn = float(min(np.nanmin(x_ref), np.nanmin(y_pred)))
    mx = float(max(np.nanmax(x_ref), np.nanmax(y_pred)))
    ax_scatter.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
    ax_scatter.set_xlabel("Validation A")
    ax_scatter.set_ylabel("Greens predicted")
    ax_scatter.set_title("Pointwise comparison")
    ax_scatter.grid(True)

    thresholds = np.array([0.011, 0.002211], dtype=float)
    match_fracs = []
    false_fracs = []
    for thr in thresholds:
        valid_mask = x_ref < thr
        pred_mask = y_pred < thr
        valid_count = int(np.count_nonzero(valid_mask))
        pred_count = int(np.count_nonzero(pred_mask))
        non_valid_count = int(np.count_nonzero(~valid_mask))
        if valid_count == 0:
            match_frac = 0.0
        else:
            match_frac = float(np.count_nonzero(pred_mask & valid_mask) / valid_count)
        if non_valid_count == 0:
            false_frac = 0.0
        else:
            false_frac = float(np.count_nonzero(pred_mask & (~valid_mask)) / non_valid_count)
        match_fracs.append(match_frac)
        false_fracs.append(false_frac)

    fit_lines = []
    for thr in thresholds:
        mask = y_pred >= thr
        slope, intercept, r2 = _linear_fit_slope_r2(x_ref[mask], y_pred[mask])
        n = int(np.count_nonzero(mask))
        print(f"Fit (Greens predicted >= {thr}): slope={slope:.6f}, R2={r2:.6f}, n={n}")
        fit_lines.append((thr, slope, intercept))

    fit_colors = ["#1f77b4", "#9467bd"]
    for (thr, slope, intercept), color in zip(fit_lines, fit_colors):
        if not np.isfinite(slope) or not np.isfinite(intercept):
            continue
        x_line = np.array([mn, mx], dtype=float)
        y_line = slope * x_line + intercept
        ax_scatter.plot(x_line, y_line, color=color, linewidth=1.5, label=f"Fit pred >= {thr}")

    x_pos = np.arange(len(thresholds))
    width = 0.35
    bars_match = ax_bar.bar(x_pos - width / 2.0, match_fracs, width=width, color="#2ca02c")
    bars_false = ax_bar.bar(x_pos + width / 2.0, false_fracs, width=width, color="#d62728")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_ylabel("fraction")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(["<0.011", "<0.002211"], rotation=90)
    ax_bar.set_title("Threshold %")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_bar.legend(["match", "false +"], loc="upper right", fontsize=8)
    for bars in (bars_match, bars_false):
        for bar in bars:
            frac = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(1.0, frac + 0.04),
                f"{frac * 100.0:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax_scatter.legend(loc="upper left", fontsize=8)
    plt.show()


def _linear_fit_slope_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return float(slope), float(intercept), float("nan")
    r2 = 1.0 - (ss_res / ss_tot)
    return float(slope), float(intercept), float(r2)


def _constant_value(rows: list[Dict[str, float]], key: str):
    values = {row[key] for row in rows}
    if not values:
        return float("nan")
    if len(values) > 1:
        raise ValueError(f"Inconsistent values for '{key}': {values}")
    return values.pop()


def _mean_std(rows: list[Dict[str, float]], key: str) -> tuple[float, float]:
    data = np.asarray([row.get(key, float("nan")) for row in rows], dtype=float)
    return float(np.nanmean(data)), float(np.nanstd(data, ddof=1))


def _pooled_mean_std_from_summaries(
    rows: list[Dict[str, float]],
    *,
    mean_key: str,
    std_key: str,
    count_key: str,
) -> tuple[float, float]:
    summaries = []
    for row in rows:
        count = float(row.get(count_key, float("nan")))
        mean = float(row.get(mean_key, float("nan")))
        std = float(row.get(std_key, float("nan")))
        if np.isfinite(count) and count > 0.0 and np.isfinite(mean):
            summaries.append((count, mean, std if np.isfinite(std) else 0.0))
    if not summaries:
        return float("nan"), float("nan")
    total = float(sum(count for count, _, _ in summaries))
    if total <= 0.0:
        return float("nan"), float("nan")
    pooled_mean = float(sum(count * mean for count, mean, _ in summaries) / total)
    if total <= 1.0:
        return pooled_mean, 0.0
    ss = 0.0
    for count, mean, std in summaries:
        ss += max(count - 1.0, 0.0) * float(std) ** 2
        ss += count * (float(mean) - pooled_mean) ** 2
    return pooled_mean, float(math.sqrt(max(ss, 0.0) / max(total - 1.0, 1.0)))


def aggregate_trials(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("No trial rows provided for aggregation.")
    result: Dict[str, float] = {}
    result["number_of_trees"] = len(rows)
    result["target_terminals"] = _constant_value(rows, "target_terminals")
    result["dlp_angle"] = _constant_value(rows, "dlp_angle")
    result["cube_side_length"] = _constant_value(rows, "cube_side_length")
    result["concentration_solver"] = _constant_value(rows, "concentration_solver")
    result["cext_accel_mode"] = _constant_value(rows, "cext_accel_mode")
    result["pressure_in_root"] = _constant_value(rows, "pressure_in_root")
    result["pressure_out_terminals"] = _constant_value(rows, "pressure_out_terminals")
    result["concentration_inlet"] = _constant_value(rows, "concentration_inlet")
    result["extravascular_concentration"] = EXTRAVASCULAR_CONCENTRATION
    result["solute_diffusivity"] = SOLUTE_DIFFUSIVITY
    # result["tissue_decay_length"] = TISSUE_DECAY_LENGTH
    result["qin_target_uL_per_min"] = _mean_std(rows, "inlet_flow_ul_per_min")[0]
    result["distance_sample_count"] = _constant_value(rows, "distance_sample_count")
    for metric in AGGREGATED_METRICS:
        mean, std = _mean_std(rows, metric)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
    cext_mean, cext_std = _pooled_mean_std_from_summaries(
        rows,
        mean_key="cext_concentration_mean",
        std_key="cext_concentration_std",
        count_key="cext_concentration_count",
    )
    result["cext_concentration_mean"] = cext_mean
    result["cext_concentration_std"] = cext_std
    return result


def _normalize_angles(values: Sequence[float] | float) -> tuple[float, ...]:
    if isinstance(values, Number):
        return (float(values),)
    if isinstance(values, Sequence):
        return tuple(float(v) for v in values)
    if hasattr(values, "__iter__"):
        return tuple(float(v) for v in values)
    raise TypeError("DLP angle list must be a number or iterable of numbers.")


def _iter_theta_values() -> tuple[float, ...]:
    return _normalize_angles(DLP_ANGLE_VALUES)


def _normalize_lengths(values: Sequence[float] | float) -> tuple[float, ...]:
    return _normalize_angles(values)


def _parse_qin_target_values(value: object) -> tuple[float, ...]:
    """Parse --qin-target as either one number or a comma/list-style sequence."""
    if isinstance(value, Number):
        values = (float(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed: list[float] = []
        for item in value:
            parsed.extend(_parse_qin_target_values(item))
        values = tuple(parsed)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("--qin-target must contain at least one positive flowrate.")
        if (text[0], text[-1]) in (("(", ")"), ("[", "]")):
            text = text[1:-1].strip()
        parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
        if not parts:
            raise ValueError("--qin-target must contain at least one positive flowrate.")
        values = tuple(float(part) for part in parts)
    for qin in values:
        if not np.isfinite(qin) or qin <= 0.0:
            raise ValueError(f"--qin-target values must be positive and finite; got {qin!r}.")
    return values


def _parse_distance_sample_count_values(value: object) -> tuple[int, ...]:
    """Parse one or more positive DISTANCE_SAMPLE_COUNT values."""
    if isinstance(value, Number):
        values = (int(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed: list[int] = []
        for item in value:
            parsed.extend(_parse_distance_sample_count_values(item))
        values = tuple(parsed)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("--distance-sample-counts must contain at least one positive integer.")
        if (text[0], text[-1]) in (("(", ")"), ("[", "]")):
            text = text[1:-1].strip()
        parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
        if not parts:
            raise ValueError("--distance-sample-counts must contain at least one positive integer.")
        values = tuple(int(float(part)) for part in parts)
    deduped = tuple(dict.fromkeys(values))
    for count in deduped:
        if count <= 0:
            raise ValueError(f"--distance-sample-counts values must be positive; got {count!r}.")
    return deduped


def append_row(path: Path, row: Optional[Dict[str, float]], write_header: bool = False) -> None:
    with path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)


def append_nondimensional_row(
    path: Path,
    row: Optional[Dict[str, float | str]],
    write_header: bool = False,
) -> None:
    with path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=NONDIMENSIONAL_FIELDNAMES)
        if write_header:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)


def _safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-30:
        return float("nan")
    return float(num / den)


def _pi_phi_gamma_columns(aggregated: Dict[str, float | str]) -> Dict[str, float]:
    """Return the Pi2/Phi2/Gamma inputs and values appended to the main CSV."""
    fluid = str(aggregated.get("fluid", "")).lower()
    c0_plasma = float(aggregated.get("concentration_inlet", float("nan")))
    avg_radius_cm = float(aggregated.get("avg_radius_mean", float("nan")))
    total_segments = float(aggregated.get("total_segments_mean", float("nan")))
    if not np.isfinite(total_segments) or total_segments <= 0.0:
        target_terminals_raw = float(aggregated.get("target_terminals", float("nan")))
        target_terminals = int(round(target_terminals_raw)) if np.isfinite(target_terminals_raw) else -1
        total_segments = float((2 * target_terminals) + 1) if target_terminals >= 0 else float("nan")
    side_len_cm = float(aggregated.get("cube_side_length", float("nan")))
    q0_ul_min = float(aggregated.get("inlet_flow_ul_per_min_mean", float("nan")))
    if not np.isfinite(q0_ul_min):
        q0_ul_min = float(aggregated.get("qin_target_uL_per_min", float("nan")))

    q0_m3_s = q0_ul_min * 1e-9 / 60.0 if np.isfinite(q0_ul_min) else float("nan")
    side_len_m = side_len_cm * CM_TO_M if np.isfinite(side_len_cm) else float("nan")
    v_t = side_len_m**3 if np.isfinite(side_len_m) else float("nan")
    avg_radius_m = avg_radius_cm * CM_TO_M if np.isfinite(avg_radius_cm) else float("nan")

    c_hb_bound = 0.0
    if fluid == "blood":
        chb_max = float(HD_DISCHARGE * O2_CAP_PER_HCT)
        if np.isfinite(c0_plasma) and ALPHA_MMHG > 0.0:
            p0_mmhg = float(c0_plasma / ALPHA_MMHG)
            c_hb_bound = float(chb_max * severinghaus_saturation(p0_mmhg))
        else:
            c_hb_bound = float("nan")

    c0_eff = c0_plasma + c_hb_bound if np.isfinite(c0_plasma) and np.isfinite(c_hb_bound) else float("nan")
    cstar = (
        float(CONC_MAX_FOR_NORMALIZATION) * 0.01
        if np.isfinite(CONC_MAX_FOR_NORMALIZATION)
        else float("nan")
    )
    solute_diffusivity_cm2_s = float(aggregated.get("solute_diffusivity", SOLUTE_DIFFUSIVITY))
    diffusivity = solute_diffusivity_cm2_s * CM2_TO_M2 if np.isfinite(solute_diffusivity_cm2_s) else float("nan")
    vmax = float(VMAX_MM)
    km = float(K_M_MM)

    l_d = float("nan")
    if np.isfinite(v_t) and np.isfinite(total_segments) and np.isfinite(avg_radius_m) and total_segments > 0.0:
        l_d = float(np.cbrt(v_t / total_segments) - avg_radius_m)

    lambda_rd = float("nan")
    if np.isfinite(diffusivity) and np.isfinite(c0_plasma) and np.isfinite(km) and np.isfinite(vmax):
        if diffusivity >= 0.0 and (c0_plasma + km) > 0.0 and vmax > 0.0:
            lambda_rd = float(np.sqrt(diffusivity * (c0_plasma + km) / vmax))

    pi2 = _safe_div(l_d, lambda_rd)
    phi2 = _safe_div(q0_m3_s * (c0_eff - cstar), v_t * (vmax * cstar / (km + cstar)))
    gamma = _safe_div(c0_eff, km)

    return {
        "C0_plasma": c0_plasma,
        "C_Hb_bound": c_hb_bound,
        "C0_eff": c0_eff,
        "Q0_inlet_m3_s": q0_m3_s,
        "Vmax": vmax,
        "Km": km,
        "Cstar": cstar,
        "D_m2_s": diffusivity,
        "Vt_m3": v_t,
        "l_d_m": l_d,
        "lambda_reaction_diffusion_m": lambda_rd,
        "Pi2": pi2,
        "Phi2": phi2,
        "Gamma": gamma,
    }


def _nondimensional_numbers_row(
    aggregated: Dict[str, float | str],
    *,
    run_name: str,
) -> Dict[str, float | str]:
    fluid = str(aggregated.get("fluid", "")).lower()
    c0_plasma = float(aggregated.get("concentration_inlet", float("nan")))
    avg_radius_cm = float(aggregated.get("avg_radius_mean", float("nan")))
    avg_length_cm = float(aggregated.get("avg_length_mean", float("nan")))
    target_terminals_raw = float(aggregated.get("target_terminals", float("nan")))
    target_terminals = int(round(target_terminals_raw)) if np.isfinite(target_terminals_raw) else -1
    n_segments = float((2 * target_terminals) + 1) if target_terminals >= 0 else float("nan")
    side_len = float(aggregated.get("cube_side_length", float("nan")))
    q0_ul_min = float(aggregated.get("inlet_flow_ul_per_min_mean", float("nan")))

    q0_m3_s = q0_ul_min * 1e-9 / 60.0 if np.isfinite(q0_ul_min) else float("nan")
    side_len_m = side_len * CM_TO_M if np.isfinite(side_len) else float("nan")
    v_t = side_len_m**3 if np.isfinite(side_len_m) else float("nan")
    avg_radius_m = avg_radius_cm * CM_TO_M if np.isfinite(avg_radius_cm) else float("nan")
    avg_length_m = avg_length_cm * CM_TO_M if np.isfinite(avg_length_cm) else float("nan")

    hct_est = float("nan")
    sat_est = float("nan")
    p0_mmhg = float("nan")
    chb_max = 0.0
    c_hb_bound = 0.0
    if fluid == "blood":
        if np.isfinite(avg_radius_cm) and avg_radius_cm > 0.0:
            hct_est = float(tube_hematocrit(avg_radius_cm, hd=HD_DISCHARGE))
        else:
            hct_est = float(HD_DISCHARGE)
        # O2_CAP_PER_HCT is already mol / m^3 per unit hematocrit, so no
        # mL/dL-to-mol/m^3 conversion belongs in the CSV/nondimensional output.
        chb_max = float(HD_DISCHARGE * O2_CAP_PER_HCT)
        if np.isfinite(c0_plasma) and ALPHA_MMHG > 0.0:
            p0_mmhg = float(c0_plasma / ALPHA_MMHG)
            sat_est = float(severinghaus_saturation(p0_mmhg))
            c_hb_bound = float(chb_max * sat_est)
        else:
            c_hb_bound = float("nan")

    c0_eff = c0_plasma + c_hb_bound if np.isfinite(c0_plasma) and np.isfinite(c_hb_bound) else float("nan")
    cstar = (
        float(CONC_MAX_FOR_NORMALIZATION) * 0.01
        if np.isfinite(CONC_MAX_FOR_NORMALIZATION)
        else float("nan")
    )
    diffusivity = float(SOLUTE_DIFFUSIVITY) * CM2_TO_M2
    vmax = float(VMAX_MM)
    km = float(K_M_MM)

    lambda_char = float("nan")
    if np.isfinite(diffusivity) and np.isfinite(vmax) and np.isfinite(km) and np.isfinite(cstar):
        if vmax > 0.0 and (km + cstar) > 0.0 and diffusivity >= 0.0:
            lambda_char = float(np.sqrt(diffusivity * (km + cstar) / vmax))

    l_d = float("nan")
    if np.isfinite(v_t) and np.isfinite(n_segments) and np.isfinite(avg_radius_m) and n_segments > 0.0:
        l_d = float(np.cbrt(v_t / n_segments) - avg_radius_m)

    pi_val = _safe_div(l_d, lambda_char)
    mm_term = _safe_div(cstar, km + cstar)
    phi_val = _safe_div(q0_m3_s * c0_eff, v_t * vmax * mm_term)
    da_val = _safe_div(vmax * (l_d**2), diffusivity * cstar)

    return {
        "run_name": run_name,
        "number_of_trees": float(aggregated.get("number_of_trees", float("nan"))),
        "target_terminals": float(target_terminals),
        "total_segments": n_segments,
        "dlp_angle": float(aggregated.get("dlp_angle", float("nan"))),
        "cube_side_length_m": side_len_m,
        "fluid": fluid,
        "C0_plasma": c0_plasma,
        "C_Hb_bound": c_hb_bound,
        "C0_eff": c0_eff,
        "Q0_inlet_m3_s": q0_m3_s,
        "Vmax": vmax,
        "Km": km,
        "Cstar": cstar,
        "Vt_m3": v_t,
        "l_d_m": l_d,
        "D_m2_s": diffusivity,
        "lambda_m": lambda_char,
        "avg_radius_m": avg_radius_m,
        "avg_length_m": avg_length_m,
        "FracAbove5pct": float(aggregated.get("FracAbove5pct_mean", float("nan"))),
        "FracAbove1pct": float(aggregated.get("FracAbove1pct_mean", float("nan"))),
        "hematocrit_est": hct_est,
        "severinghaus_saturation": sat_est,
        "Pi": pi_val,
        "Phi": phi_val,
        "Da": da_val,
    }


_DEFAULT_TREE_PARAMS: dict | None = None


def _tree_cache_dir() -> Path:
    root = Path(TREE_CACHE_DIRNAME)
    if root.is_absolute():
        return root
    return Path(__file__).resolve().parent / root


def _tree_cache_index_path() -> Path:
    return _tree_cache_dir() / TREE_CACHE_INDEX_NAME


def _as_float_list(values: Sequence[Number] | np.ndarray | None) -> list[float] | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(v) for v in arr]


def _default_tree_params() -> dict:
    global _DEFAULT_TREE_PARAMS
    if _DEFAULT_TREE_PARAMS is None:
        _require_tree_class()
        tree = Tree()
        parms = tree.parameters
        _DEFAULT_TREE_PARAMS = {
            "murray_exponent": float(parms.murray_exponent),
            "radius_exponent": float(parms.radius_exponent),
            "length_exponent": float(parms.length_exponent),
            "max_nonconvex_count": int(parms.max_nonconvex_count),
            # "dlp_early_slab_frac": None if parms.dlp_early_slab_frac is None else float(parms.dlp_early_slab_frac),
            # "dlp_early_quota_frac": None if parms.dlp_early_quota_frac is None else float(parms.dlp_early_quota_frac),
        }
    return dict(_DEFAULT_TREE_PARAMS)


def _fluid_properties(fluid: str) -> dict[str, float]:
    mode = (fluid or ACTIVE_FLUID).lower()
    if mode in {"water", "cell media", "media"}:
        return {
            "fluid_density": 0.99336,
            "kinematic_viscosity": 0.6959 / 100,
        }
    if mode == "blood":
        mu_plasma_cgs = 0.012
        return {
            "fluid_density": 1.06,
            "kinematic_viscosity": mu_plasma_cgs / 1.06,
        }
    raise ValueError(f"Unknown fluid mode for cache config: {mode}")


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash_config(config: dict) -> str:
    payload = _json_dumps(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _build_tree_cache_config(
    *,
    target_raw: int,
    target_terminals: int,
    side_length: float,
    domain_seed: int | None,
    q_inlet_cm3_s: float,
    terminal_flow_cm3_s: float,
    dlp_enable: bool,
    min_theta: float,
    weighted_sampling: bool,
    ignore_collisions: bool,
    allow_inside_vessels: bool,
    n_closest_vessels: int,
    n_points: int,
) -> tuple[dict, dict, str, str]:
    params = _default_tree_params()
    fluid_props = _fluid_properties(BUILD_FLUID)
    terminal_pressure = (
        float(ROOT_PRESSURE) - (abs(ROOT_PRESSURE - TERMINAL_PRESSURE)) * (side_length**3)
        if SCALE_dP_BY_VOLUME
        else float(TERMINAL_PRESSURE)
    )
    base_config = {
        "cube_side_length": float(side_length),
        "domain_shape": "cube",
        "domain_random_seed": int(domain_seed) if domain_seed is not None else None,
        "root_location": _as_float_list(ROOT_LOCATION),
        "root_direction": _as_float_list(ROOT_DIR),
        "root_location_scaled": _as_float_list(ROOT_LOCATION * side_length),
        "root_pressure": float(ROOT_PRESSURE),
        "terminal_pressure": float(terminal_pressure),
        "qin_target_ul_min": float(QIN_TARGET),
        "inlet_flow_cm3_s": float(q_inlet_cm3_s),
        "terminal_flow_cm3_s": float(terminal_flow_cm3_s),
        "scale_nterms_by_volume": bool(SCALE_NTERMS_BY_VOLUME),
        "scale_q_by_volume": bool(SCALE_Q_BY_VOLUME),
        "scale_dp_by_volume": bool(SCALE_dP_BY_VOLUME),
        "build_fluid": str(BUILD_FLUID),
        "build_fluid_density": float(fluid_props["fluid_density"]),
        "build_kinematic_viscosity": float(fluid_props["kinematic_viscosity"]),
        "tree_data_dtype": TREE_DATA_DTYPE_STR,
        "tree_index_dtype": TREE_INDEX_DTYPE_STR,
        # "dlp_enable": bool(dlp_enable),
        # "dlp_min_angle_deg": float(min_theta),
        # "dlp_min_adv": 0.0001,
        # "dlp_build_dir": _as_float_list((-1.0, 1.0, 1.0)),
        # "dlp_parent_below": True,
        # "dlp_build_cap_frac": None,
        # "dlp_early_slab_frac": params["dlp_early_slab_frac"],
        # "dlp_early_quota_frac": params["dlp_early_quota_frac"],
        "weighted_sampling": bool(weighted_sampling),
        "ignore_collisions": bool(ignore_collisions),
        "allow_inside_vessels": bool(allow_inside_vessels),
        "n_closest_vessels": int(n_closest_vessels),
        "n_points": int(n_points),
        "murray_exponent": params["murray_exponent"],
        "radius_exponent": params["radius_exponent"],
        "length_exponent": params["length_exponent"],
        "max_nonconvex_count": params["max_nonconvex_count"],
    }
    full_config = dict(base_config)
    full_config.update(
        {
            "target_raw": int(target_raw),
            "target_terminals": int(target_terminals),
        }
    )
    hash_config = dict(base_config)
    # Tree geometry caches are reusable across inlet-flow targets; QIN is applied
    # during flow/concentration solves after the tree is loaded or grown.
    hash_config.pop("qin_target_ul_min", None)
    hash_config.pop("inlet_flow_cm3_s", None)
    hash_config.pop("terminal_flow_cm3_s", None)
    for key in ("n_closest_vessels", "n_points", "weighted_sampling", "allow_inside_vessels"):
        hash_config.pop(key, None)
    # Preserve compatibility with pre-dtype cache config IDs.
    hash_config.pop("tree_data_dtype", None)
    hash_config.pop("tree_index_dtype", None)
    config_id = _hash_config(hash_config)
    legacy_config = dict(base_config)
    legacy_config.pop("qin_target_ul_min", None)
    legacy_config.pop("inlet_flow_cm3_s", None)
    legacy_config.pop("terminal_flow_cm3_s", None)
    for key in ("n_closest_vessels", "n_points", "weighted_sampling", "allow_inside_vessels"):
        legacy_config.pop(key, None)
    legacy_config.pop("tree_data_dtype", None)
    legacy_config.pop("tree_index_dtype", None)
    legacy_config_id = _hash_config(legacy_config)
    return base_config, full_config, config_id, legacy_config_id


def _tree_cache_row(
    *,
    config_id: str,
    tree_path: Path,
    base_config: dict,
    full_config: dict,
    target_raw: int,
    target_terminals: int,
) -> dict:
    row = {
        "config_id": config_id,
        "target_terminals": int(target_terminals),
        "target_raw": int(target_raw),
        "tree_path": _tree_cache_relpath(tree_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_json": _json_dumps(full_config),
    }
    for field in TREE_CACHE_FIELDNAMES:
        if field in row:
            continue
        if field in base_config:
            value = base_config[field]
            if isinstance(value, (list, tuple, dict)):
                row[field] = _json_dumps(value)
            elif value is None:
                row[field] = ""
            else:
                row[field] = value
        else:
            row[field] = ""
    return row


def _tree_cache_relpath(path: Path) -> str:
    base = Path(__file__).resolve().parent
    try:
        return path.resolve().relative_to(base).as_posix()
    except Exception:
        return path.as_posix()


def _resolve_tree_cache_path(path_str: str) -> Path:
    normalized = path_str.replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _load_tree_cache_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    # Ensure accel uses only cube-domain caches
    filtered = []
    for row in rows:
        shape = (row.get("domain_shape") or "").strip().lower()
        if shape and shape != "cube":
            continue
        filtered.append(row)
    return filtered


def _append_tree_cache_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TREE_CACHE_FIELDNAMES)
    write_header = not path.exists()
    if not write_header:
        try:
            with path.open("r", newline="") as readfile:
                reader = csv.reader(readfile)
                header = next(reader, None)
                if header:
                    fieldnames = header
        except Exception:
            fieldnames = list(TREE_CACHE_FIELDNAMES)
    with path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def _normalize_tree_cache_lookup_config(config: dict, *, legacy: bool) -> dict:
    normalized = dict(config)
    normalized.pop("target_raw", None)
    normalized.pop("target_terminals", None)
    normalized.pop("qin_target_ul_min", None)
    normalized.pop("inlet_flow_cm3_s", None)
    normalized.pop("terminal_flow_cm3_s", None)
    for key in ("n_closest_vessels", "n_points", "weighted_sampling", "allow_inside_vessels"):
        normalized.pop(key, None)
    normalized.pop("tree_data_dtype", None)
    normalized.pop("tree_index_dtype", None)
    return normalized


def _tree_cache_row_matches_config_id(
    row: dict[str, str],
    config_id: str,
    *,
    legacy: bool,
) -> bool:
    if row.get("config_id") == config_id:
        return True
    cache_field = "_compat_legacy_config_id" if legacy else "_compat_current_config_id"
    cached_id = row.get(cache_field)
    if cached_id is None:
        config_json = row.get("config_json", "")
        if not config_json:
            row[cache_field] = ""
            return False
        try:
            full_config = json.loads(config_json)
            normalized = _normalize_tree_cache_lookup_config(full_config, legacy=legacy)
            cached_id = _hash_config(normalized)
        except Exception:
            cached_id = ""
        row[cache_field] = cached_id
    return bool(cached_id) and cached_id == config_id


def _find_cached_tree(
    rows: Iterable[dict[str, str]],
    config_id: str,
    target_terminals: int,
) -> tuple[dict[str, str], Path] | None:
    for row in rows:
        if not (
            _tree_cache_row_matches_config_id(row, config_id, legacy=False)
            or _tree_cache_row_matches_config_id(row, config_id, legacy=True)
        ):
            continue
        row_target = _parse_int(row.get("target_terminals"))
        if row_target != int(target_terminals):
            continue
        path_str = row.get("tree_path", "")
        if not path_str:
            continue
        path = _resolve_tree_cache_path(path_str)
        if path.exists():
            return row, path
    return None


def _find_cached_tree_lower(
    rows: Iterable[dict[str, str]],
    config_id: str,
    target_terminals: int,
) -> tuple[dict[str, str], int, Path] | None:
    best = None
    best_target = -1
    best_path = None
    for row in rows:
        if not (
            _tree_cache_row_matches_config_id(row, config_id, legacy=False)
            or _tree_cache_row_matches_config_id(row, config_id, legacy=True)
        ):
            continue
        row_target = _parse_int(row.get("target_terminals"))
        if row_target is None or row_target >= int(target_terminals):
            continue
        path_str = row.get("tree_path", "")
        if not path_str:
            continue
        path = _resolve_tree_cache_path(path_str)
        if not path.exists():
            continue
        if row_target > best_target:
            best = row
            best_target = row_target
            best_path = path
    if best is None or best_path is None:
        return None
    return best, best_target, best_path


def _prepare_loaded_tree(tree: Tree) -> None:
    domain = getattr(tree, "domain", None)
    # Cached trees may be loaded without the Domain/mesh to keep memory usage bounded for large targets.
    if domain is None:
        return
    if domain.mesh is None or getattr(domain, "volume", None) is None:
        domain.build()
    if getattr(domain, "boundary", None) is None:
        domain.get_boundary()
    if domain.mesh is not None:
        prob = domain.mesh.cell_data.get("probability")
        if prob is None:
            prob = domain.mesh.cell_data.get("Normalized_Volume")
        if prob is not None:
            tree.probability = np.array(prob, copy=True)
            domain.cumulative_probability = np.cumsum(tree.probability)


def _sync_loaded_tree_params(
    tree: Tree | None,
    *,
    side_length: float | None,
    terminal_flow_override: float | None,
) -> None:
    if tree is None:
        return
    try:
        params = tree.parameters
    except Exception:
        params = None
    if params is None:
        _require_tree_class()
        params = Tree().parameters
        try:
            tree.parameters = params
        except Exception:
            return

    params.root_pressure = ROOT_PRESSURE
    if SCALE_dP_BY_VOLUME and side_length is not None:
        params.terminal_pressure = ROOT_PRESSURE - (abs(ROOT_PRESSURE - TERMINAL_PRESSURE)) * (side_length**3)
        if params.terminal_pressure < 0:
            raise ValueError("Terminal pressure is negative after scaling; aborting run.")
    else:
        params.terminal_pressure = TERMINAL_PRESSURE

    defaults = _default_tree_params()
    params.murray_exponent = float(defaults["murray_exponent"])
    params.radius_exponent = float(defaults["radius_exponent"])
    params.length_exponent = float(defaults["length_exponent"])
    params.max_nonconvex_count = int(defaults["max_nonconvex_count"])

    set_tree_fluid(tree, BUILD_FLUID)
    _apply_equal_bifurcation(tree)
    if terminal_flow_override is not None:
        params.terminal_flow = terminal_flow_override


def _ensure_tree_domain(tree: Tree | None, domain: Domain | None) -> None:
    if tree is None or domain is None:
        return
    if getattr(tree, "domain", None) is not None:
        return
    try:
        if hasattr(tree, "set_domain"):
            tree.set_domain(domain)
        else:
            setattr(tree, "domain", domain)
    except Exception:
        return
    _prepare_loaded_tree(tree)


def _fast_tree_path(path: Path) -> Path:
    name = path.name
    if name.endswith(".fast.tree.npz"):
        return path
    if name.endswith(".tree.npz"):
        return path.with_name(name[:-len(".tree.npz")] + ".fast.tree.npz")
    return path.with_suffix(path.suffix + ".fast")


def _infer_loaded_terminal_count(data: np.ndarray) -> int:
    if data.ndim == 2 and data.shape[1] > 16:
        try:
            terminals = np.isnan(data[:, 15]) & np.isnan(data[:, 16])
            count = int(np.count_nonzero(terminals))
            if count > 0:
                return count
        except Exception:
            pass
    return max((int(getattr(data, "shape", (0,))[0]) + 1) // 2, 0)


def _load_tree_analysis_only(path: Path, *, data_dtype, index_dtype) -> Tree:
    """
    Load an exact cache hit for analysis without build-only spatial indices.

    Previous exact-hit loads used Tree.load(), which rebuilds the full build-time
    KDTree/HNSW/vessel_map/preallocation state even when no more vessels will be
    added. At multi-million terminal counts those duplicate structures can exceed
    memory before concentration/tissue analysis starts.
    """
    _require_tree_class()
    from svv.tree.data.data import TreeData

    resolved_data_dtype = np.dtype(data_dtype)
    resolved_index_dtype = np.dtype(index_dtype)
    with np.load(path, allow_pickle=False) as npz:
        data = npz["data"]
        if data.dtype != resolved_data_dtype:
            data = np.asarray(data, dtype=resolved_data_dtype)
        else:
            data = np.asarray(data)

    tree = Tree(
        preallocation_step=1,
        data_dtype=resolved_data_dtype,
        index_dtype=resolved_index_dtype,
    )
    tree_data = TreeData.from_array(data)
    tree.data = tree_data
    tree.preallocate = tree_data
    tree.segment_count = int(data.shape[0])
    tree.n_terminals = _infer_loaded_terminal_count(data)
    tree.preallocation_step = int(data.shape[0])
    tree.preallocate_midpoints = np.empty((0, 3), dtype=resolved_data_dtype)
    tree.midpoints = tree.preallocate_midpoints
    tree.connectivity = None
    tree.vessel_map = {}
    tree.kdtm = None
    tree.hnsw_tree = None
    tree.hnsw_tree_id = None
    tree.domain = None
    tree.probability = None
    tree._analysis_only_load = True
    return tree


def _load_tree_cached_fast(path: Path, *, data_dtype, index_dtype, analysis_only: bool = False) -> Tree:
    load_path = path
    fast_path = _fast_tree_path(path)
    if TREE_FAST_CACHE_ENABLE and fast_path.exists():
        try:
            if fast_path.stat().st_mtime_ns >= path.stat().st_mtime_ns:
                load_path = fast_path
                print(f"  Fast tree cache: using {fast_path.name}")
        except OSError:
            pass
    if analysis_only:
        print("  Tree load mode: analysis-only (skipping build-time spatial indices)")
        return _load_tree_analysis_only(
            load_path,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
        )
    return Tree.load(
        str(load_path),
        domain_path="",
        data_dtype=data_dtype,
        index_dtype=index_dtype,
    )


# ---------------------------------------------------------------------------
# Tree assembly
# ---------------------------------------------------------------------------
def assemble_tree_segments(
    tree: Tree,
    fluid: str,
    hd_per_segment: np.ndarray | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[int],
    List[int],
    np.ndarray,
    np.ndarray,
]:
    set_tree_fluid(tree, fluid)
    seg_count = int(getattr(tree, "segment_count", 0))
    # Use stored node ids (cols 18/19) when available to avoid coordinate de-duplication.
    data = np.asarray(tree.data[:seg_count])
    starts_arr = data[:, 0:3]
    ends_arr = data[:, 3:6]
    radii_arr = np.asarray(data[:, 21], dtype=float)
    lengths_arr = np.asarray(data[:, 20], dtype=float).copy()
    bad_len = ~np.isfinite(lengths_arr) | (lengths_arr <= 0.0)
    if np.any(bad_len):
        lengths_arr[bad_len] = np.linalg.norm(ends_arr[bad_len] - starts_arr[bad_len], axis=1)

    rho = float(tree.parameters.fluid_density)
    nu = float(tree.parameters.kinematic_viscosity)
    mu_base = rho * nu
    if hd_per_segment is not None:
        mu_arr = segment_viscosity_from_radius_hd(radii_arr, mu_base, fluid, hd_per_segment)
    else:
        mu_arr = segment_viscosity_from_radius(radii_arr, mu_base, fluid)

    prox_raw = np.asarray(data[:, 18], dtype=float).reshape(-1)
    dist_raw = np.asarray(data[:, 19], dtype=float).reshape(-1)
    have_node_ids = prox_raw.size == seg_count and dist_raw.size == seg_count and np.all(np.isfinite(prox_raw)) and np.all(np.isfinite(dist_raw))

    if have_node_ids:
        prox_ids = prox_raw.astype(np.int64, copy=False)
        dist_ids = dist_raw.astype(np.int64, copy=False)
        inlet_nodes = [int(prox_ids[0])] if prox_ids.size else []
        left_child = np.asarray(data[:, 15], dtype=float)
        right_child = np.asarray(data[:, 16], dtype=float)
        is_terminal = np.isnan(left_child) & np.isnan(right_child)
        outlet_nodes = [int(v) for v in dist_ids[is_terminal]]
        return starts_arr, ends_arr, radii_arr, lengths_arr, mu_arr, inlet_nodes, outlet_nodes, prox_ids, dist_ids

    # Fallback: derive node ids from rounded coordinates.
    geom = np.zeros((starts_arr.shape[0], 6), dtype=float)
    geom[:, 0:3] = starts_arr
    geom[:, 3:6] = ends_arr
    prox_ids, dist_ids, _ = _build_node_indices(geom)
    inlet_nodes = [int(prox_ids[0])] if prox_ids.size else []

    left_child = np.asarray(data[:, 15], dtype=float)
    right_child = np.asarray(data[:, 16], dtype=float)
    is_terminal = np.isnan(left_child) & np.isnan(right_child)
    outlet_nodes = [int(v) for v in dist_ids[is_terminal]]

    return starts_arr, ends_arr, radii_arr, lengths_arr, mu_arr, inlet_nodes, outlet_nodes, prox_ids, dist_ids


def recompute_tree_flows(
    tree: Tree,
    inlet_flow_cm3_s: float,
    *,
    fluid: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    (
        starts,
        ends,
        radii,
        lengths,
        mu_arr,
        inlet_nodes,
        outlet_nodes,
        prox_ids,
        dist_ids,
    ) = assemble_tree_segments(tree, fluid)
    if starts.size == 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty, inlet_nodes, outlet_nodes, starts, ends, radii, lengths, prox_ids, dist_ids

    radius_safe = np.maximum(radii, 1e-12)
    resistances = (8.0 * mu_arr * lengths) / (np.pi * radius_safe**4)
    pressures, flows, _, _, _ = solve_kirchhoff(
        prox_ids,
        dist_ids,
        resistances,
        inlet_nodes,
        inlet_flow_cm3_s,
        outlet_nodes,
    )
    if pressures.size and inlet_nodes:
        target = float(tree.parameters.root_pressure) * PA_TO_DYN_PER_CM2
        delta = target - float(pressures[inlet_nodes[0]])
        pressures = pressures + delta
    return flows, pressures, inlet_nodes, outlet_nodes, starts, ends, radii, lengths, prox_ids, dist_ids


def solve_tree_greens(
    tree: Tree,
    inlet_flow_cm3_s: float,
    *,
    fluid: str,
    inlet_concentration: float | None = None,
    diffusivity: float = SOLUTE_DIFFUSIVITY,
    vmax: float = VMAX_MM,
    km: float = K_M_MM,
    omega: float = OMEGA,
    concentration_solver: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet(fluid)
    (
        flows,
        pressures,
        inlet_nodes,
        outlet_nodes,
        starts,
        ends,
        radii,
        lengths,
        prox_ids,
        dist_ids,
    ) = recompute_tree_flows(tree, inlet_flow_cm3_s, fluid=fluid)

    cin, cout = _solve_channel_concentrations(
        tree,
        flows,
        inlet_nodes,
        outlet_nodes,
        starts,
        ends,
        radii,
        lengths,
        prox_ids=prox_ids,
        dist_ids=dist_ids,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=vmax,
        km=km,
        fluid=fluid,
        solver=concentration_solver,
    )
    p_in = float("nan")
    p_out = float("nan")
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
    return starts, ends, radii, lengths, flows, cin, cout, (p_in, p_out)


def summarize_tree(
    tree: Tree,
    sample_points: np.ndarray,
    target_terminals: int,
    *,
    side_length: Optional[float] = None,
    fluid: Optional[str] = None,
    inlet_flow_cm3_s: Optional[float] = None,
    tissue_cache: dict | None = None,
    concentration_solver: str | None = None,
    return_details: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], dict]:
    hematocrit_model = _normalize_hematocrit_model()
    seg_count = int(getattr(tree, "segment_count", 0))
    if seg_count == 0:
        raise RuntimeError("Tree contains no segments; cannot summarize.")

    # Avoid copying the full (seg_count x 31) TreeData table.
    data = np.asarray(tree.data[:seg_count])
    starts = data[:, 0:3]
    ends = data[:, 3:6]
    lengths = data[:, 20].copy()
    radii = data[:, 21]
    bad_len = ~np.isfinite(lengths) | (lengths <= 0.0)
    if np.any(bad_len):
        lengths[bad_len] = np.linalg.norm(ends[bad_len] - starts[bad_len], axis=1)

    total_length = float(np.nansum(lengths))
    total_volume = float(np.nansum(np.pi * radii**2 * lengths))
    avg_radius = float(np.nanmean(radii)) if radii.size else float("nan")
    avg_length = float(np.nanmean(lengths)) if lengths.size else float("nan")

    pressure_in = float(tree.parameters.root_pressure)
    pressure_out = float(tree.parameters.terminal_pressure)
    pressure_drop = pressure_in - pressure_out

    n_terminals = max(int(getattr(tree, "n_terminals", 0)) - 1, 0)
    flow_inlet = inlet_flow_cm3_s
    if flow_inlet is None or not np.isfinite(flow_inlet):
        q_scale = (side_length ** 3) if (side_length is not None and SCALE_Q_BY_VOLUME) else 1.0
        flow_inlet = float(QIN_TARGET) * 1e-3 / 60.0 * q_scale

    analysis_fluid = fluid or ACTIVE_FLUID
    inlet_concentration = get_concentration_inlet(analysis_fluid)
    concentration_solver_mode = _resolve_concentration_solver(concentration_solver)
    t_assemble = perf_counter()
    (
        starts_arr,
        ends_arr,
        radii_arr,
        lengths_arr,
        mu_arr,
        inlet_nodes,
        outlet_nodes,
        prox_ids,
        dist_ids,
    ) = assemble_tree_segments(tree, analysis_fluid)
    t_assemble = perf_counter() - t_assemble

    t_kirchhoff = perf_counter()
    hd_flow_iter = np.empty((0,), dtype=float)
    t_hematocrit_flow_iter = 0.0
    hematocrit_iter_log: list[dict[str, float]] = []
    hematocrit_iterations_completed = 0
    if starts_arr.size == 0:
        empty = np.empty((0,), dtype=float)
        flows = empty
        pressures = empty
    else:
        radius_safe = np.maximum(radii_arr, 1e-12)
        fixed_flow_tree_shortcut = (
            _HAVE_NUMBA
            and str(KIRCHHOFF_SOLVER).strip().lower() in ("tree", "tree_neumann", "tree-current-bc")
            and _normalize_kirchhoff_bc_mode() == "legacy_equal_terminal_flow"
            and inlet_nodes
            and flow_inlet is not None
            and np.isfinite(flow_inlet)
        )
        flows_fixed = None
        fixed_flow_s = 0.0
        if fixed_flow_tree_shortcut:
            t_fixed_flow = perf_counter()
            hct_context_fixed = _hematocrit_context_for_tree(tree)
            flows_fixed, _, fixed_term_count = _fixed_terminal_flows_numba(
                np.asarray(hct_context_fixed["order"], dtype=np.int64),
                np.asarray(hct_context_fixed["left_child"], dtype=np.int64),
                np.asarray(hct_context_fixed["right_child"], dtype=np.int64),
                float(flow_inlet),
            )
            flows_fixed = np.asarray(flows_fixed, dtype=float)
            fixed_flow_s = perf_counter() - t_fixed_flow
            if KIRCHHOFF_DIAGNOSTICS:
                print(
                    "Kirchhoff diagnostics: solver_used=tree_fixed_equal_terminal_flow "
                    f"bc=legacy_equal_terminal_flow flow_time={fixed_flow_s:.3f}s terminals={int(fixed_term_count)}"
                )
        if (
            str(analysis_fluid).lower() == "blood"
            and hematocrit_model == "pries_secomb"
            and int(HEMATOCRIT_FLOW_ITERATIONS) > 0
        ):
            t_hct_iter = perf_counter()
            hd_iter = np.full(starts_arr.shape[0], float(HD_DISCHARGE), dtype=float)
            flows_old_iter = np.zeros(starts_arr.shape[0], dtype=float)
            rho = float(tree.parameters.fluid_density)
            nu = float(tree.parameters.kinematic_viscosity)
            mu_base = rho * nu
            iter_count = max(int(HEMATOCRIT_FLOW_ITERATIONS), 0)
            relax = float(np.clip(HEMATOCRIT_RELAXATION, 0.0, 1.0))
            qtol_cgs = float(HEMATOCRIT_QTOL_NL_MIN) * 1.0e-6 / 60.0
            for h_iter in range(iter_count):
                if (h_iter + 1) % 5 == 0:
                    relax *= 0.8
                mu_iter = segment_viscosity_from_radius_hd(radii_arr, mu_base, analysis_fluid, hd_iter)
                resistances_iter = (8.0 * mu_iter * lengths_arr) / (np.pi * radius_safe**4)
                if flows_fixed is not None:
                    flows_iter = flows_fixed
                else:
                    pressures_iter, flows_iter, _, _, _ = solve_kirchhoff(
                        prox_ids,
                        dist_ids,
                        resistances_iter,
                        inlet_nodes,
                        flow_inlet,
                        outlet_nodes,
                    )
                hd_new, _ = compute_tree_hematocrit(
                    tree,
                    hd_root=HD_DISCHARGE,
                    flows=flows_iter,
                    model=hematocrit_model,
                )
                delta_hd = float(np.nanmax(np.abs(hd_new - hd_iter))) if hd_new.size else 0.0
                delta_q = float(np.nanmax(np.abs(flows_iter - flows_old_iter))) if flows_iter.size else 0.0
                hd_iter = (1.0 - relax) * hd_iter + relax * hd_new
                flows_old_iter = np.asarray(flows_iter, dtype=float).copy()
                hematocrit_iterations_completed = h_iter + 1
                hematocrit_iter_log.append(
                    {
                        "iteration": float(h_iter + 1),
                        "relax": float(relax),
                        "max_q_delta_cm3_s": float(delta_q),
                        "max_hd_delta": float(delta_hd),
                    }
                )
                if HEMATOCRIT_DIAGNOSTICS:
                    finite_hd = hd_iter[np.isfinite(hd_iter)]
                    h50 = float(np.nanmedian(finite_hd)) if finite_hd.size else float("nan")
                    print(
                        f"  Hematocrit flow iteration {h_iter + 1}/{iter_count}: "
                        f"relax={relax:.3f} max_q_delta={delta_q:.3e} max_hd_delta={delta_hd:.3e} HD_median={h50:.3f}"
                )
                if delta_q < qtol_cgs and delta_hd < float(HEMATOCRIT_HDTOL):
                    if HEMATOCRIT_DIAGNOSTICS:
                        print(
                            f"  Hematocrit flow iterations converged at {h_iter + 1}/{iter_count}: "
                            f"max_q_delta={delta_q:.3e} max_hd_delta={delta_hd:.3e}"
                        )
                    break
            mu_arr = segment_viscosity_from_radius_hd(radii_arr, mu_base, analysis_fluid, hd_iter)
            hd_flow_iter = hd_iter
            _store_tree_hematocrit_cache(
                tree,
                hd_iter,
                _tube_hematocrit_from_hd_radius(radii_arr, hd_iter),
                model=hematocrit_model,
                flows=flows_old_iter,
                fixed_flow_bc=(_normalize_kirchhoff_bc_mode() == "legacy_equal_terminal_flow"),
            )
            t_hematocrit_flow_iter = perf_counter() - t_hct_iter
        resistances = (8.0 * mu_arr * lengths_arr) / (np.pi * radius_safe**4)
        if flows_fixed is not None:
            flows = np.asarray(flows_fixed, dtype=float)
            num_nodes = int(max(int(np.max(prox_ids)), int(np.max(dist_ids))) + 1) if prox_ids.size else 0
            root_pressure_dyn = float(tree.parameters.root_pressure) * PA_TO_DYN_PER_CM2
            t_pressure = perf_counter()
            pressures, assigned_edges = _pressures_from_tree_flows_numba(
                np.asarray(_hematocrit_context_for_tree(tree)["order"], dtype=np.int64),
                np.asarray(prox_ids, dtype=np.int64),
                np.asarray(dist_ids, dtype=np.int64),
                np.asarray(flows, dtype=np.float64),
                np.maximum(np.asarray(resistances, dtype=np.float64), 1.0e-30),
                int(inlet_nodes[0]),
                float(root_pressure_dyn),
                int(num_nodes),
            )
            pressure_s = perf_counter() - t_pressure
            if int(assigned_edges) != int(prox_ids.size) or not np.all(np.isfinite(pressures)):
                if KIRCHHOFF_DIAGNOSTICS:
                    print(
                        "Kirchhoff fixed-flow pressure reconstruction incomplete; "
                        f"assigned_edges={int(assigned_edges)}/{int(prox_ids.size)}. Falling back to tree solve."
                    )
                pressures, flows, _, _, _ = solve_kirchhoff(
                    prox_ids,
                    dist_ids,
                    resistances,
                    inlet_nodes,
                    flow_inlet,
                    outlet_nodes,
                )
                if pressures.size and inlet_nodes:
                    target = float(tree.parameters.root_pressure) * PA_TO_DYN_PER_CM2
                    delta = target - float(pressures[inlet_nodes[0]])
                    pressures = pressures + delta
            elif KIRCHHOFF_DIAGNOSTICS:
                print(
                    "Kirchhoff diagnostics: solver_used=tree_fixed_pressure_reconstruct "
                    f"bc=legacy_equal_terminal_flow solve_time={pressure_s:.3f}s true_rel_resid=0.000e+00"
                )
        else:
            pressures, flows, _, _, _ = solve_kirchhoff(
                prox_ids,
                dist_ids,
                resistances,
                inlet_nodes,
                flow_inlet,
                outlet_nodes,
            )
            if pressures.size and inlet_nodes:
                target = float(tree.parameters.root_pressure) * PA_TO_DYN_PER_CM2
                delta = target - float(pressures[inlet_nodes[0]])
                pressures = pressures + delta
    t_kirchhoff = perf_counter() - t_kirchhoff
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
        pressure_in = p_in
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
        pressure_out = p_out
    pressure_drop = pressure_in - pressure_out

    t_conc = perf_counter()
    cin, cout = _solve_channel_concentrations(
        tree,
        flows,
        inlet_nodes,
        outlet_nodes,
        starts_arr,
        ends_arr,
        radii_arr,
        lengths_arr,
        prox_ids=prox_ids,
        dist_ids=dist_ids,
        inlet_concentration=inlet_concentration,
        diffusivity=SOLUTE_DIFFUSIVITY,
        vmax=VMAX_MM,
        km=K_M_MM,
        fluid=analysis_fluid,
        solver=concentration_solver,
    )
    t_conc = perf_counter() - t_conc
    concentration_timing_details = dict(_LAST_CONCENTRATION_TIMINGS)

    q_inlet = float(np.abs(flows[0])) if flows.size else float("nan")
    q_inlet_ul_min = float(q_inlet * 60000.0) if np.isfinite(q_inlet) else float("nan")
    total_flow = float(q_inlet) if np.isfinite(q_inlet) else float("nan")
    positive_flows = np.abs(flows[flows != 0]) if flows.size else np.empty((0,))
    q_min = float(np.min(positive_flows)) if positive_flows.size else float("nan")

    q_ratio = float("nan")
    if np.isfinite(q_min) and np.isfinite(q_inlet) and q_inlet != 0.0:
        q_ratio = q_min / q_inlet

    dR_net = float("nan")
    r_net = float("nan")
    reference_flow = q_inlet if np.isfinite(q_inlet) else total_flow
    if (
        np.isfinite(pressure_drop)
        and np.isfinite(reference_flow)
        and reference_flow != 0.0
        and total_length > 0.0
    ):
        r_net = pressure_drop / reference_flow
        dR_net = r_net / total_length

    avg_distance = compute_average_distance(sample_points, starts, ends) if COMPUTE_AVG_DISTANCE_TO_CHANNEL else float("nan")
    finite_radii = radii[np.isfinite(radii)]
    radius_min = float(np.nanmin(finite_radii)) if finite_radii.size else float("nan")
    radius_max = float(np.nanmax(finite_radii)) if finite_radii.size else float("nan")

    vessel_map = getattr(tree, "vessel_map", {}) or {}
    terminals = [idx for idx in range(cout.size) if not vessel_map.get(idx, {}).get("downstream")] if vessel_map else []
    if not terminals:
        left_child = data[:, 15]
        right_child = data[:, 16]
        terminals = list(np.where(np.isnan(left_child) & np.isnan(right_child))[0])
    target_concs = cout[terminals] if terminals else cout
    finite = target_concs[np.isfinite(target_concs)]
    c_lq = float(np.percentile(finite, 25.0)) if finite.size else float("nan")

    t_tissue = perf_counter()
    cext_source_state = _LAST_CEXT_SOURCE_STATE if concentration_solver_mode.startswith("topdown_ext") else None
    if isinstance(cext_source_state, dict):
        mask, tissue_conc = compute_tissue_samples_greens_from_cext_state(
            sample_points,
            starts_arr,
            ends_arr,
            radii_arr,
            cext_source_state,
            tissue_cache=tissue_cache,
        )
    else:
        if SOLVER_TIMING_DETAILS:
            print("  Tissue Greens solve: source_mode=old_reconstructed_flux")
        mask, tissue_conc = compute_tissue_samples_greens(
            sample_points,
            starts_arr,
            ends_arr,
            radii_arr,
            cin,
            flows,
            diffusivity=SOLUTE_DIFFUSIVITY,
            vmax=VMAX_MM,
            km=K_M_MM,
            window_factor=WINDOW_FACTOR,
            inlet_concentration=inlet_concentration,
            tissue_cache=tissue_cache,
        )
    t_tissue = perf_counter() - t_tissue
    tissue_timing_details = dict(_LAST_TISSUE_TIMINGS)
    tissue_backend = str(tissue_timing_details.get("backend", _resolve_tissue_accel_mode()))
    t_tissue_geometry = float(tissue_timing_details.get("t_tissue_geometry_s", 0.0) or 0.0)
    t_tissue_oxygen = float(tissue_timing_details.get("t_tissue_oxygen_s", t_tissue) or 0.0)
    tissue_vals = tissue_conc[mask]
    tissue_pts = sample_points[mask]
    tissue_avg = float(np.nanmean(tissue_vals)) if tissue_vals.size else float("nan")
    cext_concentration_mean = float("nan")
    cext_concentration_std = float("nan")
    cext_concentration_count = 0
    if isinstance(cext_source_state, dict) and "c_ext_gl" in cext_source_state:
        cext_vals = np.asarray(cext_source_state.get("c_ext_gl"), dtype=np.float32).reshape(-1)
        cext_vals = cext_vals[np.isfinite(cext_vals)]
        cext_concentration_count = int(cext_vals.size)
        if cext_vals.size:
            cext_concentration_mean = float(np.nanmean(cext_vals))
            cext_concentration_std = float(np.nanstd(cext_vals, ddof=1)) if cext_vals.size > 1 else 0.0

    if np.isfinite(CONC_MAX_FOR_NORMALIZATION) and CONC_MAX_FOR_NORMALIZATION != 0.0:
        ratio_lq = c_lq / CONC_MAX_FOR_NORMALIZATION if np.isfinite(c_lq) else float("nan")
        ratio_tiss = tissue_avg / CONC_MAX_FOR_NORMALIZATION if np.isfinite(tissue_avg) else float("nan")
        fractions = {}
        if tissue_vals.size:
            normalized = tissue_vals / CONC_MAX_FOR_NORMALIZATION
            for threshold, key in [
                (0.50, "FracAbove50pct"),
                (0.25, "FracAbove25pct"),
                (0.10, "FracAbove10pct"),
                (0.05, "FracAbove5pct"),
                (0.01, "FracAbove1pct"),
            ]:
                fractions[key] = float(np.mean(normalized >= threshold))
        else:
            fractions = {
                k: float("nan")
                for k in [
                    "FracAbove50pct",
                    "FracAbove25pct",
                    "FracAbove10pct",
                    "FracAbove5pct",
                    "FracAbove1pct",
                ]
            }
    else:
        ratio_lq = float("nan")
        ratio_tiss = float("nan")
        fractions = {
            k: float("nan")
            for k in [
                "FracAbove50pct",
                "FracAbove25pct",
                "FracAbove10pct",
                "FracAbove5pct",
                "FracAbove1pct",
            ]
        }

    area = np.pi * radii * radii
    velocities = np.divide(
        np.abs(flows),
        area,
        out=np.full_like(flows, np.nan),
        where=area > 0,
    )
    bbox_length = _characteristic_length(tree, fallback_side_length=side_length)
    radius_sq = radii * radii
    permeation_rates = np.divide(
        3.66 * SOLUTE_DIFFUSIVITY,
        radius_sq,
        out=np.zeros_like(radius_sq),
        where=radius_sq > 0,
    )
    valid = (lengths > 1e-5) & np.isfinite(velocities) & (velocities > 0)
    damkohler = np.full_like(lengths, np.nan)
    damkohler[valid] = (permeation_rates[valid] * bbox_length) / velocities[valid]
    avg_damkohler = float(np.nanmean(damkohler)) if np.isnan(damkohler).sum() < damkohler.size else float("nan")

    parms = getattr(tree, "parameters", None)
    dlp_enabled = bool(getattr(parms, "dlp_enable", False)) if parms is not None else False
    dlp_angle = float(getattr(parms, "dlp_min_angle_deg", 0.0)) if dlp_enabled else 0.0

    result = {
        "target_terminals": int(target_terminals),
        "cube_side_length": _characteristic_length(tree, fallback_side_length=side_length),
        "distance_sample_count": int(DISTANCE_SAMPLE_COUNT),
        "concentration_solver": concentration_solver_mode,
        "finite_radius_o2_terms": str(FINITE_RADIUS_O2_TERMS),
        "lumen_wall_closure": str(LUMEN_WALL_CLOSURE),
        "graetz_n_radial": int(GRAETZ_N_RADIAL),
        "graetz_n_modes": int(GRAETZ_N_MODES),
        "total_volume": total_volume,
        "total_flowrate": reference_flow,
        "pressure_in_root": pressure_in,
        "pressure_out_terminals": pressure_out,
        "pressure_drop": pressure_drop,
        "avg_radius": avg_radius,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "avg_length": avg_length,
        "total_length": total_length,
        "avg_distance_to_channel": avg_distance,
        "terminal_segments": n_terminals,
        "total_segments": seg_count,
        "dlp_angle": dlp_angle,
        "Rnet": r_net,
        "dRnet": dR_net,
        "inlet_flow_ul_per_min": q_inlet_ul_min,
        "Qmin_over_Qinlet": q_ratio,
        "C_LQ_over_Cmax": ratio_lq,
        "C_tiss_over_Cmax": ratio_tiss,
        "Damkohler": avg_damkohler,
        # Public timing categories:
        # - t_assembly_s includes segment assembly plus tissue geometry/query/refine/setup.
        # - t_tissue_s is only the final tissue Greens oxygen kernel.
        "t_load_s": 0.0,
        "t_assembly_s": t_assemble + t_tissue_geometry,
        "t_kirchhoff_s": t_kirchhoff,
        "t_hematocrit_flow_iteration_s": t_hematocrit_flow_iter,
        "t_concentration_s": t_conc,
        "t_cext_total_s": float(concentration_timing_details.get("t_cext_total_s", 0.0) or 0.0),
        "t_cext_hct_setup_s": float(concentration_timing_details.get("t_cext_hct_setup_s", 0.0) or 0.0),
        "t_cext_context_build_s": float(concentration_timing_details.get("t_cext_context_build_s", 0.0) or 0.0),
        "t_cext_solver_setup_s": float(concentration_timing_details.get("t_cext_solver_setup_s", 0.0) or 0.0),
        "t_cext_query_s": float(concentration_timing_details.get("t_cext_query_s", 0.0) or 0.0),
        "t_cext_kernel_s": float(concentration_timing_details.get("t_cext_kernel_s", 0.0) or 0.0),
        "t_cext_gpu_transfer_s": float(concentration_timing_details.get("t_cext_gpu_transfer_s", 0.0) or 0.0),
        "t_cext_hybrid_deposit_s": float(concentration_timing_details.get("t_cext_hybrid_deposit_s", 0.0) or 0.0),
        "t_cext_hybrid_fft_s": float(concentration_timing_details.get("t_cext_hybrid_fft_s", 0.0) or 0.0),
        "t_cext_hybrid_o2_terms_s": float(concentration_timing_details.get("t_cext_hybrid_o2_terms_s", 0.0) or 0.0),
        "t_cext_hybrid_self_subtract_s": float(concentration_timing_details.get("t_cext_hybrid_self_subtract_s", 0.0) or 0.0),
        "t_cext_hybrid_local_corr_s": float(concentration_timing_details.get("t_cext_hybrid_local_corr_s", 0.0) or 0.0),
        "t_cext_hybrid_sample_s": float(concentration_timing_details.get("t_cext_hybrid_sample_s", 0.0) or 0.0),
        "t_cext_frozen_gpu_transfer_s": float(concentration_timing_details.get("t_cext_frozen_gpu_transfer_s", 0.0) or 0.0),
        "t_cext_init_s": float(concentration_timing_details.get("t_cext_init_s", 0.0) or 0.0),
        "t_cext_init_kernel_s": float(concentration_timing_details.get("t_cext_init_kernel_s", 0.0) or 0.0),
        "t_cext_init_gpu_transfer_s": float(concentration_timing_details.get("t_cext_init_gpu_transfer_s", 0.0) or 0.0),
        "t_cext_init_frozen_gpu_transfer_s": float(concentration_timing_details.get("t_cext_init_frozen_gpu_transfer_s", 0.0) or 0.0),
        "t_cext_init_frozen_kernel_s": float(concentration_timing_details.get("t_cext_init_frozen_kernel_s", 0.0) or 0.0),
        "t_cext_init_cache_s": float(concentration_timing_details.get("t_cext_init_cache_s", 0.0) or 0.0),
        "t_cext_init_hybrid_call_s": float(concentration_timing_details.get("t_cext_init_hybrid_call_s", 0.0) or 0.0),
        "t_cext_final_frozen_gpu_transfer_s": float(concentration_timing_details.get("t_cext_final_frozen_gpu_transfer_s", 0.0) or 0.0),
        "t_cext_final_frozen_kernel_s": float(concentration_timing_details.get("t_cext_final_frozen_kernel_s", 0.0) or 0.0),
        "t_cext_final_cache_s": float(concentration_timing_details.get("t_cext_final_cache_s", 0.0) or 0.0),
        "t_cext_final_source_state_s": float(concentration_timing_details.get("t_cext_final_source_state_s", 0.0) or 0.0),
        "t_cext_backend": str(concentration_timing_details.get("backend", "none")),
        "cext_init_mode": str(concentration_timing_details.get("cext_init_mode", "zero")),
        "cext_init_performed": bool(concentration_timing_details.get("cext_init_performed", False)),
        "t_tissue_s": t_tissue_oxygen,
        "t_tissue_geometry_s": t_tissue_geometry,
        "t_tissue_oxygen_s": t_tissue_oxygen,
        "t_tissue_total_s": t_tissue,
        "t_tissue_kdtree_query_s": float(tissue_timing_details.get("t_tissue_kdtree_query_s", float("nan"))),
        "t_tissue_gpu_refine_s": float(tissue_timing_details.get("t_tissue_gpu_refine_s", float("nan"))),
        "t_tissue_gpu_transfer_s": float(tissue_timing_details.get("t_tissue_gpu_transfer_s", float("nan"))),
        "t_tissue_backend": tissue_backend,
        "hematocrit_model": hematocrit_model,
        "hematocrit_flow_iterations": int(HEMATOCRIT_FLOW_ITERATIONS),
        "hematocrit_flow_iterations_completed": int(hematocrit_iterations_completed),
        "cext_outer_iterations_completed": int(concentration_timing_details.get("cext_outer_iterations_completed", 0) or 0),
        "cext_total_iterations_effective": int(concentration_timing_details.get("cext_total_iterations_effective", 0) or 0),
        "cext_accel_mode": str(concentration_timing_details.get("cext_accel_mode", "none")),
        "cext_accel_step_last": str(concentration_timing_details.get("cext_accel_step_last", "picard")),
        "cext_accel_rejections": int(concentration_timing_details.get("cext_accel_rejections", 0) or 0),
        "cext_accel_restarts": int(concentration_timing_details.get("cext_accel_restarts", 0) or 0),
        "cext_anderson_depth_used": int(concentration_timing_details.get("cext_anderson_depth_used", 0) or 0),
        "cext_omega_last": float(concentration_timing_details.get("cext_omega_last", CEXT_VESS_COUPLING_OMEGA) or CEXT_VESS_COUPLING_OMEGA),
        "cext_rel_residual_last": float(concentration_timing_details.get("cext_rel_residual_last", 0.0) or 0.0),
        "cext_max_delta_last": float(concentration_timing_details.get("cext_max_delta_last", 0.0) or 0.0),
        "cext_active_source_count": int(concentration_timing_details.get("cext_active_source_count", 0) or 0),
        "cext_frozen_source_count": int(concentration_timing_details.get("cext_frozen_source_count", 0) or 0),
        "cext_active_target_count": int(concentration_timing_details.get("cext_active_target_count", 0) or 0),
        "cext_frozen_target_count": int(concentration_timing_details.get("cext_frozen_target_count", 0) or 0),
        "cext_active_component_count": int(concentration_timing_details.get("cext_active_component_count", 0) or 0),
        "cext_largest_component_size": int(concentration_timing_details.get("cext_largest_component_size", 0) or 0),
        "cext_target_freeze_events": int(concentration_timing_details.get("cext_target_freeze_events", 0) or 0),
        "cext_target_reactivations": int(concentration_timing_details.get("cext_target_reactivations", 0) or 0),
        "cext_hybrid_bg_grid": int(concentration_timing_details.get("cext_hybrid_bg_grid", 0) or 0),
        "cext_hybrid_lambda_bins": int(concentration_timing_details.get("cext_hybrid_lambda_bins", 0) or 0),
        "cext_hybrid_lambda_bin_policy": str(concentration_timing_details.get("cext_hybrid_lambda_bin_policy", "")),
        "cext_hybrid_lambda_bin_edges_hash": str(concentration_timing_details.get("cext_hybrid_lambda_bin_edges_hash", "")),
        "cext_hybrid_bg_solver": str(concentration_timing_details.get("cext_hybrid_bg_solver", "")),
        "cext_concentration_mean": cext_concentration_mean,
        "cext_concentration_std": cext_concentration_std,
        "cext_concentration_count": int(cext_concentration_count),
        "concentration_inlet": inlet_concentration,
        **fractions,
    }
    cext_source_state = None
    _clear_cext_runtime_state(tree)
    if not return_details:
        return result
    details = {
        "starts": starts_arr,
        "ends": ends_arr,
        "radii": radii_arr,
        "lengths": lengths_arr,
        "mu": mu_arr,
        "pressures": pressures,
        "flows": flows,
        "cin": cin,
        "cout": cout,
        "concentration_timing_details": concentration_timing_details,
        "flow_iteration_hematocrit": hd_flow_iter,
        "hematocrit_iteration_log": hematocrit_iter_log,
        "tissue_points": tissue_pts,
        "tissue_values": tissue_vals,
        "inlet_concentration": inlet_concentration,
        "cext_source_state": cext_source_state,
    }
    HD_detail = np.asarray(getattr(tree, "discharge_hematocrit", np.empty((0,), dtype=float)), dtype=float)
    HT_detail = np.asarray(getattr(tree, "tube_hematocrit", np.empty((0,), dtype=float)), dtype=float)
    if HD_detail.shape[0] != seg_count or HT_detail.shape[0] != seg_count:
        if str(analysis_fluid).lower() == "blood":
            HD_detail, HT_detail = compute_tree_hematocrit(
                tree,
                hd_root=HD_DISCHARGE,
                flows=flows,
                model=hematocrit_model,
            )
        else:
            HD_detail = np.empty((0,), dtype=float)
            HT_detail = np.empty((0,), dtype=float)
    details["discharge_hematocrit"] = HD_detail
    details["tube_hematocrit"] = HT_detail
    return result, details


def summarize_tissue_only_from_details(
    base_metrics: Dict[str, float],
    details: dict,
    sample_points: np.ndarray,
    *,
    tissue_cache: dict | None,
    tissue_cache_build_elapsed: float,
    base_core_assembly_s: float,
) -> Dict[str, float]:
    """Reuse a solved tree/flow/concentration state and recompute only tissue metrics."""
    metrics = dict(base_metrics)
    cext_source_state = details.get("cext_source_state")
    t_tissue = perf_counter()
    if isinstance(cext_source_state, dict):
        mask, tissue_conc = compute_tissue_samples_greens_from_cext_state(
            sample_points,
            np.asarray(details["starts"], dtype=float),
            np.asarray(details["ends"], dtype=float),
            np.asarray(details["radii"], dtype=float),
            cext_source_state,
            tissue_cache=tissue_cache,
        )
    else:
        mask, tissue_conc = compute_tissue_samples_greens(
            sample_points,
            np.asarray(details["starts"], dtype=float),
            np.asarray(details["ends"], dtype=float),
            np.asarray(details["radii"], dtype=float),
            np.asarray(details["cin"], dtype=float),
            np.asarray(details["flows"], dtype=float),
            diffusivity=SOLUTE_DIFFUSIVITY,
            vmax=VMAX_MM,
            km=K_M_MM,
            window_factor=WINDOW_FACTOR,
            inlet_concentration=float(details.get("inlet_concentration", get_concentration_inlet())),
            tissue_cache=tissue_cache,
        )
    t_tissue = perf_counter() - t_tissue
    tissue_timing_details = dict(_LAST_TISSUE_TIMINGS)
    tissue_vals = tissue_conc[mask]
    tissue_avg = float(np.nanmean(tissue_vals)) if tissue_vals.size else float("nan")
    if np.isfinite(CONC_MAX_FOR_NORMALIZATION) and CONC_MAX_FOR_NORMALIZATION != 0.0:
        metrics["C_tiss_over_Cmax"] = (
            tissue_avg / CONC_MAX_FOR_NORMALIZATION if np.isfinite(tissue_avg) else float("nan")
        )
        if tissue_vals.size:
            normalized = tissue_vals / CONC_MAX_FOR_NORMALIZATION
            for threshold, key in [
                (0.50, "FracAbove50pct"),
                (0.25, "FracAbove25pct"),
                (0.10, "FracAbove10pct"),
                (0.05, "FracAbove5pct"),
                (0.01, "FracAbove1pct"),
            ]:
                metrics[key] = float(np.mean(normalized >= threshold))
        else:
            for key in ["FracAbove50pct", "FracAbove25pct", "FracAbove10pct", "FracAbove5pct", "FracAbove1pct"]:
                metrics[key] = float("nan")
    else:
        metrics["C_tiss_over_Cmax"] = float("nan")
        for key in ["FracAbove50pct", "FracAbove25pct", "FracAbove10pct", "FracAbove5pct", "FracAbove1pct"]:
            metrics[key] = float("nan")

    metrics["distance_sample_count"] = int(sample_points.shape[0])
    t_tissue_geometry = float(tissue_timing_details.get("t_tissue_geometry_s", 0.0) or 0.0)
    metrics["t_assembly_s"] = (
        float(base_core_assembly_s)
        + float(tissue_cache_build_elapsed)
        + t_tissue_geometry
    )
    metrics["t_tissue_s"] = float(tissue_timing_details.get("t_tissue_oxygen_s", t_tissue) or 0.0)
    metrics["t_tissue_geometry_s"] = t_tissue_geometry
    metrics["t_tissue_oxygen_s"] = float(tissue_timing_details.get("t_tissue_oxygen_s", t_tissue) or 0.0)
    metrics["t_tissue_total_s"] = float(tissue_timing_details.get("t_tissue_total_s", t_tissue) or t_tissue)
    metrics["t_tissue_kdtree_query_s"] = float(tissue_timing_details.get("t_tissue_kdtree_query_s", float("nan")))
    metrics["t_tissue_gpu_refine_s"] = float(tissue_timing_details.get("t_tissue_gpu_refine_s", float("nan")))
    metrics["t_tissue_gpu_transfer_s"] = float(tissue_timing_details.get("t_tissue_gpu_transfer_s", float("nan")))
    metrics["t_tissue_backend"] = str(tissue_timing_details.get("backend", _resolve_tissue_accel_mode()))
    if COMPUTE_AVG_DISTANCE_TO_CHANNEL:
        metrics["avg_distance_to_channel"] = compute_average_distance(
            sample_points,
            np.asarray(details["starts"], dtype=float),
            np.asarray(details["ends"], dtype=float),
        )
    return metrics


# ---------------------------------------------------------------------------
# Concentration solve on network
# ---------------------------------------------------------------------------
def solve_network_concentrations(
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    flows: np.ndarray,
    inlet_nodes: Sequence[int],
    outlet_nodes: Optional[Sequence[int]],
    inlet_concentration: float,
    *,
    fluid: str,
    prox_ids: np.ndarray | None = None,
    dist_ids: np.ndarray | None = None,
    diffusivity: float = SOLUTE_DIFFUSIVITY,
    vmax: float = VMAX_MM,
    km: float = K_M_MM,
    max_iter: int = 100,
    tol: float = 1e-3,
    omega: float = OMEGA,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[float]]]:
    if prox_ids is None or dist_ids is None:
        geom = np.zeros((starts.shape[0], 6), dtype=float)
        geom[:, 0:3] = starts
        geom[:, 3:6] = ends
        prox_ids, dist_ids, _ = _build_node_indices(geom)
    else:
        prox_ids = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
        dist_ids = np.asarray(dist_ids, dtype=np.int64).reshape(-1)

    q = np.abs(flows)
    up = prox_ids.copy()
    down = dist_ids.copy()
    flip = flows < 0
    up[flip] = dist_ids[flip]
    down[flip] = prox_ids[flip]

    num_nodes = int(max(up.max(), down.max()) + 1) if up.size else 0
    if num_nodes == 0:
        empty_hist: Dict[str, List[float]] = {"iter": [], "max_delta": [], "M_in": [], "M_out": [], "M_drop": [], "MB_resid": []}
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), empty_hist

    valid_edges = q > 0.0
    outgoing: List[List[int]] = [[] for _ in range(num_nodes)]
    incoming: List[List[int]] = [[] for _ in range(num_nodes)]
    for i in range(q.size):
        if not valid_edges[i]:
            continue
        outgoing[up[i]].append(i)
        incoming[down[i]].append(i)

    inlet_edges = np.array(
        [edge for node in inlet_nodes for edge in outgoing[int(node)]],
        dtype=int,
    )
    if outlet_nodes:
        sink_nodes = [int(n) for n in outlet_nodes]
    else:
        sink_nodes = [n for n in range(num_nodes) if incoming[n] and not outgoing[n]]
    if sink_nodes:
        sink_edges = np.concatenate([np.array(incoming[n], dtype=int) for n in sink_nodes])
    else:
        sink_edges = np.array([], dtype=int)

    sum_out = np.zeros(num_nodes, dtype=float)
    sum_in = np.zeros(num_nodes, dtype=float)
    for i in range(q.size):
        if not valid_edges[i]:
            continue
        sum_out[up[i]] += q[i]
        sum_in[down[i]] += q[i]

    dirichlet = {int(n): float(inlet_concentration) for n in inlet_nodes}
    C = np.full(num_nodes, float(inlet_concentration), dtype=float)
    for node, value in dirichlet.items():
        C[node] = value

    fluid_mode = (fluid or ACTIVE_FLUID).lower()
    Chb_max = np.zeros_like(radii)
    if fluid_mode == "blood":
        for i, r in enumerate(radii):
            HT = tube_hematocrit(r, hd=HD_DISCHARGE)
            Chb_max[i] = segment_O2_capacity_from_HT(HT)

    history: Dict[str, List[float]] = {
        "iter": [],
        "max_delta": [],
        "M_in": [],
        "M_out": [],
        "M_drop": [],
        "MB_resid": [],
    }

    for it in range(1, max_iter + 1):
        decay = np.ones_like(q)
        for i in range(q.size):
            if not valid_edges[i]:
                continue
            cin = C[up[i]]
            if fluid_mode == "blood":
                decay[i] = _blood_greens_decay_factor(
                    q[i] * CM3_TO_M3,
                    radii[i] * CM_TO_M,
                    lengths[i] * CM_TO_M,
                    diffusivity * CM2_TO_M2,
                    vmax,
                    km,
                    cin,
                    Chb_max[i],
                )
            else:
                decay[i] = _greens_decay_factor(
                    q[i] * CM3_TO_M3,
                    radii[i] * CM_TO_M,
                    lengths[i] * CM_TO_M,
                    diffusivity * CM2_TO_M2,
                    vmax,
                    km,
                    cin,
                )

        if _HAVE_SCIPY_SPARSE:
            rows: List[int] = []
            cols: List[int] = []
            data: List[float] = []
            for i in range(q.size):
                if not valid_edges[i]:
                    continue
                rows.append(int(down[i]))
                cols.append(int(up[i]))
                data.append(float(-q[i] * decay[i]))

            sink_set = set(sink_nodes)
            for n in range(num_nodes):
                if n in sink_set:
                    diag = sum_in[n] if sum_in[n] > 0.0 else 1.0
                elif sum_out[n] > 0.0:
                    diag = sum_out[n]
                elif sum_in[n] > 0.0:
                    diag = sum_in[n]
                else:
                    diag = 1.0
                rows.append(n)
                cols.append(n)
                data.append(float(diag))

            A = _sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tolil()
            b = np.zeros(num_nodes, dtype=float)
            for node, value in dirichlet.items():
                col = np.asarray(A[:, node].todense()).ravel()
                b -= col * value
                A[:, node] = 0.0
                A[node, :] = 0.0
                A[node, node] = 1.0
                b[node] = value
            C_new = _splinalg.spsolve(A.tocsr(), b)
        else:
            A = np.zeros((num_nodes, num_nodes), dtype=float)
            for i in range(q.size):
                if not valid_edges[i]:
                    continue
                A[down[i], up[i]] -= q[i] * decay[i]

            sink_set = set(sink_nodes)
            for n in range(num_nodes):
                if n in sink_set:
                    if sum_in[n] > 0.0:
                        A[n, n] += sum_in[n]
                    else:
                        A[n, n] += 1.0
                elif sum_out[n] > 0.0:
                    A[n, n] += sum_out[n]
                elif sum_in[n] > 0.0:
                    A[n, n] += sum_in[n]
                else:
                    A[n, n] += 1.0

            b = np.zeros(num_nodes, dtype=float)
            for node, value in dirichlet.items():
                b -= A[:, node] * value
                A[:, node] = 0.0
                A[node, :] = 0.0
                A[node, node] = 1.0
                b[node] = value

            C_new = np.linalg.solve(A, b)
        dC = C_new - C
        delta = float(np.nanmax(np.abs(dC)))

        cin = C[up]
        cout = cin * decay
        M_in = float(np.sum(q[inlet_edges] * cin[inlet_edges])) if inlet_edges.size else 0.0
        M_out = float(np.sum(q[sink_edges] * cout[sink_edges])) if sink_edges.size else 0.0
        M_drop = float(np.sum(q * (cin - cout)))
        MB_resid = (M_in - M_out) - M_drop

        history["iter"].append(it)
        history["max_delta"].append(delta)
        history["M_in"].append(M_in)
        history["M_out"].append(M_out)
        history["M_drop"].append(M_drop)
        history["MB_resid"].append(MB_resid)
        C = C + omega * dC
        if delta < tol:
            break

    decay = np.ones_like(q)
    for i in range(q.size):
        if not valid_edges[i]:
            continue
        cin_edge = C[up[i]]
        if fluid_mode == "blood":
            decay[i] = _blood_greens_decay_factor(
                q[i] * CM3_TO_M3,
                radii[i] * CM_TO_M,
                lengths[i] * CM_TO_M,
                diffusivity * CM2_TO_M2,
                vmax,
                km,
                cin_edge,
                Chb_max[i],
            )
        else:
            decay[i] = _greens_decay_factor(
                q[i] * CM3_TO_M3,
                radii[i] * CM_TO_M,
                lengths[i] * CM_TO_M,
                diffusivity * CM2_TO_M2,
                vmax,
                km,
                cin_edge,
            )
    cin = C[up]
    cout = cin * decay
    return cin, cout, C, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tree flow + Greens concentration script.")
    parser.add_argument("--n-closest-vessels", type=int, default=2, help="n_closest_vessels for growth.")
    parser.add_argument("--n-points", type=int, default=50, help="n_points for candidate terminal sampling.")
    parser.add_argument(
        "--weighted-sampling",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Use weighted mesh-cell sampling: true/false (default: false).",
    )
    parser.add_argument(
        "--ignore-collisions",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Skip collision checks during add_vessel: true/false (default: false).",
    )
    parser.add_argument(
        "--allow-inside-vessels",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Allow candidate growth points inside existing vessels.",
    )
    parser.add_argument("--ef", type=int, default=20, help="USearchTree expansion_search (ef) for kNN queries.")
    parser.add_argument("--max-iter", type=int, default=20, help="Optimizer max_iter passed to add_vessel.")
    parser.add_argument(
        "--n-equal-bifurcations",
        type=int,
        default=200000,
        help="Switch to equal-bifurcation growth after this many terminals (disable with < 0).",
    )
    parser.add_argument(
        "--line-profile",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Enable line-level profiling (requires line_profiler).",
    )
    parser.add_argument(
        "--line-profile-out",
        type=str,
        default="TissueSim_line_profile.txt",
        help="Output path for line-level profiler stats.",
    )
    parser.add_argument("--profile", type=str, default="false", choices=("true", "false"),
                        help="Enable cProfile for the run.")
    parser.add_argument("--profile-out", type=str, default="TissueSim_profile.prof",
                        help="Output path for profiler stats.")
    parser.add_argument(
        "--plot-flow",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Plot flow magnitude on each tree before continuing.",
    )
    parser.add_argument(
        "--plot-concentration",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Plot vessel concentration on each tree before continuing.",
    )
    parser.add_argument(
        "--plot-concentration-points",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Plot vessel + tissue concentration on each tree before continuing.",
    )
    parser.add_argument(
        "--plot-viability",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Plot live/dead vessels + points on each tree before continuing.",
    )

    parser.add_argument(
        "--concentration-solver",
        type=str,
        default=CONCENTRATION_SOLVER,
        choices=("topdown", "network", "topdown_ext", "topdown_ext_hybrid_bg", "topdown_ext_treecode"),
        help="Channel concentration solver: topdown, network, topdown_ext, topdown_ext_hybrid_bg, or topdown_ext_treecode.",
    )
    parser.add_argument(
        "--cext-accel",
        type=str,
        default=CEXT_ACCEL_MODE,
        choices=("cpu", "gpu", "auto"),
        help="External-field backend for topdown_ext. Auto uses GPU when available.",
    )
    parser.add_argument(
        "--cext-frozen-accel",
        type=str,
        default=CEXT_FROZEN_ACCEL_MODE,
        choices=("cpu", "gpu", "auto"),
        help="Frozen intravessel backend for topdown_ext. Auto uses GPU when available.",
    )
    parser.add_argument(
        "--finite-radius-o2-terms",
        type=str,
        default=FINITE_RADIUS_O2_TERMS,
        choices=("none", "monopole", "dipole", "both"),
        help="Finite-radius O(a^2) source terms to include in Cext/tissue Green's fields.",
    )
    parser.add_argument(
        "--lumen-wall-closure",
        type=str,
        default=LUMEN_WALL_CLOSURE,
        choices=("wellmixed", "graetz"),
        help="Lumen wall closure. c_iv_gl remains bulk; Graetz stores true c_wall_gl.",
    )
    parser.add_argument(
        "--graetz-n-radial",
        type=int,
        default=GRAETZ_N_RADIAL,
        help="Radial FE nodes for the Graetz lumen closure.",
    )
    parser.add_argument(
        "--graetz-n-modes",
        type=int,
        default=GRAETZ_N_MODES,
        help="Retained radial eigenmodes for the Graetz lumen closure.",
    )
    parser.add_argument(
        "--graetz-max-fp-iters",
        type=int,
        default=GRAETZ_MAX_FP_ITERS,
        help="Fixed-point iterations for wall-lambda/Biot coupling in each frozen Graetz step.",
    )
    parser.add_argument(
        "--graetz-profile",
        type=str,
        default=GRAETZ_VELOCITY_PROFILE,
        choices=("poiseuille", "plug"),
        help="Velocity profile used in the Graetz eigenproblem.",
    )
    parser.add_argument(
        "--lumen-diffusivity-cm2-s",
        type=float,
        default=None,
        help="Override lumen molecular diffusivity in cm^2/s for Graetz radial diffusion.",
    )
    parser.add_argument(
        "--cext-init-mode",
        type=str,
        default=CEXT_INIT_MODE,
        choices=("zero", "decoupled_greens"),
        help="topdown_ext Cext initialization mode. 'decoupled_greens' runs one zero-Cext predictor before Picard.",
    )
    parser.add_argument(
        "--cext-lambda-source",
        type=str,
        default=CEXT_LAMBDA_SOURCE,
        choices=("lambda_vv", "lambda_t"),
        help="Lambda field used by the Cext Green's evaluator. lambda_t uses lambda_tissue(c_wall, Cext).",
    )
    parser.add_argument(
        "--cext-window-factor",
        type=float,
        default=CEXT_WINDOW_FACTOR,
        help="External-field interaction window factor. Set 0 to collapse topdown_ext to topdown.",
    )
    parser.add_argument(
        "--cext-vess-coupling-max-iter",
        type=int,
        default=CEXT_VESS_COUPLING_MAX_ITER,
        help="Maximum outer Picard iterations for topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-tol",
        type=float,
        default=CEXT_VESS_COUPLING_TOL,
        help="Convergence tolerance for topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-rel-tol",
        type=float,
        default=CEXT_VESS_COUPLING_REL_TOL,
        help="Relative convergence tolerance for topdown_ext vessel coupling. Set 0 to disable.",
    )
    parser.add_argument(
        "--cext-vess-coupling-omega",
        type=float,
        default=CEXT_VESS_COUPLING_OMEGA,
        help="Relaxation factor for topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-accel",
        type=str,
        default=CEXT_VESS_COUPLING_ACCEL,
        choices=("none", "aitken", "anderson"),
        help="Nonlinear acceleration mode for topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-omega-min",
        type=float,
        default=CEXT_VESS_COUPLING_OMEGA_MIN,
        help="Minimum adaptive relaxation factor for topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-omega-max",
        type=float,
        default=CEXT_VESS_COUPLING_OMEGA_MAX,
        help="Maximum adaptive relaxation factor for topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-trust-abs",
        type=float,
        default=CEXT_VESS_COUPLING_TRUST_ABS,
        help="Absolute per-entry trust-region cap for topdown_ext Cext updates.",
    )
    parser.add_argument(
        "--cext-vess-coupling-trust-rel",
        type=float,
        default=CEXT_VESS_COUPLING_TRUST_REL,
        help="Relative per-entry trust-region cap for topdown_ext Cext updates.",
    )
    parser.add_argument(
        "--cext-vess-coupling-anderson-depth",
        type=int,
        default=CEXT_VESS_COUPLING_ANDERSON_DEPTH,
        help="History depth for Anderson acceleration in topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-anderson-reg",
        type=float,
        default=CEXT_VESS_COUPLING_ANDERSON_REG,
        help="Tikhonov regularization for Anderson acceleration in topdown_ext vessel coupling.",
    )
    parser.add_argument(
        "--cext-vess-coupling-anderson-start",
        type=int,
        default=CEXT_VESS_COUPLING_ANDERSON_START,
        help="Iteration index at which Anderson acceleration becomes eligible.",
    )
    parser.add_argument(
        "--cext-vess-coupling-anderson-gate-ratio",
        type=float,
        default=CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO,
        help="Residual-ratio threshold below which Anderson acceleration becomes eligible.",
    )
    parser.add_argument(
        "--cext-vess-coupling-anderson-min-stable-iters",
        type=int,
        default=CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS,
        help="Number of improving iterations required before Anderson acceleration is allowed.",
    )
    parser.add_argument(
        "--cext-vess-coupling-anderson-omega-gate-factor",
        type=float,
        default=CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR,
        help="Disable Anderson when adaptive omega falls below omega_min times this factor.",
    )
    parser.add_argument(
        "--cext-vess-coupling-accept-factor",
        type=float,
        default=CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR,
        help="Safeguard factor for accepting Anderson candidate residual predictions.",
    )
    parser.add_argument(
        "--cext-vess-coupling-step-factor",
        type=float,
        default=CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR,
        help="Maximum multiple of the fallback step allowed for accelerated topdown_ext updates.",
    )
    parser.add_argument(
        "--cext-vess-coupling-restart-factor",
        type=float,
        default=CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR,
        help="Residual growth factor that triggers an Anderson history restart.",
    )
    parser.add_argument(
        "--cext-vess-coupling-best-stall-iters",
        type=int,
        default=CEXT_VESS_COUPLING_BEST_STALL_ITERS,
        help="Stop and revert to the best Cext iterate after this many non-improving iterations. Set 0 to disable.",
    )
    parser.add_argument(
        "--cext-vess-coupling-best-revert-factor",
        type=float,
        default=CEXT_VESS_COUPLING_BEST_REVERT_FACTOR,
        help="Residual growth over the best seen Cext iterate that permits best-seen rollback.",
    )
    parser.add_argument(
        "--cext-vess-coupling-step-reject-factor",
        type=float,
        default=CEXT_VESS_COUPLING_STEP_REJECT_FACTOR,
        help="Reject and retry a Cext update when the next residual exceeds the previous evaluated residual by this factor.",
    )
    parser.add_argument(
        "--cext-vess-coupling-step-retry-factor",
        type=float,
        default=CEXT_VESS_COUPLING_STEP_RETRY_FACTOR,
        help="Multiplier applied to omega when retrying a rejected Cext update.",
    )
    parser.add_argument(
        "--cext-active-set-enable",
        type=str,
        default="true" if CEXT_ACTIVE_SET_ENABLE else "false",
        choices=("true", "false"),
        help="Freeze locally converged source segments and keep their Cext contribution in a cached field.",
    )
    parser.add_argument(
        "--cext-active-set-start",
        type=int,
        default=CEXT_ACTIVE_SET_START,
        help="First outer iteration where converged Cext sources may be moved into the frozen active set cache.",
    )
    parser.add_argument(
        "--cext-active-set-stable-iters",
        type=int,
        default=CEXT_ACTIVE_SET_STABLE_ITERS,
        help="Consecutive locally stable iterations required before a source segment is frozen.",
    )
    parser.add_argument(
        "--cext-active-set-rel-tol",
        type=float,
        default=CEXT_ACTIVE_SET_REL_TOL,
        help="Relative local source-state tolerance used to freeze converged Cext source segments.",
    )
    parser.add_argument(
        "--cext-active-set-abs-tol",
        type=float,
        default=CEXT_ACTIVE_SET_ABS_TOL,
        help="Absolute local source-state tolerance used to freeze converged Cext source segments.",
    )
    parser.add_argument(
        "--cext-active-set-refresh-period",
        type=int,
        default=CEXT_ACTIVE_SET_REFRESH_PERIOD,
        help="Rebuild the frozen-source Cext field every N outer iterations. Set 0 to disable periodic refresh.",
    )
    parser.add_argument(
        "--cext-active-set-min-active-count",
        type=int,
        default=CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT,
        help="Absolute minimum number of active Cext source segments to keep in the nonlinear solve.",
    )
    parser.add_argument(
        "--cext-active-set-min-active-fraction",
        type=float,
        default=CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION,
        help="Legacy minimum active-source fraction safeguard. Set 0 to disable; absolute count is primary.",
    )
    parser.add_argument(
        "--cext-hybrid-bg-mode",
        type=str,
        default=CEXT_HYBRID_BG_MODE,
        choices=("local_only_nlambda", "local_only", "fft", "hybrid"),
        help=(
            "Cext background mode for topdown_ext_hybrid_bg. "
            "fft is the default regular-grid solve with O2 correction and self-subtraction; "
            "local_only_nlambda is the screened pairwise solve with r <= N*lambda; "
            "hybrid is FFT plus configured local correction."
        ),
    )
    parser.add_argument(
        "--cext-hybrid-bg-grid",
        type=int,
        default=CEXT_HYBRID_BG_GRID,
        help="Regular-grid resolution per axis for local cell lists or the FFT Cext background field.",
    )
    parser.add_argument(
        "--cext-hybrid-bg-lambda-bins",
        type=int,
        default=CEXT_HYBRID_BG_LAMBDA_BINS,
        help="Number of fixed lambda bins used by topdown_ext_hybrid_bg.",
    )
    parser.add_argument(
        "--cext-hybrid-bg-near-radius-mult",
        type=float,
        default=CEXT_HYBRID_BG_NEAR_RADIUS_MULT,
        help="Exact-local correction radius in background-grid spacings for topdown_ext_hybrid_bg. Set 0 for pure-grid fast mode.",
    )
    parser.add_argument(
        "--cext-hybrid-bg-vcycles",
        type=int,
        default=CEXT_HYBRID_BG_VCYCLES,
        help="Hybrid background solve work budget per outer iteration.",
    )
    parser.add_argument(
        "--cext-hybrid-bg-assignment",
        type=str,
        default=CEXT_HYBRID_BG_ASSIGNMENT,
        choices=("cic", "tsc"),
        help="Source deposition assignment for topdown_ext_hybrid_bg.",
    )
    parser.add_argument(
        "--cext-hybrid-bg-solver",
        type=str,
        default=CEXT_HYBRID_BG_SOLVER,
        choices=("auto", "fft", "jacobi"),
        help="Background-grid solver for topdown_ext_hybrid_bg. 'auto' prefers FFT when cuFFT is available.",
    )
    parser.add_argument(
        "--cext-hybrid-bg-enable-source-freezing",
        type=str,
        default="true" if CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING else "false",
        choices=("true", "false"),
        help="Allow frozen-source caching in the hybrid Cext solver.",
    )
    parser.add_argument(
        "--cext-hybrid-fft-quantile-bins",
        type=str,
        default="true" if CEXT_HYBRID_FFT_QUANTILE_BINS else "false",
        choices=("true", "false"),
        help="Use dynamic quantile lambda bins for the FFT Cext solver.",
    )
    parser.add_argument(
        "--cext-hybrid-fft-o2-correction",
        type=str,
        default="true" if CEXT_HYBRID_FFT_O2_CORRECTION else "false",
        choices=("true", "false"),
        help="Apply the finite-radius O2 moment correction in FFT mode.",
    )
    parser.add_argument(
        "--cext-hybrid-fft-self-subtract",
        type=str,
        default="true" if CEXT_HYBRID_FFT_SELF_SUBTRACT else "false",
        choices=("true", "false"),
        help="Subtract the grid-consistent same-segment self contribution in FFT mode.",
    )
    parser.add_argument(
        "--cext-hybrid-fft-self-sub-target-sampling",
        type=str,
        default=CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING,
        choices=("cic", "tsc", "matched", "assignment", "source"),
        help="Target stencil used for FFT same-segment self-subtraction.",
    )
    parser.add_argument(
        "--cext-hybrid-fft-self-sub-scale",
        type=float,
        default=CEXT_HYBRID_FFT_SELF_SUB_SCALE,
        help="Scale factor for FFT same-segment self-subtraction.",
    )
    parser.add_argument(
        "--cext-treecode-theta",
        type=float,
        default=CEXT_TREECODE_THETA,
        help="Treecode acceptance ratio theta for topdown_ext_treecode.",
    )
    parser.add_argument(
        "--cext-treecode-order",
        type=int,
        default=CEXT_TREECODE_ORDER,
        help="Treecode far-field order for topdown_ext_treecode. 0=monopole, 1=monopole+dipole.",
    )
    parser.add_argument(
        "--cext-treecode-leaf-nodes",
        type=int,
        default=CEXT_TREECODE_LEAF_NODES,
        help="Maximum source GL nodes per treecode leaf.",
    )
    parser.add_argument(
        "--cext-treecode-lambda-bins",
        type=int,
        default=CEXT_TREECODE_LAMBDA_BINS,
        help="Number of lambda bins used by topdown_ext_treecode.",
    )
    parser.add_argument(
        "--cext-treecode-near-radius-mult",
        type=float,
        default=CEXT_TREECODE_NEAR_RADIUS_MULT,
        help="Near-field exact radius in leaf-cell widths for topdown_ext_treecode.",
    )
    parser.add_argument(
        "--cext-treecode-gpu-local",
        type=str,
        default="true" if CEXT_TREECODE_GPU_LOCAL else "false",
        choices=("true", "false"),
        help="Reserved compatibility switch for the treecode GPU local path.",
    )
    parser.add_argument(
        "--cext-tail-solver",
        type=str,
        default=CEXT_TAIL_SOLVER,
        choices=("none", "active_core_nk"),
        help="Tail solver used by topdown_ext_treecode after fixed-point stall detection.",
    )
    parser.add_argument(
        "--cext-tail-trigger-start-iter",
        type=int,
        default=CEXT_TAIL_TRIGGER_START_ITER,
        help="First outer iteration where the treecode active-core tail solver may activate.",
    )
    parser.add_argument(
        "--cext-tail-trigger-stall-iters",
        type=int,
        default=CEXT_TAIL_TRIGGER_STALL_ITERS,
        help="Number of recent iterations inspected for stalled treecode residual improvement.",
    )
    parser.add_argument(
        "--cext-tail-trigger-active-count",
        type=int,
        default=CEXT_TAIL_TRIGGER_ACTIVE_COUNT,
        help="Maximum active-core size eligible for the treecode Newton-Krylov tail solve.",
    )
    parser.add_argument(
        "--cext-tail-max-nonlinear-iters",
        type=int,
        default=CEXT_TAIL_MAX_NONLINEAR_ITERS,
        help="Maximum nonlinear Newton-Krylov iterations in the treecode tail solve.",
    )
    parser.add_argument(
        "--cext-tail-gmres-restart",
        type=int,
        default=CEXT_TAIL_GMRES_RESTART,
        help="GMRES restart used by the treecode active-core tail solve.",
    )
    parser.add_argument(
        "--cext-tail-gmres-maxiter",
        type=int,
        default=CEXT_TAIL_GMRES_MAXITER,
        help="GMRES maxiter used by the treecode active-core tail solve.",
    )
    parser.add_argument(
        "--cext-target-active-set-enable",
        type=str,
        default="true" if CEXT_TARGET_ACTIVE_SET_ENABLE else "false",
        choices=("true", "false"),
        help="Freeze locally converged target segments when nearby active-source influence is weak.",
    )
    parser.add_argument(
        "--cext-target-active-set-start",
        type=int,
        default=CEXT_TARGET_ACTIVE_SET_START,
        help="First outer iteration where target freezing becomes eligible.",
    )
    parser.add_argument(
        "--cext-target-active-set-stable-iters",
        type=int,
        default=CEXT_TARGET_ACTIVE_SET_STABLE_ITERS,
        help="Consecutive locally stable iterations required before a target can freeze.",
    )
    parser.add_argument(
        "--cext-target-active-set-rel-tol",
        type=float,
        default=CEXT_TARGET_ACTIVE_SET_REL_TOL,
        help="Relative local target-update tolerance for target freezing.",
    )
    parser.add_argument(
        "--cext-target-active-set-abs-tol",
        type=float,
        default=CEXT_TARGET_ACTIVE_SET_ABS_TOL,
        help="Absolute local target-update tolerance for target freezing.",
    )
    parser.add_argument(
        "--cext-target-active-set-min-active-count",
        type=int,
        default=CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT,
        help="Absolute minimum number of active Cext targets to keep in the nonlinear solve.",
    )
    parser.add_argument(
        "--cext-target-active-set-neighbor-pad",
        type=int,
        default=CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD,
        help="Extra GPU-direct cell padding used when checking conservative target freeze/reactivation influence.",
    )
    parser.add_argument(
        "--cext-grid-cell-factor",
        type=float,
        default=CEXT_GRID_CELL_FACTOR,
        help="Uniform-grid cell size multiplier for Cext candidate search.",
    )
    parser.add_argument(
        "--cext-streaming-target-candidate-slots",
        type=int,
        default=CEXT_STREAMING_TARGET_CANDIDATE_SLOTS,
        help="Approximate max target-to-source candidate slots per streamed Cext batch.",
    )
    parser.add_argument(
        "--cext-gpu-validate-segments",
        type=int,
        default=CEXT_GPU_VALIDATE_SEGMENTS,
        help="Validate the first N target segments in each GPU Cext batch against the CPU kernel.",
    )
    parser.add_argument(
        "--cext-approx-window-scale",
        type=float,
        default=CEXT_APPROX_WINDOW_SCALE,
        help="Optional coarse-query reach scale for Cext candidate precompute. 1.0 keeps the exact default behavior.",
    )
    parser.add_argument(
        "--cext-max-candidates-per-target",
        type=int,
        default=CEXT_MAX_CANDIDATES_PER_TARGET,
        help="Optional coarse-query cap on source candidates per target segment. 0 disables trimming.",
    )
    parser.add_argument(
        "--hematocrit-model",
        type=str,
        default=HEMATOCRIT_MODEL,
        choices=("uniform_tube", "pries_secomb"),
        help="Blood hematocrit model: current uniform discharge-to-tube correction or NetFlowV2/Pries-Secomb bifurcation partitioning.",
    )
    parser.add_argument(
        "--hematocrit-flow-iterations",
        type=int,
        default=HEMATOCRIT_FLOW_ITERATIONS,
        help="Optional viscosity-flow feedback iterations for pries_secomb hematocrit; 0 keeps flow viscosity behavior unchanged.",
    )
    parser.add_argument(
        "--hematocrit-relaxation",
        type=float,
        default=HEMATOCRIT_RELAXATION,
        help="Initial NetFlow-style relaxation factor for optional hematocrit-flow feedback iterations; reduced by 0.8 every fifth iteration.",
    )
    parser.add_argument(
        "--hematocrit-qtol-nl-min",
        type=float,
        default=HEMATOCRIT_QTOL_NL_MIN,
        help="NetFlow-style nonlinear flow convergence tolerance in nl/min.",
    )
    parser.add_argument(
        "--hematocrit-hdtol",
        type=float,
        default=HEMATOCRIT_HDTOL,
        help="NetFlow-style nonlinear discharge hematocrit convergence tolerance.",
    )
    parser.add_argument(
        "--gl-order",
        type=int,
        default=GL_ORDER,
        choices=(5, 9, 20),
        help="Gauss-Legendre points per segment for Greens integral.",
    )
    parser.add_argument(
        "--cext-gl-order",
        type=int,
        default=GL_ORDER_CEXT,
        help="Gauss-Legendre points per segment for explicit vessel Cext coupling.",
    )
    parser.add_argument(
        "--distance-sample-count",
        type=int,
        default=DISTANCE_SAMPLE_COUNT,
        help="Override DISTANCE_SAMPLE_COUNT for tissue/DNC sample points.",
    )
    parser.add_argument(
        "--distance-sample-counts",
        type=str,
        default=None,
        help=(
            "Comma-separated DISTANCE_SAMPLE_COUNT values for timing scaling. "
            "The tree/flow/concentration solve is reused and only tissue sampling is repeated per M."
        ),
    )
    parser.add_argument(
        "--compute-avg-distance-to-channel",
        choices=("true", "false"),
        default="true" if COMPUTE_AVG_DISTANCE_TO_CHANNEL else "false",
        help="Compute avg_distance_to_channel summary metric. Disable to skip the expensive nearest-channel distance diagnostic.",
    )
    parser.add_argument(
        "--target-counts",
        type=str,
        default=None,
        help="Comma-separated override for TARGET_TERMINAL_COUNTS, e.g. 5000000.",
    )
    parser.add_argument(
        "--qin-target",
        type=str,
        nargs="+",
        default=str(QIN_TARGET),
        help=(
            "Target inlet flow rate(s) in uL/min. Accepts one value or a comma/list string, "
            "e.g. 900, 100 200 300, or '(100, 200, 300)'. "
            "Tree cache lookup ignores this value so geometry caches can be reused."
        ),
    )
    parser.add_argument(
        "--nearest-tissue-vessels",
        type=int,
        default=NEAREST_TISSUE_VESSELS,
        help="Override NEAREST_TISSUE_VESSELS.",
    )
    parser.add_argument(
        "--window-factor",
        type=float,
        default=WINDOW_FACTOR,
        help="Override WINDOW_FACTOR.",
    )
    parser.add_argument(
        "--solver-timing-details",
        type=str,
        default="true" if SOLVER_TIMING_DETAILS else "false",
        choices=("true", "false"),
        help="Print detailed sub-timings for concentration and tissue geometry/Greens solves.",
    )
    parser.add_argument(
        "--tissue-cache-chunk-size",
        type=int,
        default=TISSUE_CACHE_CHUNK_SIZE,
        help="Point chunk size for exact tissue geometry cache construction.",
    )
    parser.add_argument(
        "--tissue-kdtree-workers",
        type=int,
        default=TISSUE_KDTREE_WORKERS,
        help="Workers passed to scipy cKDTree.query during tissue cache construction; use -1 for all cores.",
    )
    parser.add_argument(
        "--tissue-kdtree-candidate-mult",
        type=int,
        default=TISSUE_KDTREE_CANDIDATE_MULT,
        help="Multiplier for midpoint KDTree candidates before exact point-to-segment refinement in tissue solves.",
    )
    parser.add_argument(
        "--tissue-accel",
        type=str,
        default=TISSUE_ACCEL_MODE,
        choices=("cpu", "gpu", "auto"),
        help="Tissue solve backend. GPU mode keeps segment arrays on GPU and streams point chunks.",
    )
    parser.add_argument(
        "--tissue-gpu-chunk-points",
        type=int,
        default=TISSUE_GPU_CHUNK_POINTS,
        help="Point chunk size for GPU tissue refinement/Greens solve.",
    )
    parser.add_argument(
        "--tissue-gpu-validate-points",
        type=int,
        default=TISSUE_GPU_VALIDATE_POINTS,
        help="If >0, compare GPU tissue results against CPU for this many leading points.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=OUTPUT_CSV_NAME,
        help="Override output CSV filename.",
    )
    parser.add_argument(
        "--tree-fast-cache",
        type=str,
        default="true" if TREE_FAST_CACHE_ENABLE else "false",
        choices=("true", "false"),
        help="If a sibling .fast.tree.npz exists, load it instead of the compressed .tree.npz.",
    )
    parser.add_argument(
        "--kirchhoff-diagnostics",
        type=str,
        default="true" if KIRCHHOFF_DIAGNOSTICS else "false",
        choices=("true", "false"),
        help="Print detailed Kirchhoff/GMRES diagnostics to CLI (no CSV changes).",
    )
    parser.add_argument(
        "--kirchhoff-solver",
        type=str,
        default=KIRCHHOFF_SOLVER,
        choices=("tree", "auto", "cg", "spsolve", "gmres_ilu"),
        help="Kirchhoff solver. 'tree' is the O(N) tree-specific solver for the selected Kirchhoff BC mode.",
    )
    parser.add_argument(
        "--kirchhoff-bc-mode",
        type=str,
        default=KIRCHHOFF_BC_MODE,
        choices=(
            "terminal_pressure",
            "mixed",
            "legacy_equal_terminal_flow",
            "legacy",
            "equal_terminal_flow",
            "current_outlets",
            "neumann",
        ),
        help=(
            "Kirchhoff boundary condition mode. terminal_pressure prescribes inlet flow and terminal pressures; "
            "legacy_equal_terminal_flow matches TissueSim_cube_new by prescribing inlet flow and equal terminal outflow sinks."
        ),
    )
    parser.add_argument(
        "--kirchhoff-validate-tree",
        type=str,
        default="true" if KIRCHHOFF_VALIDATE_TREE else "false",
        choices=("true", "false"),
        help="When using --kirchhoff-solver tree, also run sparse spsolve and print speed/error comparison.",
    )
    parser.add_argument(
        "--kirchhoff-validation-sparse-solver",
        type=str,
        default=KIRCHHOFF_VALIDATE_SPARSE_SOLVER,
        choices=("auto", "cg", "spsolve", "gmres_ilu"),
        help="Sparse solver used as the reference for --kirchhoff-validate-tree.",
    )
    parser.add_argument(
        "--tissue-streaming",
        type=str,
        default="true" if TISSUE_STREAMING_ENABLED else "false",
        choices=("true", "false"),
        help="Use chunked streaming tissue oxygen solve for large sample counts.",
    )
    parser.add_argument(
        "--tissue-streaming-min-points",
        type=int,
        default=TISSUE_STREAMING_MIN_POINTS,
        help="Minimum sample-point count that triggers streaming tissue solve.",
    )
    parser.add_argument(
        "--tissue-streaming-target-candidate-slots",
        type=int,
        default=TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS,
        help="Approximate max point*candidate slots per streaming chunk.",
    )
    parser.add_argument(
        "--tissue-streaming-max-chunk-points",
        type=int,
        default=TISSUE_STREAMING_MAX_CHUNK_POINTS,
        help="Maximum points per streaming tissue chunk.",
    )
    parser.add_argument(
        "--tissue-streaming-chunk-workers",
        type=int,
        default=TISSUE_STREAMING_CHUNK_WORKERS,
        help="Parallel outer chunk workers for streaming tissue solve.",
    )
    parser.add_argument(
        "--tissue-streaming-numba-threads-per-worker",
        type=int,
        default=TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER,
        help="Numba threads used inside each streaming chunk worker.",
    )
    parser.add_argument(
        "--tissue-streaming-prune-by-window",
        type=str,
        default="true" if TISSUE_STREAMING_PRUNE_BY_WINDOW else "false",
        choices=("true", "false"),
        help="Conservatively remove candidates that cannot pass the Greens window cutoff before the kernel.",
    )
    return parser.parse_args()


def main() -> None:
    global ACTIVE_FLUID, QIN_TARGET
    args = parse_args()
    qin_target_values = _parse_qin_target_values(getattr(args, "qin_target", QIN_TARGET))
    QIN_TARGET = float(qin_target_values[0])
    if len(qin_target_values) > 1:
        print(
            "Qin target sweep enabled: "
            + ", ".join(f"{float(q):g}" for q in qin_target_values)
            + " uL/min"
        )
    global GL_ORDER, GL_ORDER_CEXT
    GL_ORDER = int(args.gl_order)
    GL_ORDER_CEXT = int(getattr(args, "cext_gl_order", GL_ORDER_CEXT))
    if GL_ORDER_CEXT <= 0:
        raise ValueError(f"Unsupported --cext-gl-order={GL_ORDER_CEXT}. Use a positive integer.")
    global DISTANCE_SAMPLE_COUNT, COMPUTE_AVG_DISTANCE_TO_CHANNEL, TARGET_TERMINAL_COUNTS, NEAREST_TISSUE_VESSELS, WINDOW_FACTOR, OUTPUT_CSV_NAME
    global TREE_FAST_CACHE_ENABLE
    DISTANCE_SAMPLE_COUNT = int(getattr(args, "distance_sample_count", DISTANCE_SAMPLE_COUNT))
    distance_sample_count_values = _parse_distance_sample_count_values(
        getattr(args, "distance_sample_counts", None) or DISTANCE_SAMPLE_COUNT
    )
    DISTANCE_SAMPLE_COUNT = max(distance_sample_count_values)
    if len(distance_sample_count_values) > 1:
        print(
            "Distance sample count sweep enabled: "
            + ", ".join(f"{int(m):g}" for m in distance_sample_count_values)
            + f" points (max sampled once: {int(DISTANCE_SAMPLE_COUNT):g})."
        )
    COMPUTE_AVG_DISTANCE_TO_CHANNEL = (
        str(getattr(args, "compute_avg_distance_to_channel", "true" if COMPUTE_AVG_DISTANCE_TO_CHANNEL else "false")).lower()
        == "true"
    )
    NEAREST_TISSUE_VESSELS = int(getattr(args, "nearest_tissue_vessels", NEAREST_TISSUE_VESSELS))
    WINDOW_FACTOR = float(getattr(args, "window_factor", WINDOW_FACTOR))
    OUTPUT_CSV_NAME = str(getattr(args, "output_csv", OUTPUT_CSV_NAME))
    TREE_FAST_CACHE_ENABLE = str(getattr(args, "tree_fast_cache", "true")).lower() == "true"
    global SOLVER_TIMING_DETAILS, TISSUE_CACHE_CHUNK_SIZE, TISSUE_KDTREE_WORKERS, TISSUE_KDTREE_CANDIDATE_MULT
    global TISSUE_ACCEL_MODE, TISSUE_GPU_CHUNK_POINTS, TISSUE_GPU_VALIDATE_POINTS
    global CEXT_ACCEL_MODE, CEXT_FROZEN_ACCEL_MODE, CEXT_INIT_MODE, CEXT_LAMBDA_SOURCE, CEXT_WINDOW_FACTOR, CEXT_VESS_COUPLING_MAX_ITER
    global CEXT_VESS_COUPLING_TOL, CEXT_VESS_COUPLING_REL_TOL, CEXT_VESS_COUPLING_OMEGA, CEXT_VESS_COUPLING_ACCEL
    global CEXT_VESS_COUPLING_OMEGA_MIN, CEXT_VESS_COUPLING_OMEGA_MAX
    global CEXT_VESS_COUPLING_TRUST_ABS, CEXT_VESS_COUPLING_TRUST_REL
    global CEXT_VESS_COUPLING_ANDERSON_DEPTH, CEXT_VESS_COUPLING_ANDERSON_REG, CEXT_VESS_COUPLING_ANDERSON_START
    global CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO, CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS
    global CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR
    global CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR, CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR, CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR
    global CEXT_VESS_COUPLING_BEST_STALL_ITERS, CEXT_VESS_COUPLING_BEST_REVERT_FACTOR
    global CEXT_VESS_COUPLING_STEP_REJECT_FACTOR, CEXT_VESS_COUPLING_STEP_RETRY_FACTOR
    global CEXT_ACTIVE_SET_ENABLE, CEXT_ACTIVE_SET_START, CEXT_ACTIVE_SET_STABLE_ITERS
    global CEXT_ACTIVE_SET_REL_TOL, CEXT_ACTIVE_SET_ABS_TOL, CEXT_ACTIVE_SET_REFRESH_PERIOD
    global CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT, CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION
    global CEXT_TARGET_ACTIVE_SET_ENABLE, CEXT_TARGET_ACTIVE_SET_START, CEXT_TARGET_ACTIVE_SET_STABLE_ITERS
    global CEXT_TARGET_ACTIVE_SET_REL_TOL, CEXT_TARGET_ACTIVE_SET_ABS_TOL
    global CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT, CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD
    global CEXT_HYBRID_BG_GRID, CEXT_HYBRID_BG_LAMBDA_BINS, CEXT_HYBRID_BG_NEAR_RADIUS_MULT
    global CEXT_HYBRID_BG_VCYCLES, CEXT_HYBRID_BG_ASSIGNMENT, CEXT_HYBRID_BG_SOLVER, CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING, CEXT_HYBRID_BG_MODE
    global CEXT_HYBRID_FFT_QUANTILE_BINS, CEXT_HYBRID_FFT_O2_CORRECTION, CEXT_HYBRID_FFT_SELF_SUBTRACT
    global CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING, CEXT_HYBRID_FFT_SELF_SUB_SCALE
    global CEXT_TREECODE_THETA, CEXT_TREECODE_ORDER, CEXT_TREECODE_LEAF_NODES
    global CEXT_TREECODE_LAMBDA_BINS, CEXT_TREECODE_NEAR_RADIUS_MULT, CEXT_TREECODE_GPU_LOCAL
    global CEXT_TAIL_SOLVER, CEXT_TAIL_TRIGGER_START_ITER, CEXT_TAIL_TRIGGER_STALL_ITERS
    global CEXT_TAIL_TRIGGER_ACTIVE_COUNT, CEXT_TAIL_MAX_NONLINEAR_ITERS
    global CEXT_TAIL_GMRES_RESTART, CEXT_TAIL_GMRES_MAXITER
    global CEXT_GRID_CELL_FACTOR
    global CEXT_STREAMING_TARGET_CANDIDATE_SLOTS, CEXT_GPU_VALIDATE_SEGMENTS
    global CEXT_APPROX_WINDOW_SCALE, CEXT_MAX_CANDIDATES_PER_TARGET
    global FINITE_RADIUS_O2_TERMS, LUMEN_WALL_CLOSURE, GRAETZ_N_RADIAL, GRAETZ_N_MODES
    global GRAETZ_MAX_FP_ITERS, GRAETZ_VELOCITY_PROFILE, LUMEN_DIFFUSIVITY_CM2_S
    global LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S
    SOLVER_TIMING_DETAILS = str(getattr(args, "solver_timing_details", "true")).lower() == "true"
    TISSUE_CACHE_CHUNK_SIZE = int(getattr(args, "tissue_cache_chunk_size", TISSUE_CACHE_CHUNK_SIZE))
    TISSUE_KDTREE_WORKERS = int(getattr(args, "tissue_kdtree_workers", TISSUE_KDTREE_WORKERS))
    TISSUE_KDTREE_CANDIDATE_MULT = int(getattr(args, "tissue_kdtree_candidate_mult", TISSUE_KDTREE_CANDIDATE_MULT))
    TISSUE_ACCEL_MODE = str(getattr(args, "tissue_accel", TISSUE_ACCEL_MODE)).lower()
    TISSUE_GPU_CHUNK_POINTS = int(getattr(args, "tissue_gpu_chunk_points", TISSUE_GPU_CHUNK_POINTS))
    TISSUE_GPU_VALIDATE_POINTS = int(getattr(args, "tissue_gpu_validate_points", TISSUE_GPU_VALIDATE_POINTS))
    CEXT_ACCEL_MODE = str(getattr(args, "cext_accel", CEXT_ACCEL_MODE)).lower()
    CEXT_FROZEN_ACCEL_MODE = str(getattr(args, "cext_frozen_accel", CEXT_FROZEN_ACCEL_MODE)).lower()
    FINITE_RADIUS_O2_TERMS = str(getattr(args, "finite_radius_o2_terms", FINITE_RADIUS_O2_TERMS)).lower()
    LUMEN_WALL_CLOSURE = str(getattr(args, "lumen_wall_closure", LUMEN_WALL_CLOSURE)).lower()
    GRAETZ_N_RADIAL = int(getattr(args, "graetz_n_radial", GRAETZ_N_RADIAL))
    GRAETZ_N_MODES = int(getattr(args, "graetz_n_modes", GRAETZ_N_MODES))
    GRAETZ_MAX_FP_ITERS = int(getattr(args, "graetz_max_fp_iters", GRAETZ_MAX_FP_ITERS))
    GRAETZ_VELOCITY_PROFILE = str(getattr(args, "graetz_profile", GRAETZ_VELOCITY_PROFILE)).lower()
    if getattr(args, "lumen_diffusivity_cm2_s", None) is not None:
        LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S = float(args.lumen_diffusivity_cm2_s)
        LUMEN_DIFFUSIVITY_CM2_S = float(args.lumen_diffusivity_cm2_s)
    _finite_radius_o2_term_flags()
    if LUMEN_WALL_CLOSURE == "graetz" and CEXT_FROZEN_ACCEL_MODE == "cpu":
        raise ValueError("Graetz lumen closure in TissueSim_cube_expanded.py requires --cext-frozen-accel gpu or auto.")
    CEXT_INIT_MODE = str(getattr(args, "cext_init_mode", CEXT_INIT_MODE)).lower()
    CEXT_LAMBDA_SOURCE = _normalize_cext_lambda_source(getattr(args, "cext_lambda_source", CEXT_LAMBDA_SOURCE))
    CEXT_WINDOW_FACTOR = float(getattr(args, "cext_window_factor", CEXT_WINDOW_FACTOR))
    CEXT_VESS_COUPLING_MAX_ITER = int(getattr(args, "cext_vess_coupling_max_iter", CEXT_VESS_COUPLING_MAX_ITER))
    CEXT_VESS_COUPLING_TOL = float(getattr(args, "cext_vess_coupling_tol", CEXT_VESS_COUPLING_TOL))
    CEXT_VESS_COUPLING_REL_TOL = float(getattr(args, "cext_vess_coupling_rel_tol", CEXT_VESS_COUPLING_REL_TOL))
    CEXT_VESS_COUPLING_OMEGA = float(getattr(args, "cext_vess_coupling_omega", CEXT_VESS_COUPLING_OMEGA))
    CEXT_VESS_COUPLING_ACCEL = str(getattr(args, "cext_vess_coupling_accel", CEXT_VESS_COUPLING_ACCEL)).lower()
    CEXT_VESS_COUPLING_OMEGA_MIN = float(getattr(args, "cext_vess_coupling_omega_min", CEXT_VESS_COUPLING_OMEGA_MIN))
    CEXT_VESS_COUPLING_OMEGA_MAX = float(getattr(args, "cext_vess_coupling_omega_max", CEXT_VESS_COUPLING_OMEGA_MAX))
    CEXT_VESS_COUPLING_TRUST_ABS = float(getattr(args, "cext_vess_coupling_trust_abs", CEXT_VESS_COUPLING_TRUST_ABS))
    CEXT_VESS_COUPLING_TRUST_REL = float(getattr(args, "cext_vess_coupling_trust_rel", CEXT_VESS_COUPLING_TRUST_REL))
    CEXT_VESS_COUPLING_ANDERSON_DEPTH = int(getattr(args, "cext_vess_coupling_anderson_depth", CEXT_VESS_COUPLING_ANDERSON_DEPTH))
    CEXT_VESS_COUPLING_ANDERSON_REG = float(getattr(args, "cext_vess_coupling_anderson_reg", CEXT_VESS_COUPLING_ANDERSON_REG))
    CEXT_VESS_COUPLING_ANDERSON_START = int(getattr(args, "cext_vess_coupling_anderson_start", CEXT_VESS_COUPLING_ANDERSON_START))
    CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO = float(getattr(args, "cext_vess_coupling_anderson_gate_ratio", CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO))
    CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS = int(getattr(args, "cext_vess_coupling_anderson_min_stable_iters", CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS))
    CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR = float(getattr(args, "cext_vess_coupling_anderson_omega_gate_factor", CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR))
    CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR = float(getattr(args, "cext_vess_coupling_accept_factor", CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR))
    CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR = float(getattr(args, "cext_vess_coupling_step_factor", CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR))
    CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR = float(getattr(args, "cext_vess_coupling_restart_factor", CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR))
    CEXT_VESS_COUPLING_BEST_STALL_ITERS = int(getattr(args, "cext_vess_coupling_best_stall_iters", CEXT_VESS_COUPLING_BEST_STALL_ITERS))
    CEXT_VESS_COUPLING_BEST_REVERT_FACTOR = float(getattr(args, "cext_vess_coupling_best_revert_factor", CEXT_VESS_COUPLING_BEST_REVERT_FACTOR))
    CEXT_VESS_COUPLING_STEP_REJECT_FACTOR = float(getattr(args, "cext_vess_coupling_step_reject_factor", CEXT_VESS_COUPLING_STEP_REJECT_FACTOR))
    CEXT_VESS_COUPLING_STEP_RETRY_FACTOR = float(getattr(args, "cext_vess_coupling_step_retry_factor", CEXT_VESS_COUPLING_STEP_RETRY_FACTOR))
    CEXT_ACTIVE_SET_ENABLE = str(getattr(args, "cext_active_set_enable", "true" if CEXT_ACTIVE_SET_ENABLE else "false")).lower() == "true"
    CEXT_ACTIVE_SET_START = int(getattr(args, "cext_active_set_start", CEXT_ACTIVE_SET_START))
    CEXT_ACTIVE_SET_STABLE_ITERS = int(getattr(args, "cext_active_set_stable_iters", CEXT_ACTIVE_SET_STABLE_ITERS))
    CEXT_ACTIVE_SET_REL_TOL = float(getattr(args, "cext_active_set_rel_tol", CEXT_ACTIVE_SET_REL_TOL))
    CEXT_ACTIVE_SET_ABS_TOL = float(getattr(args, "cext_active_set_abs_tol", CEXT_ACTIVE_SET_ABS_TOL))
    CEXT_ACTIVE_SET_REFRESH_PERIOD = int(getattr(args, "cext_active_set_refresh_period", CEXT_ACTIVE_SET_REFRESH_PERIOD))
    CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT = int(getattr(args, "cext_active_set_min_active_count", CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT))
    CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION = float(getattr(args, "cext_active_set_min_active_fraction", CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION))
    CEXT_TARGET_ACTIVE_SET_ENABLE = str(getattr(args, "cext_target_active_set_enable", "true" if CEXT_TARGET_ACTIVE_SET_ENABLE else "false")).lower() == "true"
    CEXT_TARGET_ACTIVE_SET_START = int(getattr(args, "cext_target_active_set_start", CEXT_TARGET_ACTIVE_SET_START))
    CEXT_TARGET_ACTIVE_SET_STABLE_ITERS = int(getattr(args, "cext_target_active_set_stable_iters", CEXT_TARGET_ACTIVE_SET_STABLE_ITERS))
    CEXT_TARGET_ACTIVE_SET_REL_TOL = float(getattr(args, "cext_target_active_set_rel_tol", CEXT_TARGET_ACTIVE_SET_REL_TOL))
    CEXT_TARGET_ACTIVE_SET_ABS_TOL = float(getattr(args, "cext_target_active_set_abs_tol", CEXT_TARGET_ACTIVE_SET_ABS_TOL))
    CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT = int(getattr(args, "cext_target_active_set_min_active_count", CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT))
    CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD = int(getattr(args, "cext_target_active_set_neighbor_pad", CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD))
    CEXT_HYBRID_BG_MODE = str(getattr(args, "cext_hybrid_bg_mode", CEXT_HYBRID_BG_MODE)).lower()
    CEXT_HYBRID_BG_GRID = int(getattr(args, "cext_hybrid_bg_grid", CEXT_HYBRID_BG_GRID))
    CEXT_HYBRID_BG_LAMBDA_BINS = int(getattr(args, "cext_hybrid_bg_lambda_bins", CEXT_HYBRID_BG_LAMBDA_BINS))
    CEXT_HYBRID_BG_NEAR_RADIUS_MULT = float(getattr(args, "cext_hybrid_bg_near_radius_mult", CEXT_HYBRID_BG_NEAR_RADIUS_MULT))
    CEXT_HYBRID_BG_VCYCLES = int(getattr(args, "cext_hybrid_bg_vcycles", CEXT_HYBRID_BG_VCYCLES))
    CEXT_HYBRID_BG_ASSIGNMENT = str(getattr(args, "cext_hybrid_bg_assignment", CEXT_HYBRID_BG_ASSIGNMENT)).lower()
    CEXT_HYBRID_BG_SOLVER = str(getattr(args, "cext_hybrid_bg_solver", CEXT_HYBRID_BG_SOLVER)).lower()
    CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING = str(
        getattr(args, "cext_hybrid_bg_enable_source_freezing", "true" if CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING else "false")
    ).lower() == "true"
    CEXT_HYBRID_FFT_QUANTILE_BINS = str(
        getattr(args, "cext_hybrid_fft_quantile_bins", "true" if CEXT_HYBRID_FFT_QUANTILE_BINS else "false")
    ).lower() == "true"
    CEXT_HYBRID_FFT_O2_CORRECTION = str(
        getattr(args, "cext_hybrid_fft_o2_correction", "true" if CEXT_HYBRID_FFT_O2_CORRECTION else "false")
    ).lower() == "true"
    CEXT_HYBRID_FFT_SELF_SUBTRACT = str(
        getattr(args, "cext_hybrid_fft_self_subtract", "true" if CEXT_HYBRID_FFT_SELF_SUBTRACT else "false")
    ).lower() == "true"
    CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING = str(
        getattr(args, "cext_hybrid_fft_self_sub_target_sampling", CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING)
    ).lower()
    CEXT_HYBRID_FFT_SELF_SUB_SCALE = float(getattr(args, "cext_hybrid_fft_self_sub_scale", CEXT_HYBRID_FFT_SELF_SUB_SCALE))
    CEXT_TREECODE_THETA = float(getattr(args, "cext_treecode_theta", CEXT_TREECODE_THETA))
    CEXT_TREECODE_ORDER = int(getattr(args, "cext_treecode_order", CEXT_TREECODE_ORDER))
    CEXT_TREECODE_LEAF_NODES = int(getattr(args, "cext_treecode_leaf_nodes", CEXT_TREECODE_LEAF_NODES))
    CEXT_TREECODE_LAMBDA_BINS = int(getattr(args, "cext_treecode_lambda_bins", CEXT_TREECODE_LAMBDA_BINS))
    CEXT_TREECODE_NEAR_RADIUS_MULT = float(getattr(args, "cext_treecode_near_radius_mult", CEXT_TREECODE_NEAR_RADIUS_MULT))
    CEXT_TREECODE_GPU_LOCAL = str(getattr(args, "cext_treecode_gpu_local", "true" if CEXT_TREECODE_GPU_LOCAL else "false")).lower() == "true"
    CEXT_TAIL_SOLVER = str(getattr(args, "cext_tail_solver", CEXT_TAIL_SOLVER)).lower()
    CEXT_TAIL_TRIGGER_START_ITER = int(getattr(args, "cext_tail_trigger_start_iter", CEXT_TAIL_TRIGGER_START_ITER))
    CEXT_TAIL_TRIGGER_STALL_ITERS = int(getattr(args, "cext_tail_trigger_stall_iters", CEXT_TAIL_TRIGGER_STALL_ITERS))
    CEXT_TAIL_TRIGGER_ACTIVE_COUNT = int(getattr(args, "cext_tail_trigger_active_count", CEXT_TAIL_TRIGGER_ACTIVE_COUNT))
    CEXT_TAIL_MAX_NONLINEAR_ITERS = int(getattr(args, "cext_tail_max_nonlinear_iters", CEXT_TAIL_MAX_NONLINEAR_ITERS))
    CEXT_TAIL_GMRES_RESTART = int(getattr(args, "cext_tail_gmres_restart", CEXT_TAIL_GMRES_RESTART))
    CEXT_TAIL_GMRES_MAXITER = int(getattr(args, "cext_tail_gmres_maxiter", CEXT_TAIL_GMRES_MAXITER))
    CEXT_GRID_CELL_FACTOR = float(getattr(args, "cext_grid_cell_factor", CEXT_GRID_CELL_FACTOR))
    CEXT_STREAMING_TARGET_CANDIDATE_SLOTS = int(
        getattr(args, "cext_streaming_target_candidate_slots", CEXT_STREAMING_TARGET_CANDIDATE_SLOTS)
    )
    CEXT_GPU_VALIDATE_SEGMENTS = int(getattr(args, "cext_gpu_validate_segments", CEXT_GPU_VALIDATE_SEGMENTS))
    CEXT_APPROX_WINDOW_SCALE = float(getattr(args, "cext_approx_window_scale", CEXT_APPROX_WINDOW_SCALE))
    CEXT_MAX_CANDIDATES_PER_TARGET = int(getattr(args, "cext_max_candidates_per_target", CEXT_MAX_CANDIDATES_PER_TARGET))
    global HEMATOCRIT_MODEL, HEMATOCRIT_FLOW_ITERATIONS, HEMATOCRIT_RELAXATION
    global HEMATOCRIT_QTOL_NL_MIN, HEMATOCRIT_HDTOL
    HEMATOCRIT_MODEL = _normalize_hematocrit_model(getattr(args, "hematocrit_model", HEMATOCRIT_MODEL))
    HEMATOCRIT_FLOW_ITERATIONS = int(getattr(args, "hematocrit_flow_iterations", HEMATOCRIT_FLOW_ITERATIONS))
    HEMATOCRIT_RELAXATION = float(getattr(args, "hematocrit_relaxation", HEMATOCRIT_RELAXATION))
    HEMATOCRIT_QTOL_NL_MIN = float(getattr(args, "hematocrit_qtol_nl_min", HEMATOCRIT_QTOL_NL_MIN))
    HEMATOCRIT_HDTOL = float(getattr(args, "hematocrit_hdtol", HEMATOCRIT_HDTOL))
    target_counts_arg = getattr(args, "target_counts", None)
    if target_counts_arg:
        TARGET_TERMINAL_COUNTS = tuple(
            int(part.strip()) for part in str(target_counts_arg).split(",") if part.strip()
        )
    global KIRCHHOFF_DIAGNOSTICS, KIRCHHOFF_SOLVER, KIRCHHOFF_BC_MODE, KIRCHHOFF_VALIDATE_TREE
    global KIRCHHOFF_VALIDATE_SPARSE_SOLVER, KIRCHHOFF_SPARSE_SOLVER
    KIRCHHOFF_DIAGNOSTICS = str(getattr(args, "kirchhoff_diagnostics", "false")).lower() == "true"
    KIRCHHOFF_SOLVER = str(getattr(args, "kirchhoff_solver", KIRCHHOFF_SOLVER)).lower()
    KIRCHHOFF_BC_MODE = _normalize_kirchhoff_bc_mode(getattr(args, "kirchhoff_bc_mode", KIRCHHOFF_BC_MODE))
    KIRCHHOFF_VALIDATE_TREE = str(getattr(args, "kirchhoff_validate_tree", "false")).lower() == "true"
    KIRCHHOFF_VALIDATE_SPARSE_SOLVER = str(
        getattr(args, "kirchhoff_validation_sparse_solver", KIRCHHOFF_VALIDATE_SPARSE_SOLVER)
    ).lower()
    if KIRCHHOFF_SOLVER != "tree":
        KIRCHHOFF_SPARSE_SOLVER = KIRCHHOFF_SOLVER
    global TISSUE_STREAMING_ENABLED, TISSUE_STREAMING_MIN_POINTS
    global TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS, TISSUE_STREAMING_MAX_CHUNK_POINTS
    global TISSUE_STREAMING_PRUNE_BY_WINDOW, TISSUE_STREAMING_CHUNK_WORKERS
    global TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER
    TISSUE_STREAMING_ENABLED = str(getattr(args, "tissue_streaming", "true")).lower() == "true"
    TISSUE_STREAMING_MIN_POINTS = int(getattr(args, "tissue_streaming_min_points", TISSUE_STREAMING_MIN_POINTS))
    TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS = int(
        getattr(args, "tissue_streaming_target_candidate_slots", TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS)
    )
    TISSUE_STREAMING_MAX_CHUNK_POINTS = int(
        getattr(args, "tissue_streaming_max_chunk_points", TISSUE_STREAMING_MAX_CHUNK_POINTS)
    )
    TISSUE_STREAMING_CHUNK_WORKERS = int(
        getattr(args, "tissue_streaming_chunk_workers", TISSUE_STREAMING_CHUNK_WORKERS)
    )
    TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER = int(
        getattr(args, "tissue_streaming_numba_threads_per_worker", TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER)
    )
    TISSUE_STREAMING_PRUNE_BY_WINDOW = (
        str(getattr(args, "tissue_streaming_prune_by_window", "true")).lower() == "true"
    )
    global N_EQUAL_BIFURCATIONS
    if args.n_equal_bifurcations is not None and int(args.n_equal_bifurcations) >= 0:
        N_EQUAL_BIFURCATIONS = int(args.n_equal_bifurcations)
    else:
        N_EQUAL_BIFURCATIONS = None
    global CONCENTRATION_SOLVER
    CONCENTRATION_SOLVER = args.concentration_solver.lower()
    global PLOT_FLOW, PLOT_CONC_VESSELS, PLOT_CONC_VESSELS_POINTS, PLOT_VIABILITY_POINTS
    PLOT_FLOW = args.plot_flow.lower() == "true"
    PLOT_CONC_VESSELS = args.plot_concentration.lower() == "true"
    PLOT_CONC_VESSELS_POINTS = args.plot_concentration_points.lower() == "true"
    PLOT_VIABILITY_POINTS = args.plot_viability.lower() == "true"
    weighted_sampling = args.weighted_sampling.lower() == "true"
    ignore_collisions = args.ignore_collisions.lower() == "true"
    allow_inside_vessels = args.allow_inside_vessels.lower() == "true"
    if TISSUE_USE_NUMBA and not _HAVE_NUMBA:
        print("WARNING: Numba is not available; tissue kernel will run without numba.")
    analysis_fluids = get_analysis_fluids()
    ACTIVE_FLUID = analysis_fluids[0]
    solver_logged = False
    cache_enabled = USE_TREE_CACHE or SAVE_TREES
    cache_rows: list[dict[str, str]] = []
    cache_index_path: Path | None = None
    if cache_enabled:
        cache_index_path = _tree_cache_index_path()
        cache_rows = _load_tree_cache_index(cache_index_path)

    run_configs = [
        {
            "name": "baseline",
            "output_csv_name": OUTPUT_CSV_NAME,
            "target_counts": TARGET_TERMINAL_COUNTS,
            "trials_per_combo": TRIALS_PER_COMBO,
            "fluid": FLUID,
            "cube_side_lengths": CUBE_SIDE_LENGTHS,
            "dlp_angles": DLP_ANGLE_VALUES,
        },
        # {
        #     "name": "baseline",
        #     "output_csv_name": "redoblood_30_lowmid.csv",
        #     "target_counts": (1,2,3,4,5,8,10,15,25,35,50,75,100,150,250,350,500,750),
        #     "trials_per_combo": 3,
        #     "fluid": "blood",
        #     "cube_side_lengths": (1.),
        #     "dlp_angles": (30.),
        # },
        # {
        #     "name": "baseline",
        #     "output_csv_name": "redoblood_30_high.csv",
        #     "target_counts": (1000,2000,5000,10000),
        #     "trials_per_combo": 1,
        #     "fluid": "blood",
        #     "cube_side_lengths": (1.),
        #     "dlp_angles": (30.),
        # },
        # {
        #     "name": "baseline",
        #     "output_csv_name": "redoblood_45_low.csv",
        #     "target_counts": (1,2,3,4,5,8,10,15,25,35,50,75,100,150,250),
        #     "trials_per_combo": 3,
        #     "fluid": "blood",
        #     "cube_side_lengths": (1.),
        #     "dlp_angles": (45.),
        # },
        # {
        #     "name": "baseline",
        #     "output_csv_name": "redoblood_45_10konly.csv",
        #     "target_counts": (1,10000),
        #     "trials_per_combo": 1,
        #     "fluid": "blood",
        #     "cube_side_lengths": (1.),
        #     "dlp_angles": (45.),
        # },
        # {
        #     "name": "baseline",
        #     "output_csv_name": "blood_cubesizesweep_45.csv",
        #     "target_counts": TARGET_TERMINAL_COUNTS,
        #     "trials_per_combo": 1,
        #     "fluid": "blood",
        #     "cube_side_lengths": (1.,2.,3.,4.,5.,6.),
        #     "dlp_angles": (45.),
        # },
        # {
        #     "name": "baseline",
        #     "output_csv_name": "media_cubesizesweep_45.csv",
        #     "target_counts": TARGET_TERMINAL_COUNTS,
        #     "trials_per_combo": 1,
        #     "fluid": "water",
        #     "cube_side_lengths": (1.,2.,3.,4.),
        #     "dlp_angles": (45.),
        # },
    ]

    nondim_output_path: Path | None = None
    if NONDIMENSIONAL_NUMBERS:
        nondim_output_path = Path(__file__).with_name(NONDIMENSIONAL_CSV_NAME)
        if nondim_output_path.exists():
            nondim_output_path.unlink()
        append_nondimensional_row(nondim_output_path, row=None, write_header=True)
        print(
            f"Nondimensional mode: enabled. Writing dimensional/nondimensional metrics to "
            f"{nondim_output_path.name}."
        )

    for run in run_configs:
        _set_progress("run_start", run_name=run["name"], output_csv=run["output_csv_name"])
        side_lengths = _normalize_lengths(run.get("cube_side_lengths", CUBE_SIDE_LENGTHS))
        if BUILD_ON_PREVIOUS:
            if len(side_lengths) > 1:
                raise ValueError("BUILD_ON_PREVIOUS requires a single cube side length per run.")
            trials = int(run.get("trials_per_combo", TRIALS_PER_COMBO))
            if trials > 1:
                raise ValueError("BUILD_ON_PREVIOUS requires trials_per_combo == 1.")
        output_path = Path(__file__).with_name(run["output_csv_name"])
        if output_path.exists():
            output_path.unlink()

        append_row(output_path, row=None, write_header=True)

        for side_len in side_lengths:
            _set_progress("side_start", side_len=side_len, theta=None, target=None, trial=None, fluid=None)
            domain = build_domain(side_len)
            sample_points = sample_domain_points(domain, DISTANCE_SAMPLE_COUNT)

            checker_done = False
            if "dlp_angles" in run:
                angles = _normalize_angles(run["dlp_angles"])
            else:
                angles = _iter_theta_values()
            for min_theta in angles:
                _set_progress("theta_start", theta=min_theta, target=None, trial=None, fluid=None)
                dlp_enable = False  # DLP disabled in this build
                previous_tree = None
                previous_target = 0
                dnc_history: dict[str, object] = {}
                if PLOT_HISTOGRAM:
                    dnc_history = {"targets": [], "mu_sigma": []}
                violin_path = Path(__file__).with_name(VIOLIN_CSV_NAME)
                if VIOLIN_DNC:
                    _init_violin_points_csv(
                        violin_path,
                        target_counts=list(run["target_counts"]),
                        n_rows=int(sample_points.shape[0]),
                    )
                for target in run["target_counts"]:
                    _set_progress("target_start", target=target, trial=None, fluid=None)
                    trials = int(run["trials_per_combo"])
                    print(
                        f"\nRun '{run['name']}': build_fluid={BUILD_FLUID}, analysis={','.join(analysis_fluids)}, "
                        f"theta={min_theta} deg, target={target}, trials={trials}, side={side_len}, "
                        f"qin={','.join(f'{float(q):g}' for q in qin_target_values)} uL/min"
                    )
                    qin_trial_rows = {
                        float(qin): {
                            int(m): {fluid: [] for fluid in analysis_fluids}
                            for m in distance_sample_count_values
                        }
                        for qin in qin_target_values
                    }
                    need_dnc_fit = bool(PLOT_HISTOGRAM or SAVE_DNC_GAUSSIAN_FITS)
                    need_dnc_violin = bool(VIOLIN_DNC)
                    need_dnc_any = bool(need_dnc_fit or need_dnc_violin)
                    dnc_mu_trials: list[float] = []
                    dnc_sigma_trials: list[float] = []
                    dnc_violin_trials: list[np.ndarray] = []
                    for trial_idx in range(1, trials + 1):
                        _set_progress("trial_start", trial=trial_idx)
                        _dbg(f"trial {trial_idx}: start")
                        volume_scale = side_len * side_len * side_len
                        scale_factor = volume_scale if SCALE_NTERMS_BY_VOLUME else 1.0
                        effective_target = max(int(round(target * scale_factor)), 1)
                        q_scale = volume_scale if SCALE_Q_BY_VOLUME else 1.0
                        build_qin_target = float(qin_target_values[0])
                        QIN_TARGET = build_qin_target
                        q_inlet_cm3_s = build_qin_target * 1e-3 / 60.0 * q_scale
                        per_seg_flow = q_inlet_cm3_s / (effective_target + 1)
                        terminal_flow_override = per_seg_flow if SCALE_Q_BY_VOLUME else None
                        terminal_flow_cm3_s = per_seg_flow
                        base_config = None
                        full_config = None
                        config_id = None
                        legacy_config_id = None
                        if cache_enabled:
                            base_config, full_config, config_id, legacy_config_id = _build_tree_cache_config(
                                target_raw=target,
                                target_terminals=effective_target,
                                side_length=side_len,
                                domain_seed=getattr(domain, "random_seed", None),
                                q_inlet_cm3_s=q_inlet_cm3_s,
                                terminal_flow_cm3_s=terminal_flow_cm3_s,
                                dlp_enable=dlp_enable,
                                min_theta=min_theta,
                                weighted_sampling=weighted_sampling,
                                ignore_collisions=ignore_collisions,
                                allow_inside_vessels=allow_inside_vessels,
                                n_closest_vessels=args.n_closest_vessels,
                                n_points=args.n_points,
                            )
                        t_tree_build_start = perf_counter()
                        tree = None
                        cache_exact_hit = False
                        cache_status = "miss"
                        cache_note = ""
                        cached = None
                        lower = None
                        if USE_TREE_CACHE and config_id is not None:
                            cached = _find_cached_tree(cache_rows, config_id, effective_target)
                            if cached is None and legacy_config_id and legacy_config_id != config_id:
                                cached = _find_cached_tree(cache_rows, legacy_config_id, effective_target)
                            if cached is None:
                                lower = _find_cached_tree_lower(cache_rows, config_id, effective_target)
                        if cached is not None:
                            _, tree_path = cached
                            # Avoid loading the cached Domain/mesh (can be extremely large) unless explicitly needed.
                            # We'll reuse the already-built run domain when we need the boundary/mesh (growth/plots).
                            need_tree_domain = bool(
                                CHECKER_PLOT
                                or PLOT_FLOW
                                or PLOT_CONC_VESSELS
                                or PLOT_CONC_VESSELS_POINTS
                                or PLOT_VIABILITY_POINTS
                            )
                            tree = _load_tree_cached_fast(
                                Path(tree_path),
                                data_dtype=TREE_DATA_DTYPE,
                                index_dtype=TREE_INDEX_DTYPE,
                                analysis_only=not need_tree_domain,
                            )
                            if need_tree_domain:
                                _ensure_tree_domain(tree, domain)
                            _prepare_loaded_tree(tree)
                            _sync_loaded_tree_params(
                                tree,
                                side_length=side_len,
                                terminal_flow_override=terminal_flow_override,
                            )
                            cache_exact_hit = True
                            cache_status = "hit"
                        else:
                            prev_ok = (
                                BUILD_ON_PREVIOUS
                                and previous_tree is not None
                                and not bool(getattr(previous_tree, "_analysis_only_load", False))
                            )
                            if prev_ok and effective_target < previous_target:
                                raise ValueError("BUILD_ON_PREVIOUS requires non-decreasing target counts.")
                            use_previous = False
                            use_lower = False
                            cached_target = None
                            cached_path = None
                            if lower is not None:
                                _, cached_target, cached_path = lower
                            if prev_ok:
                                if cached_target is not None:
                                    prev_gap = effective_target - previous_target
                                    cache_gap = effective_target - cached_target
                                    if cache_gap < prev_gap:
                                        use_lower = True
                                    else:
                                        use_previous = True
                                else:
                                    use_previous = True
                            elif cached_target is not None:
                                use_lower = True

                            if use_lower and cached_path is not None and cached_target is not None:
                                vessels_to_add = effective_target - cached_target
                                tree = _load_tree_cached_fast(
                                    Path(cached_path),
                                    data_dtype=TREE_DATA_DTYPE,
                                    index_dtype=TREE_INDEX_DTYPE,
                                )
                                _prepare_loaded_tree(tree)
                                _sync_loaded_tree_params(
                                    tree,
                                    side_length=side_len,
                                    terminal_flow_override=terminal_flow_override,
                                )
                                if vessels_to_add > 0:
                                    _ensure_tree_domain(tree, domain)
                                    tree.n_add(
                                        vessels_to_add,
                                        n_closest_vessels=args.n_closest_vessels,
                                        n_points=args.n_points,
                                        use_random_int=not weighted_sampling,
                                        ignore_collisions=ignore_collisions,
                                        allow_inside_vessels=allow_inside_vessels,
                                        debug_add_vessel=DEBUG_ADD_VESSEL,
                                    )
                                cache_status = "extend"
                            elif use_previous:
                                tree = previous_tree
                                _sync_loaded_tree_params(
                                    tree,
                                    side_length=side_len,
                                    terminal_flow_override=terminal_flow_override,
                                )
                                vessels_to_add = effective_target - previous_target
                                if vessels_to_add > 0:
                                    _ensure_tree_domain(tree, domain)
                                    tree.n_add(
                                        vessels_to_add,
                                        n_closest_vessels=args.n_closest_vessels,
                                        n_points=args.n_points,
                                        use_random_int=not weighted_sampling,
                                        ignore_collisions=ignore_collisions,
                                        allow_inside_vessels=allow_inside_vessels,
                                        debug_add_vessel=DEBUG_ADD_VESSEL,
                                    )
                                cache_note = "previous"

                            if tree is None:
                                tree = grow_tree(
                                    domain,
                                    effective_target,
                                    dlp_enable=dlp_enable,
                                    min_theta=min_theta,
                                    side_length=side_len,
                                    terminal_flow_override=terminal_flow_override,
                                    fluid=BUILD_FLUID,
                                    n_closest_vessels=args.n_closest_vessels,
                                    n_points=args.n_points,
                                    weighted_sampling=weighted_sampling,
                                    ignore_collisions=ignore_collisions,
                                )
                                cache_note = "build"
                        _apply_run_tree_params(tree, side_len, terminal_flow_override)
                        if USE_TREE_CACHE:
                            suffix = f" ({cache_note})" if cache_note else ""
                            print(f"  Cache: {cache_status}{suffix}")
                        if SAVE_TREES and not cache_exact_hit:
                            t_tree_save_start = perf_counter()
                            if tree is None:
                                raise RuntimeError("Tree build failed; nothing to save.")
                            if cache_index_path is None or base_config is None or full_config is None or config_id is None:
                                raise RuntimeError("Tree cache config missing; cannot save.")
                            set_tree_fluid(tree, BUILD_FLUID)
                            domain_obj = getattr(tree, "domain", None)
                            if domain_obj is None:
                                raise ValueError("Tree has no domain; cannot save to cache.")
                            if domain_obj.mesh is None or getattr(domain_obj, "volume", None) is None:
                                domain_obj.build()
                            if getattr(domain_obj, "boundary", None) is None:
                                domain_obj.get_boundary()
                            cache_dir = _tree_cache_dir()
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            tree_filename = f"tree_{config_id}_t{effective_target}.tree.npz"
                            tree_path = cache_dir / tree_filename
                            tree.save(
                                str(tree_path),
                                include_domain=True,
                                domain_save_kwargs={"include_mesh": True, "include_boundary": True},
                            )
                            cache_row = _tree_cache_row(
                                config_id=config_id,
                                tree_path=tree_path,
                                base_config=base_config,
                                full_config=full_config,
                                target_raw=target,
                                target_terminals=effective_target,
                            )
                            _append_tree_cache_row(cache_index_path, cache_row)
                            cache_rows.append(cache_row)
                            print(f"  Tree save time: {_fmt_seconds(perf_counter() - t_tree_save_start)}")
                        tree_build_elapsed = perf_counter() - t_tree_build_start
                        if cache_status == "hit":
                            print(f"  Tree load time (cache hit): {_fmt_seconds(tree_build_elapsed)}")
                        elif cache_status == "extend":
                            print(f"  Tree extend time (cache lower): {_fmt_seconds(tree_build_elapsed)}")
                        elif cache_status == "miss" and cache_note == "build":
                            print(f"  Tree build time (cache miss): {_fmt_seconds(tree_build_elapsed)}")
                        elif cache_status == "miss" and cache_note == "previous":
                            print(f"  Tree extend time (previous tree): {_fmt_seconds(tree_build_elapsed)}")
                        else:
                            print(f"  Tree build/load time: {_fmt_seconds(tree_build_elapsed)}")
                        _set_progress("tree_ready", trial=trial_idx)
                        metrics_by_qin: dict[float, dict[int, dict[str, dict]]] = {}
                        try:
                            if CHECKER_PLOT and not checker_done and target == 250:
                                checker_done = True
                                print("Checker: displaying first 250-terminal tree for inspection...")
                                ACTIVE_FLUID = analysis_fluids[0]
                                plot_checker_tree(
                                    tree,
                                    sample_points,
                                    inlet_flow_cm3_s=q_inlet_cm3_s,
                                    fluid=analysis_fluids[0],
                                )
                            if need_dnc_any:
                                seg_count = int(getattr(tree, "segment_count", 0))
                                data = np.asarray(tree.data[:seg_count]) if seg_count > 0 else np.empty((0, 0), dtype=float)
                                starts = data[:, 0:3] if data.size else np.empty((0, 3), dtype=float)
                                ends = data[:, 3:6] if data.size else np.empty((0, 3), dtype=float)
                                radii = data[:, 21] if (data.size and data.shape[1] > 21) else np.zeros((starts.shape[0],), dtype=float)
                                print("Starting DNC Calculations")
                                dnc_values_cm = compute_distance_to_nearest_channel(sample_points, starts, ends, radii)
                                print("Ending DNC Calculations")
                                dnc_values_um = dnc_values_cm * CM_TO_UM
                                finite = dnc_values_um[np.isfinite(dnc_values_um)]
                                if need_dnc_fit:
                                    mu, sigma = _fit_truncnorm_mu_sigma_lower0(finite)
                                    dnc_mu_trials.append(float(mu))
                                    dnc_sigma_trials.append(float(sigma))
                                if need_dnc_violin:
                                    # Save per-point DNC (one value per sample point) for violin plots.
                                    dnc_violin_trials.append(np.asarray(dnc_values_um, dtype=float).reshape(-1))
                            sample_points_by_count = {
                                int(m): sample_points[: int(m)]
                                for m in distance_sample_count_values
                            }
                            max_sample_count = int(DISTANCE_SAMPLE_COUNT)
                            for qin_target_value in qin_target_values:
                                qin_key = float(qin_target_value)
                                QIN_TARGET = qin_key
                                q_inlet_cm3_s_analysis = qin_key * 1e-3 / 60.0 * q_scale
                                per_seg_flow_analysis = q_inlet_cm3_s_analysis / (effective_target + 1)
                                terminal_flow_override_analysis = (
                                    per_seg_flow_analysis if SCALE_Q_BY_VOLUME else None
                                )
                                _apply_run_tree_params(tree, side_len, terminal_flow_override_analysis)
                                metrics_by_sample = {
                                    int(m): {} for m in distance_sample_count_values
                                }
                                metrics_by_qin[qin_key] = metrics_by_sample
                                print(f"  Flowrate analysis: qin={qin_key:g} uL/min")
                                for fluid in analysis_fluids:
                                    _set_progress("analyze", fluid=fluid)
                                    ACTIVE_FLUID = fluid
                                    if not solver_logged:
                                        solver_kind = (
                                            "tree"
                                            if str(KIRCHHOFF_SOLVER).strip().lower() in ("tree", "tree_neumann", "tree-current-bc")
                                            else ("sparse" if _HAVE_SCIPY_SPARSE else "dense")
                                        )
                                        print(
                                            f"Analysis solver: {solver_kind} Kirchhoff "
                                            f"(bc={_normalize_kirchhoff_bc_mode()}) "
                                            f"concentration={_resolve_concentration_solver(CONCENTRATION_SOLVER)} "
                                            f"finite_radius_o2={FINITE_RADIUS_O2_TERMS} "
                                            f"lumen_wall={LUMEN_WALL_CLOSURE}"
                                        )
                                        if LUMEN_WALL_CLOSURE == "graetz":
                                            print(
                                                f"  Graetz closure: profile={GRAETZ_VELOCITY_PROFILE} "
                                                f"n_radial={GRAETZ_N_RADIAL} n_modes={GRAETZ_N_MODES} "
                                                f"max_fp_iters={GRAETZ_MAX_FP_ITERS} "
                                                f"D_lumen={_lumen_diffusivity_cm2_s_for_fluid(ACTIVE_FLUID):.4g} cm^2/s"
                                            )
                                        solver_logged = True
                                    plot_any = PLOT_FLOW or PLOT_CONC_VESSELS or PLOT_CONC_VESSELS_POINTS or PLOT_VIABILITY_POINTS
                                    max_points = sample_points_by_count[max_sample_count]
                                    t_tissue_cache_start = perf_counter()
                                    tissue_cache = build_tissue_cache_from_tree(tree, max_points)
                                    tissue_cache_build_elapsed = perf_counter() - t_tissue_cache_start
                                    print(
                                        f"  Tissue cache build M={max_sample_count:g}: "
                                        f"{_fmt_seconds(tissue_cache_build_elapsed)}"
                                    )
                                    if CEXT_TRACE_CALLBACK is not None and bool(CEXT_TRACE_TISSUE_ENABLED):
                                        globals()["CEXT_TRACE_TISSUE_POINTS"] = max_points
                                        globals()["CEXT_TRACE_TISSUE_CACHE"] = tissue_cache
                                    print(f"Starting summarize_tree. Fluid: {ACTIVE_FLUID}")
                                    summary = summarize_tree(
                                        tree,
                                        max_points,
                                        effective_target,
                                        side_length=side_len,
                                        fluid=fluid,
                                        inlet_flow_cm3_s=q_inlet_cm3_s_analysis,
                                        tissue_cache=tissue_cache,
                                        return_details=True,
                                    )
                                    print(f"Ending summarize_tree. Fluid: {ACTIVE_FLUID}")
                                    base_metrics, details = summary
                                    base_core_assembly_s = max(
                                        float(base_metrics.get("t_assembly_s", 0.0))
                                        - float(base_metrics.get("t_tissue_geometry_s", 0.0) or 0.0),
                                        0.0,
                                    )
                                    base_metrics["t_load_s"] = float(tree_build_elapsed)
                                    base_metrics["t_assembly_s"] = (
                                        base_core_assembly_s
                                        + float(tissue_cache_build_elapsed)
                                        + float(base_metrics.get("t_tissue_geometry_s", 0.0) or 0.0)
                                    )
                                    metrics_by_sample[max_sample_count][fluid] = base_metrics
                                    if not (CEXT_TRACE_CALLBACK is not None and bool(CEXT_TRACE_TISSUE_ENABLED)):
                                        del tissue_cache
                                    if plot_any:
                                        domain_obj = getattr(tree, "domain", None)
                                        if domain_obj is None:
                                            raise RuntimeError("Tree has no domain; cannot plot.")
                                        if PLOT_FLOW:
                                            _plot_flow(
                                                domain_obj,
                                                details["starts"],
                                                details["ends"],
                                                details["radii"],
                                                details["flows"],
                                            )
                                        if PLOT_CONC_VESSELS:
                                            _plot_concentration(
                                                domain_obj,
                                                details["starts"],
                                                details["ends"],
                                                details["radii"],
                                                details["cin"],
                                                details["cout"],
                                                details["tissue_points"],
                                                details["tissue_values"],
                                                show_points=False,
                                                inlet_concentration=details["inlet_concentration"],
                                            )
                                        if PLOT_CONC_VESSELS_POINTS:
                                            _plot_concentration(
                                                domain_obj,
                                                details["starts"],
                                                details["ends"],
                                                details["radii"],
                                                details["cin"],
                                                details["cout"],
                                                details["tissue_points"],
                                                details["tissue_values"],
                                                show_points=True,
                                                inlet_concentration=details["inlet_concentration"],
                                            )
                                        if PLOT_VIABILITY_POINTS:
                                            _plot_viability_points(
                                                domain_obj,
                                                details["starts"],
                                                details["ends"],
                                                details["radii"],
                                                details["cin"],
                                                details["cout"],
                                                details["tissue_points"],
                                                details["tissue_values"],
                                                inlet_concentration=details["inlet_concentration"],
                                            )
                                    for sample_count in distance_sample_count_values:
                                        sample_count = int(sample_count)
                                        if sample_count == max_sample_count:
                                            continue
                                        points_m = sample_points_by_count[sample_count]
                                        t_tissue_cache_start = perf_counter()
                                        tissue_cache_m = build_tissue_cache_from_tree(tree, points_m)
                                        tissue_cache_build_elapsed_m = perf_counter() - t_tissue_cache_start
                                        print(
                                            f"  Tissue cache build M={sample_count:g}: "
                                            f"{_fmt_seconds(tissue_cache_build_elapsed_m)}"
                                        )
                                        metrics_by_sample[sample_count][fluid] = summarize_tissue_only_from_details(
                                            base_metrics,
                                            details,
                                            points_m,
                                            tissue_cache=tissue_cache_m,
                                            tissue_cache_build_elapsed=tissue_cache_build_elapsed_m,
                                            base_core_assembly_s=base_core_assembly_s,
                                        )
                                        del tissue_cache_m
                        finally:
                            if not BUILD_ON_PREVIOUS:
                                del tree
                        for qin_target_value in qin_target_values:
                            qin_key = float(qin_target_value)
                            metrics_by_sample = metrics_by_qin.get(qin_key, {})
                            for sample_count in distance_sample_count_values:
                                sample_count = int(sample_count)
                                metrics = metrics_by_sample.get(sample_count, {})
                                for fluid in analysis_fluids:
                                    if metrics.get(fluid) is not None:
                                        qin_trial_rows[qin_key][sample_count][fluid].append(metrics[fluid])
                                        _dbg(
                                            f"trial {trial_idx} qin={qin_key:g} M={sample_count:g} ({fluid}): "
                                            f"metrics keys={list(metrics[fluid].keys())}"
                                        )
                                        print(
                                            f"  trial {trial_idx} qin={qin_key:g} M={sample_count:g} ({fluid}): "
                                            f"segs={metrics[fluid]['total_segments']} "
                                            f"dRnet={metrics[fluid]['dRnet']:.4e} "
                                            f"C_LQ={metrics[fluid]['C_LQ_over_Cmax']:.4f} "
                                            f"C_tiss={metrics[fluid]['C_tiss_over_Cmax']:.4f}"
                                        )
                                        print(
                                            f"    timings: load={_fmt_seconds(metrics[fluid].get('t_load_s'))} "
                                            f"assembly={_fmt_seconds(metrics[fluid].get('t_assembly_s'))} "
                                            f"kirchhoff={_fmt_seconds(metrics[fluid].get('t_kirchhoff_s'))} "
                                            f"conc={_fmt_seconds(metrics[fluid].get('t_concentration_s'))} "
                                            f"tissue_greens={_fmt_seconds(metrics[fluid].get('t_tissue_s'))}"
                                        )
                                        if str(metrics[fluid].get("concentration_solver", "")).startswith("topdown_ext"):
                                            print(
                                                f"    cext timing ({metrics[fluid].get('concentration_solver')} / "
                                                f"{metrics[fluid].get('t_cext_backend', metrics[fluid].get('concentration_solver'))}): "
                                                f"total={_fmt_seconds(metrics[fluid].get('t_cext_total_s'))} "
                                                f"iters={int(metrics[fluid].get('cext_outer_iterations_completed', 0) or 0)} "
                                                f"delta={float(metrics[fluid].get('cext_max_delta_last', 0.0) or 0.0):.3e} "
                                                f"rel={float(metrics[fluid].get('cext_rel_residual_last', 0.0) or 0.0):.3e}"
                                            )
                                        print(
                                            f"    tissue timing ({metrics[fluid].get('t_tissue_backend', 'unknown')}): "
                                            f"geometry/refine/cache={_fmt_seconds(metrics[fluid].get('t_tissue_geometry_s'))} "
                                            f"oxygen_greens={_fmt_seconds(metrics[fluid].get('t_tissue_oxygen_s'))} "
                                            f"total={_fmt_seconds(metrics[fluid].get('t_tissue_total_s'))}"
                                        )
                                    else:
                                        print(
                                            f"  trial {trial_idx} qin={qin_key:g} M={sample_count:g} ({fluid}): "
                                            "FAILED (metrics None)"
                                        )
                    dnc_mu = float("nan")
                    dnc_sigma = float("nan")
                    if need_dnc_fit and dnc_mu_trials and dnc_sigma_trials:
                        dnc_mu = float(np.nanmean(np.asarray(dnc_mu_trials, dtype=float)))
                        dnc_sigma = float(np.nanmean(np.asarray(dnc_sigma_trials, dtype=float)))
                    if PLOT_HISTOGRAM and need_dnc_fit:
                        dnc_history["targets"].append(int(effective_target))
                        dnc_history["mu_sigma"].append((dnc_mu, dnc_sigma))
                    if VIOLIN_DNC:
                        col_idx = list(run["target_counts"]).index(target)
                        if dnc_violin_trials:
                            values = np.nanmean(np.vstack(dnc_violin_trials), axis=0)
                        else:
                            values = np.full((int(sample_points.shape[0]),), np.nan, dtype=float)
                        min_keep = float(VIOLIN_MIN_KEEP_UM)
                        if np.isfinite(min_keep):
                            values = np.where(values <= min_keep, np.nan, values)
                        keep = values[np.isfinite(values)]
                        if keep.size:
                            keep = np.sort(keep)
                        sorted_values = np.full_like(values, np.nan, dtype=float)
                        sorted_values[: keep.size] = keep
                        _update_violin_points_csv(
                            violin_path,
                            target_counts=list(run["target_counts"]),
                            col_idx=col_idx,
                            values_um=sorted_values,
                        )
                    if BUILD_ON_PREVIOUS:
                        previous_tree = tree
                        previous_target = effective_target
                    any_rows = False
                    for qin_target_value in qin_target_values:
                        qin_key = float(qin_target_value)
                        trial_rows_by_sample = qin_trial_rows[qin_key]
                        for sample_count in distance_sample_count_values:
                            sample_count = int(sample_count)
                            trial_rows = trial_rows_by_sample[sample_count]
                            for fluid in analysis_fluids:
                                if not trial_rows[fluid]:
                                    _dbg(
                                        f"No trial rows collected for qin={qin_key:g} "
                                        f"M={sample_count:g} {fluid}; skipping aggregation."
                                    )
                                    continue
                                any_rows = True
                                aggregated = aggregate_trials(trial_rows[fluid])
                                aggregated["fluid"] = fluid
                                aggregated.update(_pi_phi_gamma_columns(aggregated))
                                if SAVE_DNC_GAUSSIAN_FITS:
                                    aggregated["dnc_gaussian_mu_um"] = dnc_mu
                                    aggregated["dnc_gaussian_sigma_um"] = dnc_sigma
                                else:
                                    aggregated["dnc_gaussian_mu_um"] = float("nan")
                                    aggregated["dnc_gaussian_sigma_um"] = float("nan")
                                _dbg(
                                    f"Aggregated keys qin={qin_key:g} M={sample_count:g} "
                                    f"({fluid})={list(aggregated.keys())}"
                                )
                                append_row(output_path, aggregated)
                                if NONDIMENSIONAL_NUMBERS and nondim_output_path is not None:
                                    append_nondimensional_row(
                                        nondim_output_path,
                                        _nondimensional_numbers_row(aggregated, run_name=str(run["name"])),
                                    )
                                print(
                                    f"  -> averages qin={qin_key:g} M={sample_count:g} ({fluid}): "
                                    f"vol={aggregated['total_volume_mean']:.4e}+/-{aggregated['total_volume_std']:.1e} "
                                    f"Qmin/Qin={aggregated['Qmin_over_Qinlet_mean']:.4f}+/-{aggregated['Qmin_over_Qinlet_std']:.2e} "
                                    f"C_LQ={aggregated['C_LQ_over_Cmax_mean']:.4f}+/-{aggregated['C_LQ_over_Cmax_std']:.2e} "
                                    f"C_tiss={aggregated['C_tiss_over_Cmax_mean']:.4f}+/-{aggregated['C_tiss_over_Cmax_std']:.2e}"
                                )
                    if not any_rows:
                        _dbg("No trial rows collected; skipping aggregation.")
                    _set_progress("target_done", trial=None, fluid=None)
                if PLOT_HISTOGRAM:
                    mu_sigma = dnc_history.get("mu_sigma") or []
                    targets = dnc_history.get("targets") or []
                    if mu_sigma and targets:
                        _plot_dnc_gaussians(
                            mu_sigma=mu_sigma,
                            target_terminals=targets,
                            title=f"DNC Gaussian fits (geometry) | run={run['name']} | side={side_len} | theta={min_theta}",
                        )

    print(f"\nSaved metrics to: {output_path}")
    if NONDIMENSIONAL_NUMBERS and nondim_output_path is not None:
        print(f"Saved nondimensional metrics to: {nondim_output_path}")


def _run_main_with_profile(profile_out: str) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    finally:
        profiler.disable()
        profiler.dump_stats(profile_out)
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        print(f"Profile written to: {profile_out}")
        stats.print_stats(40)


def _run_main_with_line_profile(profile_out: str) -> None:
    try:
        import line_profiler  # type: ignore
    except Exception:
        print("line_profiler not installed; install with `pip install line_profiler` or use `kernprof`.")
        main()
        return

    profiler = line_profiler.LineProfiler()

    def _maybe_add(name: str) -> None:
        fn = globals().get(name)
        if callable(fn):
            profiler.add_function(fn)

    for fn_name in (
        "compute_tissue_samples_greens",
        "solve_tree_greens",
        "compute_concentration_profiles",
        "solve_network_concentrations",
        "_solve_channel_concentrations",
        "solve_kirchhoff",
        "build_tissue_cache_from_tree",
    ):
        _maybe_add(fn_name)

    profiler.enable_by_count()
    try:
        main()
    finally:
        profiler.disable_by_count()
        if profile_out:
            with open(profile_out, "w", encoding="ascii") as handle:
                profiler.print_stats(stream=handle)
            print(f"Line profile written to: {profile_out}")
        else:
            profiler.print_stats()


if __name__ == "__main__":
    try:
        args = parse_args()
        if str(getattr(args, 'line_profile', 'false')).lower() == 'true':
            _run_main_with_line_profile(getattr(args, 'line_profile_out', 'TissueSim_line_profile.txt'))
        elif str(getattr(args, 'profile', 'false')).lower() == 'true':
            _run_main_with_profile(getattr(args, 'profile_out', 'TissueSim_profile.prof'))
        else:
            main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        if code == 0:
            raise
        print("\nFATAL ERROR: unhandled exception in TissueSim.")
        print(f"Last progress: {_progress_snapshot()}")
        _write_crash_log(exc)
        traceback.print_exc()
        raise
    except BaseException as exc:
        print("\nFATAL ERROR: unhandled exception in TissueSim.")
        print(f"Last progress: {_progress_snapshot()}")
        _write_crash_log(exc)
        traceback.print_exc()
        raise
