"""Mutable solver settings shared by CASCADE's numerical subsystems.

A run's validated configuration is applied here before solver dispatch. Keeping
the active values in one module gives CPU, GPU, CLI, and GUI entry points the
same settings for the duration of a simulation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from cascade.runtime._dependencies import (  # noqa: F401
    Domain,
    Tree,
    _HAVE_CUPY,
    _HAVE_NUMBA,
    _HAVE_SCIPY,
    _HAVE_SCIPY_NDIMAGE,
    _HAVE_SCIPY_SPARSE,
    _HAVE_SCIPY_SPATIAL,
    _IMPORT_ERROR,
    _compute_cext_batch_numba,
    _cp,
    _scipy_linalg,
    _scipy_ndimage,
    get_num_threads,
    set_num_threads,
)

# Expensive default objects are created on first use. Declaring the slots here
# keeps direct subsystem imports deterministic.
_DEFAULT_TREE_PARAMS: dict | None = None

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


TARGET_TERMINAL_COUNTS: tuple[int, ...] = (
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    8,
    10,
    15,
    25,
    35,
    50,
    75,
    100,
    150,
    250,
    350,
    500,
    750,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    300000,
    400000,
    500000,
    650000,
    1000000,
    2000000,
    3000000,
    4000000,
    5000000,
)
ROOT_LOCATION = np.array(
    [[0.49, -0.49, -0.49]]
)  # scaled by cube side length at runtime
ROOT_DIR = np.array([[-0.49, 0.49, 0.49]])
OUTPUT_CSV_NAME = "cascade_results.csv"
NONDIMENSIONAL_NUMBERS = False
NONDIMENSIONAL_CSV_NAME = "cascade_dimensionless_results.csv"
FLUID = "both"  # analysis mode: "water", "blood", or "both"
BUILD_FLUID = "blood"
ACTIVE_FLUID = FLUID
CUSTOM_FLUID_DENSITY_G_CM3 = 1.0
CUSTOM_FLUID_DYNAMIC_VISCOSITY_CP = 1.0

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
NEAREST_TISSUE_VESSELS = 250  ### was 1000 !!!
WINDOW_FACTOR = 6  ### was 8 !!!
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
TISSUE_STREAMING_CHUNK_WORKERS = max(os.cpu_count() - 2 or 1, 1)
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
CEXT_LAMBDA_SOURCE = (
    os.environ.get("SVV_CEXT_LAMBDA_SOURCE", "lambda_t").strip().lower()
)
CEXT_WINDOW_FACTOR = WINDOW_FACTOR
CEXT_VESS_COUPLING_MAX_ITER = 1
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
CEXT_HYBRID_BG_NEAR_RADIUS_MULT = 0.0
CEXT_HYBRID_BG_VCYCLES = 2
CEXT_HYBRID_BG_ASSIGNMENT = "tsc"  # "cic" or "tsc"
CEXT_HYBRID_BG_SOLVER = "auto"  # "auto", "fft", or "jacobi"
CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING = False
# Best 500k diagnostic FFT path: dynamic quantile lambda bins, TSC source deposit,
# CIC target sampling, finite-radius O2 correction, and grid-consistent self-subtraction.
CEXT_HYBRID_BG_MODE = os.environ.get("SVV_CEXT_HYBRID_BG_MODE", "fft").strip().lower()
CEXT_HYBRID_FFT_QUANTILE_BINS = str(
    os.environ.get("SVV_CEXT_HYBRID_FFT_QUANTILE_BINS", "true")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_O2_CORRECTION = str(
    os.environ.get("SVV_CEXT_HYBRID_FFT_O2_CORRECTION", "true")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_SELF_SUBTRACT = str(
    os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUBTRACT", "true")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING = (
    os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING", "cic")
    .strip()
    .lower()
)
CEXT_HYBRID_FFT_SELF_SUB_SCALE = float(
    os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_SCALE", "1.0")
)
CEXT_HYBRID_FFT_BIN_EPOCH_CACHE = str(
    os.environ.get("SVV_CEXT_HYBRID_FFT_BIN_EPOCH_CACHE", "true")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_RESPONSE_BATCHED = str(
    os.environ.get("SVV_CEXT_HYBRID_FFT_RESPONSE_BATCHED", "false")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_O2_FUSED_IFFT = str(
    os.environ.get("SVV_CEXT_HYBRID_FFT_O2_FUSED_IFFT", "true")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_FFT_SELF_SUB_FUSED = str(
    os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_FUSED", "false")
).strip().lower() in ("1", "true", "yes", "on")
try:
    CEXT_HYBRID_FFT_O2_MOMENT_BATCH = max(
        int(os.environ.get("SVV_CEXT_HYBRID_FFT_O2_MOMENT_BATCH", "1")), 1
    )
except ValueError:
    CEXT_HYBRID_FFT_O2_MOMENT_BATCH = 1
CEXT_HYBRID_GPU_ITERATION_CACHE = str(
    os.environ.get("SVV_CEXT_HYBRID_GPU_ITERATION_CACHE", "false")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_GPU_RUNTIME_WEIGHTS = str(
    os.environ.get("SVV_CEXT_HYBRID_GPU_RUNTIME_WEIGHTS", "true")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_GPU_RUNTIME_STENCIL = str(
    os.environ.get("SVV_CEXT_HYBRID_GPU_RUNTIME_STENCIL", "false")
).strip().lower() in ("1", "true", "yes", "on")
CEXT_HYBRID_GPU_RUNTIME_MOMENTS = str(
    os.environ.get("SVV_CEXT_HYBRID_GPU_RUNTIME_MOMENTS", "false")
).strip().lower() in ("1", "true", "yes", "on")
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
CEXT_PRECOMPUTE_WORKERS = max(1, os.cpu_count() - 2 or 1)
CEXT_LOCAL_EXCLUDE_HOPS = 2
VESS_CONC_FLOOR = 1.0e-12
CEXT_FLOAT_DTYPE = np.float32
CEXT_INDEX_DTYPE = np.int32

# Expanded oxygen transport controls.
# c_iv_gl remains the cup/bulk lumen concentration for compatibility with the
# original topdown solver.  c_wall_gl is the true wall concentration used by
# q_line and by the finite-radius dipole source coefficient.
FINITE_RADIUS_O2_TERMS = (
    str(os.environ.get("SVV_FINITE_RADIUS_O2_TERMS", "both")).strip().lower()
)
LUMEN_WALL_CLOSURE = (
    str(os.environ.get("SVV_LUMEN_WALL_CLOSURE", "graetz")).strip().lower()
)
GRAETZ_N_RADIAL = 8
GRAETZ_N_MODES = 4
GRAETZ_MAX_FP_ITERS = int(os.environ.get("SVV_GRAETZ_MAX_FP_ITERS", "4"))
GRAETZ_FP_TOL = float(os.environ.get("SVV_GRAETZ_FP_TOL", "1e-5"))
GRAETZ_VELOCITY_PROFILE = (
    str(os.environ.get("SVV_GRAETZ_VELOCITY_PROFILE", "poiseuille")).strip().lower()
)
GRAETZ_BI_CACHE_PER_DECADE = 128
GRAETZ_MIN_BI = 1.0e-8
GRAETZ_MAX_BI = 1.0e6
GRAETZ_DEBUG_DIAGNOSTICS = str(
    os.environ.get("SVV_GRAETZ_DEBUG_DIAGNOSTICS", "0")
).strip().lower() in ("1", "true", "yes", "on")
LUMEN_DIFFUSIVITY_WATER_CM2_S = float(
    os.environ.get("SVV_LUMEN_DIFFUSIVITY_WATER_CM2_S", "3.2e-5")
)
LUMEN_DIFFUSIVITY_BLOOD_CM2_S = float(
    os.environ.get("SVV_LUMEN_DIFFUSIVITY_BLOOD_CM2_S", "2.41e-5")
)
LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S = (
    float(os.environ["SVV_LUMEN_DIFFUSIVITY_CM2_S"])
    if "SVV_LUMEN_DIFFUSIVITY_CM2_S" in os.environ
    else None
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


TREE_DATA_DTYPE_STR = _normalize_tree_float_dtype(
    os.environ.get("SVV_TREE_DATA_DTYPE", "float64")
)
TREE_INDEX_DTYPE_STR = _normalize_tree_int_dtype(
    os.environ.get("SVV_TREE_INDEX_DTYPE", "int64")
)
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
K_M_MM = 0.0069
AXIAL_BLOOD_STEPS = 5
OMEGA = 0.7

CONCENTRATION_SOLVER = (
    "network_ext"  # topdown/network, with optional direct or FFT Cext coupling
)
HEMATOCRIT_MODEL = "pries_secomb"  # "uniform_tube" or "pries_secomb".
HEMATOCRIT_FLOW_ITERATIONS = 2
HEMATOCRIT_RELAXATION = 1.0
HEMATOCRIT_QTOL_NL_MIN = 1.0e-3
HEMATOCRIT_HDTOL = 1.0e-3
HEMATOCRIT_MIN = 0.0
HEMATOCRIT_MAX = 0.95
HEMATOCRIT_DIAGNOSTICS = True
PRIES_SECOMB_BIFPAR_1 = 0.964
PRIES_SECOMB_BIFPAR_2 = 6.98
PRIES_SECOMB_BIFPAR_3 = -13.29
PRIES_SECOMB_CPAR_1 = 0.80
PRIES_SECOMB_CPAR_2 = -0.075
PRIES_SECOMB_CPAR_3 = -11.0
PRIES_SECOMB_CPAR_4 = 12.0
PRIES_SECOMB_VISCPAR_1 = 6.0
PRIES_SECOMB_VISCPAR_2 = -0.085
PRIES_SECOMB_VISCPAR_3 = 3.2
PRIES_SECOMB_VISCPAR_4 = -2.44
PRIES_SECOMB_VISCPAR_5 = -0.06
PRIES_SECOMB_VISCPAR_6 = 0.645
PRIES_SECOMB_OPTW_UM = 1.1
PRIES_SECOMB_VPLAS_CP = 1.0466
PRIES_SECOMB_MCV_FL = 55.0
PRIES_SECOMB_MCV_CORR = (92.0 / PRIES_SECOMB_MCV_FL) ** (1.0 / 3.0)

# Sparse Kirchhoff solver selection.
# Options: "auto" | "cg" | "spsolve" | "gmres_ilu".
KIRCHHOFF_SOLVER = "spsolve"  # "tree" or one of the sparse solver options below.
KIRCHHOFF_BC_MODE = (
    "legacy_equal_terminal_flow"  # "terminal_pressure" or "legacy_equal_terminal_flow".
)
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

CONC_MAX_FOR_NORMALIZATION = 0.14

EXTRAVASCULAR_CONCENTRATION = 0.0
SOLUTE_DIFFUSIVITY = 2.41e-5
TISSUE_DECAY_LENGTH = 0.2
POROSITY = 0.9
GL_ORDER = 5  # Gauss-Legendre points per segment for Greens integral (5, 9, or 20)
GL_ORDER_CEXT = 1  # Gauss-Legendre points per segment for explicit vessel Cext coupling
CEXT_TISSUE_QUADRATURE_MODE = "independent"

HD_DISCHARGE = 0.42
# Hemoglobin-bound O2 capacity in mol / m^3 blood per unit tube hematocrit.
# Therefore Chb_max = HT * O2_CAP_PER_HCT is already in mol / m^3.
O2_CAP_PER_HCT = 20.3
ALPHA_MMHG = 1.408e-3
P50_MMHG = 26.5
N_HILL = 2.7

# Stable output schemas and shared numerical constants.
CONCENTRATION_INLET_BY_FLUID = {
    "water": 0.2211,
    "cell media": 0.2211,
    "media": 0.2211,
    "blood": 0.14,
    "custom": 0.2211,
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
VIOLIN_CSV_NAME = "cascade_dnc_samples.csv"
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
_NUMERICAL_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "numerics"
KRATIO_LUT_PATH = _NUMERICAL_ASSET_DIR / "bessel_k_v1.npz"
GRAETZ_BASIS_PATH = _NUMERICAL_ASSET_DIR / "graetz_basis_8x4_v1.npz"

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

# Process progress state used by diagnostics.
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


__all__ = [name for name in globals() if not name.startswith("__")]
