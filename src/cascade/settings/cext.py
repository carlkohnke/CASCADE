from __future__ import annotations

import os
import numpy as np

DEFAULTS = {
    # Device used for the main explicit extravascular concentration solve: "gpu", "cpu", or "auto".
    "CEXT_ACCEL_MODE": "gpu",
    # Device used for frozen-source Cext steps, which reuse fixed vessel source strengths.
    "CEXT_FROZEN_ACCEL_MODE": "gpu",
    # Initial guess for Cext before vessel/extravascular coupling iterations begin.
    "CEXT_INIT_MODE": "decoupled_greens",
    # Source of the screening length lambda used by Cext kernels, usually tissue lambda from local uptake.
    "CEXT_LAMBDA_SOURCE": os.environ.get("SVV_CEXT_LAMBDA_SOURCE", "lambda_t").strip().lower(),
    # Spatial interaction cutoff measured in oxygen decay lengths for local Cext calculations.
    "CEXT_WINDOW_FACTOR": 6,
    # Maximum outer iterations coupling intravascular oxygen to extravascular concentration.
    "CEXT_VESS_COUPLING_MAX_ITER": 1,
    # Absolute convergence tolerance for the maximum Cext update between outer iterations.
    "CEXT_VESS_COUPLING_TOL": 1.0e-3,
    # Base relaxation factor for vessel-Cext coupling updates before acceleration modifies it.
    "CEXT_VESS_COUPLING_OMEGA": 1.0,
    # Relative convergence tolerance for vessel-Cext coupling; 0 disables this relative stop test.
    "CEXT_VESS_COUPLING_REL_TOL": 0.0,
    # Iteration acceleration method: "none", "aitken", or "anderson".
    "CEXT_VESS_COUPLING_ACCEL": "anderson",
    # Lower bound on relaxation after acceleration; prevents updates from becoming too tiny.
    "CEXT_VESS_COUPLING_OMEGA_MIN": 0.025,
    # Upper bound on relaxation after acceleration; prevents overly aggressive updates.
    "CEXT_VESS_COUPLING_OMEGA_MAX": 1.4,
    # Absolute trust limit for accepting an accelerated coupling step.
    "CEXT_VESS_COUPLING_TRUST_ABS": 1.0e-3,
    # Relative trust limit for accepting an accelerated coupling step.
    "CEXT_VESS_COUPLING_TRUST_REL": 0.4,
    # Number of previous residual vectors Anderson acceleration can use.
    "CEXT_VESS_COUPLING_ANDERSON_DEPTH": 4,
    # Small regularization added to Anderson least-squares problems for numerical stability.
    "CEXT_VESS_COUPLING_ANDERSON_REG": 1.0e-10,
    # First outer iteration where Anderson acceleration is allowed to start.
    "CEXT_VESS_COUPLING_ANDERSON_START": 2,
    # Residual-improvement gate required before accepting Anderson acceleration.
    "CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO": 0.9,
    # Number of stable iterations required before Anderson acceleration is trusted.
    "CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS": 2,
    # Additional gate on Anderson steps based on the current relaxation factor.
    "CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR": 1.5,
    # Required improvement factor for accepting any accelerated step.
    "CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR": 0.9,
    # Scale applied to candidate accelerated step sizes.
    "CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR": 2.0,
    # Residual-growth factor that triggers acceleration history restart.
    "CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR": 1.1,
    # Iterations allowed without improving the best residual before best-state recovery logic can act.
    "CEXT_VESS_COUPLING_BEST_STALL_ITERS": 20,
    # Residual increase over the best residual that can trigger reverting toward the best state.
    "CEXT_VESS_COUPLING_BEST_REVERT_FACTOR": 1.02,
    # Residual-growth factor that causes an accelerated step to be rejected.
    "CEXT_VESS_COUPLING_STEP_REJECT_FACTOR": 1.02,
    # Factor used to shrink and retry a rejected coupling step.
    "CEXT_VESS_COUPLING_STEP_RETRY_FACTOR": 0.5,
    # If true, freeze source vessels whose Cext contribution has converged enough.
    "CEXT_ACTIVE_SET_ENABLE": True,
    # First outer iteration where source-vessel active-set freezing is allowed.
    "CEXT_ACTIVE_SET_START": 6,
    # Consecutive stable iterations required before a source vessel can be frozen.
    "CEXT_ACTIVE_SET_STABLE_ITERS": 3,
    # Relative change threshold used to decide that a source vessel is stable.
    "CEXT_ACTIVE_SET_REL_TOL": 5.0e-3,
    # Absolute change threshold used to decide that a source vessel is stable.
    "CEXT_ACTIVE_SET_ABS_TOL": 2.5e-4,
    # Iteration period for refreshing frozen/active source-vessel status.
    "CEXT_ACTIVE_SET_REFRESH_PERIOD": 8,
    # Minimum number of still-active source vessels required before freezing is allowed.
    "CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT": 1024,
    # Minimum active-source fraction required before freezing is allowed; 0 means use only the count limit.
    "CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION": 0.0,
    # If true, freeze target vessels whose received Cext field has converged enough.
    "CEXT_TARGET_ACTIVE_SET_ENABLE": True,
    # First outer iteration where target-vessel freezing is allowed.
    "CEXT_TARGET_ACTIVE_SET_START": 6,
    # Consecutive stable iterations required before a target vessel can be frozen.
    "CEXT_TARGET_ACTIVE_SET_STABLE_ITERS": 3,
    # Relative change threshold used to decide that a target vessel is stable.
    "CEXT_TARGET_ACTIVE_SET_REL_TOL": 5.0e-3,
    # Absolute change threshold used to decide that a target vessel is stable.
    "CEXT_TARGET_ACTIVE_SET_ABS_TOL": 2.5e-4,
    # Minimum number of still-active target vessels required before target freezing is allowed.
    "CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT": 1024,
    # Extra neighbor layers kept active around active targets so local interactions remain accurate.
    "CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD": 1,
    # Grid resolution per dimension for the hybrid background-field approximation.
    "CEXT_HYBRID_BG_GRID": 256,
    # Number of lambda bins used when grouping source vessels for the hybrid background solve.
    "CEXT_HYBRID_BG_LAMBDA_BINS": 5,
    # Radius multiplier separating near-field direct corrections from background-field approximation.
    "CEXT_HYBRID_BG_NEAR_RADIUS_MULT": 0.0,
    # Number of multigrid V-cycles used by background solvers that support V-cycles.
    "CEXT_HYBRID_BG_VCYCLES": 2,
    # Source-to-grid assignment stencil for the hybrid background field, such as CIC or TSC.
    "CEXT_HYBRID_BG_ASSIGNMENT": "tsc",
    # Solver used for the hybrid background field: "auto", "fft", or "jacobi".
    "CEXT_HYBRID_BG_SOLVER": "auto",
    # If true, allow background-source freezing during hybrid background iterations.
    "CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING": False,
    # Hybrid background mode; "fft" uses grid convolution, while local-only modes use direct local corrections.
    "CEXT_HYBRID_BG_MODE": os.environ.get("SVV_CEXT_HYBRID_BG_MODE", "fft").strip().lower(),
    # If true, choose lambda-bin edges from quantiles of the current vessel lambda values.
    "CEXT_HYBRID_FFT_QUANTILE_BINS": os.environ.get("SVV_CEXT_HYBRID_FFT_QUANTILE_BINS", "true").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, include finite-radius oxygen source corrections in the FFT background path.
    "CEXT_HYBRID_FFT_O2_CORRECTION": os.environ.get("SVV_CEXT_HYBRID_FFT_O2_CORRECTION", "true").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, subtract each vessel's own gridded background contribution before adding direct self terms.
    "CEXT_HYBRID_FFT_SELF_SUBTRACT": os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUBTRACT", "true").strip().lower()
    in ("1", "true", "yes", "on"),
    # Grid interpolation method used when sampling the self-subtraction correction at target vessels.
    "CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING": os.environ.get(
        "SVV_CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING", "cic"
    ).strip().lower(),
    # Multiplicative scale for the FFT self-subtraction correction.
    "CEXT_HYBRID_FFT_SELF_SUB_SCALE": float(os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_SCALE", "1.0")),
    # If true, cache FFT bin data across coupling iterations when bin definitions do not change.
    "CEXT_HYBRID_FFT_BIN_EPOCH_CACHE": os.environ.get("SVV_CEXT_HYBRID_FFT_BIN_EPOCH_CACHE", "true").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, batch FFT response calculations to reduce Python overhead at the cost of more temporary memory.
    "CEXT_HYBRID_FFT_RESPONSE_BATCHED": os.environ.get("SVV_CEXT_HYBRID_FFT_RESPONSE_BATCHED", "false").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, fuse some oxygen-correction work into inverse FFT operations.
    "CEXT_HYBRID_FFT_O2_FUSED_IFFT": os.environ.get("SVV_CEXT_HYBRID_FFT_O2_FUSED_IFFT", "true").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, fuse self-subtraction operations where supported by the FFT implementation.
    "CEXT_HYBRID_FFT_SELF_SUB_FUSED": os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_FUSED", "false").strip().lower()
    in ("1", "true", "yes", "on"),
    # Batch size for FFT oxygen moment calculations; larger batches can improve throughput but use more memory.
    "CEXT_HYBRID_FFT_O2_MOMENT_BATCH": max(int(os.environ.get("SVV_CEXT_HYBRID_FFT_O2_MOMENT_BATCH", "1")), 1),
    # If true, cache GPU hybrid-background data between coupling iterations.
    "CEXT_HYBRID_GPU_ITERATION_CACHE": os.environ.get("SVV_CEXT_HYBRID_GPU_ITERATION_CACHE", "false").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, update hybrid GPU work weights from measured runtime instead of static estimates.
    "CEXT_HYBRID_GPU_RUNTIME_WEIGHTS": os.environ.get("SVV_CEXT_HYBRID_GPU_RUNTIME_WEIGHTS", "true").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, compute grid assignment stencils inside CUDA kernels instead of storing large stencil arrays.
    "CEXT_HYBRID_GPU_RUNTIME_STENCIL": os.environ.get("SVV_CEXT_HYBRID_GPU_RUNTIME_STENCIL", "false").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, compute FFT O2 moment weights inside runtime-stencil kernels instead of caching moment arrays.
    "CEXT_HYBRID_GPU_RUNTIME_MOMENTS": os.environ.get("SVV_CEXT_HYBRID_GPU_RUNTIME_MOMENTS", "false").strip().lower()
    in ("1", "true", "yes", "on"),
    # If true, use a faster self-interaction path for local-only Cext modes.
    "CEXT_LOCAL_ONLY_FAST_SELF": True,
    # Treecode opening angle; smaller values are more accurate and slower.
    "CEXT_TREECODE_THETA": 0.5,
    # Multipole expansion order used by treecode Cext approximations.
    "CEXT_TREECODE_ORDER": 1,
    # Maximum source count per leaf node in the treecode spatial hierarchy.
    "CEXT_TREECODE_LEAF_NODES": 128,
    # Number of lambda bins used by treecode approximations.
    "CEXT_TREECODE_LAMBDA_BINS": 8,
    # Radius multiplier defining direct near-field work around treecode targets.
    "CEXT_TREECODE_NEAR_RADIUS_MULT": 4.0,
    # If true, compute treecode near-field/local corrections on the GPU when available.
    "CEXT_TREECODE_GPU_LOCAL": True,
    # Tail solver used when active-set iterations stall; "none" disables the tail correction.
    "CEXT_TAIL_SOLVER": "active_core_nk",
    # Earliest outer iteration where the tail solver can be triggered.
    "CEXT_TAIL_TRIGGER_START_ITER": 8,
    # Number of stalled iterations required before the tail solver is considered.
    "CEXT_TAIL_TRIGGER_STALL_ITERS": 6,
    # Minimum active unknown count before the tail solver is worth using.
    "CEXT_TAIL_TRIGGER_ACTIVE_COUNT": 4096,
    # Maximum nonlinear iterations inside the tail correction solve.
    "CEXT_TAIL_MAX_NONLINEAR_ITERS": 6,
    # Krylov restart length for GMRES inside the tail solver.
    "CEXT_TAIL_GMRES_RESTART": 32,
    # Maximum GMRES iterations inside the tail solver.
    "CEXT_TAIL_GMRES_MAXITER": 96,
    # Required improvement ratio for accepting tail-solver progress.
    "CEXT_TAIL_TRIGGER_IMPROVEMENT_RATIO": 0.8,
    # Relative rebound limit used to reject tail updates that worsen the residual too much.
    "CEXT_TAIL_REBOUND_REL": 0.9,
    # Armijo-style relative rebound limit used in guarded tail-solver step selection.
    "CEXT_TAIL_REBOUND_ARM_REL": 0.4,
    # Relative residual threshold defining the active core for tail correction.
    "CEXT_TAIL_CORE_REL_THRESHOLD": 5.0e-3,
    # Absolute residual threshold defining the active core for tail correction.
    "CEXT_TAIL_CORE_ABS_THRESHOLD": 2.5e-4,
    # Relative flow-change tolerance used when deciding whether treecode sources can remain frozen.
    "CEXT_TREECODE_FREEZE_QREL_TOL": 2.5e-2,
    # Multiplier connecting vessel interaction length scales to hybrid/grid cell sizes.
    "CEXT_GRID_CELL_FACTOR": 1.0,
    # Target number of source-target candidate slots when streaming Cext calculations.
    "CEXT_STREAMING_TARGET_CANDIDATE_SLOTS": 500000,
    # Maximum number of grid cells a single segment may touch during Cext spatial indexing.
    "CEXT_MAX_CELLS_PER_SEG": 128,
    # Number of vessel segments to validate against a reference GPU path; 0 disables validation.
    "CEXT_GPU_VALIDATE_SEGMENTS": 0,
    # Scale factor applied to approximate-window distances for Cext candidate pruning.
    "CEXT_APPROX_WINDOW_SCALE": 1.0,
    # Maximum candidate source vessels retained per target in approximate Cext searches.
    "CEXT_MAX_CANDIDATES_PER_TARGET": 100,
    # Worker count for CPU precomputation before Cext GPU or hybrid solves.
    "CEXT_PRECOMPUTE_WORKERS": max(1, (os.cpu_count() or 1) - 2),
    # Number of graph hops excluded from local Cext coupling to avoid near-self double counting.
    "CEXT_LOCAL_EXCLUDE_HOPS": 2,
    # Lower bound for vessel concentration used to avoid zero or negative concentration in Cext formulas.
    "VESS_CONC_FLOOR": 1.0e-12,
    # Floating-point dtype used in Cext working arrays.
    "CEXT_FLOAT_DTYPE": np.float32,
    # Integer dtype used in Cext index arrays.
    "CEXT_INDEX_DTYPE": np.int32,
}

ALIASES = {
    "accel": "CEXT_ACCEL_MODE",
    "accel_mode": "CEXT_ACCEL_MODE",
    "frozen_accel": "CEXT_FROZEN_ACCEL_MODE",
    "frozen_accel_mode": "CEXT_FROZEN_ACCEL_MODE",
    "init_mode": "CEXT_INIT_MODE",
    "lambda_source": "CEXT_LAMBDA_SOURCE",
    "window_factor": "CEXT_WINDOW_FACTOR",
    "vess_coupling_max_iter": "CEXT_VESS_COUPLING_MAX_ITER",
    "vess_coupling_tol": "CEXT_VESS_COUPLING_TOL",
    "vess_coupling_omega": "CEXT_VESS_COUPLING_OMEGA",
    "vess_coupling_omega_min": "CEXT_VESS_COUPLING_OMEGA_MIN",
    "vess_coupling_omega_max": "CEXT_VESS_COUPLING_OMEGA_MAX",
    "vess_coupling_accel": "CEXT_VESS_COUPLING_ACCEL",
    "hybrid_bg_mode": "CEXT_HYBRID_BG_MODE",
    "hybrid_bg_grid": "CEXT_HYBRID_BG_GRID",
    "hybrid_bg_lambda_bins": "CEXT_HYBRID_BG_LAMBDA_BINS",
    "hybrid_gpu_iteration_cache": "CEXT_HYBRID_GPU_ITERATION_CACHE",
    "hybrid_gpu_runtime_weights": "CEXT_HYBRID_GPU_RUNTIME_WEIGHTS",
    "hybrid_gpu_runtime_stencil": "CEXT_HYBRID_GPU_RUNTIME_STENCIL",
    "hybrid_gpu_runtime_moments": "CEXT_HYBRID_GPU_RUNTIME_MOMENTS",
    "treecode_theta": "CEXT_TREECODE_THETA",
    "treecode_order": "CEXT_TREECODE_ORDER",
    "float_dtype": "CEXT_FLOAT_DTYPE",
    "index_dtype": "CEXT_INDEX_DTYPE",
    "vess_conc_floor": "VESS_CONC_FLOOR",
}

PREFIXES = ("CEXT_",)

DEPRECATED = {}
