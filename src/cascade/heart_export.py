#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from time import perf_counter

import numpy as np
import pyvista as pv
from tqdm import tqdm

from .gpu import preload_cuda_component_libraries
from .execution import guard_simulation
from .svv_adapter import Domain, Forest


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FOREST_DIR = Path.cwd()
DEFAULT_OUT_DIR = Path("cascade_heart_export")
UNC_PREFIX = "\\\\wsl.localhost\\Ubuntu\\"

# Oxygen/concentration defaults. These intentionally mirror TissueSim_heart_accel.py
# so this exporter can be configured from the top of the script or from CLI.
TISSUE_ACCEL_MODE_DEFAULT = "gpu"
TISSUE_GPU_CHUNK_POINTS_DEFAULT = 8192
TISSUE_GPU_VALIDATE_POINTS_DEFAULT = 0
SOLUTE_DIFFUSIVITY_DEFAULT = 2.41e-5
# VMAX_MM_DEFAULT = 2e-16 * 2.2e13
VMAX_MM_DEFAULT = 0.04
K_M_MM_DEFAULT = 0.0069
NEAREST_TISSUE_VESSELS_DEFAULT = 128
WINDOW_FACTOR_DEFAULT = 6.0
GL_ORDER_DEFAULT = 5
TISSUE_KDTREE_CANDIDATE_MULT_DEFAULT = 6
AXIAL_BLOOD_STEPS_DEFAULT = 5
CONC_MAX_FOR_NORMALIZATION_DEFAULT = 0.14
HEMATOCRIT_MODEL_DEFAULT = "pries_secomb"
HEMATOCRIT_FLOW_ITERATIONS_DEFAULT = 10
HEMATOCRIT_RELAXATION_DEFAULT = 1.0
HEMATOCRIT_QTOL_NL_MIN_DEFAULT = 1.0e-3
HEMATOCRIT_HDTOL_DEFAULT = 1.0e-3
EXPORT_FLOAT_DTYPE_DEFAULT = "float32"
EXPORT_INDEX_DTYPE_DEFAULT = "int64"
WORKING_FLOAT_DTYPE_DEFAULT = "float32"
WORKING_INDEX_DTYPE_DEFAULT = "int32"
CEXT_FOREST_MODE_DEFAULT = "shared-global"
CEXT_CONCENTRATION_SOLVER_DEFAULT = "topdown_ext_hybrid_bg"
CEXT_ACCEL_MODE_DEFAULT = "gpu"
CEXT_FROZEN_ACCEL_MODE_DEFAULT = "gpu"
CEXT_INIT_MODE_DEFAULT = "decoupled_greens"
CEXT_LAMBDA_SOURCE_DEFAULT = "lambda_t"
CEXT_BG_MODE_DEFAULT = "fft"
CEXT_BG_GRID_DEFAULT = 256
CEXT_BG_LAMBDA_BINS_DEFAULT = 5
CEXT_BG_ASSIGNMENT_DEFAULT = "tsc"
CEXT_BG_SOLVER_DEFAULT = "auto"
CEXT_GL_ORDER_DEFAULT = 1
CEXT_BG_NEAR_RADIUS_MULT_DEFAULT = 0.0
CEXT_BG_VCYCLES_DEFAULT = 2
CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING_DEFAULT = False
CEXT_VESS_COUPLING_MAX_ITER_DEFAULT = 1
CEXT_VESS_COUPLING_TOL_DEFAULT = 1.0e-3
CEXT_VESS_COUPLING_REL_TOL_DEFAULT = 0.0
CEXT_VESS_COUPLING_OMEGA_DEFAULT = 1.0
CEXT_VESS_COUPLING_ACCEL_DEFAULT = "anderson"
CEXT_VESS_COUPLING_OMEGA_MIN_DEFAULT = 0.025
CEXT_VESS_COUPLING_OMEGA_MAX_DEFAULT = 1.4
CEXT_VESS_COUPLING_TRUST_ABS_DEFAULT = 1.0e-3
CEXT_VESS_COUPLING_TRUST_REL_DEFAULT = 0.4
CEXT_VESS_COUPLING_ANDERSON_DEPTH_DEFAULT = 4
CEXT_VESS_COUPLING_ANDERSON_REG_DEFAULT = 1.0e-10
CEXT_VESS_COUPLING_ANDERSON_START_DEFAULT = 2
CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO_DEFAULT = 0.9
CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS_DEFAULT = 2
CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR_DEFAULT = 1.5
CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR_DEFAULT = 0.9
CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR_DEFAULT = 2.0
CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR_DEFAULT = 1.1
CEXT_VESS_COUPLING_BEST_STALL_ITERS_DEFAULT = 20
CEXT_VESS_COUPLING_BEST_REVERT_FACTOR_DEFAULT = 1.02
CEXT_VESS_COUPLING_STEP_REJECT_FACTOR_DEFAULT = 1.02
CEXT_VESS_COUPLING_STEP_RETRY_FACTOR_DEFAULT = 0.5
CEXT_ACTIVE_SET_ENABLE_DEFAULT = True
CEXT_ACTIVE_SET_START_DEFAULT = 6
CEXT_ACTIVE_SET_STABLE_ITERS_DEFAULT = 3
CEXT_ACTIVE_SET_REL_TOL_DEFAULT = 5.0e-3
CEXT_ACTIVE_SET_ABS_TOL_DEFAULT = 2.5e-4
CEXT_ACTIVE_SET_REFRESH_PERIOD_DEFAULT = 8
CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT = 1024
CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION_DEFAULT = 0.0
CEXT_TARGET_ACTIVE_SET_ENABLE_DEFAULT = True
CEXT_TARGET_ACTIVE_SET_START_DEFAULT = 6
CEXT_TARGET_ACTIVE_SET_STABLE_ITERS_DEFAULT = 3
CEXT_TARGET_ACTIVE_SET_REL_TOL_DEFAULT = 5.0e-3
CEXT_TARGET_ACTIVE_SET_ABS_TOL_DEFAULT = 2.5e-4
CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT = 1024
CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD_DEFAULT = 1
CEXT_HYBRID_FFT_QUANTILE_BINS_DEFAULT = True
CEXT_HYBRID_FFT_O2_CORRECTION_DEFAULT = True
CEXT_HYBRID_FFT_SELF_SUBTRACT_DEFAULT = True
CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING_DEFAULT = "cic"
CEXT_HYBRID_FFT_SELF_SUB_SCALE_DEFAULT = 1.0
CEXT_HYBRID_FFT_BIN_EPOCH_CACHE_DEFAULT = True
CEXT_HYBRID_FFT_RESPONSE_BATCHED_DEFAULT = False
CEXT_HYBRID_FFT_O2_FUSED_IFFT_DEFAULT = True
CEXT_HYBRID_FFT_SELF_SUB_FUSED_DEFAULT = False
CEXT_HYBRID_FFT_O2_MOMENT_BATCH_DEFAULT = 1
CEXT_HYBRID_GPU_ITERATION_CACHE_DEFAULT = False
CEXT_HYBRID_GPU_RUNTIME_WEIGHTS_DEFAULT = True
CEXT_HYBRID_GPU_RUNTIME_STENCIL_DEFAULT = False
CEXT_HYBRID_GPU_RUNTIME_MOMENTS_DEFAULT = False
CEXT_GRID_CELL_FACTOR_DEFAULT = 1.0
CEXT_STREAMING_TARGET_CANDIDATE_SLOTS_DEFAULT = 500_000
CEXT_GPU_VALIDATE_SEGMENTS_DEFAULT = 0
CEXT_APPROX_WINDOW_SCALE_DEFAULT = 1.0
CEXT_MAX_CANDIDATES_PER_TARGET_DEFAULT = 100
FINITE_RADIUS_O2_TERMS_DEFAULT = "both"
LUMEN_WALL_CLOSURE_DEFAULT = "graetz"
GRAETZ_N_RADIAL_DEFAULT = 6
GRAETZ_N_MODES_DEFAULT = 3
GRAETZ_MAX_FP_ITERS_DEFAULT = 4
GRAETZ_VELOCITY_PROFILE_DEFAULT = "poiseuille"


def _normalize_path(value: str | Path) -> Path:
    s = str(value).strip()
    if s.startswith(UNC_PREFIX):
        s = "/" + s[len(UNC_PREFIX):].replace("\\", "/")
    return Path(s).expanduser()


def _float_dtype_from_name(value: str) -> np.dtype:
    text = str(value or "float32").strip().lower()
    if text in {"float32", "f32", "32"}:
        return np.dtype(np.float32)
    if text in {"float64", "f64", "64"}:
        return np.dtype(np.float64)
    raise ValueError(f"Unknown export float dtype: {value!r}")


def _int_dtype_from_name(value: str) -> np.dtype:
    text = str(value or "int32").strip().lower()
    if text in {"int32", "i32", "32"}:
        return np.dtype(np.int32)
    if text in {"int64", "i64", "64"}:
        return np.dtype(np.int64)
    raise ValueError(f"Unknown export index dtype: {value!r}")


def _resolve_forest_path(value: str | Path) -> Path:
    raw = _normalize_path(value)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw
    return DEFAULT_FOREST_DIR / raw.name


def _default_simulation_cache_path(forest_path: Path) -> Path:
    return forest_path.with_name(forest_path.name + ".simcache")


def _should_use_simulation_cache(source_path: Path, cache_path: Path) -> bool:
    return (
        cache_path.exists()
        and Forest._is_simulation_cache(str(cache_path))
        and cache_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    )


def _load_tissuesim():
    from .runtime import tissuesim

    return tissuesim


def _load_cext_tissuesim(module_path: str | Path | None = None):
    if module_path:
        path = _normalize_path(module_path)
        if not path.exists():
            raise FileNotFoundError(f"Cext runtime module not found: {path}")
        spec = importlib.util.spec_from_file_location("TissueSim_cube_local_gfm_export", str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Cext TissueSim module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.__svv_export_source_path__ = str(path)
        return module

    from .runtime import tissuesim as cext_ts

    cext_ts.__svv_export_source_path__ = "cascade.runtime.tissuesim"
    return cext_ts


def _apply_tissuesim_overrides(ts, args) -> dict:
    overrides = {
        "TISSUE_ACCEL_MODE": str(args.tissue_accel),
        "TISSUE_GPU_CHUNK_POINTS": int(args.tissue_gpu_chunk_points),
        "TISSUE_GPU_VALIDATE_POINTS": int(args.tissue_gpu_validate_points),
        "SOLUTE_DIFFUSIVITY": float(args.solute_diffusivity),
        "VMAX_MM": float(args.vmax_mm),
        "K_M_MM": float(args.km_mm),
        "NEAREST_TISSUE_VESSELS": int(args.nearest_tissue_vessels),
        "WINDOW_FACTOR": float(args.window_factor),
        "GL_ORDER": int(args.gl_order),
        "CEXT_TISSUE_QUADRATURE_MODE": str(args.cext_tissue_quadrature_mode),
        "TISSUE_KDTREE_CANDIDATE_MULT": int(args.tissue_kdtree_candidate_mult),
        "AXIAL_BLOOD_STEPS": int(args.axial_blood_steps),
        "CONC_MAX_FOR_NORMALIZATION": float(args.conc_max_for_normalization),
        "HEMATOCRIT_MODEL": str(args.hematocrit_model),
        "HEMATOCRIT_FLOW_ITERATIONS": int(args.hematocrit_flow_iterations),
        "HEMATOCRIT_RELAXATION": float(args.hematocrit_relaxation),
        "HEMATOCRIT_QTOL_NL_MIN": float(args.hematocrit_qtol_nl_min),
        "HEMATOCRIT_HDTOL": float(args.hematocrit_hdtol),
    }
    if args.kirchhoff_bc_mode is not None:
        overrides["KIRCHHOFF_BC_MODE"] = str(args.kirchhoff_bc_mode)
    for name, value in overrides.items():
        if hasattr(ts, name):
            setattr(ts, name, value)
    if hasattr(ts, "CONCENTRATION_SOLVER"):
        ts.CONCENTRATION_SOLVER = str(args.concentration_solver)
    return {name: getattr(ts, name, value) for name, value in overrides.items()}


def _bool_arg(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_cext_overrides(cext_ts, ts, args) -> dict:
    solver_requested = str(args.cext_hybrid_bg_solver).strip().lower()
    if solver_requested not in {"auto", "fft", "jacobi"}:
        solver_requested = "auto"
    solver_effective = "fft" if solver_requested == "auto" else solver_requested
    overrides = {
        "TISSUE_ACCEL_MODE": str(args.tissue_accel),
        "TISSUE_GPU_CHUNK_POINTS": int(args.tissue_gpu_chunk_points),
        "TISSUE_GPU_VALIDATE_POINTS": int(args.tissue_gpu_validate_points),
        "SOLUTE_DIFFUSIVITY": float(ts.SOLUTE_DIFFUSIVITY),
        "VMAX_MM": float(ts.VMAX_MM),
        "K_M_MM": float(ts.K_M_MM),
        "NEAREST_TISSUE_VESSELS": int(ts.NEAREST_TISSUE_VESSELS),
        "WINDOW_FACTOR": float(ts.WINDOW_FACTOR),
        "CEXT_WINDOW_FACTOR": float(ts.WINDOW_FACTOR),
        "TISSUE_KDTREE_CANDIDATE_MULT": int(getattr(ts, "TISSUE_KDTREE_CANDIDATE_MULT", TISSUE_KDTREE_CANDIDATE_MULT_DEFAULT)),
        "AXIAL_BLOOD_STEPS": int(ts.AXIAL_BLOOD_STEPS),
        "CONC_MAX_FOR_NORMALIZATION": float(ts.CONC_MAX_FOR_NORMALIZATION),
        "HEMATOCRIT_MODEL": str(ts.HEMATOCRIT_MODEL),
        "HEMATOCRIT_FLOW_ITERATIONS": int(ts.HEMATOCRIT_FLOW_ITERATIONS),
        "HEMATOCRIT_RELAXATION": float(ts.HEMATOCRIT_RELAXATION),
        "HEMATOCRIT_QTOL_NL_MIN": float(ts.HEMATOCRIT_QTOL_NL_MIN),
        "HEMATOCRIT_HDTOL": float(ts.HEMATOCRIT_HDTOL),
        "GL_ORDER_CEXT": int(args.cext_gl_order),
        "CEXT_TISSUE_QUADRATURE_MODE": str(args.cext_tissue_quadrature_mode),
        "CEXT_ACCEL_MODE": str(args.cext_accel).strip().lower(),
        "CEXT_FROZEN_ACCEL_MODE": str(args.cext_frozen_accel).strip().lower(),
        "CEXT_INIT_MODE": str(args.cext_init_mode).strip().lower(),
        "CEXT_LAMBDA_SOURCE": str(args.cext_lambda_source).strip().lower(),
        "CEXT_WINDOW_FACTOR": float(args.cext_window_factor),
        "CEXT_VESS_COUPLING_MAX_ITER": int(args.cext_vess_coupling_max_iter),
        "CEXT_VESS_COUPLING_TOL": float(args.cext_vess_coupling_tol),
        "CEXT_VESS_COUPLING_REL_TOL": float(args.cext_vess_coupling_rel_tol),
        "CEXT_VESS_COUPLING_OMEGA": float(args.cext_vess_coupling_omega),
        "CEXT_VESS_COUPLING_ACCEL": str(args.cext_vess_coupling_accel).strip().lower(),
        "CEXT_VESS_COUPLING_OMEGA_MIN": float(args.cext_vess_coupling_omega_min),
        "CEXT_VESS_COUPLING_OMEGA_MAX": float(args.cext_vess_coupling_omega_max),
        "CEXT_VESS_COUPLING_TRUST_ABS": float(args.cext_vess_coupling_trust_abs),
        "CEXT_VESS_COUPLING_TRUST_REL": float(args.cext_vess_coupling_trust_rel),
        "CEXT_VESS_COUPLING_ANDERSON_DEPTH": int(args.cext_vess_coupling_anderson_depth),
        "CEXT_VESS_COUPLING_ANDERSON_REG": float(args.cext_vess_coupling_anderson_reg),
        "CEXT_VESS_COUPLING_ANDERSON_START": int(args.cext_vess_coupling_anderson_start),
        "CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO": float(args.cext_vess_coupling_anderson_gate_ratio),
        "CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS": int(args.cext_vess_coupling_anderson_min_stable_iters),
        "CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR": float(args.cext_vess_coupling_anderson_omega_gate_factor),
        "CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR": float(args.cext_vess_coupling_accept_factor),
        "CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR": float(args.cext_vess_coupling_step_factor),
        "CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR": float(args.cext_vess_coupling_restart_factor),
        "CEXT_VESS_COUPLING_BEST_STALL_ITERS": int(args.cext_vess_coupling_best_stall_iters),
        "CEXT_VESS_COUPLING_BEST_REVERT_FACTOR": float(args.cext_vess_coupling_best_revert_factor),
        "CEXT_VESS_COUPLING_STEP_REJECT_FACTOR": float(args.cext_vess_coupling_step_reject_factor),
        "CEXT_VESS_COUPLING_STEP_RETRY_FACTOR": float(args.cext_vess_coupling_step_retry_factor),
        "CEXT_ACTIVE_SET_ENABLE": _bool_arg(args.cext_active_set_enable),
        "CEXT_ACTIVE_SET_START": int(args.cext_active_set_start),
        "CEXT_ACTIVE_SET_STABLE_ITERS": int(args.cext_active_set_stable_iters),
        "CEXT_ACTIVE_SET_REL_TOL": float(args.cext_active_set_rel_tol),
        "CEXT_ACTIVE_SET_ABS_TOL": float(args.cext_active_set_abs_tol),
        "CEXT_ACTIVE_SET_REFRESH_PERIOD": int(args.cext_active_set_refresh_period),
        "CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT": int(args.cext_active_set_min_active_count),
        "CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION": float(args.cext_active_set_min_active_fraction),
        "CEXT_TARGET_ACTIVE_SET_ENABLE": _bool_arg(args.cext_target_active_set_enable),
        "CEXT_TARGET_ACTIVE_SET_START": int(args.cext_target_active_set_start),
        "CEXT_TARGET_ACTIVE_SET_STABLE_ITERS": int(args.cext_target_active_set_stable_iters),
        "CEXT_TARGET_ACTIVE_SET_REL_TOL": float(args.cext_target_active_set_rel_tol),
        "CEXT_TARGET_ACTIVE_SET_ABS_TOL": float(args.cext_target_active_set_abs_tol),
        "CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT": int(args.cext_target_active_set_min_active_count),
        "CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD": int(args.cext_target_active_set_neighbor_pad),
        "CEXT_HYBRID_BG_MODE": str(args.cext_hybrid_bg_mode).strip().lower(),
        "CEXT_HYBRID_BG_GRID": int(args.cext_hybrid_bg_grid),
        "CEXT_HYBRID_BG_LAMBDA_BINS": int(args.cext_hybrid_bg_lambda_bins),
        "CEXT_HYBRID_BG_ASSIGNMENT": str(args.cext_hybrid_bg_assignment).strip().lower(),
        "CEXT_HYBRID_BG_SOLVER": solver_effective,
        "CEXT_HYBRID_BG_NEAR_RADIUS_MULT": float(args.cext_hybrid_bg_near_radius_mult),
        "CEXT_HYBRID_BG_VCYCLES": int(args.cext_hybrid_bg_vcycles),
        "CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING": _bool_arg(args.cext_hybrid_bg_enable_source_freezing),
        "CEXT_HYBRID_FFT_QUANTILE_BINS": _bool_arg(args.cext_hybrid_fft_quantile_bins),
        "CEXT_HYBRID_FFT_O2_CORRECTION": _bool_arg(args.cext_hybrid_fft_o2_correction),
        "CEXT_HYBRID_FFT_SELF_SUBTRACT": _bool_arg(args.cext_hybrid_fft_self_subtract),
        "CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING": str(args.cext_hybrid_fft_self_sub_target_sampling).strip().lower(),
        "CEXT_HYBRID_FFT_SELF_SUB_SCALE": float(args.cext_hybrid_fft_self_sub_scale),
        "CEXT_HYBRID_FFT_BIN_EPOCH_CACHE": _bool_arg(args.cext_hybrid_fft_bin_epoch_cache),
        "CEXT_HYBRID_FFT_RESPONSE_BATCHED": _bool_arg(args.cext_hybrid_fft_response_batched),
        "CEXT_HYBRID_FFT_O2_FUSED_IFFT": _bool_arg(args.cext_hybrid_fft_o2_fused_ifft),
        "CEXT_HYBRID_FFT_SELF_SUB_FUSED": _bool_arg(args.cext_hybrid_fft_self_sub_fused),
        "CEXT_HYBRID_FFT_O2_MOMENT_BATCH": int(args.cext_hybrid_fft_o2_moment_batch),
        "CEXT_HYBRID_GPU_ITERATION_CACHE": _bool_arg(args.cext_hybrid_gpu_iteration_cache),
        "CEXT_HYBRID_GPU_RUNTIME_WEIGHTS": _bool_arg(args.cext_hybrid_gpu_runtime_weights),
        "CEXT_HYBRID_GPU_RUNTIME_STENCIL": _bool_arg(args.cext_hybrid_gpu_runtime_stencil),
        "CEXT_HYBRID_GPU_RUNTIME_MOMENTS": _bool_arg(args.cext_hybrid_gpu_runtime_moments),
        "CEXT_GRID_CELL_FACTOR": float(args.cext_grid_cell_factor),
        "CEXT_STREAMING_TARGET_CANDIDATE_SLOTS": int(args.cext_streaming_target_candidate_slots),
        "CEXT_GPU_VALIDATE_SEGMENTS": int(args.cext_gpu_validate_segments),
        "CEXT_APPROX_WINDOW_SCALE": float(args.cext_approx_window_scale),
        "CEXT_MAX_CANDIDATES_PER_TARGET": int(args.cext_max_candidates_per_target),
        "FINITE_RADIUS_O2_TERMS": str(args.finite_radius_o2_terms).strip().lower(),
        "LUMEN_WALL_CLOSURE": str(args.lumen_wall_closure).strip().lower(),
        "GRAETZ_N_RADIAL": int(args.graetz_n_radial),
        "GRAETZ_N_MODES": int(args.graetz_n_modes),
        "GRAETZ_MAX_FP_ITERS": int(args.graetz_max_fp_iters),
        "GRAETZ_VELOCITY_PROFILE": str(args.graetz_profile).strip().lower(),
    }
    if args.lumen_diffusivity_cm2_s is not None:
        overrides["LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S"] = float(args.lumen_diffusivity_cm2_s)
        overrides["LUMEN_DIFFUSIVITY_CM2_S"] = float(args.lumen_diffusivity_cm2_s)
    if hasattr(ts, "HD_DISCHARGE") and hasattr(cext_ts, "HD_DISCHARGE"):
        overrides["HD_DISCHARGE"] = float(ts.HD_DISCHARGE)
    if hasattr(ts, "O2_CAP_PER_HCT") and hasattr(cext_ts, "O2_CAP_PER_HCT"):
        overrides["O2_CAP_PER_HCT"] = float(ts.O2_CAP_PER_HCT)
    for name, value in overrides.items():
        if hasattr(cext_ts, name):
            setattr(cext_ts, name, value)
    return {
        "enabled": not bool(args.no_cext),
        "tissuesim_module": str(getattr(cext_ts, "__svv_export_source_path__", "")),
        "forest_mode": str(args.cext_forest_mode),
        "concentration_solver": str(args.cext_concentration_solver),
        "requested_bg_solver": solver_requested,
        "effective_bg_solver": solver_effective,
        "bg_mode": str(args.cext_hybrid_bg_mode).strip().lower(),
        "bg_grid": int(args.cext_hybrid_bg_grid),
        "bg_lambda_bins": int(args.cext_hybrid_bg_lambda_bins),
        "bg_assignment": str(args.cext_hybrid_bg_assignment).strip().lower(),
        "gl_order": int(args.cext_gl_order),
        "near_radius_mult": float(args.cext_hybrid_bg_near_radius_mult),
        "max_iter": int(args.cext_vess_coupling_max_iter),
        "omega": float(args.cext_vess_coupling_omega),
        "accel": str(args.cext_vess_coupling_accel).strip().lower(),
        "finite_radius_o2_terms": str(args.finite_radius_o2_terms).strip().lower(),
        "lumen_wall_closure": str(args.lumen_wall_closure).strip().lower(),
    }


def _log(message: str) -> None:
    print(message, flush=True)


def _build_domain(ts, domain_path: Path, side_length: float):
    suffix = str(domain_path.suffix).strip().lower()
    if suffix == ".dmn":
        return Domain.load(str(domain_path))
    if hasattr(ts, "build_domain_from_pyvista"):
        mesh = pv.read(str(domain_path))
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        return ts.build_domain_from_pyvista(mesh)
    if hasattr(ts, "DEFAULT_STL"):
        ts.DEFAULT_STL = str(domain_path)
    if hasattr(ts, "DOMAIN_CACHE_PATH"):
        # The heart forest can use arbitrary imported root geometry; avoid accidentally
        # reusing an incompatible single-tree domain cache.
        ts.DOMAIN_CACHE_PATH = None
    return ts.build_domain(float(side_length))


def _attach_domain(forest: Forest, domain) -> None:
    try:
        forest.attach_domain_for_simulation(domain)
    except Exception:
        forest.domain = domain
        forest.geodesic = None
        for tree in forest.networks[0]:
            tree.set_domain(domain)
            tree.domain = domain


def _normalize_forest_dtype_attrs(forest: Forest) -> list[dict]:
    summaries: list[dict] = []
    for tree_id, tree in enumerate(forest.networks[0]):
        data = np.asarray(tree.data)
        conn = getattr(tree, "connectivity", None)
        conn_arr = np.asarray(conn) if conn is not None else np.empty((0, 3), dtype=np.int64)
        data_dtype = data.dtype if data.dtype.kind == "f" else np.dtype(np.float64)
        index_dtype = conn_arr.dtype if conn is not None and conn_arr.dtype.kind == "i" else np.dtype(getattr(tree, "index_dtype", np.int64))
        tree.data_dtype = np.dtype(data_dtype)
        tree.index_dtype = np.dtype(index_dtype)
        summaries.append(
            {
                "tree_id": int(tree_id),
                "segments": int(getattr(tree, "segment_count", 0) or 0),
                "terminals": int(getattr(tree, "n_terminals", 0) or 0),
                "data_dtype": str(data.dtype),
                "preallocate_dtype": str(np.asarray(getattr(tree, "preallocate", [])).dtype),
                "connectivity_dtype": str(conn_arr.dtype),
                "data_dtype_attr": str(tree.data_dtype),
                "index_dtype_attr": str(tree.index_dtype),
            }
        )
    return summaries


def _make_tree_analysis_only(tree) -> None:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    data = np.asarray(tree.data[:seg_count])
    conn = getattr(tree, "connectivity", None)
    if conn is not None:
        conn_arr = np.asarray(conn)
        if conn_arr.dtype.kind == "i":
            tree.index_dtype = conn_arr.dtype
    tree.preallocate = tree.data
    tree.preallocation_step = int(seg_count)
    tree.preallocate_midpoints = np.empty((0, 3), dtype=getattr(tree, "data_dtype", data.dtype))
    tree.midpoints = tree.preallocate_midpoints
    tree.vessel_map = {}
    tree.kdtm = None
    tree.hnsw_tree = None
    tree.hnsw_tree_id = None
    tree.connectivity = None
    tree._analysis_only_load = True


def _make_forest_analysis_only(forest: Forest) -> None:
    for tree in forest.networks[0]:
        _make_tree_analysis_only(tree)


def _repair_tree_parent_columns_from_children(tree) -> int:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    if seg_count <= 0:
        return 0
    data = np.asarray(tree.data[:seg_count])
    index_dtype = getattr(tree, "index_dtype", np.int64)
    conn = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(index_dtype)
    # Some older forest-export repairs wrote -1 into tree.data child slots.
    # SVV's solver code treats NaN child slots as terminal leaves, so keep -1
    # only in tree.connectivity and preserve/restore NaNs in tree.data.
    child_conn = conn[:, 0:2]
    child_conn[data[:, 15:17] < 0] = -1
    conn[:, 0:2] = child_conn

    rows = np.arange(seg_count, dtype=np.int64)
    parent = np.full(seg_count, -1, dtype=np.int64)
    child_counts = np.zeros(seg_count, dtype=np.uint8)
    for col in (0, 1):
        children = conn[:, col].astype(np.int64, copy=False)
        valid = children >= 0
        if not np.any(valid):
            continue
        valid_children = children[valid]
        max_child = int(valid_children.max())
        if max_child >= seg_count:
            row = int(rows[valid][int(np.argmax(valid_children))])
            raise RuntimeError(f"Connectivity child index out of range: row={row} child={max_child}")
        np.add.at(child_counts, valid_children, 1)
        parent[valid_children] = rows[valid]
    duplicate_children = np.flatnonzero(child_counts > 1)
    if duplicate_children.size:
        raise RuntimeError(f"Connectivity has duplicate child references: {duplicate_children[:5].tolist()}")

    changed = conn[:, 2].astype(np.int64, copy=False) != parent
    n_changed = int(np.count_nonzero(changed))
    conn[:, 2] = parent.astype(conn.dtype, copy=False)
    data_conn = conn.astype(float)
    data_conn[data_conn < 0] = np.nan
    tree.data[:seg_count, 15:18] = data_conn.astype(np.asarray(tree.data).dtype, copy=False)
    try:
        tree.preallocate[:seg_count, 15:18] = data_conn.astype(np.asarray(tree.preallocate).dtype, copy=False)
    except Exception:
        pass
    tree.connectivity = conn.astype(index_dtype, copy=False)
    return n_changed


def _repair_forest_connectivity(forest: Forest) -> list[int]:
    repairs = []
    for tree in forest.networks[0]:
        repairs.append(_repair_tree_parent_columns_from_children(tree))
    return repairs


def _connectivity_report(tree, *, geometry_atol: float = 1e-6) -> dict:
    n = int(getattr(tree, "segment_count", 0) or 0)
    data = np.asarray(tree.data[:n])
    if n == 0:
        return {"segments": 0, "roots": [], "reachable": 0, "bad_parent": 0, "bad_child": 0, "bad_geom": 0}
    conn = np.nan_to_num(data[:, 15:18], nan=-1).astype(int)
    roots = np.flatnonzero(conn[:, 2] < 0)
    row_ids = np.arange(n, dtype=np.int64)

    parent = conn[:, 2].astype(np.int64, copy=False)
    valid_parent = parent >= 0
    bad_parent_range = valid_parent & (parent >= n)
    good_parent = valid_parent & (parent < n)
    bad_parent = int(np.count_nonzero(bad_parent_range))
    bad_geom = 0
    if np.any(good_parent):
        rows = row_ids[good_parent]
        parents = parent[good_parent]
        parent_children = conn[parents, 0:2]
        parent_links_back = np.any(parent_children == rows[:, None], axis=1)
        bad_parent += int(np.count_nonzero(~parent_links_back))
        geom_ok = np.all(
            np.abs(data[parents, 3:6] - data[rows, 0:3]) <= float(geometry_atol),
            axis=1,
        )
        bad_geom += int(np.count_nonzero(parent_links_back & ~geom_ok))

    child_conn = conn[:, 0:2].astype(np.int64, copy=False)
    valid_child = child_conn >= 0
    bad_child_range = valid_child & (child_conn >= n)
    bad_child = int(np.count_nonzero(bad_child_range))
    good_child = valid_child & (child_conn < n)
    if np.any(good_child):
        parent_rows, child_cols = np.nonzero(good_child)
        child_ids = child_conn[parent_rows, child_cols]
        child_links_back = conn[child_ids, 2] == parent_rows
        bad_child += int(np.count_nonzero(~child_links_back))
        geom_ok = np.all(
            np.abs(data[parent_rows, 3:6] - data[child_ids, 0:3]) <= float(geometry_atol),
            axis=1,
        )
        bad_geom += int(np.count_nonzero(child_links_back & ~geom_ok))

    # For a repaired rooted tree, one root plus valid reciprocal parent/child
    # references is enough for the exporter. Avoid a Python DFS over tens of
    # millions of rows; disconnected cycles would already be non-tree input.
    reachable = n if len(roots) == 1 and bad_parent == 0 and bad_child == 0 else 0
    return {
        "segments": n,
        "roots": roots.tolist(),
        "reachable": int(reachable),
        "bad_parent": int(bad_parent),
        "bad_child": int(bad_child),
        "bad_geom": int(bad_geom),
    }


def _validate_forest_connectivity(forest: Forest, *, fail: bool, geometry_atol: float) -> list[dict]:
    reports = []
    for tree_id, tree in enumerate(forest.networks[0]):
        report = _connectivity_report(tree, geometry_atol=float(geometry_atol))
        report["tree_id"] = int(tree_id)
        reports.append(report)
        _log(
            "Connectivity tree={tree_id}: segments={segments} roots={roots} "
            "reachable={reachable}/{segments} bad_parent={bad_parent} "
            "bad_child={bad_child} bad_geom={bad_geom}".format(**report)
        )
        bad = (
            len(report["roots"]) != 1
            or int(report["reachable"]) != int(report["segments"])
            or int(report["bad_parent"]) > 0
            or int(report["bad_child"]) > 0
            or int(report["bad_geom"]) > 0
        )
        if fail and bad:
            raise RuntimeError(f"Connectivity validation failed for tree {tree_id}: {report}")
    return reports


def _tree_root_flow_cm3_s(tree) -> float:
    params = getattr(tree, "parameters", None)
    if params is not None:
        val = getattr(params, "root_flow", None)
        if val is not None and np.isfinite(float(val)) and float(val) > 0.0:
            return float(val)
    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0)])
    if data.size:
        val = float(data[0, 22])
        if np.isfinite(val) and val > 0.0:
            return val
    return 1.0


def _flow_inputs(args, forest: Forest) -> list[float]:
    trees = list(forest.networks[0])
    root_flows = np.array([_tree_root_flow_cm3_s(tree) for tree in trees], dtype=float)
    if str(args.flow_source).lower() == "tree-root-flow":
        return [float(v) for v in root_flows]
    if args.total_qin_ul_min is None:
        raise ValueError("--total-qin-ul-min is required when --flow-source total-qin-split")
    total_qin_cm3_s = float(args.total_qin_ul_min) * 1.0e-3 / 60.0
    weights = np.maximum(root_flows, 0.0)
    if not np.any(weights > 0.0):
        weights = np.ones(len(trees), dtype=float)
    flows = total_qin_cm3_s * weights / float(np.sum(weights))
    return [float(v) for v in flows]


def _safe_index_array(values, dtype: np.dtype) -> np.ndarray:
    arr64 = np.asarray(values, dtype=np.int64)
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.int32):
        info = np.iinfo(np.int32)
        if arr64.size and (int(np.nanmax(arr64)) > info.max or int(np.nanmin(arr64)) < info.min):
            raise OverflowError("Index array cannot be represented as int32")
    return arr64.astype(dtype, copy=False)


def _collect_downstream_segment_ids(tree, segment_id: int, seg_count: int) -> np.ndarray:
    if seg_count <= 0 or segment_id < 0 or segment_id >= seg_count:
        return np.empty((0,), dtype=np.int64)
    conn = getattr(tree, "connectivity", None)
    if conn is None or np.asarray(conn).size == 0:
        data = np.asarray(tree.data[:seg_count])
        conn = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(np.int64)
    else:
        conn = np.asarray(conn[:seg_count], dtype=np.int64)
    seen = np.zeros(seg_count, dtype=bool)
    stack = [int(segment_id)]
    ordered: list[int] = []
    while stack:
        current = int(stack.pop())
        if current < 0 or current >= seg_count or seen[current]:
            continue
        seen[current] = True
        ordered.append(current)
        children = np.asarray(conn[current, 0:2], dtype=np.int64).reshape(-1)
        for child in children[::-1]:
            if child >= 0:
                stack.append(int(child))
    return np.asarray(ordered, dtype=np.int64)


def _resolve_global_segment_id(forest: Forest, global_segment_id: int) -> tuple[int, int]:
    offset = 0
    target = int(global_segment_id)
    for tree_id, tree in enumerate(forest.networks[0]):
        seg_count = int(getattr(tree, "segment_count", 0) or 0)
        if target < offset + seg_count:
            return int(tree_id), int(target - offset)
        offset += seg_count
    raise ValueError(f"Global segment id {global_segment_id} is out of range for {offset} total segments")


def _solve_tree(ts, tree, inlet_flow_cm3_s: float, *, fluid: str, concentration_solver: str | None) -> dict:
    inlet_concentration = float(ts.get_concentration_inlet(fluid))
    _log(
        f"  Solving tree: segments={int(tree.segment_count)} terminals={int(tree.n_terminals)} "
        f"qin={inlet_flow_cm3_s:.9g} cm3/s"
    )
    t0 = perf_counter()
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
    ) = ts.recompute_tree_flows(tree, inlet_flow_cm3_s, fluid=fluid)
    _log(f"    flow recompute completed in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    cin, cout = ts._solve_channel_concentrations(
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
        diffusivity=float(getattr(ts, "SOLUTE_DIFFUSIVITY", 3e-5)),
        vmax=float(getattr(ts, "VMAX_MM", 1.0)),
        km=float(getattr(ts, "K_M_MM", 1.0)),
        fluid=fluid,
        solver=concentration_solver,
    )
    _log(f"    concentration solve completed in {perf_counter() - t0:.2f}s")
    p_in = float("nan")
    p_out = float("nan")
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
    return {
        "starts": np.asarray(starts),
        "ends": np.asarray(ends),
        "radii": np.asarray(radii),
        "lengths": np.asarray(lengths),
        "flows": np.asarray(flows),
        "cin": np.asarray(cin),
        "cout": np.asarray(cout),
        "pressures": np.asarray(pressures),
        "p_in": p_in,
        "p_out": p_out,
        "inlet_concentration": inlet_concentration,
    }


def _cext_chb_max_for_tree(cext_ts, tree, nseg: int, flows: np.ndarray, *, fluid: str) -> np.ndarray:
    if str(fluid).lower() != "blood":
        return np.zeros((int(nseg),), dtype=np.float32)
    cached = cext_ts._get_tree_hematocrit_cache(
        tree,
        int(nseg),
        model=str(cext_ts.HEMATOCRIT_MODEL),
        flows=flows,
    )
    if cached is None:
        hct_context = cext_ts._hematocrit_context_for_tree(tree)
        HD, HT = cext_ts.compute_tree_hematocrit(
            tree,
            hd_root=float(cext_ts.HD_DISCHARGE),
            flows=flows,
            model=str(cext_ts.HEMATOCRIT_MODEL),
            order=np.asarray(hct_context["order"], dtype=np.int64),
        )
        cext_ts._store_tree_hematocrit_cache(
            tree,
            HD,
            HT,
            model=str(cext_ts.HEMATOCRIT_MODEL),
            flows=flows,
        )
    else:
        _, HT = cached
    return np.asarray(HT, dtype=np.float32) * np.float32(float(cext_ts.O2_CAP_PER_HCT))


def _run_cext_frozen_step(cext_ts, sol: dict, *, fluid: str) -> None:
    ext_state = sol["cext_state"]
    context = sol["cext_context"]
    frozen_backend = "gpu"
    if hasattr(cext_ts, "_resolve_cext_frozen_accel_mode"):
        frozen_backend = str(cext_ts._resolve_cext_frozen_accel_mode())
    cin, cout, c_iv, backend, transfer = cext_ts._run_topdown_ext_frozen_step(
        context,
        ext_state,
        inlet_concentration=float(sol["inlet_concentration"]),
        vmax=float(cext_ts.VMAX_MM),
        km=float(cext_ts.K_M_MM),
        chb_max=np.asarray(sol["chb_max"], dtype=np.float32),
        fluid_mode=str(fluid),
        frozen_backend=frozen_backend,
    )
    ext_state["cin_seg"] = np.asarray(cin, dtype=np.float32)
    ext_state["cout_seg"] = np.asarray(cout, dtype=np.float32)
    ext_state["c_iv_gl"] = np.asarray(c_iv, dtype=np.float32)
    cext_ts._build_cext_iteration_cache(context, ext_state)
    sol["cin"] = ext_state["cin_seg"]
    sol["cout"] = ext_state["cout_seg"]
    sol["cext_frozen_backend"] = str(backend)
    sol["cext_frozen_transfer_s"] = float(transfer)


def _solve_tree_cext_prepare(cext_ts, ts, tree, inlet_flow_cm3_s: float, *, fluid: str) -> dict:
    inlet_concentration = float(ts.get_concentration_inlet(fluid))
    _log(
        f"  Cext tree prep: segments={int(tree.segment_count)} terminals={int(tree.n_terminals)} "
        f"qin={inlet_flow_cm3_s:.9g} cm3/s"
    )
    t0 = perf_counter()
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
    ) = ts.recompute_tree_flows(tree, inlet_flow_cm3_s, fluid=fluid)
    _log(f"    flow recompute completed in {perf_counter() - t0:.2f}s")
    nseg = int(np.asarray(starts).shape[0])
    t0 = perf_counter()
    context = cext_ts._build_cext_geometry_context(
        tree,
        np.asarray(flows),
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(radii),
        np.asarray(lengths),
        inlet_concentration=inlet_concentration,
        diffusivity=float(ts.SOLUTE_DIFFUSIVITY),
        vmax=float(ts.VMAX_MM),
        km=float(ts.K_M_MM),
        build_candidate_index=False,
    )
    cache_key = (
        str(fluid),
        int(cext_ts.GL_ORDER_CEXT),
        float(inlet_concentration),
        float(ts.SOLUTE_DIFFUSIVITY),
        float(ts.VMAX_MM),
        float(ts.K_M_MM),
    )
    ext_state = cext_ts._initialize_cext_state(
        tree,
        cache_key=cache_key,
        nseg=nseg,
        inlet_concentration=inlet_concentration,
        vmax=float(ts.VMAX_MM),
        km=float(ts.K_M_MM),
    )
    chb_max = _cext_chb_max_for_tree(cext_ts, tree, nseg, np.asarray(flows), fluid=fluid)
    p_in = float("nan")
    p_out = float("nan")
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
    sol = {
        "starts": np.asarray(starts),
        "ends": np.asarray(ends),
        "radii": np.asarray(radii),
        "lengths": np.asarray(lengths),
        "flows": np.asarray(flows),
        "cin": np.asarray(ext_state["cin_seg"]),
        "cout": np.asarray(ext_state["cout_seg"]),
        "pressures": np.asarray(pressures),
        "p_in": p_in,
        "p_out": p_out,
        "inlet_concentration": inlet_concentration,
        "cext_context": context,
        "cext_state": ext_state,
        "chb_max": chb_max,
    }
    _run_cext_frozen_step(cext_ts, sol, fluid=fluid)
    _log(f"    initial frozen topdown and Cext geometry completed in {perf_counter() - t0:.2f}s")
    return sol


def _global_cext_box(cext_ts, contexts: list[dict]) -> dict:
    grid_n = max(int(cext_ts.CEXT_HYBRID_BG_GRID), 16)
    mins = None
    maxs = None
    max_reach = 0.0
    for context in contexts:
        gl_points = np.asarray(context["gl_points_si"], dtype=np.float32).reshape(-1, 3)
        if gl_points.size:
            cmin = np.min(gl_points, axis=0)
            cmax = np.max(gl_points, axis=0)
            mins = cmin if mins is None else np.minimum(mins, cmin)
            maxs = cmax if maxs is None else np.maximum(maxs, cmax)
        max_reach = max(max_reach, float(context.get("max_reach_si", 0.0)))
    if mins is None or maxs is None:
        mins = np.zeros((3,), dtype=np.float32)
        maxs = np.ones((3,), dtype=np.float32)
    center = np.asarray(0.5 * (mins + maxs), dtype=np.float32)
    span = max(float(np.max(maxs - mins)), 1.0e-8)
    pad = max(float(max_reach), 0.1 * span)
    side = span + 2.0 * pad
    spacing = side / float(grid_n)
    origin = np.asarray(center - 0.5 * side, dtype=np.float32)
    kfreq = 2.0 * np.pi * np.fft.fftfreq(grid_n, d=spacing)
    k2 = (
        kfreq[:, None, None] ** 2
        + kfreq[None, :, None] ** 2
        + kfreq[None, None, :] ** 2
    ).astype(np.float32)
    return {
        "grid_n": int(grid_n),
        "origin": origin,
        "side": float(side),
        "spacing": float(spacing),
        "near_radius_si": 0.0,
        "k2": np.asarray(k2, dtype=np.float32),
        "bounds_min": np.asarray(mins, dtype=np.float32),
        "bounds_max": np.asarray(maxs, dtype=np.float32),
        "padding_si": float(pad),
    }


def _global_lambda_bins(cext_ts, states: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pieces = [np.asarray(state["lambda_iv_gl"], dtype=np.float32).reshape(-1) for state in states]
    lambda_gl = np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.float32)
    finite = lambda_gl[np.isfinite(lambda_gl) & (lambda_gl > 0.0)]
    if finite.size <= 0:
        finite = np.asarray([1.0e-6], dtype=np.float32)
    lam_min = max(float(np.min(finite)), 1.0e-8)
    lam_max = max(float(np.max(finite)), lam_min * (1.0 + 1.0e-6))
    n_bins = max(int(cext_ts.CEXT_HYBRID_BG_LAMBDA_BINS), 1)
    if n_bins <= 1 or lam_max <= lam_min * (1.0 + 1.0e-6):
        edges = np.asarray([lam_min, lam_max], dtype=np.float32)
        centers = np.asarray([np.sqrt(lam_min * lam_max)], dtype=np.float32)
    else:
        edges = np.geomspace(lam_min, lam_max, n_bins + 1).astype(np.float32)
        centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
    return edges, centers


def _make_tree_hybrid_for_global_box(cext_ts, context: dict, box: dict, edges: np.ndarray, centers: np.ndarray) -> dict:
    assignment = str(cext_ts.CEXT_HYBRID_BG_ASSIGNMENT).strip().lower()
    if assignment not in {"cic", "tsc"}:
        assignment = "tsc"
    gl_points = np.asarray(context["gl_points_si"], dtype=np.float32).reshape(-1, 3)
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    hybrid = {
        "grid_n": int(box["grid_n"]),
        "lambda_bins": max(int(cext_ts.CEXT_HYBRID_BG_LAMBDA_BINS), 1),
        "near_radius_mult": 0.0,
        "assignment": assignment,
        "origin": np.asarray(box["origin"], dtype=np.float32),
        "side": float(box["side"]),
        "spacing": float(box["spacing"]),
        "near_radius_si": 0.0,
        "k2": np.asarray(box["k2"], dtype=np.float32),
        "node_seg_ids": np.repeat(np.arange(nseg, dtype=np.int32), gl_order),
        "gl_points_flat": np.asarray(gl_points, dtype=np.float32),
        "lambda_bin_edges": np.asarray(edges, dtype=np.float32),
        "lambda_bin_centers": np.asarray(centers, dtype=np.float32),
        "gpu_static": None,
    }
    context["hybrid_bg_context"] = hybrid
    return hybrid


def _solve_shared_cext_fft(cext_ts, box: dict, centers: np.ndarray, mass_grids_g, diffusivity_si: float):
    cp = cext_ts._cp
    if cp is None:
        raise RuntimeError("Cext mode requires CuPy/GPU support.")
    t0 = perf_counter()
    mass_arr = cp.asarray(mass_grids_g, dtype=cp.float32)
    phi_grids = cp.empty_like(mass_arr)
    k2_g = cp.asarray(np.asarray(box["k2"], dtype=np.float32))
    lambda_centers_g = cp.asarray(np.asarray(centers, dtype=np.float32))
    spacing = float(box["spacing"])
    cell_vol = spacing ** 3
    rhs_scale = np.float32(max(cell_vol * float(diffusivity_si), 1.0e-30))
    for bin_idx in range(int(mass_arr.shape[0])):
        rhs = mass_arr[bin_idx] / rhs_scale
        rhs_hat = cp.fft.fftn(rhs, axes=(0, 1, 2))
        lam = cp.maximum(lambda_centers_g[bin_idx], cp.float32(1.0e-8))
        denom = k2_g + cp.reciprocal(lam * lam)
        phi_hat = rhs_hat / denom
        phi_grids[bin_idx] = cp.real(cp.fft.ifftn(phi_hat, axes=(0, 1, 2))).astype(cp.float32)
        del rhs, rhs_hat, denom, phi_hat
    cp.cuda.Stream.null.synchronize()
    return phi_grids, float(perf_counter() - t0), "fft"


def _compute_shared_plain_box_cext(cext_ts, solutions: list[dict]) -> dict:
    if not solutions:
        return {"timings": {}, "grid": {}}
    cp = cext_ts._cp
    if cp is None:
        raise RuntimeError("Cext mode requires CuPy/GPU support, but TissueSim_cube_accel_ext._cp is unavailable.")
    contexts = [sol["cext_context"] for sol in solutions]
    states = [sol["cext_state"] for sol in solutions]
    box = _global_cext_box(cext_ts, contexts)
    edges, centers = _global_lambda_bins(cext_ts, states)
    hybrids = [
        _make_tree_hybrid_for_global_box(cext_ts, sol["cext_context"], box, edges, centers)
        for sol in solutions
    ]

    total_t0 = perf_counter()
    t_deposit = 0.0
    t_sample = 0.0
    global_mass_g = None
    for sol, hybrid in zip(solutions, hybrids):
        context = sol["cext_context"]
        state_cpu = sol["cext_state"]
        runtime = cext_ts._ensure_cext_hybrid_bg_runtime_state(context, hybrid, state_cpu)
        cext_ts._sync_cext_hybrid_bg_runtime_state(state_cpu, runtime)
        mass_g, dt = cext_ts._cext_hybrid_deposit_sources_gpu(
            context,
            hybrid,
            state_cpu,
            runtime_state=runtime,
            out_mass_grids_g=runtime["active_mass_grids_g"],
        )
        if global_mass_g is None:
            global_mass_g = cp.zeros_like(mass_g)
        global_mass_g += mass_g
        t_deposit += float(dt)
        hybrid["gpu_static"] = None
        hybrid["runtime_state"] = None
        context["gpu_static"] = None
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    if global_mass_g is None:
        return {"timings": {}, "grid": box}

    phi_g, t_fft, solver_mode = _solve_shared_cext_fft(
        cext_ts,
        box,
        centers,
        global_mass_g,
        float(solutions[0]["cext_context"]["diffusivity_si"]),
    )
    for sol, hybrid in zip(solutions, hybrids):
        context = sol["cext_context"]
        runtime = cext_ts._ensure_cext_hybrid_bg_runtime_state(context, hybrid, sol["cext_state"])
        sample_g, dt = cext_ts._sample_cext_hybrid_bg_gpu(
            context,
            hybrid,
            phi_g,
            global_mass_g,
            runtime_state=runtime,
        )
        sol["cext_state"]["c_ext_gl"] = np.asarray(cp.asnumpy(sample_g), dtype=np.float32)
        t_sample += float(dt)
        hybrid["gpu_static"] = None
        hybrid["runtime_state"] = None
        context["gpu_static"] = None
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    timings = {
        "deposit_s": float(t_deposit),
        "fft_s": float(t_fft),
        "sample_s": float(t_sample),
        "total_s": float(perf_counter() - total_t0),
        "solver_mode": str(solver_mode),
    }
    return {
        "timings": timings,
        "grid": {
            "grid_n": int(box["grid_n"]),
            "origin_si": np.asarray(box["origin"], dtype=float).tolist(),
            "side_si": float(box["side"]),
            "spacing_si": float(box["spacing"]),
            "bounds_min_si": np.asarray(box["bounds_min"], dtype=float).tolist(),
            "bounds_max_si": np.asarray(box["bounds_max"], dtype=float).tolist(),
            "padding_si": float(box["padding_si"]),
            "lambda_bin_edges_si": np.asarray(edges, dtype=float).tolist(),
            "lambda_bin_centers_si": np.asarray(centers, dtype=float).tolist(),
        },
    }


def _solve_forest_cext_legacy_shared_one_shot(
    cext_ts,
    ts,
    forest: Forest,
    inlet_flows: list[float],
    *,
    fluid: str,
) -> tuple[list[dict], dict]:
    if cext_ts._cp is None:
        raise RuntimeError("Cext mode requires GPU/CuPy support. Install CuPy or rerun with --no-cext.")
    t_total = perf_counter()
    solutions = []
    initial_cache_releases = []
    for tree_id, (tree, inlet_flow) in enumerate(zip(forest.networks[0], inlet_flows)):
        _log(f"Solving Cext initial state for tree {tree_id}...")
        sol = _solve_tree_cext_prepare(cext_ts, ts, tree, inlet_flow, fluid=fluid)
        solutions.append(sol)
        initial_cache_releases.append(_release_cext_transient_gpu_cache(cext_ts, sol.get("cext_context")))
    _log(
        "Computing legacy shared plain-box FFT Cext field: "
        f"trees={len(solutions)} grid={cext_ts.CEXT_HYBRID_BG_GRID} "
        f"bins={cext_ts.CEXT_HYBRID_BG_LAMBDA_BINS} assignment={cext_ts.CEXT_HYBRID_BG_ASSIGNMENT}"
    )
    cext_meta = _compute_shared_plain_box_cext(cext_ts, solutions)
    for tree_id, sol in enumerate(solutions):
        _log(f"Running reflected topdown for tree {tree_id}...")
        t0 = perf_counter()
        _run_cext_frozen_step(cext_ts, sol, fluid=fluid)
        sol["cext_reflected_topdown_s"] = float(perf_counter() - t0)
        _snapshot_sol_cext_state(cext_ts, sol, solver="legacy_shared_plain_box_fft_one_shot", backend="gpu")
    cext_vals = [
        np.asarray(sol["cext_state"]["c_ext_gl"], dtype=np.float32).reshape(-1)
        for sol in solutions
        if np.asarray(sol["cext_state"]["c_ext_gl"]).size
    ]
    all_cext = np.concatenate(cext_vals) if cext_vals else np.empty((0,), dtype=np.float32)
    cext_meta.update(
        {
            "enabled": True,
            "mode": "legacy_shared_plain_box_fft_one_shot",
            "near_radius_mult": 0.0,
            "max_iter": 0,
            "max_sampled_cext": float(np.nanmax(all_cext)) if all_cext.size else 0.0,
            "mean_sampled_cext": float(np.nanmean(all_cext)) if all_cext.size else 0.0,
            "total_s": float(perf_counter() - t_total),
        }
    )
    return solutions, cext_meta


def _concat_global_cext_context(solutions: list[dict]) -> tuple[dict, list[int]]:
    contexts = [sol["cext_context"] for sol in solutions]
    counts = [int(np.asarray(ctx["gl_points_si"]).shape[0]) for ctx in contexts]
    total = int(sum(counts))

    def cat(key: str, dtype) -> np.ndarray:
        arrays = [np.asarray(ctx[key], dtype=dtype) for ctx in contexts]
        if not arrays:
            return np.empty((0,), dtype=dtype)
        return np.concatenate(arrays, axis=0)

    parent_parts = []
    left_parts = []
    right_parts = []
    order_parts = []
    exclude_parts = []
    exclude_count_parts = []
    exclude_width = max(
        [int(np.asarray(ctx.get("exclude_idx", np.empty((0, 0), dtype=np.int32))).shape[1]) for ctx in contexts]
        or [32]
    )
    offset = 0
    for ctx, count in zip(contexts, counts):
        for source_key, target in (("parents", parent_parts), ("left_child", left_parts), ("right_child", right_parts)):
            arr = np.asarray(ctx.get(source_key, np.full((count,), -1, dtype=np.int32)), dtype=np.int32).copy()
            valid = arr >= 0
            arr[valid] += np.int32(offset)
            target.append(arr)
        order = np.asarray(ctx.get("order", np.arange(count, dtype=np.int32)), dtype=np.int32).copy()
        order_parts.append(order + np.int32(offset))
        ex = np.asarray(ctx.get("exclude_idx", np.full((count, exclude_width), -1, dtype=np.int32)), dtype=np.int32)
        ex_out = np.full((count, exclude_width), -1, dtype=np.int32)
        width = min(int(ex.shape[1]) if ex.ndim == 2 else 0, exclude_width)
        if width > 0:
            ex_slice = ex[:, :width].copy()
            valid_ex = ex_slice >= 0
            ex_slice[valid_ex] += np.int32(offset)
            ex_out[:, :width] = ex_slice
        ex_count = np.asarray(ctx.get("exclude_count", np.zeros((count,), dtype=np.uint8)), dtype=np.uint8).reshape(-1)
        if ex_count.shape[0] != count:
            ex_count = np.zeros((count,), dtype=np.uint8)
        exclude_parts.append(ex_out)
        exclude_count_parts.append(np.minimum(ex_count, np.uint8(exclude_width)))
        offset += count

    gl_points = cat("gl_points_si", np.float32)
    gl_order = int(gl_points.shape[1]) if gl_points.ndim >= 2 else 0
    exclude_idx = np.concatenate(exclude_parts, axis=0) if exclude_parts else np.full((0, exclude_width), -1, dtype=np.int32)
    exclude_count = np.concatenate(exclude_count_parts, axis=0) if exclude_count_parts else np.zeros((0,), dtype=np.uint8)

    out = {
        "parents": np.concatenate(parent_parts) if parent_parts else np.empty((0,), dtype=np.int32),
        "left_child": np.concatenate(left_parts) if left_parts else np.empty((0,), dtype=np.int32),
        "right_child": np.concatenate(right_parts) if right_parts else np.empty((0,), dtype=np.int32),
        "order": np.concatenate(order_parts) if order_parts else np.empty((0,), dtype=np.int32),
        "level_order": np.arange(total, dtype=np.int32),
        "level_offsets": np.asarray([0, total], dtype=np.int32),
        "depth": 1,
        "flow_starts_si": cat("flow_starts_si", np.float32),
        "flow_ends_si": cat("flow_ends_si", np.float32),
        "flow_vectors_si": cat("flow_vectors_si", np.float32),
        "segment_vectors": cat("segment_vectors", np.float32),
        "midpoints_si": cat("midpoints_si", np.float32),
        "flows_si": cat("flows_si", np.float32),
        "lengths_si": cat("lengths_si", np.float32),
        "radii_si": cat("radii_si", np.float32),
        "gl_t": np.asarray(contexts[0]["gl_t"], dtype=np.float32) if contexts else np.empty((0,), dtype=np.float32),
        "gl_weights": np.asarray(contexts[0]["gl_weights"], dtype=np.float32) if contexts else np.empty((0,), dtype=np.float32),
        "gl_points_si": gl_points,
        "ds_gl": cat("ds_gl", np.float32).reshape((total, gl_order)) if gl_order else np.empty((total, 0), dtype=np.float32),
        "exclude_idx": exclude_idx,
        "exclude_count": exclude_count,
        "diffusivity_si": float(contexts[0]["diffusivity_si"]) if contexts else 0.0,
        "lambda_inlet": max((float(ctx.get("lambda_inlet", 0.0)) for ctx in contexts), default=0.0),
        "cell_size": 0.0,
        "reach_si": cat("reach_si", np.float32) if contexts else np.empty((0,), dtype=np.float32),
        "max_reach_si": max((float(ctx.get("max_reach_si", 0.0)) for ctx in contexts), default=0.0),
        "candidate_query_mode": "hybrid_bg",
        "candidate_kdtree": None,
        "gpu_direct_available": False,
        "gpu_direct_cell_origin": np.zeros((3,), dtype=np.int32),
        "gpu_direct_cell_dims": np.zeros((3,), dtype=np.int32),
        "gpu_direct_cell_ptr": np.zeros((1,), dtype=np.int32),
        "gpu_direct_cell_seg_ids": np.zeros((0,), dtype=np.int32),
        "gpu_direct_cell_flat_sorted": np.zeros((0,), dtype=np.int64),
        "gpu_direct_home_cell_flat": np.zeros((total,), dtype=np.int64),
        "gpu_direct_n_cells": 0,
        "grid": {},
        "overflow_segments": np.empty((0,), dtype=np.int32),
    }
    return out, counts


def _concat_global_cext_state(solutions: list[dict]) -> dict:
    states = [sol["cext_state"] for sol in solutions]

    def state_array(state: dict, key: str) -> np.ndarray:
        if key in state:
            return np.asarray(state[key], dtype=np.float32)
        if key in {"c_bulk_gl", "c_wall_gl"}:
            return np.asarray(state["c_iv_gl"], dtype=np.float32)
        if key in {"mono2_weight_gl", "dipole2_weight_gl"}:
            return np.zeros_like(np.asarray(state["q_weighted_gl"], dtype=np.float32))
        raise KeyError(key)

    def cat(key: str) -> np.ndarray:
        pieces = [state_array(state, key) for state in states]
        return np.concatenate(pieces, axis=0) if pieces else np.empty((0,), dtype=np.float32)

    first = states[0] if states else {}
    return {
        "solver": "shared_global_gfm",
        "backend": "gpu",
        "c_ext_gl": cat("c_ext_gl"),
        "c_iv_gl": cat("c_iv_gl"),
        "c_bulk_gl": cat("c_bulk_gl"),
        "c_wall_gl": cat("c_wall_gl"),
        "lambda_iv_gl": cat("lambda_iv_gl"),
        "k_if_gl": cat("k_if_gl"),
        "q_line_gl": cat("q_line_gl"),
        "q_weighted_gl": cat("q_weighted_gl"),
        "mono2_weight_gl": cat("mono2_weight_gl"),
        "dipole2_weight_gl": cat("dipole2_weight_gl"),
        "seg_cap_gl": cat("seg_cap_gl"),
        "vmax": float(first.get("vmax", VMAX_MM_DEFAULT)),
        "km": float(first.get("km", K_M_MM_DEFAULT)),
        "window_factor": float(first.get("window_factor", WINDOW_FACTOR_DEFAULT)),
        "_lambda_bin_epoch": max((int(state.get("_lambda_bin_epoch", 0)) for state in states), default=0),
    }


def _compute_global_gfm_cext(cext_ts, solutions: list[dict], context: dict | None = None) -> tuple[np.ndarray, dict, dict]:
    if cext_ts._cp is None:
        raise RuntimeError("Cext mode requires GPU/CuPy support.")
    if context is None:
        context, _ = _concat_global_cext_context(solutions)
    state = _concat_global_cext_state(solutions)
    hybrid = cext_ts._ensure_cext_hybrid_bg_context(context)
    runtime = cext_ts._ensure_cext_hybrid_bg_runtime_state(context, hybrid, state)
    cext_ts._sync_cext_hybrid_bg_runtime_state(state, runtime)
    c_ext_new, timings = cext_ts._compute_cext_hybrid_bg_gpu(
        context,
        state,
        hybrid,
        runtime_state=runtime,
    )
    grid_meta = {
        "grid_n": int(hybrid.get("grid_n", 0)),
        "origin_si": np.asarray(hybrid.get("origin", np.zeros((3,), dtype=np.float32)), dtype=float).tolist(),
        "side_si": float(hybrid.get("side", 0.0)),
        "spacing_si": float(hybrid.get("spacing", 0.0)),
        "near_radius_si": float(hybrid.get("near_radius_si", 0.0)),
        "lambda_bin_edges_si": (
            np.asarray(hybrid.get("lambda_bin_edges"), dtype=float).tolist()
            if hybrid.get("lambda_bin_edges") is not None
            else []
        ),
        "lambda_bin_centers_si": (
            np.asarray(hybrid.get("lambda_bin_centers"), dtype=float).tolist()
            if hybrid.get("lambda_bin_centers") is not None
            else []
        ),
        "lambda_bin_policy": str(hybrid.get("lambda_bin_policy", "")),
        "lambda_bin_edges_hash": str(hybrid.get("lambda_bin_edges_hash", "")),
    }
    try:
        cext_ts._cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return np.asarray(c_ext_new, dtype=np.float32), dict(timings), grid_meta


def _apply_global_cext_update(solutions: list[dict], c_ext_target: np.ndarray, omega: float) -> tuple[float, float]:
    counts = [int(np.asarray(sol["cext_state"]["c_ext_gl"]).shape[0]) for sol in solutions]
    old_global = np.concatenate([np.asarray(sol["cext_state"]["c_ext_gl"], dtype=np.float32) for sol in solutions], axis=0)
    target = np.asarray(c_ext_target, dtype=np.float32)
    if old_global.shape != target.shape:
        raise ValueError(f"Global Cext target shape {target.shape} does not match current state {old_global.shape}.")
    omega_eff = float(omega)
    updated = old_global + np.float32(omega_eff) * (target - old_global)
    delta = target - old_global
    finite_delta = delta[np.isfinite(delta)]
    max_delta = float(np.max(np.abs(finite_delta))) if finite_delta.size else 0.0
    denom = float(np.linalg.norm(target.reshape(-1)))
    rel_delta = float(np.linalg.norm(delta.reshape(-1)) / max(denom, 1.0e-30)) if delta.size else 0.0
    offset = 0
    for sol, count in zip(solutions, counts):
        sol["cext_state"]["c_ext_gl"] = np.asarray(updated[offset: offset + count], dtype=np.float32)
        offset += count
    return max_delta, rel_delta


def _snapshot_sol_cext_state(cext_ts, sol: dict, *, solver: str, backend: str) -> None:
    if hasattr(cext_ts, "_snapshot_cext_source_state"):
        snap = cext_ts._snapshot_cext_source_state(
            sol["cext_context"],
            sol["cext_state"],
            solver=solver,
            backend=backend,
        )
        snap["cin_seg"] = np.asarray(sol["cin"], dtype=np.float32)
        snap["cout_seg"] = np.asarray(sol["cout"], dtype=np.float32)
        sol["cext_state"] = snap


def _solve_tree_cext_backend(
    cext_ts,
    ts,
    tree,
    inlet_flow_cm3_s: float,
    *,
    fluid: str,
    concentration_solver: str,
) -> dict:
    inlet_concentration = float(ts.get_concentration_inlet(fluid))
    _log(
        f"  CASCADE Cext backend tree: segments={int(tree.segment_count)} terminals={int(tree.n_terminals)} "
        f"qin={inlet_flow_cm3_s:.9g} cm3/s solver={concentration_solver}"
    )
    t0 = perf_counter()
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
    ) = ts.recompute_tree_flows(tree, inlet_flow_cm3_s, fluid=fluid)
    _log(f"    flow recompute completed in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    cin, cout = cext_ts._solve_channel_concentrations(
        tree,
        np.asarray(flows),
        inlet_nodes,
        outlet_nodes,
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(radii),
        np.asarray(lengths),
        prox_ids=prox_ids,
        dist_ids=dist_ids,
        inlet_concentration=inlet_concentration,
        diffusivity=float(ts.SOLUTE_DIFFUSIVITY),
        vmax=float(ts.VMAX_MM),
        km=float(ts.K_M_MM),
        fluid=fluid,
        solver=concentration_solver,
    )
    _log(f"    CASCADE concentration/Cext solve completed in {perf_counter() - t0:.2f}s")
    cext_state = getattr(cext_ts, "_LAST_CEXT_SOURCE_STATE", None)
    if not isinstance(cext_state, dict):
        raise RuntimeError("CASCADE Cext backend did not expose _LAST_CEXT_SOURCE_STATE.")
    cext_state = dict(cext_state)
    cext_state["cin_seg"] = np.asarray(cin, dtype=np.float32)
    cext_state["cout_seg"] = np.asarray(cout, dtype=np.float32)
    p_in = float("nan")
    p_out = float("nan")
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
    return {
        "starts": np.asarray(starts),
        "ends": np.asarray(ends),
        "radii": np.asarray(radii),
        "lengths": np.asarray(lengths),
        "flows": np.asarray(flows),
        "cin": np.asarray(cin),
        "cout": np.asarray(cout),
        "pressures": np.asarray(pressures),
        "p_in": p_in,
        "p_out": p_out,
        "inlet_concentration": inlet_concentration,
        "cext_state": cext_state,
        "cext_timings": dict(getattr(cext_ts, "_LAST_CONCENTRATION_TIMINGS", {}) or {}),
    }


def _solve_forest_cext_backend_per_tree(
    cext_ts,
    ts,
    forest: Forest,
    inlet_flows: list[float],
    *,
    fluid: str,
    concentration_solver: str,
) -> tuple[list[dict], dict]:
    t_total = perf_counter()
    solutions = []
    for tree_id, (tree, inlet_flow) in enumerate(zip(forest.networks[0], inlet_flows)):
        _log(f"Solving CASCADE Cext backend tree {tree_id}...")
        solutions.append(
            _solve_tree_cext_backend(
                cext_ts,
                ts,
                tree,
                inlet_flow,
                fluid=fluid,
                concentration_solver=concentration_solver,
            )
        )
    cext_vals = [
        np.asarray(sol["cext_state"]["c_ext_gl"], dtype=np.float32).reshape(-1)
        for sol in solutions
        if np.asarray(sol["cext_state"].get("c_ext_gl", ())).size
    ]
    all_cext = np.concatenate(cext_vals) if cext_vals else np.empty((0,), dtype=np.float32)
    timings = [dict(sol.get("cext_timings", {})) for sol in solutions]
    return solutions, {
        "enabled": True,
        "mode": "backend_per_tree",
        "solver": str(concentration_solver),
        "backend": str(getattr(cext_ts, "CEXT_ACCEL_MODE", "")),
        "max_sampled_cext": float(np.nanmax(all_cext)) if all_cext.size else 0.0,
        "mean_sampled_cext": float(np.nanmean(all_cext)) if all_cext.size else 0.0,
        "tree_timings": timings,
        "total_s": float(perf_counter() - t_total),
    }



def _release_cext_transient_gpu_cache(cext_ts, *contexts) -> dict:
    released = {
        "contexts_cleared": 0,
        "module_attrs_cleared": [],
        "cupy_pool_trimmed": False,
    }
    for context in contexts:
        if not isinstance(context, dict):
            continue
        for key in ("hybrid_bg_context", "gpu_static", "gpu_geometry_static"):
            cached = context.pop(key, None)
            if isinstance(cached, dict):
                cached.clear()
                released["contexts_cleared"] += 1
    if cext_ts is not None:
        for attr in ("_LAST_CEXT_SOURCE_STATE", "_LAST_CEXT_CONTEXT", "_LAST_TISSUE_TIMINGS"):
            if hasattr(cext_ts, attr):
                try:
                    setattr(cext_ts, attr, None)
                    released["module_attrs_cleared"].append(attr)
                except Exception:
                    pass
        cp = getattr(cext_ts, "_cp", None)
        if cp is not None:
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
            try:
                cp.get_default_memory_pool().free_all_blocks()
                released["cupy_pool_trimmed"] = True
            except Exception:
                pass
            try:
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
    gc.collect()
    return released


def _solve_forest_cext_shared_global(
    cext_ts,
    ts,
    forest: Forest,
    inlet_flows: list[float],
    *,
    fluid: str,
) -> tuple[list[dict], dict]:
    if cext_ts._cp is None:
        raise RuntimeError("Cext mode requires GPU/CuPy support. Install CuPy or rerun with --no-cext.")
    t_total = perf_counter()
    solutions = []
    initial_cache_releases = []
    for tree_id, (tree, inlet_flow) in enumerate(zip(forest.networks[0], inlet_flows)):
        _log(f"Solving Cext initial state for tree {tree_id}...")
        sol = _solve_tree_cext_prepare(cext_ts, ts, tree, inlet_flow, fluid=fluid)
        solutions.append(sol)
        initial_cache_releases.append(_release_cext_transient_gpu_cache(cext_ts, sol.get("cext_context")))
    if str(getattr(cext_ts, "CEXT_VESS_COUPLING_ACCEL", "none")).strip().lower() != "none":
        _log(
            "Shared-global forest Cext uses the CASCADE field evaluator with omega relaxation; "
            "use --cext-forest-mode backend-per-tree for the backend's Anderson/active-set loop."
        )
    bg_mode = (
        str(cext_ts._resolve_cext_hybrid_bg_mode())
        if hasattr(cext_ts, "_resolve_cext_hybrid_bg_mode")
        else str(getattr(cext_ts, "CEXT_HYBRID_BG_MODE", "fft")).strip().lower()
    )
    _log(
        "Computing shared CASCADE forest-global Cext field: "
        f"trees={len(solutions)} grid={cext_ts.CEXT_HYBRID_BG_GRID} "
        f"bins={cext_ts.CEXT_HYBRID_BG_LAMBDA_BINS} assignment={cext_ts.CEXT_HYBRID_BG_ASSIGNMENT} "
        f"bg_mode={bg_mode} max_iter={cext_ts.CEXT_VESS_COUPLING_MAX_ITER}"
    )

    max_iter = max(int(getattr(cext_ts, "CEXT_VESS_COUPLING_MAX_ITER", 0)), 0)
    n_evals = max(max_iter, 1)
    omega = float(getattr(cext_ts, "CEXT_VESS_COUPLING_OMEGA", 1.0))
    tol = float(getattr(cext_ts, "CEXT_VESS_COUPLING_TOL", 0.0))
    rel_tol = float(getattr(cext_ts, "CEXT_VESS_COUPLING_REL_TOL", 0.0))
    iteration_timings = []
    grid_meta = {}
    completed_iters = 0
    max_delta_last = float("inf")
    rel_delta_last = float("inf")
    global_context, _ = _concat_global_cext_context(solutions)
    for iter_idx in range(1, n_evals + 1):
        t_iter = perf_counter()
        c_ext_new, timings, grid_meta = _compute_global_gfm_cext(cext_ts, solutions, global_context)
        max_delta_last, rel_delta_last = _apply_global_cext_update(solutions, c_ext_new, omega)
        reflected_total = 0.0
        for tree_id, sol in enumerate(solutions):
            t_reflect = perf_counter()
            _run_cext_frozen_step(cext_ts, sol, fluid=fluid)
            elapsed = float(perf_counter() - t_reflect)
            sol["cext_reflected_topdown_s"] = float(sol.get("cext_reflected_topdown_s", 0.0) + elapsed)
            reflected_total += elapsed
        completed_iters = iter_idx
        timings.update(
            {
                "iteration": int(iter_idx),
                "max_delta": float(max_delta_last),
                "rel_delta": float(rel_delta_last),
                "reflected_topdown_s": float(reflected_total),
                "total_iteration_s": float(perf_counter() - t_iter),
            }
        )
        iteration_timings.append(timings)
        _log(
            f"  Shared Cext iter {iter_idx}/{n_evals}: "
            f"max_delta={max_delta_last:.3e} rel={rel_delta_last:.3e} "
            f"deposit={timings.get('deposit_s', 0.0):.2f}s fft={timings.get('fft_s', 0.0):.2f}s "
            f"local={timings.get('local_corr_s', 0.0):.2f}s sample={timings.get('sample_s', 0.0):.2f}s "
            f"reflected={reflected_total:.2f}s"
        )
        if max_delta_last < tol:
            break
        if rel_tol > 0.0 and rel_delta_last < rel_tol:
            break

    for sol in solutions:
        _snapshot_sol_cext_state(cext_ts, sol, solver=f"shared_global_gfm_{bg_mode}", backend="gpu")

    cext_vals = [
        np.asarray(sol["cext_state"]["c_ext_gl"], dtype=np.float32).reshape(-1)
        for sol in solutions
        if np.asarray(sol["cext_state"]["c_ext_gl"]).size
    ]
    all_cext = np.concatenate(cext_vals) if cext_vals else np.empty((0,), dtype=np.float32)
    cext_meta = {"timings": {}, "grid": grid_meta}
    cext_meta.update(
        {
            "enabled": True,
            "mode": "shared_global_gfm",
            "bg_mode": str(bg_mode),
            "near_radius_mult": float(getattr(cext_ts, "CEXT_HYBRID_BG_NEAR_RADIUS_MULT", 0.0)),
            "max_iter": int(max_iter),
            "completed_iters": int(completed_iters),
            "omega": float(omega),
            "max_delta_last": float(max_delta_last),
            "rel_delta_last": float(rel_delta_last),
            "iteration_timings": iteration_timings,
            "initial_tree_gpu_cache_releases": initial_cache_releases,
            "grid": grid_meta,
            "max_sampled_cext": float(np.nanmax(all_cext)) if all_cext.size else 0.0,
            "mean_sampled_cext": float(np.nanmean(all_cext)) if all_cext.size else 0.0,
            "total_s": float(perf_counter() - t_total),
        }
    )
    cext_meta["post_cext_gpu_cache_release"] = _release_cext_transient_gpu_cache(cext_ts, global_context)
    return solutions, cext_meta


def _solve_forest_cext(
    cext_ts,
    ts,
    forest: Forest,
    inlet_flows: list[float],
    *,
    fluid: str,
    args,
) -> tuple[list[dict], dict]:
    mode = str(args.cext_forest_mode).strip().lower()
    if mode == "backend-per-tree":
        return _solve_forest_cext_backend_per_tree(
            cext_ts,
            ts,
            forest,
            inlet_flows,
            fluid=fluid,
            concentration_solver=str(args.cext_concentration_solver),
        )
    if mode == "legacy-shared-one-shot":
        return _solve_forest_cext_legacy_shared_one_shot(cext_ts, ts, forest, inlet_flows, fluid=fluid)
    return _solve_forest_cext_shared_global(cext_ts, ts, forest, inlet_flows, fluid=fluid)


def _compact_cext_state_for_concat(sol: dict) -> dict:
    context = sol.get("cext_context")
    state = sol["cext_state"]
    if context is not None:
        gl_points = np.asarray(context["gl_points_si"], dtype=np.float32)
        diffusivity_si = float(context["diffusivity_si"])
        segment_vectors = np.asarray(context.get("segment_vectors", np.zeros((gl_points.shape[0], 3), dtype=np.float32)), dtype=np.float32)
    else:
        gl_points = np.asarray(state["gl_points_si"], dtype=np.float32)
        diffusivity_si = float(state["diffusivity_si"])
        segment_vectors = np.asarray(state.get("segment_vectors", np.zeros((gl_points.shape[0], 3), dtype=np.float32)), dtype=np.float32)
    q_weighted = np.asarray(state["q_weighted_gl"], dtype=np.float32)
    return {
        "solver": str(state.get("solver", "cext_state")),
        "backend": str(state.get("backend", "gpu")),
        "gl_points_si": gl_points,
        "diffusivity_si": diffusivity_si,
        "window_factor": float(state.get("window_factor", WINDOW_FACTOR_DEFAULT)),
        "c_iv_gl": np.asarray(state["c_iv_gl"], dtype=np.float32),
        "c_bulk_gl": np.asarray(state.get("c_bulk_gl", state["c_iv_gl"]), dtype=np.float32),
        "c_wall_gl": np.asarray(state.get("c_wall_gl", state["c_iv_gl"]), dtype=np.float32),
        "c_ext_gl": np.asarray(state["c_ext_gl"], dtype=np.float32),
        "lambda_iv_gl": np.asarray(state["lambda_iv_gl"], dtype=np.float32),
        "k_if_gl": np.asarray(state["k_if_gl"], dtype=np.float32),
        "q_line_gl": np.asarray(state["q_line_gl"], dtype=np.float32),
        "q_weighted_gl": q_weighted,
        "mono2_weight_gl": np.asarray(state.get("mono2_weight_gl", np.zeros_like(q_weighted)), dtype=np.float32),
        "dipole2_weight_gl": np.asarray(state.get("dipole2_weight_gl", np.zeros_like(q_weighted)), dtype=np.float32),
        "seg_cap_gl": np.asarray(state["seg_cap_gl"], dtype=np.float32),
        "segment_vectors": segment_vectors,
    }


def _concat_tree_solutions(solutions: list[dict], *, index_dtype: np.dtype) -> dict:
    out = {}
    for key in ("starts", "ends", "radii", "lengths", "flows", "cin", "cout"):
        out[key] = np.concatenate([s[key] for s in solutions], axis=0) if solutions else np.empty((0,))
    if solutions and all(isinstance(s.get("cext_state"), dict) for s in solutions):
        compact = [_compact_cext_state_for_concat(s) for s in solutions]
        out["cext_state"] = {
            "solver": str(compact[0]["solver"]),
            "backend": str(compact[0]["backend"]),
            "gl_points_si": np.concatenate([s["gl_points_si"] for s in compact], axis=0),
            "diffusivity_si": float(compact[0]["diffusivity_si"]),
            "window_factor": float(compact[0]["window_factor"]),
            "c_iv_gl": np.concatenate([s["c_iv_gl"] for s in compact], axis=0),
            "c_bulk_gl": np.concatenate([s["c_bulk_gl"] for s in compact], axis=0),
            "c_wall_gl": np.concatenate([s["c_wall_gl"] for s in compact], axis=0),
            "c_ext_gl": np.concatenate([s["c_ext_gl"] for s in compact], axis=0),
            "lambda_iv_gl": np.concatenate([s["lambda_iv_gl"] for s in compact], axis=0),
            "k_if_gl": np.concatenate([s["k_if_gl"] for s in compact], axis=0),
            "q_line_gl": np.concatenate([s["q_line_gl"] for s in compact], axis=0),
            "q_weighted_gl": np.concatenate([s["q_weighted_gl"] for s in compact], axis=0),
            "mono2_weight_gl": np.concatenate([s["mono2_weight_gl"] for s in compact], axis=0),
            "dipole2_weight_gl": np.concatenate([s["dipole2_weight_gl"] for s in compact], axis=0),
            "seg_cap_gl": np.concatenate([s["seg_cap_gl"] for s in compact], axis=0),
            "segment_vectors": np.concatenate([s["segment_vectors"] for s in compact], axis=0),
            "export_slim_state": True,
        }
        out["cext_mean"] = np.concatenate(
            [np.mean(np.asarray(s["c_ext_gl"], dtype=np.float32), axis=1) for s in compact],
            axis=0,
        )
        out["c_iv_minus_cext_mean"] = np.concatenate(
            [
                np.mean(np.asarray(s["c_iv_gl"], dtype=np.float32) - np.asarray(s["c_ext_gl"], dtype=np.float32), axis=1)
                for s in compact
            ],
            axis=0,
        )
    tree_ids = []
    local_ids = []
    for tree_id, sol in enumerate(solutions):
        n = int(sol["starts"].shape[0])
        tree_ids.append(np.full(n, tree_id, dtype=np.int16))
        local_ids.append(_safe_index_array(np.arange(n, dtype=np.int64), index_dtype))
    out["tree_id"] = np.concatenate(tree_ids) if tree_ids else np.empty((0,), dtype=np.int16)
    out["local_segment_id"] = np.concatenate(local_ids) if local_ids else np.empty((0,), dtype=index_dtype)
    out["global_segment_id"] = _safe_index_array(np.arange(out["tree_id"].shape[0], dtype=np.int64), index_dtype)
    return out


def _build_vessel_polydata(combo: dict, *, resolution: int, float_dtype: np.dtype, index_dtype: np.dtype) -> pv.PolyData:
    starts = np.asarray(combo["starts"], dtype=float_dtype)
    ends = np.asarray(combo["ends"], dtype=float_dtype)
    nseg = int(starts.shape[0])
    if nseg == 0:
        return pv.PolyData()
    res = max(int(resolution), 2)
    t = np.linspace(0.0, 1.0, res, dtype=float_dtype)
    _log(f"Building vessel VTP arrays: segments={nseg} resolution={res}")
    points = (starts[:, None, :] + t[None, :, None] * (ends - starts)[:, None, :]).reshape(nseg * res, 3)
    line_ids = np.arange(nseg * res, dtype=np.int64).reshape(nseg, res)
    lines = np.empty((nseg, res + 1), dtype=np.int64)
    lines[:, 0] = res
    lines[:, 1:] = line_ids
    poly = pv.PolyData(points, lines=lines.reshape(-1))
    cin = np.asarray(combo["cin"], dtype=float_dtype)
    cout = np.asarray(combo["cout"], dtype=float_dtype)
    flow = np.asarray(combo["flows"], dtype=float_dtype)
    radii = np.asarray(combo["radii"], dtype=float_dtype)
    lengths = np.asarray(combo["lengths"], dtype=float_dtype)
    tree_ids = np.asarray(combo["tree_id"], dtype=np.int16)
    local_ids = np.asarray(combo["local_segment_id"], dtype=index_dtype)
    global_ids = np.asarray(combo["global_segment_id"], dtype=index_dtype)
    conc_samples = (cin[:, None] + t[None, :] * (cout - cin)[:, None]).reshape(nseg * res)
    poly.point_data["concentration"] = conc_samples
    if "cext_mean" in combo:
        cext_mean = np.asarray(combo["cext_mean"], dtype=float_dtype)
        poly.point_data["cext_mean"] = np.repeat(cext_mean, res)
    if "c_iv_minus_cext_mean" in combo:
        delta_mean = np.asarray(combo["c_iv_minus_cext_mean"], dtype=float_dtype)
        poly.point_data["c_iv_minus_cext_mean"] = np.repeat(delta_mean, res)
    poly.point_data["flow_cm3_s"] = np.repeat(flow, res)
    poly.point_data["flow_ul_min"] = np.repeat(flow * 60000.0, res).astype(float_dtype, copy=False)
    poly.point_data["radius"] = np.repeat(radii, res)
    poly.point_data["length"] = np.repeat(lengths, res)
    poly.point_data["tree_id"] = np.repeat(tree_ids, res)
    poly.point_data["local_segment_id"] = np.repeat(local_ids, res)
    poly.point_data["global_segment_id"] = np.repeat(global_ids, res)
    return poly


def _build_points_polydata(points: np.ndarray, metrics: dict[str, np.ndarray], *, float_dtype: np.dtype, index_dtype: np.dtype) -> pv.PolyData:
    points = np.asarray(points, dtype=float_dtype)
    if points.size == 0:
        return pv.PolyData()
    n = points.shape[0]
    verts = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).reshape(-1)
    poly = pv.PolyData(points, verts=verts)
    poly.point_data["point_id"] = _safe_index_array(np.arange(n, dtype=np.int64), index_dtype)
    poly.point_data["x"] = points[:, 0]
    poly.point_data["y"] = points[:, 1]
    poly.point_data["z"] = points[:, 2]
    for key, values in metrics.items():
        arr = np.asarray(values)
        if arr.shape[0] != n:
            raise ValueError(f"Metric {key} length {arr.shape[0]} != point count {n}")
        if arr.dtype.kind == "f":
            arr = arr.astype(float_dtype, copy=False)
        elif key in {"tree_id", "closest_tree_id"}:
            arr = arr.astype(np.int16, copy=False)
        elif key.endswith("_id") or key in {"point_id", "local_segment_id", "global_segment_id"}:
            arr = _safe_index_array(arr.astype(np.int64, copy=False), index_dtype)
        poly.point_data[key] = arr
    return poly


def _filter_export_points(points: np.ndarray, metrics: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    if points.shape[0] != keep.shape[0]:
        raise ValueError(f"Export mask length {keep.shape[0]} != point count {points.shape[0]}")
    filtered_metrics: dict[str, np.ndarray] = {}
    for key, values in metrics.items():
        arr = np.asarray(values)
        if arr.shape[0] != keep.shape[0]:
            raise ValueError(f"Metric {key} length {arr.shape[0]} != export mask length {keep.shape[0]}")
        filtered_metrics[key] = arr[keep]
    return points[keep], filtered_metrics


def _get_boundary(domain, boundary_resolution: int) -> pv.PolyData:
    boundary = getattr(domain, "boundary", None)
    if boundary is None and getattr(domain, "mesh", None) is not None:
        boundary = domain.mesh.extract_surface()
    if boundary is None:
        domain.build(resolution=int(boundary_resolution), skip_boundary=False)
        boundary = domain.boundary
    if boundary is None:
        raise RuntimeError("Could not obtain domain boundary.")
    if not boundary.is_all_triangles:
        boundary = boundary.triangulate()
    return boundary.clean()


def _grid_axes(boundary: pv.PolyData, nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bmin = np.min(boundary.points, axis=0).astype(np.float64)
    bmax = np.max(boundary.points, axis=0).astype(np.float64)
    x = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    y = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    z = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)
    return x, y, z


def _grid_points_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def _grid_points(boundary: pv.PolyData, nx: int, ny: int, nz: int) -> np.ndarray:
    return _grid_points_from_axes(*_grid_axes(boundary, nx, ny, nz))


def _inside_mask(domain, boundary: pv.PolyData, points: np.ndarray, args) -> np.ndarray:
    implicit = np.asarray(domain(points)).reshape(-1) <= -float(args.implicit_margin)
    if bool(args.disable_enclosed_check):
        return implicit
    try:
        cloud = pv.PolyData(points.astype(np.float64))
        selected = cloud.select_enclosed_points(
            boundary,
            tolerance=float(args.enclosed_tolerance),
            check_surface=False,
        )
        enclosed = np.asarray(selected.point_data["SelectedPoints"]).astype(bool).reshape(-1)
        if args.inside_combine_mode == "or":
            return np.logical_or(implicit, enclosed)
        return np.logical_and(implicit, enclosed)
    except Exception as exc:
        _log(f"Warning: enclosed-point check failed ({exc}); using implicit-only mask.")
        return implicit


def _inside_grid_points_chunked(domain, boundary: pv.PolyData, args) -> np.ndarray:
    x, y, z = _grid_axes(boundary, int(args.nx), int(args.ny), int(args.nz))
    total_points = int(x.size * y.size * z.size)
    chunk_points = max(int(getattr(args, "tissue_grid_chunk_points", 250_000)), 1)
    yz_count = max(int(y.size * z.size), 1)
    x_step = max(1, min(int(x.size), chunk_points // yz_count))
    chunks = []
    inside_total = 0
    with tqdm(total=total_points, desc="Filtering tissue grid", unit="pt") as progress:
        for start in range(0, int(x.size), x_step):
            stop = min(start + x_step, int(x.size))
            chunk = _grid_points_from_axes(x[start:stop], y, z)
            implicit = np.asarray(domain(chunk)).reshape(-1) <= -float(args.implicit_margin)
            if np.any(implicit):
                inside_chunk = chunk[implicit]
                chunks.append(inside_chunk)
                inside_total += int(inside_chunk.shape[0])
            progress.update(int(chunk.shape[0]))
            del chunk, implicit
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    _log(f"Chunked inside-domain filter retained {inside_total} / {total_points} points")
    return np.concatenate(chunks, axis=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tissue_points(path_value: str | Path) -> tuple[np.ndarray, dict]:
    path = _normalize_path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tissue point file does not exist: {path}")
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Tissue point array must have shape (N, 3); got {points.shape} from {path}"
        )
    if points.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"Tissue point array must be numeric; got dtype {points.dtype} from {path}")
    points = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Tissue point array contains NaN or infinite coordinates: {path}")
    return points, {
        "mode": "explicit_npy",
        "path": str(path),
        "sha256": _sha256_file(path),
        "count": int(points.shape[0]),
        "coordinate_units": "cm",
    }


def _compute_tissue(
    ts,
    cext_ts,
    combo: dict,
    domain,
    args,
    *,
    float_dtype: np.dtype,
    index_dtype: np.dtype,
) -> tuple[pv.PolyData, dict]:
    if args.tissue_points is not None:
        points, point_source = _load_tissue_points(args.tissue_points)
        total_grid_points = int(points.shape[0])
        _log(
            f"Explicit tissue points: {total_grid_points} from {point_source['path']} "
            f"(sha256={point_source['sha256']})"
        )
    else:
        boundary = _get_boundary(domain, int(args.boundary_resolution))
        total_grid_points = int(args.nx) * int(args.ny) * int(args.nz)
        point_source = {
            "mode": "generated_grid",
            "shape": [int(args.nx), int(args.ny), int(args.nz)],
            "candidate_count": total_grid_points,
        }
        _log(f"Tissue grid: {args.nx} x {args.ny} x {args.nz} = {total_grid_points} points")
        if bool(args.disable_enclosed_check):
            points = _inside_grid_points_chunked(domain, boundary, args)
        else:
            _log("Using PyVista enclosed-point check; this is CPU-heavy for large grids.")
            grid = _grid_points(boundary, int(args.nx), int(args.ny), int(args.nz))
            inside = _inside_mask(domain, boundary, grid, args)
            points = grid[inside]
            del grid, inside
    _log(f"Inside-domain tissue points: {points.shape[0]}")
    gc.collect()

    tissue_cache = None
    if bool(args.include_tissue_nearest_fields):
        max_nearby = min(int(getattr(ts, "NEAREST_TISSUE_VESSELS", 250)), int(combo["starts"].shape[0]))
        _log(f"Building tissue geometry cache for nearest fields (max_nearby={max_nearby})...")
        tissue_cache = ts._prepare_tissue_geometry(
            points,
            combo["starts"],
            combo["ends"],
            combo["radii"],
            max_nearby=max_nearby,
        )

    if "cext_state" in combo:
        if cext_ts is None:
            raise RuntimeError("Combined solution contains Cext state, but the Cext module was not loaded.")
        _log("Computing tissue oxygen field from Cext source state...")
        keep_mask, tissue_conc = cext_ts.compute_tissue_samples_greens_from_cext_state(
            points,
            combo["starts"],
            combo["ends"],
            combo["radii"],
            combo["cext_state"],
            tissue_cache=tissue_cache,
        )
        tissue_timings = dict(getattr(cext_ts, "_LAST_TISSUE_TIMINGS", {}) or {})
    else:
        _log("Computing tissue oxygen field...")
        keep_mask, tissue_conc = ts.compute_tissue_samples_greens(
            points,
            combo["starts"],
            combo["ends"],
            combo["radii"],
            combo["cin"],
            combo["flows"],
            diffusivity=float(ts.SOLUTE_DIFFUSIVITY),
            vmax=float(ts.VMAX_MM),
            km=float(ts.K_M_MM),
            window_factor=float(ts.WINDOW_FACTOR),
            inlet_concentration=float(ts.get_concentration_inlet(args.fluid)),
            tissue_cache=tissue_cache,
        )
        tissue_timings = dict(getattr(ts, "_LAST_TISSUE_TIMINGS", {}) or {})
    conc = np.asarray(tissue_conc, dtype=float_dtype)
    keep = np.asarray(keep_mask, dtype=bool)
    conc_masked = conc.copy()
    conc_masked[~keep] = np.nan
    conc_max = float(getattr(ts, "CONC_MAX_FOR_NORMALIZATION", np.nan))
    if np.isfinite(conc_max) and conc_max > 0.0:
        conc_norm = (conc_masked / conc_max).astype(float_dtype, copy=False)
        viability = (conc >= 0.01 * conc_max).astype(np.int8)
    else:
        conc_norm = np.full(conc_masked.shape, np.nan, dtype=float_dtype)
        viability = np.zeros(conc_masked.shape, dtype=np.int8)

    metrics: dict[str, np.ndarray] = {
        "inside_tissue": keep.astype(np.uint8),
        "local_concentration": conc_masked,
        "local_concentration_raw": conc,
        "local_concentration_norm": conc_norm,
        "viability": viability,
    }

    if tissue_cache is not None:
        nearest_idx = np.asarray(tissue_cache["nearest_idx"])
        d_center = np.asarray(tissue_cache["d_center"], dtype=np.float64)
        valid = np.asarray(tissue_cache["valid_mask"], dtype=bool)
        valid_global = np.flatnonzero(valid).astype(np.int64)
        if nearest_idx.size and d_center.size:
            local = np.argmin(d_center, axis=1)
            row = np.arange(nearest_idx.shape[0])
            seg_valid = nearest_idx[row, local].astype(np.int64)
            seg_valid = np.clip(seg_valid, 0, max(valid_global.size - 1, 0))
            closest_global = valid_global[seg_valid]
            metrics["closest_segment_id"] = _safe_index_array(closest_global, index_dtype)
            metrics["closest_tree_id"] = np.asarray(combo["tree_id"], dtype=np.int16)[closest_global]
            metrics["closest_flow_ul_min"] = (
                np.asarray(combo["flows"], dtype=np.float64)[closest_global] * 60000.0
            ).astype(float_dtype)
        _log("Computing DNC field...")
        metrics["dnc_cm"] = ts.compute_distance_to_nearest_channel(
            points,
            combo["starts"],
            combo["ends"],
            combo["radii"],
        ).astype(float_dtype)

    export_keep = keep & np.all(np.isfinite(points), axis=1)
    dropped = int(points.shape[0] - np.count_nonzero(export_keep))
    if dropped:
        _log(f"Dropping {dropped} tissue export points with invalid samples or non-finite coordinates.")
    export_points, export_metrics = _filter_export_points(points, metrics, export_keep)

    points_poly = _build_points_polydata(export_points, export_metrics, float_dtype=float_dtype, index_dtype=index_dtype)
    meta = {
        "n_grid_points_inside": int(points.shape[0]),
        "n_tissue_points": int(np.sum(keep)),
        "n_tissue_points_exported": int(export_points.shape[0]),
        "conc_max_for_normalization": conc_max,
        "source_mode": "cext_state" if "cext_state" in combo else "legacy_vessel_cin_flow",
        "point_source": point_source,
        "tissue_timings": tissue_timings,
    }
    return points_poly, meta


def _save_domain_outputs(domain, out_dir: Path, prefix: str) -> dict:
    outputs = {}
    boundary = getattr(domain, "boundary", None)
    if boundary is None and getattr(domain, "mesh", None) is not None:
        boundary = domain.mesh.extract_surface()
    if boundary is not None:
        path = out_dir / f"{prefix}_domain_boundary.vtp"
        boundary.save(str(path))
        outputs["domain_boundary"] = str(path)
    mesh = getattr(domain, "mesh", None)
    if mesh is not None:
        path = out_dir / f"{prefix}_domain_mesh.vtu"
        mesh.save(str(path))
        outputs["domain_mesh"] = str(path)
    return outputs


@guard_simulation("cascade export-heart")
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cascade export-heart",
        description="Export a two-tree heart .forest plus tissue oxygen grid with optional shared FFT Cext for ParaView/ParaFlow.",
    )
    parser.add_argument("--forest", required=True, help="Input .forest path.")
    parser.add_argument("--no-simulation-cache", action="store_true", help="Disable automatic simulation-cache read/write.")
    parser.add_argument("--domain", dest="domain_path", required=True, help="Heart domain path (.stl or .dmn).")
    parser.add_argument("--domain-stl", dest="domain_path", help="Deprecated alias for --domain.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to forest stem plus _cext when Cext is enabled.")
    parser.add_argument("--side-length", type=float, default=1.0)
    parser.add_argument("--fluid", default="blood", choices=("blood", "water", "media", "cell media"))
    parser.add_argument(
        "--concentration-solver",
        default="topdown",
        choices=("topdown", "network", "topdown_ext", "topdown_ext_hybrid_bg", "topdown_ext_treecode"),
        help="Legacy no-Cext concentration solver. Cext mode uses --cext-concentration-solver where applicable.",
    )
    parser.add_argument("--no-cext", action="store_true", help="Disable Cext and use the legacy per-tree exporter solve.")
    parser.add_argument(
        "--cext-tissuesim",
        default=None,
        help="Optional external TissueSim-compatible Cext module. Defaults to CASCADE's packaged runtime.",
    )
    parser.add_argument(
        "--cext-forest-mode",
        default=CEXT_FOREST_MODE_DEFAULT,
        choices=("shared-global", "shared-global-fft", "backend-per-tree", "legacy-shared-one-shot"),
        help=(
            "shared-global uses one combined two-tree CASCADE source/target context for FFT, pairwise, or hybrid modes; "
            "shared-global-fft is a compatibility alias requiring bg_mode=fft; "
            "backend-per-tree uses TissueSim_cube_local's exact per-tree solver loop."
        ),
    )
    parser.add_argument(
        "--cext-concentration-solver",
        default=CEXT_CONCENTRATION_SOLVER_DEFAULT,
        choices=("topdown_ext_hybrid_bg", "topdown_ext_treecode", "topdown_ext"),
        help="Concentration solver used by --cext-forest-mode backend-per-tree.",
    )
    parser.add_argument("--cext-accel", default=CEXT_ACCEL_MODE_DEFAULT, choices=("cpu", "gpu", "auto"))
    parser.add_argument("--cext-frozen-accel", default=CEXT_FROZEN_ACCEL_MODE_DEFAULT, choices=("cpu", "gpu", "auto"))
    parser.add_argument("--cext-init-mode", default=CEXT_INIT_MODE_DEFAULT, choices=("zero", "decoupled_greens"))
    parser.add_argument("--cext-lambda-source", default=CEXT_LAMBDA_SOURCE_DEFAULT, choices=("lambda_vv", "lambda_t"))
    parser.add_argument("--cext-window-factor", type=float, default=WINDOW_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-max-iter", type=int, default=CEXT_VESS_COUPLING_MAX_ITER_DEFAULT)
    parser.add_argument("--cext-vess-coupling-tol", type=float, default=CEXT_VESS_COUPLING_TOL_DEFAULT)
    parser.add_argument("--cext-vess-coupling-rel-tol", type=float, default=CEXT_VESS_COUPLING_REL_TOL_DEFAULT)
    parser.add_argument("--cext-vess-coupling-omega", type=float, default=CEXT_VESS_COUPLING_OMEGA_DEFAULT)
    parser.add_argument("--cext-vess-coupling-accel", default=CEXT_VESS_COUPLING_ACCEL_DEFAULT, choices=("none", "aitken", "anderson"))
    parser.add_argument("--cext-vess-coupling-omega-min", type=float, default=CEXT_VESS_COUPLING_OMEGA_MIN_DEFAULT)
    parser.add_argument("--cext-vess-coupling-omega-max", type=float, default=CEXT_VESS_COUPLING_OMEGA_MAX_DEFAULT)
    parser.add_argument("--cext-vess-coupling-trust-abs", type=float, default=CEXT_VESS_COUPLING_TRUST_ABS_DEFAULT)
    parser.add_argument("--cext-vess-coupling-trust-rel", type=float, default=CEXT_VESS_COUPLING_TRUST_REL_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-depth", type=int, default=CEXT_VESS_COUPLING_ANDERSON_DEPTH_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-reg", type=float, default=CEXT_VESS_COUPLING_ANDERSON_REG_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-start", type=int, default=CEXT_VESS_COUPLING_ANDERSON_START_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-gate-ratio", type=float, default=CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-min-stable-iters", type=int, default=CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-omega-gate-factor", type=float, default=CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-accept-factor", type=float, default=CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-step-factor", type=float, default=CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-restart-factor", type=float, default=CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-best-stall-iters", type=int, default=CEXT_VESS_COUPLING_BEST_STALL_ITERS_DEFAULT)
    parser.add_argument("--cext-vess-coupling-best-revert-factor", type=float, default=CEXT_VESS_COUPLING_BEST_REVERT_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-step-reject-factor", type=float, default=CEXT_VESS_COUPLING_STEP_REJECT_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-step-retry-factor", type=float, default=CEXT_VESS_COUPLING_STEP_RETRY_FACTOR_DEFAULT)
    parser.add_argument("--cext-active-set-enable", default="true" if CEXT_ACTIVE_SET_ENABLE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-active-set-start", type=int, default=CEXT_ACTIVE_SET_START_DEFAULT)
    parser.add_argument("--cext-active-set-stable-iters", type=int, default=CEXT_ACTIVE_SET_STABLE_ITERS_DEFAULT)
    parser.add_argument("--cext-active-set-rel-tol", type=float, default=CEXT_ACTIVE_SET_REL_TOL_DEFAULT)
    parser.add_argument("--cext-active-set-abs-tol", type=float, default=CEXT_ACTIVE_SET_ABS_TOL_DEFAULT)
    parser.add_argument("--cext-active-set-refresh-period", type=int, default=CEXT_ACTIVE_SET_REFRESH_PERIOD_DEFAULT)
    parser.add_argument("--cext-active-set-min-active-count", type=int, default=CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT)
    parser.add_argument("--cext-active-set-min-active-fraction", type=float, default=CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION_DEFAULT)
    parser.add_argument("--cext-target-active-set-enable", default="true" if CEXT_TARGET_ACTIVE_SET_ENABLE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-target-active-set-start", type=int, default=CEXT_TARGET_ACTIVE_SET_START_DEFAULT)
    parser.add_argument("--cext-target-active-set-stable-iters", type=int, default=CEXT_TARGET_ACTIVE_SET_STABLE_ITERS_DEFAULT)
    parser.add_argument("--cext-target-active-set-rel-tol", type=float, default=CEXT_TARGET_ACTIVE_SET_REL_TOL_DEFAULT)
    parser.add_argument("--cext-target-active-set-abs-tol", type=float, default=CEXT_TARGET_ACTIVE_SET_ABS_TOL_DEFAULT)
    parser.add_argument("--cext-target-active-set-min-active-count", type=int, default=CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT)
    parser.add_argument("--cext-target-active-set-neighbor-pad", type=int, default=CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD_DEFAULT)
    parser.add_argument("--cext-hybrid-bg-mode", default=CEXT_BG_MODE_DEFAULT, choices=("local_only_nlambda", "local_only", "fft", "hybrid"))
    parser.add_argument("--cext-bg-grid", "--cext-hybrid-bg-grid", dest="cext_hybrid_bg_grid", type=int, default=CEXT_BG_GRID_DEFAULT)
    parser.add_argument("--cext-bg-lambda-bins", "--cext-hybrid-bg-lambda-bins", dest="cext_hybrid_bg_lambda_bins", type=int, default=CEXT_BG_LAMBDA_BINS_DEFAULT)
    parser.add_argument("--cext-bg-assignment", "--cext-hybrid-bg-assignment", dest="cext_hybrid_bg_assignment", default=CEXT_BG_ASSIGNMENT_DEFAULT, choices=("cic", "tsc"))
    parser.add_argument("--cext-bg-solver", "--cext-hybrid-bg-solver", dest="cext_hybrid_bg_solver", default=CEXT_BG_SOLVER_DEFAULT, choices=("auto", "fft", "jacobi"))
    parser.add_argument("--cext-hybrid-bg-near-radius-mult", type=float, default=CEXT_BG_NEAR_RADIUS_MULT_DEFAULT)
    parser.add_argument("--cext-hybrid-bg-vcycles", type=int, default=CEXT_BG_VCYCLES_DEFAULT)
    parser.add_argument("--cext-hybrid-bg-enable-source-freezing", default="true" if CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-quantile-bins", default="true" if CEXT_HYBRID_FFT_QUANTILE_BINS_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-o2-correction", default="true" if CEXT_HYBRID_FFT_O2_CORRECTION_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-self-subtract", default="true" if CEXT_HYBRID_FFT_SELF_SUBTRACT_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-self-sub-target-sampling", default=CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING_DEFAULT, choices=("cic", "tsc", "matched", "assignment", "source"))
    parser.add_argument("--cext-hybrid-fft-self-sub-scale", type=float, default=CEXT_HYBRID_FFT_SELF_SUB_SCALE_DEFAULT)
    parser.add_argument("--cext-hybrid-fft-bin-epoch-cache", default="true" if CEXT_HYBRID_FFT_BIN_EPOCH_CACHE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-response-batched", default="true" if CEXT_HYBRID_FFT_RESPONSE_BATCHED_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-o2-fused-ifft", default="true" if CEXT_HYBRID_FFT_O2_FUSED_IFFT_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-self-sub-fused", default="true" if CEXT_HYBRID_FFT_SELF_SUB_FUSED_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-o2-moment-batch", type=int, default=CEXT_HYBRID_FFT_O2_MOMENT_BATCH_DEFAULT)
    parser.add_argument("--cext-hybrid-gpu-iteration-cache", default="true" if CEXT_HYBRID_GPU_ITERATION_CACHE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-gpu-runtime-weights", default="true" if CEXT_HYBRID_GPU_RUNTIME_WEIGHTS_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-gpu-runtime-stencil", default="true" if CEXT_HYBRID_GPU_RUNTIME_STENCIL_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-gpu-runtime-moments", default="true" if CEXT_HYBRID_GPU_RUNTIME_MOMENTS_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-grid-cell-factor", type=float, default=CEXT_GRID_CELL_FACTOR_DEFAULT)
    parser.add_argument("--cext-streaming-target-candidate-slots", type=int, default=CEXT_STREAMING_TARGET_CANDIDATE_SLOTS_DEFAULT)
    parser.add_argument("--cext-gpu-validate-segments", type=int, default=CEXT_GPU_VALIDATE_SEGMENTS_DEFAULT)
    parser.add_argument("--cext-approx-window-scale", type=float, default=CEXT_APPROX_WINDOW_SCALE_DEFAULT)
    parser.add_argument("--cext-max-candidates-per-target", type=int, default=CEXT_MAX_CANDIDATES_PER_TARGET_DEFAULT)
    parser.add_argument("--finite-radius-o2-terms", default=FINITE_RADIUS_O2_TERMS_DEFAULT, choices=("none", "monopole", "dipole", "both"))
    parser.add_argument("--lumen-wall-closure", default=LUMEN_WALL_CLOSURE_DEFAULT, choices=("wellmixed", "graetz"))
    parser.add_argument("--graetz-n-radial", type=int, default=GRAETZ_N_RADIAL_DEFAULT)
    parser.add_argument("--graetz-n-modes", type=int, default=GRAETZ_N_MODES_DEFAULT)
    parser.add_argument("--graetz-max-fp-iters", type=int, default=GRAETZ_MAX_FP_ITERS_DEFAULT)
    parser.add_argument("--graetz-profile", default=GRAETZ_VELOCITY_PROFILE_DEFAULT, choices=("poiseuille", "plug"))
    parser.add_argument("--lumen-diffusivity-cm2-s", type=float, default=None)
    parser.add_argument("--cext-gl-order", type=int, default=CEXT_GL_ORDER_DEFAULT, choices=(1, 5, 9, 20))
    parser.add_argument(
        "--kirchhoff-bc-mode",
        default=None,
        choices=("legacy_equal_terminal_flow", "terminal_pressure"),
        help=(
            "Flow boundary condition mode forwarded to TissueSim_heart_accel. "
            "legacy_equal_terminal_flow prescribes total inlet flow and equal terminal sinks; "
            "terminal_pressure prescribes inlet flow with terminal pressures."
        ),
    )
    parser.add_argument(
        "--flow-source",
        default="tree-root-flow",
        choices=("tree-root-flow", "total-qin-split"),
        help=(
            "'tree-root-flow' uses the root_flow saved in each forest tree. "
            "'total-qin-split' overrides those magnitudes by splitting --total-qin-ul-min "
            "across trees in proportion to their saved root_flow."
        ),
    )
    parser.add_argument(
        "--total-qin-ul-min",
        type=float,
        default=None,
        help="Total inlet flow in uL/min, only used with --flow-source total-qin-split.",
    )
    parser.add_argument("--tissue-accel", default=TISSUE_ACCEL_MODE_DEFAULT, choices=("cpu", "gpu", "auto"))
    parser.add_argument("--tissue-gpu-chunk-points", type=int, default=TISSUE_GPU_CHUNK_POINTS_DEFAULT)
    parser.add_argument("--tissue-gpu-validate-points", type=int, default=TISSUE_GPU_VALIDATE_POINTS_DEFAULT)
    parser.add_argument("--solute-diffusivity", type=float, default=SOLUTE_DIFFUSIVITY_DEFAULT)
    parser.add_argument("--vmax-mm", type=float, default=VMAX_MM_DEFAULT)
    parser.add_argument("--km-mm", type=float, default=K_M_MM_DEFAULT)
    parser.add_argument("--nearest-tissue-vessels", type=int, default=NEAREST_TISSUE_VESSELS_DEFAULT)
    parser.add_argument("--window-factor", type=float, default=WINDOW_FACTOR_DEFAULT)
    parser.add_argument("--gl-order", type=int, default=GL_ORDER_DEFAULT, choices=(5, 9, 20))
    parser.add_argument(
        "--cext-tissue-quadrature-mode",
        default="independent",
        choices=("independent", "legacy_cext"),
        help=(
            "Use independent tissue quadrature (production default), or reuse the "
            "Cext source nodes to reproduce the historical oracle timing/field behavior."
        ),
    )
    parser.add_argument("--tissue-kdtree-candidate-mult", type=int, default=TISSUE_KDTREE_CANDIDATE_MULT_DEFAULT)
    parser.add_argument("--axial-blood-steps", type=int, default=AXIAL_BLOOD_STEPS_DEFAULT)
    parser.add_argument("--conc-max-for-normalization", type=float, default=CONC_MAX_FOR_NORMALIZATION_DEFAULT)
    parser.add_argument(
        "--hematocrit-model",
        default=HEMATOCRIT_MODEL_DEFAULT,
        choices=("uniform_tube", "pries_secomb"),
    )
    parser.add_argument("--hematocrit-flow-iterations", type=int, default=HEMATOCRIT_FLOW_ITERATIONS_DEFAULT)
    parser.add_argument("--hematocrit-relaxation", type=float, default=HEMATOCRIT_RELAXATION_DEFAULT)
    parser.add_argument("--hematocrit-qtol-nl-min", type=float, default=HEMATOCRIT_QTOL_NL_MIN_DEFAULT)
    parser.add_argument("--hematocrit-hdtol", type=float, default=HEMATOCRIT_HDTOL_DEFAULT)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=200)
    parser.add_argument(
        "--tissue-points",
        default=None,
        help=(
            "Optional .npy array with shape (N, 3) containing explicit tissue sample coordinates in cm. "
            "When provided, these points replace generated --nx/--ny/--nz grid filtering."
        ),
    )
    parser.add_argument("--boundary-resolution", type=int, default=28)
    parser.add_argument("--implicit-margin", type=float, default=0.0)
    parser.add_argument("--disable-enclosed-check", action="store_true", default=True)
    parser.add_argument("--enable-enclosed-check", dest="disable_enclosed_check", action="store_false")
    parser.add_argument("--tissue-grid-chunk-points", type=int, default=250_000)
    parser.add_argument("--enclosed-tolerance", type=float, default=1e-6)
    parser.add_argument("--inside-combine-mode", default="and", choices=("and", "or"))
    parser.add_argument("--vessel-resolution", type=int, default=2)
    parser.add_argument(
        "--skip-vessel-output",
        action="store_true",
        help="Skip writing the forest vessel VTP while still computing downstream tissue outputs.",
    )
    parser.add_argument(
        "--fraction-blocked-infarction",
        type=float,
        default=0.0,
        help=(
            "Solve-time infarction severity for the selected global vessel segment. "
            "0.0 = none, 0.5 = 50%% radius reduction, 1.0 = full occlusion "
            "(segment + downstream subtree forced to zero flow/concentration outputs)."
        ),
    )
    parser.add_argument(
        "--infarction-global-segment-id",
        type=int,
        default=17,
        help="Exporter global_segment_id to target for solve-time infarction.",
    )
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--skip-tissue-oxygen", action="store_true")
    parser.add_argument(
        "--report-global-segment-id",
        type=int,
        default=None,
        help="Record and log the solved flow for one global segment id.",
    )
    parser.add_argument("--include-tissue-nearest-fields", action="store_true")
    parser.add_argument("--validate-connectivity", action="store_true", default=False)
    parser.add_argument("--no-validate-connectivity", dest="validate_connectivity", action="store_false")
    parser.add_argument("--no-repair-connectivity", action="store_true", help="Do not repair stale parent columns before validation/solve.")
    parser.add_argument("--connectivity-geometry-atol", type=float, default=1e-6)
    parser.add_argument("--no-fail-connectivity", action="store_true")
    parser.add_argument(
        "--working-float-dtype",
        default=WORKING_FLOAT_DTYPE_DEFAULT,
        choices=("float32", "float64"),
        help="In-memory dtype used while loading and solving the forest.",
    )
    parser.add_argument(
        "--working-index-dtype",
        default=WORKING_INDEX_DTYPE_DEFAULT,
        choices=("int32", "int64"),
        help="In-memory dtype used for reconstructed connectivity arrays.",
    )
    parser.add_argument(
        "--export-float-dtype",
        default=EXPORT_FLOAT_DTYPE_DEFAULT,
        choices=("float32", "float64"),
        help="Floating-point dtype for user-facing exported VTP arrays.",
    )
    parser.add_argument(
        "--export-index-dtype",
        default=EXPORT_INDEX_DTYPE_DEFAULT,
        choices=("int32", "int64"),
        help="Integer dtype for user-facing exported ID arrays. Default keeps vessel IDs as int64.",
    )
    args = parser.parse_args(argv)
    working_float_dtype = _float_dtype_from_name(args.working_float_dtype)
    working_index_dtype = _int_dtype_from_name(args.working_index_dtype)
    export_float_dtype = _float_dtype_from_name(args.export_float_dtype)
    export_index_dtype = _int_dtype_from_name(args.export_index_dtype)
    cext_enabled = not bool(args.no_cext)
    if cext_enabled and str(args.concentration_solver).lower() == "network":
        raise ValueError("Cext mode is topdown-only. Use --no-cext with --concentration-solver network.")
    cext_forest_mode = str(args.cext_forest_mode).lower()
    cext_bg_mode = str(args.cext_hybrid_bg_mode).lower()
    if cext_enabled and cext_forest_mode == "shared-global-fft":
        if cext_bg_mode != "fft":
            raise ValueError("--cext-forest-mode shared-global-fft requires --cext-hybrid-bg-mode fft.")
    if cext_enabled and cext_bg_mode == "fft":
        if str(args.cext_hybrid_bg_solver).lower() == "jacobi":
            raise ValueError("Shared-global heart Cext requires --cext-bg-solver auto or fft.")
    pct_blocked = float(args.fraction_blocked_infarction)
    if not np.isfinite(pct_blocked) or pct_blocked < 0.0 or pct_blocked > 1.0:
        raise ValueError("--fraction-blocked-infarction must be in [0, 1].")
    if pct_blocked > 0.0 and args.infarction_global_segment_id is None:
        raise ValueError("--infarction-global-segment-id is required when --fraction-blocked-infarction > 0.")

    t_start = perf_counter()
    forest_path = _resolve_forest_path(args.forest)
    out_dir = _normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (f"{forest_path.stem}_cext" if cext_enabled else forest_path.stem)

    _log(f"Loading TissueSim_heart_accel...")
    ts = _load_tissuesim()
    solver_parameters = _apply_tissuesim_overrides(ts, args)
    cext_ts = None
    cext_parameters = {"enabled": False}
    if cext_enabled:
        preload_cuda_component_libraries()
        _log(f"Loading CASCADE Cext TissueSim module: {args.cext_tissuesim}")
        cext_ts = _load_cext_tissuesim(args.cext_tissuesim)
        if cext_ts._cp is None:
            raise RuntimeError("Cext mode requires GPU/CuPy support. Rerun with --no-cext for legacy exporter behavior.")
        cext_parameters = _apply_cext_overrides(cext_ts, ts, args)
    _log(
        "Oxygen parameters: "
        f"D={ts.SOLUTE_DIFFUSIVITY:g} vmax={ts.VMAX_MM:g} km={ts.K_M_MM:g} "
        f"nearby={ts.NEAREST_TISSUE_VESSELS} window={ts.WINDOW_FACTOR:g} gl={ts.GL_ORDER} "
        f"hematocrit={ts.HEMATOCRIT_MODEL}/{ts.HEMATOCRIT_FLOW_ITERATIONS}"
    )
    _log(f"Tissue accel: mode={ts.TISSUE_ACCEL_MODE} gpu_chunk_points={ts.TISSUE_GPU_CHUNK_POINTS}")
    if cext_enabled:
        _log(
            f"Cext mode: {args.cext_forest_mode} "
            f"solver={args.cext_concentration_solver} bg_mode={args.cext_hybrid_bg_mode} "
            f"grid={args.cext_hybrid_bg_grid} bins={args.cext_hybrid_bg_lambda_bins} "
            f"assignment={args.cext_hybrid_bg_assignment} bg_solver={cext_parameters.get('effective_bg_solver')} "
            f"gl_order={args.cext_gl_order} max_iter={args.cext_vess_coupling_max_iter} "
            f"wall={args.lumen_wall_closure} o2_terms={args.finite_radius_o2_terms}"
        )
    else:
        _log("Cext mode: disabled (--no-cext); using legacy per-tree concentration/tissue path.")
    if args.kirchhoff_bc_mode is not None:
        _log(f"Kirchhoff BC mode: {ts.KIRCHHOFF_BC_MODE}")

    _log(f"Loading forest: {forest_path}")
    cache_path = _default_simulation_cache_path(forest_path)
    cache_enabled = not bool(args.no_simulation_cache)
    load_source = forest_path
    if cache_enabled and _should_use_simulation_cache(forest_path, cache_path):
        load_source = cache_path
        _log(f"Using simulation cache: {cache_path}")
    elif cache_enabled:
        try:
            _log(f"Building simulation cache without vessel maps: {cache_path}")
            Forest.build_simulation_cache_from_legacy(str(forest_path), str(cache_path), show_progress=True)
            load_source = cache_path
            _log(f"Wrote simulation cache: {cache_path}")
        except Exception as exc:
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"Fast simulation-cache build failed ({exc}). "
                "Aborting instead of falling back to the legacy forest loader."
            ) from exc
    forest = Forest.load(
        str(load_source),
        mode="simulation",
        data_dtype=working_float_dtype,
        index_dtype=working_index_dtype,
    )
    _make_forest_analysis_only(forest)
    _log("Forest load mode: analysis-only (skipping build-time spatial indices)")
    if cache_enabled and load_source == forest_path:
        try:
            forest.save_simulation_cache(str(cache_path))
            _log(f"Wrote simulation cache: {cache_path}")
        except Exception as exc:
            _log(f"Warning: failed to write simulation cache ({exc}).")
    input_dtype_summaries = _normalize_forest_dtype_attrs(forest)
    _log(f"Forest loaded: n_networks={forest.n_networks} n_trees_per_network={forest.n_trees_per_network}")

    _log(f"Building/loading domain: {args.domain_path}")
    t0 = perf_counter()
    domain = _build_domain(ts, _normalize_path(args.domain_path), float(args.side_length))
    _log(f"Domain loaded in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    _attach_domain(forest, domain)
    _log(f"Domain attached in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    input_dtype_summaries = _normalize_forest_dtype_attrs(forest)
    _log(f"Forest dtype attrs normalized in {perf_counter() - t0:.2f}s")

    parent_repairs = []
    if not bool(args.no_repair_connectivity):
        t0 = perf_counter()
        _log("Repairing connectivity parent columns...")
        parent_repairs = _repair_forest_connectivity(forest)
        _log(f"Connectivity repair completed in {perf_counter() - t0:.2f}s")
        if any(parent_repairs):
            _log(f"Connectivity repair: stale parent refs per tree={parent_repairs}")

    connectivity_reports = []
    if args.validate_connectivity:
        t0 = perf_counter()
        _log("Validating connectivity...")
        connectivity_reports = _validate_forest_connectivity(
            forest,
            fail=not bool(args.no_fail_connectivity),
            geometry_atol=float(args.connectivity_geometry_atol),
        )
        _log(f"Connectivity validation completed in {perf_counter() - t0:.2f}s")

    outputs: dict[str, str] = {}
    meta: dict = {
        "forest": str(forest_path),
        "domain_path": str(args.domain_path),
        "fluid": str(args.fluid),
        "flow_source": str(args.flow_source),
        "connectivity": connectivity_reports,
        "parent_connectivity_repairs": [int(v) for v in parent_repairs],
        "solver_parameters": solver_parameters,
        "cext_parameters": cext_parameters,
        "input_tree_dtypes": input_dtype_summaries,
        "working_float_dtype": str(working_float_dtype),
        "working_index_dtype": str(working_index_dtype),
        "export_float_dtype": str(export_float_dtype),
        "export_index_dtype": str(export_index_dtype),
        "tree_summaries": [],
    }
    if not cext_enabled:
        meta["cext"] = {"enabled": False}
    infarction_meta = {
        "target_global_segment_id": None if args.infarction_global_segment_id is None else int(args.infarction_global_segment_id),
        "target_tree_id": None,
        "target_local_segment_id": None,
        "fraction_blocked_infarction": pct_blocked,
        "applied": False,
        "mode": "none",
        "geometry_only_ignored": bool(args.geometry_only and pct_blocked > 0.0),
        "target_segment_original_radius": None,
        "target_segment_effective_radius": None,
        "blocked_subtree_count": 0,
        "blocked_subtree_preview_global": [],
        "blocked_subtree_preview_local": [],
    }
    if args.geometry_only and pct_blocked > 0.0:
        _log("Infarction requested, but --geometry-only is enabled; ignoring infarction for this run.")
    elif pct_blocked > 0.0:
        target_tree_id, target_local_segment_id = _resolve_global_segment_id(forest, int(args.infarction_global_segment_id))
        infarction_meta["target_tree_id"] = int(target_tree_id)
        infarction_meta["target_local_segment_id"] = int(target_local_segment_id)
        _log(
            f"Infarction target: global_segment_id={args.infarction_global_segment_id} "
            f"-> tree_id={target_tree_id} local_segment_id={target_local_segment_id}"
        )

    tree_solutions: list[dict] = []
    if args.geometry_only:
        _log("Geometry-only mode: skipping flow and concentration solves.")
        for tree_id, tree in enumerate(forest.networks[0]):
            data = np.asarray(tree.data[: int(tree.segment_count)])
            sol = {
                "starts": data[:, 0:3],
                "ends": data[:, 3:6],
                "radii": data[:, 21],
                "lengths": data[:, 20],
                "flows": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "cin": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "cout": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "p_in": float("nan"),
                "p_out": float("nan"),
                "inlet_concentration": float("nan"),
            }
            tree_solutions.append(sol)
            meta["tree_summaries"].append(
                {
                    "tree_id": tree_id,
                    "segments": int(data.shape[0]),
                    "geometry_only": True,
                    "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                    "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                }
            )
    else:
        t0 = perf_counter()
        inlet_flows = _flow_inputs(args, forest)
        _log(f"Flow inputs prepared in {perf_counter() - t0:.2f}s")
        tree_offsets: list[int] = []
        offset = 0
        for tree in forest.networks[0]:
            tree_offsets.append(offset)
            offset += int(getattr(tree, "segment_count", 0) or 0)
        if cext_enabled:
            blocked_tree_id = None
            blocked_subtree_ids = np.empty((0,), dtype=np.int64)
            restore_radii_ids = np.empty((0,), dtype=np.int64)
            restore_radii_values = np.empty((0,), dtype=float)
            if pct_blocked > 0.0:
                blocked_tree_id = int(infarction_meta["target_tree_id"])
                target_tree = forest.networks[0][blocked_tree_id]
                target_id = int(infarction_meta["target_local_segment_id"])
                target_radius = float(target_tree.data[target_id, 21])
                infarction_meta["target_segment_original_radius"] = target_radius
                if pct_blocked < 1.0:
                    effective_radius = target_radius * (1.0 - pct_blocked)
                    infarction_meta["target_segment_effective_radius"] = float(effective_radius)
                    infarction_meta["mode"] = "radius_reduction"
                    infarction_meta["applied"] = True
                    restore_radii_ids = np.array([target_id], dtype=np.int64)
                    restore_radii_values = np.array([target_radius], dtype=float)
                    target_tree.data[target_id, 21] = effective_radius
                    _log(
                        f"Infarction solve override: tree {blocked_tree_id} segment {target_id} radius "
                        f"{target_radius:.9g} -> {effective_radius:.9g}"
                    )
                else:
                    seg_count = int(getattr(target_tree, "segment_count", 0) or 0)
                    blocked_subtree_ids = _collect_downstream_segment_ids(target_tree, target_id, seg_count)
                    blocked_count = int(blocked_subtree_ids.size)
                    infarction_meta["mode"] = "full_occlusion_subtree"
                    infarction_meta["applied"] = True
                    infarction_meta["blocked_subtree_count"] = blocked_count
                    infarction_meta["blocked_subtree_preview_local"] = blocked_subtree_ids[:20].tolist()
                    infarction_meta["blocked_subtree_preview_global"] = (
                        blocked_subtree_ids[:20] + int(tree_offsets[blocked_tree_id])
                    ).tolist()
                    infarction_meta["target_segment_effective_radius"] = 0.0
                    if blocked_count > 0:
                        restore_radii_ids = blocked_subtree_ids
                        restore_radii_values = np.asarray(target_tree.data[restore_radii_ids, 21], dtype=float)
                        target_tree.data[restore_radii_ids, 21] = 0.0
                    _log(
                        f"Infarction full occlusion: tree {blocked_tree_id} segment {target_id} "
                        f"subtree size {blocked_count} set to zero effective radius for solves."
                    )
            try:
                assert cext_ts is not None
                tree_solutions, cext_meta = _solve_forest_cext(
                    cext_ts,
                    ts,
                    forest,
                    inlet_flows,
                    fluid=str(args.fluid),
                    args=args,
                )
            finally:
                if restore_radii_ids.size > 0 and blocked_tree_id is not None:
                    forest.networks[0][blocked_tree_id].data[restore_radii_ids, 21] = restore_radii_values
            if pct_blocked >= 1.0 and blocked_subtree_ids.size > 0 and blocked_tree_id is not None:
                sol = tree_solutions[int(blocked_tree_id)]
                valid_blocked = blocked_subtree_ids[blocked_subtree_ids < sol["flows"].shape[0]]
                if valid_blocked.size > 0:
                    sol["flows"][valid_blocked] = 0.0
                    sol["cin"][valid_blocked] = 0.0
                    sol["cout"][valid_blocked] = 0.0
                    sol["cext_state"]["cin_seg"][valid_blocked] = 0.0
                    sol["cext_state"]["cout_seg"][valid_blocked] = 0.0
                    sol["cext_state"]["c_iv_gl"][valid_blocked] = 0.0
                    sol["cext_state"]["q_line_gl"][valid_blocked] = 0.0
                    sol["cext_state"]["q_weighted_gl"][valid_blocked] = 0.0
                    sol["cext_state"]["seg_cap_gl"][valid_blocked] = 0.0
                    _log(
                        f"Infarction post-process: forced zero flow/concentration/Cext source on "
                        f"{valid_blocked.size} blocked subtree segments in tree {blocked_tree_id}."
                    )
            meta["cext"] = cext_meta
            for tree_id, (tree, inlet_flow, sol) in enumerate(zip(forest.networks[0], inlet_flows, tree_solutions)):
                meta["tree_summaries"].append(
                    {
                        "tree_id": int(tree_id),
                        "segments": int(sol["starts"].shape[0]),
                        "terminals": int(getattr(tree, "n_terminals", 0)),
                        "inlet_flow_cm3_s": float(inlet_flow),
                        "inlet_flow_ul_min": float(inlet_flow * 60000.0),
                        "pressure_in": float(sol["p_in"]),
                        "pressure_out": float(sol["p_out"]),
                        "cext_reflected_topdown_s": float(sol.get("cext_reflected_topdown_s", 0.0)),
                        "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                        "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                    }
                )
        for tree_id, (tree, inlet_flow) in ([] if cext_enabled else list(enumerate(zip(forest.networks[0], inlet_flows)))):
            t_tree = perf_counter()
            _log(f"Solving tree {tree_id}...")
            blocked_subtree_ids = np.empty((0,), dtype=np.int64)
            restore_radii_ids = np.empty((0,), dtype=np.int64)
            restore_radii_values = np.empty((0,), dtype=float)
            if pct_blocked > 0.0 and int(infarction_meta["target_tree_id"]) == int(tree_id):
                target_id = int(infarction_meta["target_local_segment_id"])
                target_radius = float(tree.data[target_id, 21])
                infarction_meta["target_segment_original_radius"] = target_radius
                if pct_blocked < 1.0:
                    effective_radius = target_radius * (1.0 - pct_blocked)
                    infarction_meta["target_segment_effective_radius"] = float(effective_radius)
                    infarction_meta["mode"] = "radius_reduction"
                    infarction_meta["applied"] = True
                    restore_radii_ids = np.array([target_id], dtype=np.int64)
                    restore_radii_values = np.array([target_radius], dtype=float)
                    tree.data[target_id, 21] = effective_radius
                    _log(
                        f"Infarction solve override: tree {tree_id} segment {target_id} radius "
                        f"{target_radius:.9g} -> {effective_radius:.9g}"
                    )
                else:
                    seg_count = int(getattr(tree, "segment_count", 0) or 0)
                    blocked_subtree_ids = _collect_downstream_segment_ids(tree, target_id, seg_count)
                    blocked_count = int(blocked_subtree_ids.size)
                    infarction_meta["mode"] = "full_occlusion_subtree"
                    infarction_meta["applied"] = True
                    infarction_meta["blocked_subtree_count"] = blocked_count
                    infarction_meta["blocked_subtree_preview_local"] = blocked_subtree_ids[:20].tolist()
                    infarction_meta["blocked_subtree_preview_global"] = (blocked_subtree_ids[:20] + int(tree_offsets[tree_id])).tolist()
                    infarction_meta["target_segment_effective_radius"] = 0.0
                    if blocked_count > 0:
                        restore_radii_ids = blocked_subtree_ids
                        restore_radii_values = np.asarray(tree.data[restore_radii_ids, 21], dtype=float)
                        tree.data[restore_radii_ids, 21] = 0.0
                    _log(
                        f"Infarction full occlusion: tree {tree_id} segment {target_id} "
                        f"subtree size {blocked_count} set to zero effective radius for solves."
                    )
            try:
                sol = _solve_tree(ts, tree, inlet_flow, fluid=str(args.fluid), concentration_solver=args.concentration_solver)
            finally:
                if restore_radii_ids.size > 0:
                    tree.data[restore_radii_ids, 21] = restore_radii_values
            if pct_blocked >= 1.0 and blocked_subtree_ids.size > 0:
                valid_blocked = blocked_subtree_ids[blocked_subtree_ids < sol["flows"].shape[0]]
                if valid_blocked.size > 0:
                    sol["flows"][valid_blocked] = 0.0
                    sol["cin"][valid_blocked] = 0.0
                    sol["cout"][valid_blocked] = 0.0
                    _log(
                        f"Infarction post-process: forced zero flow/cin/cout on {valid_blocked.size} "
                        f"blocked subtree segments in tree {tree_id}."
                    )
            _log(f"Tree {tree_id} solved in {perf_counter() - t_tree:.2f}s")
            tree_solutions.append(sol)
            meta["tree_summaries"].append(
                {
                    "tree_id": int(tree_id),
                    "segments": int(sol["starts"].shape[0]),
                    "terminals": int(getattr(tree, "n_terminals", 0)),
                    "inlet_flow_cm3_s": float(inlet_flow),
                    "inlet_flow_ul_min": float(inlet_flow * 60000.0),
                    "pressure_in": float(sol["p_in"]),
                    "pressure_out": float(sol["p_out"]),
                    "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                    "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                }
            )

    if args.report_global_segment_id is not None:
        report_tree_id, report_local_segment_id = _resolve_global_segment_id(
            forest, int(args.report_global_segment_id)
        )
        if 0 <= report_tree_id < len(tree_solutions):
            report_flows = np.asarray(tree_solutions[report_tree_id].get("flows", []), dtype=float)
            if 0 <= report_local_segment_id < report_flows.size:
                report_flow = float(report_flows[report_local_segment_id])
                meta["reported_segment"] = {
                    "global_segment_id": int(args.report_global_segment_id),
                    "tree_id": int(report_tree_id),
                    "local_segment_id": int(report_local_segment_id),
                    "flow_cm3_s": report_flow,
                    "flow_ul_min": report_flow * 60000.0,
                }
                _log(
                    "Reported segment flow: "
                    f"global_segment_id={int(args.report_global_segment_id)} "
                    f"tree_id={int(report_tree_id)} local_segment_id={int(report_local_segment_id)} "
                    f"flow_cm3_s={report_flow:.12g} flow_ul_min={report_flow * 60000.0:.12g}"
                )

    t0 = perf_counter()
    combo = _concat_tree_solutions(tree_solutions, index_dtype=export_index_dtype)
    tree_solutions.clear()
    gc.collect()
    if cext_enabled and cext_ts is not None:
        cache_release = _release_cext_transient_gpu_cache(cext_ts)
        meta.setdefault("cext", {}).setdefault("pre_tissue_gpu_cache_release", cache_release)
        _log(
            "Released transient Cext GPU cache before tissue solve: "
            f"cupy_pool_trimmed={cache_release.get('cupy_pool_trimmed')} "
            f"attrs={cache_release.get('module_attrs_cleared', [])}"
        )
    _log(f"Combined tree solution arrays in {perf_counter() - t0:.2f}s")
    _log(f"Combined vessel segments: {combo['starts'].shape[0]}")

    if args.skip_vessel_output:
        _log("Skipping vessel VTP output.")
    else:
        t0 = perf_counter()
        vessels = _build_vessel_polydata(
            combo,
            resolution=int(args.vessel_resolution),
            float_dtype=export_float_dtype,
            index_dtype=export_index_dtype,
        )
        _log(f"Built vessel VTP in {perf_counter() - t0:.2f}s")
        vessels_path = out_dir / f"{prefix}_forest_vessels.vtp"
        _log(f"Saving vessel VTP: {vessels_path}")
        t0 = perf_counter()
        vessels.save(str(vessels_path))
        _log(f"Saved vessel VTP in {perf_counter() - t0:.2f}s")
        outputs["vessels"] = str(vessels_path)
        del vessels
        gc.collect()

    if not args.geometry_only and not args.skip_tissue_oxygen:
        t0 = perf_counter()
        points_poly, tissue_meta = _compute_tissue(
            ts,
            cext_ts,
            combo,
            domain,
            args,
            float_dtype=export_float_dtype,
            index_dtype=export_index_dtype,
        )
        _log(f"Computed tissue oxygen in {perf_counter() - t0:.2f}s")
        points_path = out_dir / f"{prefix}_forest_oxygen_points.vtp"
        _log(f"Saving tissue oxygen VTP: {points_path}")
        t0 = perf_counter()
        points_poly.save(str(points_path))
        _log(f"Saved tissue oxygen VTP in {perf_counter() - t0:.2f}s")
        outputs["oxygen_points"] = str(points_path)
        meta["tissue"] = tissue_meta
        del points_poly
        gc.collect()
    else:
        _log("Skipping tissue oxygen output.")

    outputs.update(_save_domain_outputs(domain, out_dir, prefix))
    meta["infarction"] = infarction_meta
    meta["outputs"] = outputs
    meta["elapsed_s"] = perf_counter() - t_start
    meta["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    meta_path = out_dir / f"{prefix}_forest_paraview_export.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"Metadata: {meta_path}")
    _log(f"Done in {meta['elapsed_s']:.2f}s")
    print("Wrote:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    print(f"  meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
