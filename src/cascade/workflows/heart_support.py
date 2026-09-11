#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import importlib.metadata
import json
import os
import types
from pathlib import Path
from time import perf_counter

import numpy as np
import pyvista as pv
from tqdm import tqdm

from cascade.accelerators.backend import preload_cuda_component_libraries
from cascade.utils.execution import guard_simulation
from cascade.vessels.generation.svv_adapter import Domain, Forest


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
GRAETZ_N_RADIAL_DEFAULT = 8
GRAETZ_N_MODES_DEFAULT = 4
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
    from cascade.runtime import tissuesim

    return tissuesim


def _load_cext_tissuesim():
    from cascade.runtime import tissuesim as cext_ts

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




__all__ = ('_normalize_path', '_float_dtype_from_name', '_int_dtype_from_name', '_resolve_forest_path', '_default_simulation_cache_path', '_should_use_simulation_cache', '_load_tissuesim', '_load_cext_tissuesim', '_apply_tissuesim_overrides', '_bool_arg', '_apply_cext_overrides')
