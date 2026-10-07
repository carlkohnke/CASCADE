"""Hybrid-grid geometry, stencils, and runtime state.

It creates grid dimensions, interpolation stencils, local source plans, and
reusable device buffers for hybrid solves.
"""

from __future__ import annotations

import hashlib
from itertools import count
import math
import os

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.accelerators.cuda import load_cuda_source


_LAMBDA_BIN_STATE_TOKENS = count()


def _ensure_cext_hybrid_bg_context(context: dict) -> dict:
    hybrid = context.get("hybrid_bg_context")
    grid_n = max(int(_state.CEXT_HYBRID_BG_GRID), 16)
    bg_mode = _resolve_cext_hybrid_bg_mode()
    configured_near_radius_mult = (
        0.0
        if bg_mode in ("local_only_nlambda", "fft")
        else max(float(_state.CEXT_HYBRID_BG_NEAR_RADIUS_MULT), 0.0)
    )
    assignment = str(_state.CEXT_HYBRID_BG_ASSIGNMENT).strip().lower()
    if assignment not in ("cic", "tsc"):
        assignment = "tsc"
    if isinstance(hybrid, dict):
        if (
            int(hybrid.get("grid_n", -1)) == grid_n
            and int(hybrid.get("lambda_bins", -1))
            == max(int(_state.CEXT_HYBRID_BG_LAMBDA_BINS), 1)
            and float(hybrid.get("near_radius_mult", -1.0))
            == float(configured_near_radius_mult)
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
        "lambda_bins": max(int(_state.CEXT_HYBRID_BG_LAMBDA_BINS), 1),
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


def _cext_hybrid_axis_assignment_cpu(
    coord: np.ndarray, grid_n: int, assignment: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
                    np.float32(0.5)
                    * (np.float32(1.5) - dist)
                    * (np.float32(1.5) - dist),
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
    points = np.asarray(
        hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32)),
        dtype=np.float32,
    )
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
                    flat = (
                        ((ax[valid] * grid_n) + ay[valid]) * grid_n + az[valid]
                    ).astype(np.int32, copy=False)
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
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    points = _state._cp.asarray(
        np.asarray(
            hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
    )
    n_nodes = int(points.shape[0])
    grid_n = int(hybrid["grid_n"])
    grid_cells = int(grid_n * grid_n * grid_n)
    assignment = str(hybrid["assignment"])
    stencil_n = 27 if assignment == "tsc" else 8
    flat_idx = _state._cp.full(
        (n_nodes, stencil_n), _state._cp.int32(-1), dtype=_state._cp.int32
    )
    weights = _state._cp.zeros((n_nodes, stencil_n), dtype=_state._cp.float32)
    if n_nodes <= 0 or grid_cells <= 0:
        return {
            "stencil_flat_idx": flat_idx,
            "stencil_weight": weights,
            "stencil_n": int(stencil_n),
            "grid_cells": int(grid_cells),
        }

    origin_g = _state._cp.asarray(np.asarray(hybrid["origin"], dtype=np.float32))
    spacing = _state._cp.float32(float(hybrid["spacing"]))
    coords = ((points - origin_g[None, :]) / spacing) - _state._cp.float32(0.5)
    if assignment == "tsc":
        axis_idx = []
        axis_weight = []
        for axis in range(3):
            center = _state._cp.floor(coords[:, axis] + _state._cp.float32(0.5)).astype(
                _state._cp.int32
            )
            idx_list = []
            weight_list = []
            for offset in (-1, 0, 1):
                idx = center + np.int32(offset)
                dist = _state._cp.abs(coords[:, axis] - idx.astype(_state._cp.float32))
                weight = _state._cp.where(
                    dist < _state._cp.float32(0.5),
                    _state._cp.float32(0.75) - dist * dist,
                    _state._cp.where(
                        dist < _state._cp.float32(1.5),
                        _state._cp.float32(0.5)
                        * (_state._cp.float32(1.5) - dist)
                        * (_state._cp.float32(1.5) - dist),
                        _state._cp.float32(0.0),
                    ),
                ).astype(_state._cp.float32)
                idx_list.append(idx)
                weight_list.append(weight)
            axis_idx.append(idx_list)
            axis_weight.append(weight_list)
    else:
        axis_idx = []
        axis_weight = []
        for axis in range(3):
            base = _state._cp.floor(coords[:, axis]).astype(_state._cp.int32)
            frac = (coords[:, axis] - base.astype(_state._cp.float32)).astype(
                _state._cp.float32
            )
            axis_idx.append([base, base + np.int32(1)])
            axis_weight.append(
                [
                    (_state._cp.float32(1.0) - frac).astype(_state._cp.float32),
                    frac.astype(_state._cp.float32),
                ]
            )

    slot = 0
    for ax, wx in zip(axis_idx[0], axis_weight[0]):
        for ay, wy in zip(axis_idx[1], axis_weight[1]):
            for az, wz in zip(axis_idx[2], axis_weight[2]):
                weight = (wx * wy * wz).astype(_state._cp.float32)
                valid = (
                    (ax >= 0)
                    & (ax < grid_n)
                    & (ay >= 0)
                    & (ay < grid_n)
                    & (az >= 0)
                    & (az < grid_n)
                    & _state._cp.isfinite(weight)
                    & (weight != _state._cp.float32(0.0))
                )
                flat = (((ax * np.int32(grid_n)) + ay) * np.int32(grid_n) + az).astype(
                    _state._cp.int32
                )
                flat_idx[:, slot] = _state._cp.where(valid, flat, _state._cp.int32(-1))
                weights[:, slot] = _state._cp.where(
                    valid, weight, _state._cp.float32(0.0)
                )
                slot += 1

    return {
        "stencil_flat_idx": flat_idx,
        "stencil_weight": weights,
        "stencil_n": int(stencil_n),
        "grid_cells": int(grid_cells),
    }


def _cext_hybrid_build_local_node_cells(hybrid: dict) -> dict[str, np.ndarray | int]:
    points = np.asarray(
        hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32)),
        dtype=np.float32,
    )
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
    coords = np.floor(
        (np.asarray(points, dtype=np.float32) - origin[None, :]) / np.float32(spacing)
    ).astype(np.int32)
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
            coords[valid, 0].astype(np.int64) * np.int64(grid_n)
            + coords[valid, 1].astype(np.int64)
        ) * np.int64(grid_n) + coords[valid, 2].astype(np.int64)
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
    local_cell_ptr[1:] = np.cumsum(
        np.asarray(counts, dtype=np.int64), dtype=np.int64
    ).astype(np.int32)
    rad_cells = int(
        np.ceil(
            max(float(hybrid.get("near_radius_si", 0.0)), 0.0) / max(spacing, 1.0e-30)
        )
    )
    return {
        "local_cell_ptr": np.asarray(local_cell_ptr, dtype=np.int32),
        "local_node_ids": np.asarray(local_node_ids, dtype=np.int32),
        "local_node_cell_flat": np.asarray(node_cell_flat, dtype=np.int32),
        "local_rad_cells": int(max(rad_cells, 0)),
    }


def _build_cext_hybrid_local_source_plan(
    hybrid: dict, source_mask: np.ndarray | None
) -> dict:
    node_seg_ids = np.asarray(
        hybrid.get("node_seg_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32
    )
    node_cell_flat = np.asarray(
        hybrid.get(
            "local_node_cell_flat", np.zeros((node_seg_ids.size,), dtype=np.int32)
        ),
        dtype=np.int32,
    )
    grid_n = int(hybrid.get("grid_n", 0))
    grid_cells = int(grid_n * grid_n * grid_n)
    if (
        source_mask is None
        or node_seg_ids.size <= 0
        or node_cell_flat.size != node_seg_ids.size
        or grid_cells <= 0
    ):
        return {
            "kind": "hybrid_local_static",
            "uses_static": True,
            "active_count": int(node_seg_ids.size),
        }
    mask = np.asarray(source_mask, dtype=bool).reshape(-1)
    if mask.size <= 0 or int(np.count_nonzero(mask)) >= int(mask.size):
        return {
            "kind": "hybrid_local_static",
            "uses_static": True,
            "active_count": int(node_seg_ids.size),
        }
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
    cell_ptr[1:] = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64).astype(
        np.int32
    )
    return {
        "kind": "hybrid_local_active",
        "uses_static": False,
        "cell_ptr": np.asarray(cell_ptr, dtype=np.int32),
        "node_ids": np.asarray(sorted_node_ids, dtype=np.int32),
        "active_count": int(node_ids.size),
    }


def _resolve_cext_hybrid_bg_mode() -> str:
    mode = str(_state.CEXT_HYBRID_BG_MODE or "local_only_nlambda").strip().lower()
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
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    static = hybrid.get("gpu_static")
    if isinstance(static, dict) and "local_node_cell_flat" in static:
        return static
    grid_n = int(hybrid["grid_n"])
    if float(hybrid.get("near_radius_si", 0.0)) > 0.0:
        local_cell_meta = _cext_hybrid_build_local_node_cells(hybrid)
    else:
        n_nodes = int(
            np.asarray(
                hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32))
            ).shape[0]
        )
        grid_cells = int(grid_n * grid_n * grid_n)
        local_cell_meta = {
            "local_cell_ptr": np.zeros((grid_cells + 1,), dtype=np.int32),
            "local_node_ids": np.zeros((0,), dtype=np.int32),
            "local_node_cell_flat": np.full((n_nodes,), -1, dtype=np.int32),
            "local_rad_cells": 0,
        }
    hybrid["local_node_cell_flat"] = np.asarray(
        local_cell_meta["local_node_cell_flat"], dtype=np.int32
    )
    hybrid["local_rad_cells"] = int(local_cell_meta["local_rad_cells"])
    grid_cells = int(grid_n * grid_n * grid_n)
    static = {
        "origin": _state._cp.asarray(np.asarray(hybrid["origin"], dtype=np.float32)),
        "k2": _state._cp.asarray(np.zeros((1,), dtype=np.float32)),
        "node_seg_ids": _state._cp.asarray(
            np.asarray(hybrid["node_seg_ids"], dtype=np.int32)
        ),
        "gl_points_flat": _state._cp.asarray(
            np.asarray(hybrid["gl_points_flat"], dtype=np.float32)
        ),
        "all_target_seg_ids": _state._cp.arange(
            int(np.asarray(context["midpoints_si"]).shape[0]), dtype=_state._cp.int32
        ),
        "grid_shape": (grid_n, grid_n, grid_n),
        "stencil_flat_idx": _state._cp.asarray(np.zeros((1, 1), dtype=np.int32)),
        "stencil_weight": _state._cp.asarray(np.ones((1, 1), dtype=np.float32)),
        "stencil_n": 1,
        "grid_cells": grid_cells,
        "local_cell_ptr": _state._cp.asarray(
            np.asarray(local_cell_meta["local_cell_ptr"], dtype=np.int32)
        ),
        "local_node_ids": _state._cp.asarray(
            np.asarray(local_cell_meta["local_node_ids"], dtype=np.int32)
        ),
        "local_node_cell_flat": _state._cp.asarray(
            np.asarray(local_cell_meta["local_node_cell_flat"], dtype=np.int32)
        ),
        "local_rad_cells": int(local_cell_meta["local_rad_cells"]),
    }
    hybrid["gpu_static"] = static
    return static


def _ensure_cext_hybrid_local_only_runtime_state(
    context: dict, hybrid: dict, ext_state: dict
) -> dict:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gl_shape = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape
    nseg = int(gl_shape[0]) if len(gl_shape) >= 1 else 0
    grid_shape = (
        0,
        int(hybrid["grid_n"]),
        int(hybrid["grid_n"]),
        int(hybrid["grid_n"]),
    )
    state = hybrid.get("runtime_state")
    if isinstance(state, dict):
        if tuple(state.get("gl_shape", ())) == tuple(gl_shape) and bool(
            state.get("local_only_lean", False)
        ):
            return state
    state = {
        "gl_shape": tuple(gl_shape),
        "grid_shape": tuple(grid_shape),
        "stencil_n": int(static.get("stencil_n", 1)),
        "local_only_lean": True,
        "q_weighted_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "mono2_weight_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "dipole2_weight_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "lambda_iv_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "seg_cap_gl": _state._cp.zeros((nseg,), dtype=_state._cp.float32),
        "c_ext_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "active_source_mask_g": _state._cp.ones((nseg,), dtype=_state._cp.uint8),
        "local_total_g": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "local_cap_g": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
    }
    hybrid["runtime_state"] = state
    return state


def _ensure_cext_hybrid_bg_gpu_static(context: dict, hybrid: dict) -> dict:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    if str(os.environ.get("SVV_CEXT_GPU_POOL_TRIM", "false")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        _state._cp.cuda.Stream.null.synchronize()
        _state._cp.get_default_memory_pool().free_all_blocks()
    if _resolve_cext_hybrid_bg_mode().startswith("local_only"):
        return _ensure_cext_hybrid_local_only_gpu_static(context, hybrid)
    static = hybrid.get("gpu_static")
    if isinstance(static, dict) and "local_node_cell_flat" not in static:
        return static
    grid_n = int(hybrid["grid_n"])
    runtime_stencil = bool(_state.CEXT_HYBRID_GPU_RUNTIME_STENCIL)
    if runtime_stencil:
        grid_cells = int(grid_n * grid_n * grid_n)
        stencil_n = (
            27 if str(hybrid.get("assignment", "tsc")).strip().lower() == "tsc" else 8
        )
        stencil_meta = {
            "stencil_flat_idx": _state._cp.asarray(np.zeros((1,), dtype=np.int32)),
            "stencil_weight": _state._cp.asarray(np.ones((1,), dtype=np.float32)),
            "stencil_n": int(stencil_n),
            "grid_cells": int(grid_cells),
        }
    else:
        stencil_meta = _cext_hybrid_build_stencil_metadata_gpu(hybrid)
    if float(hybrid.get("near_radius_si", 0.0)) > 0.0:
        local_cell_meta = _cext_hybrid_build_local_node_cells(hybrid)
    else:
        n_nodes = int(
            np.asarray(
                hybrid.get("gl_points_flat", np.zeros((0, 3), dtype=np.float32))
            ).shape[0]
        )
        grid_cells = int(grid_n * grid_n * grid_n)
        local_cell_meta = {
            "local_cell_ptr": np.zeros((grid_cells + 1,), dtype=np.int32),
            "local_node_ids": np.zeros((0,), dtype=np.int32),
            "local_node_cell_flat": np.full((n_nodes,), -1, dtype=np.int32),
            "local_rad_cells": 0,
        }
    hybrid["local_node_cell_flat"] = np.asarray(
        local_cell_meta["local_node_cell_flat"], dtype=np.int32
    )
    hybrid["local_rad_cells"] = int(local_cell_meta["local_rad_cells"])
    static = {
        "origin": _state._cp.asarray(np.asarray(hybrid["origin"], dtype=np.float32)),
        "k2": _state._cp.asarray(np.asarray(hybrid["k2"], dtype=np.float32)),
        "node_seg_ids": _state._cp.asarray(
            np.asarray(hybrid["node_seg_ids"], dtype=np.int32)
        ),
        "gl_points_flat": _state._cp.asarray(
            np.asarray(hybrid["gl_points_flat"], dtype=np.float32)
        ),
        "all_target_seg_ids": _state._cp.arange(
            int(np.asarray(context["midpoints_si"]).shape[0]), dtype=_state._cp.int32
        ),
        "grid_shape": (grid_n, grid_n, grid_n),
        "stencil_flat_idx": _state._cp.asarray(
            stencil_meta["stencil_flat_idx"], dtype=_state._cp.int32
        ),
        "stencil_weight": _state._cp.asarray(
            stencil_meta["stencil_weight"], dtype=_state._cp.float32
        ),
        "stencil_n": int(stencil_meta["stencil_n"]),
        "grid_cells": int(stencil_meta["grid_cells"]),
        "runtime_stencil": bool(runtime_stencil),
        "assignment_mode": 1
        if str(hybrid.get("assignment", "tsc")).strip().lower() == "tsc"
        else 0,
        "local_cell_ptr": _state._cp.asarray(
            np.asarray(local_cell_meta["local_cell_ptr"], dtype=np.int32)
        ),
        "local_node_ids": _state._cp.asarray(
            np.asarray(local_cell_meta["local_node_ids"], dtype=np.int32)
        ),
        "local_rad_cells": int(local_cell_meta["local_rad_cells"]),
    }
    hybrid["gpu_static"] = static
    return static


def _ensure_cext_hybrid_bg_runtime_state(
    context: dict, hybrid: dict, ext_state: dict
) -> dict:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    if str(os.environ.get("SVV_CEXT_GPU_POOL_TRIM", "false")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        _state._cp.cuda.Stream.null.synchronize()
        _state._cp.get_default_memory_pool().free_all_blocks()
    if _resolve_cext_hybrid_bg_mode().startswith("local_only"):
        return _ensure_cext_hybrid_local_only_runtime_state(context, hybrid, ext_state)
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gl_shape = np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape
    nseg = int(gl_shape[0]) if len(gl_shape) >= 1 else 0
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
        "q_weighted_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "mono2_weight_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "dipole2_weight_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "lambda_iv_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "seg_cap_gl": _state._cp.zeros((nseg,), dtype=_state._cp.float32),
        "c_ext_gl": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "active_source_mask_g": _state._cp.ones((nseg,), dtype=_state._cp.uint8),
        "active_mass_grids_g": _state._cp.zeros(grid_shape, dtype=_state._cp.float32),
        "bg_sample_g": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "local_total_g": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
        "local_cap_g": _state._cp.zeros(gl_shape, dtype=_state._cp.float32),
    }
    hybrid["runtime_state"] = state
    return state


def _sync_cext_hybrid_bg_runtime_state(ext_state: dict, runtime_state: dict) -> None:
    runtime_state["q_weighted_gl"].set(
        np.asarray(ext_state["q_weighted_gl"], dtype=np.float32)
    )
    if "mono2_weight_gl" in runtime_state:
        runtime_state["mono2_weight_gl"].set(
            np.asarray(
                ext_state.get(
                    "mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])
                ),
                dtype=np.float32,
            )
        )
    if "dipole2_weight_gl" in runtime_state:
        runtime_state["dipole2_weight_gl"].set(
            np.asarray(
                ext_state.get(
                    "dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])
                ),
                dtype=np.float32,
            )
        )
    runtime_state["lambda_iv_gl"].set(
        np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32)
    )
    runtime_state["seg_cap_gl"].set(
        np.asarray(ext_state["seg_cap_gl"], dtype=np.float32)
    )
    runtime_state["c_ext_gl"].set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))


def _resolve_cext_hybrid_bg_solver_mode() -> str:
    mode = str(_state.CEXT_HYBRID_BG_SOLVER or "auto").strip().lower()
    if mode not in ("auto", "fft", "jacobi"):
        mode = "auto"
    return mode


def _cext_hybrid_init_lambda_bins(
    hybrid: dict, ext_state: dict
) -> tuple[np.ndarray, np.ndarray]:
    edges = hybrid.get("lambda_bin_edges")
    centers = hybrid.get("lambda_bin_centers")
    epoch = int(ext_state.get("_lambda_bin_epoch", -1))
    if "_lambda_bin_state_token" not in ext_state:
        ext_state["_lambda_bin_state_token"] = next(_LAMBDA_BIN_STATE_TOKENS)
    token = ext_state["_lambda_bin_state_token"]
    quantile_fft = (
        bool(_state.CEXT_HYBRID_FFT_QUANTILE_BINS)
        and str(hybrid.get("bg_mode", _resolve_cext_hybrid_bg_mode())).strip().lower()
        == "fft"
    )
    if (
        bool(_state.CEXT_HYBRID_FFT_BIN_EPOCH_CACHE)
        and quantile_fft
        and edges is not None
        and centers is not None
        and int(hybrid.get("lambda_bin_epoch", -2)) == epoch
        and hybrid.get("lambda_bin_state_token") == token
    ):
        edge_arr = np.asarray(edges, dtype=np.float32)
        center_arr = np.asarray(centers, dtype=np.float32)
        if _state._cp is not None and isinstance(hybrid.get("gpu_static"), dict):
            static = hybrid["gpu_static"]
            if int(static.get("lambda_bin_epoch", -2)) != epoch:
                static["lambda_bin_edges"] = _state._cp.asarray(edge_arr)
                static["lambda_bin_centers"] = _state._cp.asarray(center_arr)
                static["lambda_bin_epoch"] = int(epoch)
        return edge_arr, center_arr
    if not quantile_fft and edges is not None and centers is not None:
        return np.asarray(edges, dtype=np.float32), np.asarray(
            centers, dtype=np.float32
        )
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
        centers = np.sqrt(
            np.maximum(edges[:-1], 1.0e-30) * np.maximum(edges[1:], 1.0e-30)
        ).astype(np.float32)
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
    hybrid["lambda_bin_state_token"] = token
    hybrid["lambda_bin_edges_hash"] = hashlib.sha256(edge_arr.tobytes()).hexdigest()[
        :16
    ]
    hybrid["lambda_bin_meta"] = {
        "lambda_bin_policy": policy,
        "lambda_bin_edges_hash": hybrid["lambda_bin_edges_hash"],
        "lambda_bin_edge_min": float(edge_arr[0]) if edge_arr.size else float("nan"),
        "lambda_bin_edge_max": float(edge_arr[-1]) if edge_arr.size else float("nan"),
        "lambda_bin_count": int(max(edge_arr.size - 1, 0)),
        "lambda_bin_center_min": float(np.min(center_arr))
        if center_arr.size
        else float("nan"),
        "lambda_bin_center_max": float(np.max(center_arr))
        if center_arr.size
        else float("nan"),
    }
    if _state._cp is not None and isinstance(hybrid.get("gpu_static"), dict):
        hybrid["gpu_static"]["lambda_bin_edges"] = _state._cp.asarray(
            np.asarray(edges, dtype=np.float32)
        )
        hybrid["gpu_static"]["lambda_bin_centers"] = _state._cp.asarray(
            np.asarray(centers, dtype=np.float32)
        )
        hybrid["gpu_static"]["lambda_bin_epoch"] = int(epoch)
    return np.asarray(edges, dtype=np.float32), np.asarray(centers, dtype=np.float32)


def _cext_hybrid_axis_assignment_gpu(coord_g, grid_n: int, assignment: str):
    if assignment == "tsc":
        center = _state._cp.floor(coord_g + _state._cp.float32(0.5)).astype(
            _state._cp.int32
        )
        idx_list = []
        weight_list = []
        for offset in (-1, 0, 1):
            idx = center + np.int32(offset)
            dist = _state._cp.abs(coord_g - idx.astype(_state._cp.float32))
            weight = _state._cp.where(
                dist < _state._cp.float32(0.5),
                _state._cp.float32(0.75) - dist * dist,
                _state._cp.where(
                    dist < _state._cp.float32(1.5),
                    _state._cp.float32(0.5)
                    * (_state._cp.float32(1.5) - dist)
                    * (_state._cp.float32(1.5) - dist),
                    _state._cp.float32(0.0),
                ),
            ).astype(_state._cp.float32)
            idx_list.append(idx)
            weight_list.append(weight)
        return idx_list, weight_list
    base = _state._cp.floor(coord_g).astype(_state._cp.int32)
    frac = (coord_g - base.astype(_state._cp.float32)).astype(_state._cp.float32)
    return [base, base + np.int32(1)], [
        (_state._cp.float32(1.0) - frac).astype(_state._cp.float32),
        frac.astype(_state._cp.float32),
    ]


def _get_cext_hybrid_deposit_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_HYBRID_DEPOSIT_KERNEL is not None:
        return _state._CEXT_HYBRID_DEPOSIT_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("cext_hybrid_deposit_kernel.cu")
    _state._CEXT_HYBRID_DEPOSIT_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_hybrid_deposit_kernel"
    )
    return _state._CEXT_HYBRID_DEPOSIT_KERNEL


_state._CEXT_HYBRID_DEPOSIT_RUNTIME_STENCIL_KERNEL = None
_state._CEXT_FFT_MOMENT_DEPOSIT_RUNTIME_STENCIL_KERNEL = None
_state._CEXT_FFT_MOMENT_DEPOSIT_COMPUTED_RUNTIME_STENCIL_KERNEL = None
_state._CEXT_FFT_DISCRETE_SELF_RUNTIME_STENCIL_KERNEL = None
_state._CEXT_FFT_DISCRETE_SELF_RUNTIME_MOMENT_KERNEL = None


def _get_cext_hybrid_deposit_runtime_stencil_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_HYBRID_DEPOSIT_RUNTIME_STENCIL_KERNEL is not None:
        return _state._CEXT_HYBRID_DEPOSIT_RUNTIME_STENCIL_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source(
        "cext_hybrid_deposit_runtime_stencil_kernel.cu",
        prelude="cext_runtime_stencil.cuh",
    )
    _state._CEXT_HYBRID_DEPOSIT_RUNTIME_STENCIL_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_hybrid_deposit_runtime_stencil_kernel"
    )
    return _state._CEXT_HYBRID_DEPOSIT_RUNTIME_STENCIL_KERNEL


def _get_cext_fft_moment_deposit_runtime_stencil_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_FFT_MOMENT_DEPOSIT_RUNTIME_STENCIL_KERNEL is not None:
        return _state._CEXT_FFT_MOMENT_DEPOSIT_RUNTIME_STENCIL_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source(
        "cext_fft_moment_deposit_runtime_stencil_kernel.cu",
        prelude="cext_runtime_stencil.cuh",
    )
    _state._CEXT_FFT_MOMENT_DEPOSIT_RUNTIME_STENCIL_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_fft_moment_deposit_runtime_stencil_kernel"
    )
    return _state._CEXT_FFT_MOMENT_DEPOSIT_RUNTIME_STENCIL_KERNEL


def _get_cext_fft_moment_deposit_computed_runtime_stencil_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_FFT_MOMENT_DEPOSIT_COMPUTED_RUNTIME_STENCIL_KERNEL is not None:
        return _state._CEXT_FFT_MOMENT_DEPOSIT_COMPUTED_RUNTIME_STENCIL_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source(
        "cext_fft_moment_deposit_computed_runtime_stencil_kernel.cu",
        prelude="cext_runtime_stencil.cuh",
    )
    _state._CEXT_FFT_MOMENT_DEPOSIT_COMPUTED_RUNTIME_STENCIL_KERNEL = (
        _state._cp.RawKernel(
            _state.code, "cext_fft_moment_deposit_computed_runtime_stencil_kernel"
        )
    )
    return _state._CEXT_FFT_MOMENT_DEPOSIT_COMPUTED_RUNTIME_STENCIL_KERNEL


def _get_cext_fft_discrete_self_runtime_moment_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_MOMENT_KERNEL is not None:
        return _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_MOMENT_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source(
        "cext_fft_discrete_self_runtime_moment_kernel.cu",
        prelude="cext_runtime_stencil.cuh",
    )
    _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_MOMENT_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_fft_discrete_self_runtime_moment_kernel"
    )
    return _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_MOMENT_KERNEL


def _get_cext_fft_discrete_self_runtime_stencil_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_STENCIL_KERNEL is not None:
        return _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_STENCIL_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source(
        "cext_fft_discrete_self_runtime_stencil_kernel.cu",
        prelude="cext_runtime_stencil.cuh",
    )
    _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_STENCIL_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_fft_discrete_self_runtime_stencil_kernel"
    )
    return _state._CEXT_FFT_DISCRETE_SELF_RUNTIME_STENCIL_KERNEL


__all__ = [
    "_ensure_cext_hybrid_bg_context",
    "_cext_hybrid_axis_assignment_cpu",
    "_cext_hybrid_build_stencil_metadata",
    "_cext_hybrid_build_stencil_metadata_gpu",
    "_cext_hybrid_build_local_node_cells",
    "_build_cext_hybrid_local_source_plan",
    "_resolve_cext_hybrid_bg_mode",
    "_ensure_cext_hybrid_local_only_gpu_static",
    "_ensure_cext_hybrid_local_only_runtime_state",
    "_ensure_cext_hybrid_bg_gpu_static",
    "_ensure_cext_hybrid_bg_runtime_state",
    "_sync_cext_hybrid_bg_runtime_state",
    "_resolve_cext_hybrid_bg_solver_mode",
    "_cext_hybrid_init_lambda_bins",
    "_cext_hybrid_axis_assignment_gpu",
    "_get_cext_hybrid_deposit_kernel",
    "_get_cext_hybrid_deposit_runtime_stencil_kernel",
    "_get_cext_fft_moment_deposit_runtime_stencil_kernel",
    "_get_cext_fft_moment_deposit_computed_runtime_stencil_kernel",
    "_get_cext_fft_discrete_self_runtime_moment_kernel",
    "_get_cext_fft_discrete_self_runtime_stencil_kernel",
]
