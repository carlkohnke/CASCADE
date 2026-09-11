"""External-field coupling across one or more vascular networks.

This is the shared multi-network implementation used by anatomical forests
and any other case whose vessels interact through one external field.
"""

from __future__ import annotations

import gc
import logging
from time import perf_counter

import numpy as np

from cascade.configuration.settings.oxygen import DEFAULTS as _OXYGEN_DEFAULTS
from cascade.vessels.collections import VascularNetworkSet
from .preparation import (
    prepare_network_external_field,
    run_frozen_transport_step,
)

K_M_MM_DEFAULT = float(_OXYGEN_DEFAULTS["K_M_MM"])
VMAX_MM_DEFAULT = float(_OXYGEN_DEFAULTS["VMAX_MM"])
WINDOW_FACTOR_DEFAULT = 6.0
_LOGGER = logging.getLogger(__name__)


def _log(message: str) -> None:
    _LOGGER.info(message)


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
        [
            int(
                np.asarray(
                    ctx.get("exclude_idx", np.empty((0, 0), dtype=np.int32))
                ).shape[1]
            )
            for ctx in contexts
        ]
        or [32]
    )
    offset = 0
    for ctx, count in zip(contexts, counts):
        for source_key, target in (
            ("parents", parent_parts),
            ("left_child", left_parts),
            ("right_child", right_parts),
        ):
            arr = np.asarray(
                ctx.get(source_key, np.full((count,), -1, dtype=np.int32)),
                dtype=np.int32,
            ).copy()
            valid = arr >= 0
            arr[valid] += np.int32(offset)
            target.append(arr)
        order = np.asarray(
            ctx.get("order", np.arange(count, dtype=np.int32)), dtype=np.int32
        ).copy()
        order_parts.append(order + np.int32(offset))
        ex = np.asarray(
            ctx.get("exclude_idx", np.full((count, exclude_width), -1, dtype=np.int32)),
            dtype=np.int32,
        )
        ex_out = np.full((count, exclude_width), -1, dtype=np.int32)
        width = min(int(ex.shape[1]) if ex.ndim == 2 else 0, exclude_width)
        if width > 0:
            ex_slice = ex[:, :width].copy()
            valid_ex = ex_slice >= 0
            ex_slice[valid_ex] += np.int32(offset)
            ex_out[:, :width] = ex_slice
        ex_count = np.asarray(
            ctx.get("exclude_count", np.zeros((count,), dtype=np.uint8)), dtype=np.uint8
        ).reshape(-1)
        if ex_count.shape[0] != count:
            ex_count = np.zeros((count,), dtype=np.uint8)
        exclude_parts.append(ex_out)
        exclude_count_parts.append(np.minimum(ex_count, np.uint8(exclude_width)))
        offset += count

    gl_points = cat("gl_points_si", np.float32)
    gl_order = int(gl_points.shape[1]) if gl_points.ndim >= 2 else 0
    exclude_idx = (
        np.concatenate(exclude_parts, axis=0)
        if exclude_parts
        else np.full((0, exclude_width), -1, dtype=np.int32)
    )
    exclude_count = (
        np.concatenate(exclude_count_parts, axis=0)
        if exclude_count_parts
        else np.zeros((0,), dtype=np.uint8)
    )

    out = {
        "parents": np.concatenate(parent_parts)
        if parent_parts
        else np.empty((0,), dtype=np.int32),
        "left_child": np.concatenate(left_parts)
        if left_parts
        else np.empty((0,), dtype=np.int32),
        "right_child": np.concatenate(right_parts)
        if right_parts
        else np.empty((0,), dtype=np.int32),
        "order": np.concatenate(order_parts)
        if order_parts
        else np.empty((0,), dtype=np.int32),
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
        "gl_t": np.asarray(contexts[0]["gl_t"], dtype=np.float32)
        if contexts
        else np.empty((0,), dtype=np.float32),
        "gl_weights": np.asarray(contexts[0]["gl_weights"], dtype=np.float32)
        if contexts
        else np.empty((0,), dtype=np.float32),
        "gl_points_si": gl_points,
        "ds_gl": cat("ds_gl", np.float32).reshape((total, gl_order))
        if gl_order
        else np.empty((total, 0), dtype=np.float32),
        "exclude_idx": exclude_idx,
        "exclude_count": exclude_count,
        "diffusivity_si": float(contexts[0]["diffusivity_si"]) if contexts else 0.0,
        "lambda_inlet": max(
            (float(ctx.get("lambda_inlet", 0.0)) for ctx in contexts), default=0.0
        ),
        "cell_size": 0.0,
        "reach_si": cat("reach_si", np.float32)
        if contexts
        else np.empty((0,), dtype=np.float32),
        "max_reach_si": max(
            (float(ctx.get("max_reach_si", 0.0)) for ctx in contexts), default=0.0
        ),
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
        return (
            np.concatenate(pieces, axis=0)
            if pieces
            else np.empty((0,), dtype=np.float32)
        )

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
        "_lambda_bin_epoch": max(
            (int(state.get("_lambda_bin_epoch", 0)) for state in states), default=0
        ),
    }


def _compute_global_gfm_cext(
    cext_ts, solutions: list[dict], context: dict | None = None
) -> tuple[np.ndarray, dict, dict]:
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
        "origin_si": np.asarray(
            hybrid.get("origin", np.zeros((3,), dtype=np.float32)), dtype=float
        ).tolist(),
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


def _apply_global_cext_update(
    solutions: list[dict], c_ext_target: np.ndarray, omega: float
) -> tuple[float, float]:
    counts = [
        int(np.asarray(sol["cext_state"]["c_ext_gl"]).shape[0]) for sol in solutions
    ]
    old_global = np.concatenate(
        [
            np.asarray(sol["cext_state"]["c_ext_gl"], dtype=np.float32)
            for sol in solutions
        ],
        axis=0,
    )
    target = np.asarray(c_ext_target, dtype=np.float32)
    if old_global.shape != target.shape:
        raise ValueError(
            f"Global Cext target shape {target.shape} does not match current state {old_global.shape}."
        )
    omega_eff = float(omega)
    updated = old_global + np.float32(omega_eff) * (target - old_global)
    delta = target - old_global
    finite_delta = delta[np.isfinite(delta)]
    max_delta = float(np.max(np.abs(finite_delta))) if finite_delta.size else 0.0
    denom = float(np.linalg.norm(target.reshape(-1)))
    rel_delta = (
        float(np.linalg.norm(delta.reshape(-1)) / max(denom, 1.0e-30))
        if delta.size
        else 0.0
    )
    offset = 0
    for sol, count in zip(solutions, counts):
        sol["cext_state"]["c_ext_gl"] = np.asarray(
            updated[offset : offset + count], dtype=np.float32
        )
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
        for attr in (
            "_LAST_CEXT_SOURCE_STATE",
            "_LAST_CEXT_CONTEXT",
            "_LAST_TISSUE_TIMINGS",
        ):
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


def _solve_multinetwork_shared_global(
    cext_ts,
    ts,
    networks: VascularNetworkSet,
    inlet_flows: list[float],
    *,
    fluid: str,
) -> tuple[list[dict], dict]:
    if cext_ts._cp is None:
        raise RuntimeError(
            "Cext mode requires GPU/CuPy support. Install CuPy or rerun with --no-cext."
        )
    t_total = perf_counter()
    solutions = []
    initial_cache_releases = []
    for tree_id, (tree, inlet_flow) in enumerate(zip(networks.networks, inlet_flows)):
        _log(f"Solving Cext initial state for tree {tree_id}...")
        sol = prepare_network_external_field(
            cext_ts, ts, tree, inlet_flow, fluid=fluid, progress=_log
        )
        solutions.append(sol)
        initial_cache_releases.append(
            _release_cext_transient_gpu_cache(cext_ts, sol.get("cext_context"))
        )
    if (
        str(getattr(cext_ts, "CEXT_VESS_COUPLING_ACCEL", "none")).strip().lower()
        != "none"
    ):
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
        c_ext_new, timings, grid_meta = _compute_global_gfm_cext(
            cext_ts, solutions, global_context
        )
        max_delta_last, rel_delta_last = _apply_global_cext_update(
            solutions, c_ext_new, omega
        )
        reflected_total = 0.0
        for sol in solutions:
            t_reflect = perf_counter()
            run_frozen_transport_step(cext_ts, sol, fluid=fluid)
            elapsed = float(perf_counter() - t_reflect)
            sol["cext_reflected_topdown_s"] = float(
                sol.get("cext_reflected_topdown_s", 0.0) + elapsed
            )
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
        _snapshot_sol_cext_state(
            cext_ts, sol, solver=f"shared_global_gfm_{bg_mode}", backend="gpu"
        )

    cext_vals = [
        np.asarray(sol["cext_state"]["c_ext_gl"], dtype=np.float32).reshape(-1)
        for sol in solutions
        if np.asarray(sol["cext_state"]["c_ext_gl"]).size
    ]
    all_cext = (
        np.concatenate(cext_vals) if cext_vals else np.empty((0,), dtype=np.float32)
    )
    cext_meta = {"timings": {}, "grid": grid_meta}
    cext_meta.update(
        {
            "enabled": True,
            "mode": "shared_global_gfm",
            "bg_mode": str(bg_mode),
            "near_radius_mult": float(
                getattr(cext_ts, "CEXT_HYBRID_BG_NEAR_RADIUS_MULT", 0.0)
            ),
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
    cext_meta["post_cext_gpu_cache_release"] = _release_cext_transient_gpu_cache(
        cext_ts, global_context
    )
    return solutions, cext_meta


def solve_multinetwork_external_field(
    cext_ts,
    ts,
    networks: VascularNetworkSet,
    inlet_flows: list[float],
    *,
    fluid: str,
    mode: str = "shared-global",
) -> tuple[list[dict], dict]:
    """Solve Cext for one or more networks using one canonical dispatcher."""
    mode = str(mode).strip().lower()
    if mode != "shared-global":
        raise ValueError(f"Unknown multi-network external-field mode: {mode!r}")
    return _solve_multinetwork_shared_global(
        cext_ts, ts, networks, inlet_flows, fluid=fluid
    )


def compact_external_field_state(sol: dict) -> dict:
    """Return the compact Cext state required by tissue solving and export."""
    context = sol.get("cext_context")
    state = sol["cext_state"]
    if context is not None:
        gl_points = np.asarray(context["gl_points_si"], dtype=np.float32)
        diffusivity_si = float(context["diffusivity_si"])
        segment_vectors = np.asarray(
            context.get(
                "segment_vectors", np.zeros((gl_points.shape[0], 3), dtype=np.float32)
            ),
            dtype=np.float32,
        )
    else:
        gl_points = np.asarray(state["gl_points_si"], dtype=np.float32)
        diffusivity_si = float(state["diffusivity_si"])
        segment_vectors = np.asarray(
            state.get(
                "segment_vectors", np.zeros((gl_points.shape[0], 3), dtype=np.float32)
            ),
            dtype=np.float32,
        )
    q_weighted = np.asarray(state["q_weighted_gl"], dtype=np.float32)
    return {
        "solver": str(state.get("solver", "cext_state")),
        "backend": str(state.get("backend", "gpu")),
        "gl_points_si": gl_points,
        "diffusivity_si": diffusivity_si,
        "window_factor": float(state.get("window_factor", WINDOW_FACTOR_DEFAULT)),
        "c_iv_gl": np.asarray(state["c_iv_gl"], dtype=np.float32),
        "c_bulk_gl": np.asarray(
            state.get("c_bulk_gl", state["c_iv_gl"]), dtype=np.float32
        ),
        "c_wall_gl": np.asarray(
            state.get("c_wall_gl", state["c_iv_gl"]), dtype=np.float32
        ),
        "c_ext_gl": np.asarray(state["c_ext_gl"], dtype=np.float32),
        "lambda_iv_gl": np.asarray(state["lambda_iv_gl"], dtype=np.float32),
        "k_if_gl": np.asarray(state["k_if_gl"], dtype=np.float32),
        "q_line_gl": np.asarray(state["q_line_gl"], dtype=np.float32),
        "q_weighted_gl": q_weighted,
        "mono2_weight_gl": np.asarray(
            state.get("mono2_weight_gl", np.zeros_like(q_weighted)), dtype=np.float32
        ),
        "dipole2_weight_gl": np.asarray(
            state.get("dipole2_weight_gl", np.zeros_like(q_weighted)), dtype=np.float32
        ),
        "seg_cap_gl": np.asarray(state["seg_cap_gl"], dtype=np.float32),
        "segment_vectors": segment_vectors,
    }


__all__ = (
    "solve_multinetwork_external_field",
    "compact_external_field_state",
    "_release_cext_transient_gpu_cache",
)
