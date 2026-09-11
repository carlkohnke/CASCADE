"""Whole-forest external-field coupling."""

from __future__ import annotations

from .heart_support import (
    Forest,
    K_M_MM_DEFAULT,
    VMAX_MM_DEFAULT,
    WINDOW_FACTOR_DEFAULT,
    gc,
    np,
    perf_counter,
)

from .heart_domain import (
    _log,
)

from .heart_flow import (
    _run_cext_frozen_step,
    _solve_tree_cext_prepare,
)

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
        raise RuntimeError("Cext mode requires CuPy/GPU support, but the packaged CASCADE runtime is unavailable.")
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




__all__ = ('_global_cext_box', '_global_lambda_bins', '_make_tree_hybrid_for_global_box', '_solve_shared_cext_fft', '_compute_shared_plain_box_cext', '_solve_forest_cext_legacy_shared_one_shot', '_concat_global_cext_context', '_concat_global_cext_state', '_compute_global_gfm_cext', '_apply_global_cext_update', '_snapshot_sol_cext_state', '_solve_tree_cext_backend', '_solve_forest_cext_backend_per_tree', '_release_cext_transient_gpu_cache', '_solve_forest_cext_shared_global', '_solve_forest_cext', '_compact_cext_state_for_concat')
