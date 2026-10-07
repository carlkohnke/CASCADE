"""External-field source state, caches, and tissue handoff.

This module owns reusable geometry/runtime caches and converts quadrature state
into the representation consumed by tissue oxygen calculations.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from cascade.accelerators.cuda import load_cuda_source
from cascade.concentration.quadrature import _get_gl_nodes_weights
from cascade.concentration.vessel.greens import (
    _cext_green_lambda_from_fields,
    _finite_radius_o2_term_flags,
    _interfacial_transfer_coefficient,
    _lambda_if_from_civ,
    _normalize_cext_lambda_source,
)
from cascade.configuration import solver_state as _state
from cascade.flow.linear_system import njit

from .direct import _ensure_cext_gpu_static
from .frozen import _build_local_exclusion_arrays_numba


def _build_local_exclusion_lists(
    parents: np.ndarray,
    left_child: np.ndarray,
    right_child: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parents_arr = np.asarray(parents, dtype=np.int32)
    left_arr = np.asarray(left_child, dtype=np.int32)
    right_arr = np.asarray(right_child, dtype=np.int32)
    max_local = 32
    if _state._HAVE_NUMBA:
        return _build_local_exclusion_arrays_numba(
            parents_arr, left_arr, right_arr, max_local
        )
    nseg = int(parents_arr.shape[0])
    out_idx = np.full((nseg, max_local), -1, dtype=np.int32)
    out_count = np.zeros((nseg,), dtype=np.uint8)
    for seg in range(nseg):
        seen: set[int] = set()

        def add_immediate(node: int, seen_for_segment=seen) -> None:
            if node < 0 or node >= nseg:
                return
            seen_for_segment.add(node)
            parent = int(parents_arr[node])
            if 0 <= parent < nseg:
                seen_for_segment.add(parent)
                pl = int(left_arr[parent])
                pr = int(right_arr[parent])
                if pl >= 0:
                    seen_for_segment.add(pl)
                if pr >= 0:
                    seen_for_segment.add(pr)
            left = int(left_arr[node])
            right = int(right_arr[node])
            if left >= 0:
                seen_for_segment.add(left)
            if right >= 0:
                seen_for_segment.add(right)

        add_immediate(seg)
        for node in list(seen):
            add_immediate(int(node))
        values = sorted(int(v) for v in seen if 0 <= int(v) < nseg)[:max_local]
        out_count[seg] = np.uint8(len(values))
        if values:
            out_idx[seg, : len(values)] = np.asarray(values, dtype=np.int32)
    return out_idx, out_count


@njit(cache=True)
def _network_exclusions_csr(prox, dist, offsets, incident, max_local):
    """Keep the smallest unique edge IDs in each exact two-hop neighborhood."""
    out = np.full((len(prox), max_local), -1, dtype=np.int32)
    counts = np.zeros(len(prox), dtype=np.uint8)
    for edge in range(len(prox)):
        count = 0
        for endpoint in range(2):
            node = prox[edge] if endpoint == 0 else dist[edge]
            for pos in range(offsets[node], offsets[node + 1]):
                adjacent = incident[pos]
                for other in range(2):
                    neighbor = prox[adjacent] if other == 0 else dist[adjacent]
                    for slot in range(offsets[neighbor], offsets[neighbor + 1]):
                        candidate = incident[slot]
                        index = 0
                        while index < count and out[edge, index] < candidate:
                            index += 1
                        if index < count and out[edge, index] == candidate:
                            continue
                        if index >= max_local:
                            continue
                        limit = min(count, max_local - 1)
                        for move in range(limit, index, -1):
                            out[edge, move] = out[edge, move - 1]
                        out[edge, index] = candidate
                        count = min(count + 1, max_local)
        counts[edge] = count
    return out, counts


def _build_network_local_exclusion_lists(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    *,
    max_local: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    if not _state._HAVE_NUMBA:
        return _build_network_local_exclusion_lists_python(
            prox_ids, dist_ids, max_local=max_local
        )
    prox, dist = (
        np.asarray(prox_ids, dtype=np.int64),
        np.asarray(dist_ids, dtype=np.int64),
    )
    if not len(prox):
        return np.empty((0, max_local), np.int32), np.empty(0, np.uint8)
    if not 0 <= max_local <= 255:
        raise ValueError("max_local must fit the uint8 exclusion count")
    nodes = np.concatenate((prox, dist))
    incident = np.tile(np.arange(len(prox), dtype=np.int32), 2)
    order = np.argsort(nodes, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(np.bincount(nodes)))).astype(np.int64)
    return _network_exclusions_csr(prox, dist, offsets, incident[order], max_local)


def _build_network_local_exclusion_lists_python(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    *,
    max_local: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the same two-hop near-field exclusions for an arbitrary graph."""
    prox = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
    dist = np.asarray(dist_ids, dtype=np.int64).reshape(-1)
    nseg = int(prox.size)
    out_idx = np.full((nseg, max_local), -1, dtype=np.int32)
    out_count = np.zeros((nseg,), dtype=np.uint8)
    if nseg == 0:
        return out_idx, out_count

    node_edges: dict[int, list[int]] = defaultdict(list)
    for edge_idx, (node_a, node_b) in enumerate(zip(prox, dist)):
        node_edges[int(node_a)].append(edge_idx)
        node_edges[int(node_b)].append(edge_idx)
    for edge_idx in range(nseg):
        first = set(node_edges[int(prox[edge_idx])])
        first.update(node_edges[int(dist[edge_idx])])
        nearby = set(first)
        for adjacent in first:
            nearby.update(node_edges[int(prox[adjacent])])
            nearby.update(node_edges[int(dist[adjacent])])
        values = sorted(nearby)[:max_local]
        out_count[edge_idx] = np.uint8(len(values))
        if values:
            out_idx[edge_idx, : len(values)] = np.asarray(values, dtype=np.int32)
    return out_idx, out_count


def _cext_initial_chunk_targets() -> int:
    slots = max(int(_state.CEXT_STREAMING_TARGET_CANDIDATE_SLOTS), 1)
    per_target_guess = max(int(_state.GL_ORDER_CEXT) * 32, 1)
    return max(1, slots // per_target_guess)


def _cext_state_cache_key(
    *,
    fluid_mode: str,
    inlet_concentration: float,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> tuple:
    return (
        str(fluid_mode).lower(),
        int(_state.GL_ORDER_CEXT),
        float(inlet_concentration),
        float(diffusivity_si),
        float(vmax),
        float(km),
        str(_state.JUNCTION_OXYGEN_BALANCE),
        str(_state.BLOOD_CONVECTIVE_HEMATOCRIT),
        str(_state.CEXT_VESS_COUPLING_NORM),
    )


def _get_tree_cext_state_cache(
    tree: _state.Tree,
    cache_key: tuple[str, int, float, float, float, float],
) -> dict | None:
    cache_root = getattr(tree, "_cext_state_cache", None)
    if not isinstance(cache_root, dict):
        return None
    cached = cache_root.get(cache_key)
    return cached if isinstance(cached, dict) else None


def _store_tree_cext_state_cache(
    tree: _state.Tree,
    cache_key: tuple[str, int, float, float, float, float],
    ext_state: dict,
) -> None:
    cache_root = getattr(tree, "_cext_state_cache", None)
    if not isinstance(cache_root, dict):
        cache_root = {}
        tree._cext_state_cache = cache_root
    cache_root[cache_key] = {
        "c_ext_gl": np.asarray(ext_state["c_ext_gl"], dtype=np.float32).copy(),
        "c_iv_gl": np.asarray(ext_state["c_iv_gl"], dtype=np.float32).copy(),
        "c_bulk_gl": np.asarray(
            ext_state.get("c_bulk_gl", ext_state["c_iv_gl"]), dtype=np.float32
        ).copy(),
        "c_wall_gl": np.asarray(
            ext_state.get("c_wall_gl", ext_state["c_iv_gl"]), dtype=np.float32
        ).copy(),
        "cin_seg": np.asarray(ext_state["cin_seg"], dtype=np.float32).copy(),
        "cout_seg": np.asarray(ext_state["cout_seg"], dtype=np.float32).copy(),
    }


def _clear_cext_runtime_state(tree: _state.Tree | None = None) -> None:
    """Drop large per-fluid Cext arrays after tissue sampling consumes them."""
    # Mutable runtime state is centralized in configuration.solver_state.
    _state._LAST_CEXT_SOURCE_STATE = None
    if tree is not None:
        cache_root = getattr(tree, "_cext_state_cache", None)
        if isinstance(cache_root, dict):
            cache_root.clear()


def _initialize_cext_state(
    tree: _state.Tree,
    *,
    cache_key: tuple[str, int, float, float, float, float],
    nseg: int,
    inlet_concentration: float,
    vmax: float,
    km: float,
) -> dict:
    gl_order = int(_state.GL_ORDER_CEXT)
    ext_state = {
        "c_ext_gl": np.zeros((nseg, gl_order), dtype=np.float32),
        "c_iv_gl": np.full(
            (nseg, gl_order), float(inlet_concentration), dtype=np.float32
        ),
        "c_bulk_gl": np.full(
            (nseg, gl_order), float(inlet_concentration), dtype=np.float32
        ),
        "c_wall_gl": np.full(
            (nseg, gl_order), float(inlet_concentration), dtype=np.float32
        ),
        "cin_seg": np.full((nseg,), float(inlet_concentration), dtype=np.float32),
        "cout_seg": np.full((nseg,), float(inlet_concentration), dtype=np.float32),
        "vmax": float(vmax),
        "km": float(km),
        "window_factor": float(_state.CEXT_WINDOW_FACTOR),
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
    ext_state["c_ext_gl"][:prefix] = np.asarray(cached["c_ext_gl"], dtype=np.float32)[
        :prefix
    ]
    ext_state["c_iv_gl"][:prefix] = np.asarray(cached["c_iv_gl"], dtype=np.float32)[
        :prefix
    ]
    ext_state["c_bulk_gl"][:prefix] = np.asarray(
        cached.get("c_bulk_gl", cached["c_iv_gl"]), dtype=np.float32
    )[:prefix]
    ext_state["c_wall_gl"][:prefix] = np.asarray(
        cached.get("c_wall_gl", cached["c_iv_gl"]), dtype=np.float32
    )[:prefix]
    ext_state["cin_seg"][:prefix] = np.asarray(cached["cin_seg"], dtype=np.float32)[
        :prefix
    ]
    ext_state["cout_seg"][:prefix] = np.asarray(cached["cout_seg"], dtype=np.float32)[
        :prefix
    ]
    return ext_state


def _build_cext_iteration_cache(context: dict, ext_state: dict) -> dict:
    diffusivity_si = float(context["diffusivity_si"])
    c_bulk_floor = np.maximum(
        np.asarray(ext_state["c_iv_gl"], dtype=float), _state.VESS_CONC_FLOOR
    )
    c_wall_raw = ext_state.get("c_wall_gl")
    if c_wall_raw is None:
        c_wall_floor = c_bulk_floor.copy()
    else:
        c_wall_floor = np.maximum(
            np.asarray(c_wall_raw, dtype=float), _state.VESS_CONC_FLOOR
        )
    c_ext_pos = np.maximum(np.asarray(ext_state["c_ext_gl"], dtype=float), 0.0)
    lambda_if_gl = _lambda_if_from_civ(
        c_wall_floor, diffusivity_si, float(ext_state["vmax"]), float(ext_state["km"])
    )
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
    mono2_weight_gl = (
        (0.25 * a2 * q_weighted_gl) if include_mono2 else np.zeros_like(q_weighted_gl)
    )
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
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_ITERATION_CACHE_KERNEL is not None:
        return _state._CEXT_ITERATION_CACHE_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("cext_iteration_cache_kernel.cu")
    _state._CEXT_ITERATION_CACHE_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_iteration_cache_kernel"
    )
    return _state._CEXT_ITERATION_CACHE_KERNEL


def _build_cext_iteration_cache_gpu(
    context: dict, ext_state: dict, runtime_state: dict, *, materialize: bool = False
) -> None:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    gl_shape = tuple(np.asarray(ext_state["c_iv_gl"], dtype=np.float32).shape)
    if (
        runtime_state.get("cache_c_bulk_gl") is None
        or tuple(runtime_state["cache_c_bulk_gl"].shape) != gl_shape
    ):
        runtime_state["cache_c_bulk_gl"] = _state._cp.empty(
            gl_shape, dtype=_state._cp.float32
        )
        runtime_state["cache_c_wall_gl"] = _state._cp.empty(
            gl_shape, dtype=_state._cp.float32
        )
    runtime_state["cache_c_bulk_gl"].set(
        np.asarray(ext_state["c_iv_gl"], dtype=np.float32)
    )
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
    for name in ("lambda_if_gl", "k_if_gl", "q_line_gl"):
        if name not in runtime_state or tuple(runtime_state[name].shape) != gl_shape:
            runtime_state[name] = _state._cp.empty(gl_shape, dtype=_state._cp.float32)
    # A retained runtime can outlive a reset/reinitialized numerical field.
    # Refresh the authoritative field even for intermediate source updates.
    runtime_state["c_ext_gl"].set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))
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
            np.int32(int(np.asarray(_state._KRATIO_XS).size)),
            np.int32(nseg),
            np.int32(gl_order),
            np.float32(float(context["diffusivity_si"])),
            np.float32(float(ext_state["vmax"])),
            np.float32(float(ext_state["km"])),
            np.float32(float(_state.VESS_CONC_FLOOR)),
            np.int32(1 if _normalize_cext_lambda_source() == "lambda_t" else 0),
            np.int32(1 if include_mono2 else 0),
            np.int32(1 if include_dipole2 else 0),
            runtime_state["q_weighted_gl"].ravel(),
            runtime_state["mono2_weight_gl"].ravel(),
            runtime_state["dipole2_weight_gl"].ravel(),
            runtime_state["lambda_iv_gl"].ravel(),
            runtime_state["seg_cap_gl"],
            runtime_state["lambda_if_gl"].ravel(),
            runtime_state["k_if_gl"].ravel(),
            runtime_state["q_line_gl"].ravel(),
        ),
    )
    _state._cp.cuda.Stream.null.synchronize()
    ext_state["lambda_iv_gl"] = _state._cp.asnumpy(runtime_state["lambda_iv_gl"])
    if materialize:
        # Preserve the complete public source handoff, including flux checks;
        # intermediate iterations need only download lambda-bin input.
        for name in (
            "q_weighted_gl",
            "mono2_weight_gl",
            "dipole2_weight_gl",
            "seg_cap_gl",
            "lambda_if_gl",
            "k_if_gl",
            "q_line_gl",
        ):
            ext_state[name] = _state._cp.asnumpy(runtime_state[name])
        for name, device_name in (
            ("c_bulk_gl", "cache_c_bulk_gl"),
            ("c_wall_gl", "cache_c_wall_gl"),
        ):
            ext_state[name] = np.maximum(
                np.nan_to_num(
                    _state._cp.asnumpy(runtime_state[device_name]),
                    nan=_state.VESS_CONC_FLOOR,
                ),
                np.float32(_state.VESS_CONC_FLOOR),
            )
    ext_state["_lambda_bin_epoch"] = int(ext_state.get("_lambda_bin_epoch", 0)) + 1


def validate_cext_tissue_flux_consistency(
    cext_state: dict, *, rtol: float = 1e-5, atol: float = 1e-8
) -> dict[str, float | bool]:
    """Check that q_line uses the wall concentration; c_iv_gl is bulk/cup."""
    if "k_if_gl" not in cext_state:
        return {
            "ok": True,
            "max_abs": float("nan"),
            "rel_l2": float("nan"),
            "skipped": True,
        }
    c_wall = np.maximum(
        np.asarray(
            cext_state.get("c_wall_gl", cext_state["c_iv_gl"]), dtype=np.float32
        ),
        np.float32(_state.VESS_CONC_FLOOR),
    )
    c_ext = np.maximum(
        np.asarray(cext_state["c_ext_gl"], dtype=np.float32), np.float32(0.0)
    )
    k_if = np.asarray(cext_state["k_if_gl"], dtype=np.float32)
    q_line = np.asarray(cext_state["q_line_gl"], dtype=np.float32)
    expected = k_if * (c_wall - c_ext)
    diff = q_line - expected
    max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    denom = float(np.linalg.norm(expected.reshape(-1))) if expected.size else 0.0
    rel_l2 = (
        float(np.linalg.norm(diff.reshape(-1)) / max(denom, 1e-30))
        if diff.size
        else 0.0
    )
    return {
        "ok": bool(
            max_abs <= float(atol) + float(rtol) * float(np.nanmax(np.abs(expected)))
            if expected.size
            else True
        ),
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "skipped": False,
    }


def _snapshot_cext_source_state(
    context: dict,
    ext_state: dict,
    *,
    solver: str,
    backend: str,
) -> dict:
    c_ext_gl = np.asarray(ext_state["c_ext_gl"], dtype=np.float32)
    q_weighted_gl = np.asarray(
        ext_state.get("q_weighted_gl", np.zeros_like(c_ext_gl)), dtype=np.float32
    )
    return {
        "solver": str(solver),
        "backend": str(backend),
        "gl_points_si": np.asarray(context["gl_points_si"], dtype=np.float32).copy(),
        "diffusivity_si": float(context["diffusivity_si"]),
        "window_factor": float(
            ext_state.get("window_factor", _state.CEXT_WINDOW_FACTOR)
        ),
        "c_iv_gl": np.asarray(ext_state["c_iv_gl"], dtype=np.float32).copy(),
        "c_bulk_gl": np.asarray(
            ext_state.get("c_bulk_gl", ext_state["c_iv_gl"]), dtype=np.float32
        ).copy(),
        "c_wall_gl": np.asarray(
            ext_state.get("c_wall_gl", ext_state["c_iv_gl"]), dtype=np.float32
        ).copy(),
        "c_ext_gl": c_ext_gl.copy(),
        "lambda_iv_gl": np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32).copy(),
        "k_if_gl": np.asarray(ext_state["k_if_gl"], dtype=np.float32).copy(),
        "q_line_gl": np.asarray(ext_state["q_line_gl"], dtype=np.float32).copy(),
        "q_weighted_gl": q_weighted_gl.copy(),
        "mono2_weight_gl": np.asarray(
            ext_state.get("mono2_weight_gl", np.zeros_like(q_weighted_gl)),
            dtype=np.float32,
        ).copy(),
        "dipole2_weight_gl": np.asarray(
            ext_state.get("dipole2_weight_gl", np.zeros_like(q_weighted_gl)),
            dtype=np.float32,
        ).copy(),
        "seg_cap_gl": np.asarray(ext_state["seg_cap_gl"], dtype=np.float32).copy(),
        "segment_vectors": np.asarray(
            context["segment_vectors"], dtype=np.float32
        ).copy(),
    }


def _resample_cext_source_state_for_tissue(
    cext_state: dict,
    order: int,
) -> dict:
    """Resample converged vessel fields onto an independent tissue quadrature.

    Cext coupling is solved at ``GL_ORDER_CEXT``.  The final vessel-to-tissue
    Green's integral may use ``GL_ORDER`` instead; weighted source terms are
    reconstructed as line densities before being integrated with the new
    Gauss–Legendre weights.
    """
    target_order = max(int(order), 1)
    points = np.asarray(cext_state["gl_points_si"], dtype=np.float32)
    if points.ndim != 3 or points.shape[1] == target_order:
        return cext_state
    old_order = int(points.shape[1])
    if old_order <= 0:
        return cext_state

    old_nodes, old_weights = _get_gl_nodes_weights(old_order)
    new_nodes, new_weights = _get_gl_nodes_weights(target_order)
    old_t = 0.5 * (np.asarray(old_nodes, dtype=float) + 1.0)
    new_t = 0.5 * (np.asarray(new_nodes, dtype=float) + 1.0)
    vectors = np.asarray(
        cext_state.get("segment_vectors", np.zeros((points.shape[0], 3))),
        dtype=np.float32,
    )

    def interpolate_rows(values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != old_order:
            return array
        # The production heart contract resamples Cext GL1 to tissue GL5.
        # Linear interpolation from one source node is constant, so avoid a
        # Python loop over tens of millions of vessel segments.
        if old_order == 1:
            return np.repeat(array, target_order, axis=1)
        out = np.empty((array.shape[0], target_order), dtype=np.float32)
        for segment_id in range(array.shape[0]):
            out[segment_id] = np.interp(new_t, old_t, array[segment_id]).astype(
                np.float32,
                copy=False,
            )
        return out

    lengths = np.linalg.norm(vectors, axis=1).astype(np.float64, copy=False)
    old_ds = 0.5 * lengths[:, None] * np.asarray(old_weights, dtype=float)[None, :]
    new_ds = 0.5 * lengths[:, None] * np.asarray(new_weights, dtype=float)[None, :]

    def reweight(values) -> np.ndarray:
        weighted = np.asarray(values, dtype=np.float32)
        density = np.divide(
            weighted,
            old_ds,
            out=np.zeros_like(weighted, dtype=np.float64),
            where=old_ds > 0.0,
        )
        return np.asarray(interpolate_rows(density) * new_ds, dtype=np.float32)

    resampled = dict(cext_state)
    flow_start = points[:, 0, :] - np.asarray(old_t[0], dtype=np.float32) * vectors
    resampled["gl_points_si"] = np.asarray(
        flow_start[:, None, :] + new_t[None, :, None] * vectors[:, None, :],
        dtype=np.float32,
    )
    for key in (
        "c_iv_gl",
        "c_bulk_gl",
        "c_wall_gl",
        "c_ext_gl",
        "lambda_iv_gl",
        "k_if_gl",
        "q_line_gl",
    ):
        if key in cext_state:
            resampled[key] = interpolate_rows(cext_state[key])
    for key in ("q_weighted_gl", "mono2_weight_gl", "dipole2_weight_gl"):
        if key in cext_state:
            resampled[key] = reweight(cext_state[key])
    resampled["tissue_gl_order"] = target_order
    resampled["cext_gl_order"] = old_order
    return resampled


def _prepare_cext_source_state_for_tissue(cext_state: dict, order: int) -> dict:
    """Apply the explicit production/legacy tissue-quadrature policy."""
    mode = str(_state.CEXT_TISSUE_QUADRATURE_MODE).strip().lower()
    if mode == "independent":
        return _resample_cext_source_state_for_tissue(cext_state, order)
    if mode == "legacy_cext":
        legacy = dict(cext_state)
        points = np.asarray(legacy.get("gl_points_si", np.empty((0, 0, 3))))
        legacy["tissue_gl_order"] = int(points.shape[1]) if points.ndim == 3 else 0
        legacy["cext_gl_order"] = legacy["tissue_gl_order"]
        legacy["quadrature_compatibility_mode"] = "legacy_cext"
        return legacy
    raise ValueError(
        "CEXT_TISSUE_QUADRATURE_MODE must be 'independent' or 'legacy_cext', "
        f"got {_state.CEXT_TISSUE_QUADRATURE_MODE!r}."
    )


def _set_last_cext_source_state(
    context: dict,
    ext_state: dict,
    *,
    solver: str,
    backend: str,
) -> None:
    # Mutable runtime state is centralized in configuration.solver_state.
    _state._LAST_CEXT_SOURCE_STATE = _snapshot_cext_source_state(
        context,
        ext_state,
        solver=solver,
        backend=backend,
    )


__all__ = [
    "_build_cext_iteration_cache",
    "_build_cext_iteration_cache_gpu",
    "_build_local_exclusion_lists",
    "_build_network_local_exclusion_lists",
    "_cext_initial_chunk_targets",
    "_cext_state_cache_key",
    "_clear_cext_runtime_state",
    "_get_cext_iteration_cache_kernel",
    "_get_tree_cext_state_cache",
    "_initialize_cext_state",
    "_prepare_cext_source_state_for_tissue",
    "_resample_cext_source_state_for_tissue",
    "_set_last_cext_source_state",
    "_snapshot_cext_source_state",
    "_store_tree_cext_state_cache",
    "validate_cext_tissue_flux_consistency",
]
