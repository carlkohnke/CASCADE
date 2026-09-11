"""External-field geometry and candidate plans.

It prepares segment quadrature, neighbor lists, and compact CPU/GPU work plans
that are reused across coupling iterations.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from time import perf_counter

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.concentration.vessel.greens import _lambda_if_from_civ
from cascade.concentration.quadrature import _get_gl_nodes_weights
from cascade.flow.hematocrit import _hematocrit_context_for_tree

from .backend import _build_topdown_level_slices
from .state import (
    _build_local_exclusion_lists,
    _build_network_local_exclusion_lists,
    _cext_initial_chunk_targets,
)

try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:  # pragma: no cover - dependency validation handles this
    _cKDTree = None


def _coerce_candidate_batches_for_gpu(context: dict) -> list[dict]:
    batches = list(context.get("candidate_batches", []))
    if _state._cp is None:
        return batches
    for batch in batches:
        gpu_batch = batch.get("gpu")
        if isinstance(gpu_batch, dict):
            continue
        batch["gpu"] = {
            "target_seg_ids": _state._cp.asarray(
                np.asarray(batch["target_seg_ids"], dtype=np.int32)
            ),
            "row_ptr": _state._cp.asarray(np.asarray(batch["row_ptr"], dtype=np.int32)),
            "col_idx": _state._cp.asarray(np.asarray(batch["col_idx"], dtype=np.int32)),
        }
    return batches


def _maybe_trim_candidate_list(
    ordered: list[int],
    *,
    target_mid: np.ndarray,
    midpoints_si: np.ndarray,
) -> list[int]:
    max_candidates = int(_state.CEXT_MAX_CANDIDATES_PER_TARGET)
    if max_candidates <= 0 or len(ordered) <= max_candidates:
        return ordered
    ranked = sorted(
        ordered,
        key=lambda source_i: float(
            np.dot(
                midpoints_si[source_i] - target_mid, midpoints_si[source_i] - target_mid
            )
        ),
    )
    return ranked[:max_candidates]


def _maybe_trim_candidate_array(
    candidates: np.ndarray,
    *,
    target_mid: np.ndarray,
    midpoints_si: np.ndarray,
) -> np.ndarray:
    candidate_ids = np.asarray(candidates, dtype=np.int32)
    max_candidates = int(_state.CEXT_MAX_CANDIDATES_PER_TARGET)
    if candidate_ids.size == 0:
        return candidate_ids
    if max_candidates <= 0 or candidate_ids.size <= max_candidates:
        return np.sort(candidate_ids)
    delta = np.asarray(midpoints_si[candidate_ids] - target_mid[None, :], dtype=float)
    dist2 = np.einsum("ij,ij->i", delta, delta)
    keep_idx = np.argsort(dist2, kind="stable")[:max_candidates]
    return np.asarray(candidate_ids[keep_idx], dtype=np.int32)


def _build_cext_geometry_context(
    tree: _state.Tree,
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
    network_topology: dict | None = None,
) -> dict:
    if network_topology is None:
        hct_context = _hematocrit_context_for_tree(tree)
        parents = np.asarray(hct_context["parents"], dtype=np.int32)
        left_child = np.asarray(hct_context["left_child"], dtype=np.int32)
        right_child = np.asarray(hct_context["right_child"], dtype=np.int32)
        order = np.asarray(hct_context["order"], dtype=np.int64)
        level_order, level_offsets, depth = _build_topdown_level_slices(order, parents)
        exclude_idx, exclude_count = _build_local_exclusion_lists(
            parents, left_child, right_child
        )
    else:
        nseg = int(np.asarray(flows).size)
        parents = np.full((nseg,), -1, dtype=np.int32)
        left_child = np.full((nseg,), -1, dtype=np.int32)
        right_child = np.full((nseg,), -1, dtype=np.int32)
        order = np.arange(nseg, dtype=np.int64)
        level_order = np.asarray(order, dtype=np.int32)
        level_offsets = np.asarray([0, nseg], dtype=np.int32)
        depth = np.zeros((nseg,), dtype=np.int32)
        exclude_idx, exclude_count = _build_network_local_exclusion_lists(
            np.asarray(network_topology["prox_ids"], dtype=np.int64),
            np.asarray(network_topology["dist_ids"], dtype=np.int64),
        )

    flow_starts = np.asarray(starts, dtype=float).copy()
    flow_ends = np.asarray(ends, dtype=float).copy()
    flip = np.asarray(flows, dtype=float) < 0.0
    flow_starts[flip] = np.asarray(ends, dtype=float)[flip]
    flow_ends[flip] = np.asarray(starts, dtype=float)[flip]

    flow_starts_si = np.asarray(
        flow_starts * _state.CM_TO_M, dtype=_state.CEXT_FLOAT_DTYPE
    )
    flow_ends_si = np.asarray(flow_ends * _state.CM_TO_M, dtype=_state.CEXT_FLOAT_DTYPE)
    flow_vectors_si = np.asarray(
        flow_ends_si - flow_starts_si, dtype=_state.CEXT_FLOAT_DTYPE
    )
    midpoints_si = np.asarray(
        0.5 * (flow_starts_si + flow_ends_si), dtype=_state.CEXT_FLOAT_DTYPE
    )
    flows_si = np.asarray(
        np.asarray(flows, dtype=float) * _state.CM3_TO_M3, dtype=_state.CEXT_FLOAT_DTYPE
    )
    lengths_si = np.asarray(
        np.maximum(np.asarray(lengths, dtype=float), 0.0) * _state.CM_TO_M,
        dtype=_state.CEXT_FLOAT_DTYPE,
    )
    radii_si = np.asarray(
        np.maximum(np.asarray(radii, dtype=float), 0.0) * _state.CM_TO_M,
        dtype=_state.CEXT_FLOAT_DTYPE,
    )
    gl_nodes, gl_weights = _get_gl_nodes_weights(_state.GL_ORDER_CEXT)
    gl_t = np.asarray(0.5 * (gl_nodes + 1.0), dtype=_state.CEXT_FLOAT_DTYPE)
    gl_weights_arr = np.asarray(gl_weights, dtype=_state.CEXT_FLOAT_DTYPE)
    gl_points_si = np.asarray(
        flow_starts_si[:, None, :] + gl_t[None, :, None] * flow_vectors_si[:, None, :],
        dtype=_state.CEXT_FLOAT_DTYPE,
    )
    ds_gl = np.asarray(
        0.5 * lengths_si[:, None] * gl_weights_arr[None, :],
        dtype=_state.CEXT_FLOAT_DTYPE,
    )

    diffusivity_si = float(diffusivity * _state.CM2_TO_M2)
    lambda_inlet = float(
        _lambda_if_from_civ(float(inlet_concentration), diffusivity_si, vmax, km)
    )
    window = max(float(_state.CEXT_WINDOW_FACTOR), 0.0)
    cell_size = float(_state.CEXT_GRID_CELL_FACTOR) * window * lambda_inlet
    reach_si = np.asarray(
        0.5 * lengths_si + radii_si + window * lambda_inlet,
        dtype=_state.CEXT_FLOAT_DTYPE,
    )

    gpu_direct_cell_origin = np.zeros((3,), dtype=np.int32)
    gpu_direct_cell_dims = np.zeros((3,), dtype=np.int32)
    gpu_direct_cell_ptr = np.zeros((1,), dtype=np.int32)
    gpu_direct_cell_seg_ids = np.zeros((0,), dtype=np.int32)
    gpu_direct_cell_flat_sorted = np.zeros((0,), dtype=np.int64)
    gpu_direct_home_cell_flat = np.zeros((midpoints_si.shape[0],), dtype=np.int64)
    gpu_direct_available = False
    gpu_direct_n_cells = 0
    if (
        build_candidate_index
        and cell_size > 0.0
        and np.isfinite(cell_size)
        and midpoints_si.shape[0] > 0
    ):
        home_idx = np.floor(np.asarray(midpoints_si, dtype=float) / cell_size).astype(
            np.int32
        )
        cell_origin = np.min(home_idx, axis=0).astype(np.int32)
        shifted = np.asarray(home_idx - cell_origin[None, :], dtype=np.int32)
        cell_dims = np.max(shifted, axis=0).astype(np.int32) + np.int32(1)
        n_cells = int(cell_dims[0]) * int(cell_dims[1]) * int(cell_dims[2])
        if n_cells > 0:
            flat_ids = shifted[:, 0].astype(np.int64) + np.int64(cell_dims[0]) * (
                shifted[:, 1].astype(np.int64)
                + np.int64(cell_dims[1]) * shifted[:, 2].astype(np.int64)
            )
            order_home = np.argsort(flat_ids, kind="stable")
            sorted_flat = np.asarray(flat_ids[order_home], dtype=np.int64)
            counts = np.bincount(sorted_flat, minlength=n_cells)
            cell_ptr = np.zeros((n_cells + 1,), dtype=np.int32)
            cell_ptr[1:] = np.cumsum(
                np.asarray(counts, dtype=np.int64), dtype=np.int64
            ).astype(np.int32)
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
    if (
        build_candidate_index
        and _state._HAVE_SCIPY_SPATIAL
        and midpoints_si.shape[0] > 0
    ):
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
            if n_cells > int(_state.CEXT_MAX_CELLS_PER_SEG):
                overflow.append(seg_idx)
                continue
            for ix in range(int(lo[0]), int(hi[0]) + 1):
                for iy in range(int(lo[1]), int(hi[1]) + 1):
                    for iz in range(int(lo[2]), int(hi[2]) + 1):
                        builders[(ix, iy, iz)].append(seg_idx)
        grid = {
            key: np.asarray(values, dtype=np.int32) for key, values in builders.items()
        }
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
        "network_topology": network_topology,
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
            "cell_ptr": np.asarray(
                context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)),
                dtype=np.int32,
            ),
            "cell_seg_ids": np.asarray(
                context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)),
                dtype=np.int32,
            ),
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
            "cell_ptr": np.asarray(
                context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)),
                dtype=np.int32,
            ),
            "cell_seg_ids": np.asarray(
                context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)),
                dtype=np.int32,
            ),
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
            "cell_ptr": np.asarray(
                context.get("gpu_direct_cell_ptr", np.zeros((1,), dtype=np.int32)),
                dtype=np.int32,
            ),
            "cell_seg_ids": np.asarray(
                context.get("gpu_direct_cell_seg_ids", np.zeros((0,), dtype=np.int32)),
                dtype=np.int32,
            ),
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
    home_flat = np.asarray(
        context.get("gpu_direct_home_cell_flat", np.zeros((nseg,), dtype=np.int64)),
        dtype=np.int64,
    )
    active_flat = np.asarray(home_flat[source_ids], dtype=np.int64)
    order = np.argsort(active_flat, kind="stable")
    cell_seg_ids = np.asarray(source_ids[order], dtype=np.int32)
    active_flat = np.asarray(active_flat[order], dtype=np.int64)
    counts = (
        np.bincount(active_flat, minlength=n_cells)
        if active_flat.size
        else np.zeros((n_cells,), dtype=np.int64)
    )
    cell_ptr = np.zeros((n_cells + 1,), dtype=np.int32)
    if counts.size:
        cell_ptr[1:] = np.cumsum(
            np.asarray(counts, dtype=np.int64), dtype=np.int64
        ).astype(np.int32)
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
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            perf_counter() - t0,
            False,
        )
    reach_si = np.asarray(context["reach_si"], dtype=float)
    max_reach = float(context.get("max_reach_si", 0.0))
    midpoints_si = np.asarray(context["midpoints_si"], dtype=float)
    exclude_idx = np.asarray(context["exclude_idx"], dtype=np.int32)
    exclude_count = np.asarray(context["exclude_count"], dtype=np.uint8)
    reach_scale = max(float(_state.CEXT_APPROX_WINDOW_SCALE), 0.0)
    target_ids = np.asarray(target_seg_ids, dtype=np.int32)
    row_ptr = [0]
    col_values: list[int] = []
    max_slots_i = max(int(max_slots), 1)
    too_many = False
    if target_ids.size == 0:
        return (
            np.asarray(row_ptr, dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            perf_counter() - t0,
            False,
        )

    query_radii = np.asarray(
        reach_scale * reach_si[target_ids] + reach_scale * max_reach, dtype=float
    )
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
            delta = np.asarray(
                midpoints_si[neighbors] - target_mid[None, :], dtype=float
            )
            dist2 = np.einsum("ij,ij->i", delta, delta)
            pair_limit = np.square(target_reach + reach_scale * reach_si[neighbors])
            neighbors = neighbors[dist2 <= pair_limit]
        ordered = _maybe_trim_candidate_array(
            neighbors,
            target_mid=np.asarray(midpoints_si[target_i], dtype=float),
            midpoints_si=midpoints_si,
        )
        next_size = row_ptr[-1] + int(ordered.size)
        if next_size > max_slots_i and target_ids.size > 1:
            too_many = True
            break
        if ordered.size:
            col_values.extend(int(val) for val in ordered)
        row_ptr.append(next_size)

    if too_many:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            perf_counter() - t0,
            True,
        )
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
    reach_scale = max(float(_state.CEXT_APPROX_WINDOW_SCALE), 0.0)
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
                            if float(np.dot(dx, dx)) > float(
                                (target_reach + source_reach) ** 2
                            ):
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

        ordered = _maybe_trim_candidate_list(
            sorted(candidates), target_mid=target_mid, midpoints_si=midpoints_si
        )
        next_size = row_ptr[-1] + len(ordered)
        if next_size > max_slots_i and target_seg_ids.size > 1:
            too_many = True
            break
        col_values.extend(ordered)
        row_ptr.append(next_size)

    if too_many:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            perf_counter() - t0,
            True,
        )
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
    target_ranges = max(1, min(int(_state.CEXT_PRECOMPUTE_WORKERS), nseg))
    range_span = max(chunk_targets, int(math.ceil(float(nseg) / float(target_ranges))))
    ranges = [
        (start, min(start + range_span, nseg)) for start in range(0, nseg, range_span)
    ]
    workers = min(int(_state.CEXT_PRECOMPUTE_WORKERS), len(ranges))
    if workers <= 1 or len(ranges) <= 1:
        _, batches, query_time = _build_cext_candidate_batches_for_range(
            context,
            0,
            nseg,
            max_slots=_state.CEXT_STREAMING_TARGET_CANDIDATE_SLOTS,
            query_workers=max(int(_state.CEXT_PRECOMPUTE_WORKERS), 1),
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
                max_slots=_state.CEXT_STREAMING_TARGET_CANDIDATE_SLOTS,
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
            max_slots=_state.CEXT_STREAMING_TARGET_CANDIDATE_SLOTS,
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


__all__ = [
    "_coerce_candidate_batches_for_gpu",
    "_maybe_trim_candidate_list",
    "_maybe_trim_candidate_array",
    "_build_cext_geometry_context",
    "_build_cext_gpu_direct_source_plan",
    "_build_cext_gpu_direct_source_plan_from_ids",
    "_build_cext_gpu_direct_target_plan",
    "_build_cext_include_lists_kdtree",
    "_build_cext_include_lists",
    "_build_cext_candidate_batches_for_range",
    "_build_cext_candidate_batches",
    "_ensure_cext_candidate_batches",
    "_split_cext_candidate_batch",
]
