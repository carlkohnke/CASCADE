"""Green's Function Method tissue oxygen solvers.

CPU, streaming, and GPU paths share the same vessel source formulation and
return concentrations only for tissue points inside the interaction window.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
import traceback

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.concentration.properties import get_concentration_inlet
from cascade.concentration.external_field.state import (
    _prepare_cext_source_state_for_tissue,
    validate_cext_tissue_flux_consistency,
)
from cascade.concentration.vessel.greens import (
    _cext_tissue_kernel_numba,
    _k_ratio,
    _tissue_dense_kernel_numba,
    _tissue_kernel_numba,
)
from cascade.diagnostics.runtime import _fmt_seconds, _resolve_tissue_accel_mode
from cascade.concentration.quadrature import _get_gl_nodes_weights

from .cache import _compute_tissue_samples_greens_streaming
from .geometry import (
    _build_tissue_geometry_context,
    _diagnostic_stats,
    _ensure_tissue_context_kdtree,
    _prepare_tissue_geometry,
    _process_tissue_chunk,
    _streaming_tissue_chunk_size,
)
from .gpu import (
    _compute_tissue_samples_greens_from_cext_state_gpu,
    _compute_tissue_samples_greens_gpu,
)
from .streaming import _compute_cext_streaming_tissue_chunk, _process_cext_tissue_chunk


def profile(func):
    """No-op hook retained for compatibility with line-profiler instrumentation."""
    return func


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
    # Mutable runtime state is centralized in configuration.solver_state.
    _state._LAST_TISSUE_TIMINGS = {}
    if points.size == 0 or starts.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=float)

    t_total = perf_counter()
    t0 = perf_counter()
    validation = validate_cext_tissue_flux_consistency(cext_state)
    cext_state = _prepare_cext_source_state_for_tissue(cext_state, _state.GL_ORDER)
    gl_points_si = np.asarray(cext_state["gl_points_si"], dtype=np.float32)
    lambda_iv_gl = np.asarray(cext_state["lambda_iv_gl"], dtype=np.float32)
    q_weighted_gl = np.asarray(cext_state["q_weighted_gl"], dtype=np.float32)
    mono2_weight_gl = np.asarray(
        cext_state.get("mono2_weight_gl", np.zeros_like(q_weighted_gl)),
        dtype=np.float32,
    )
    dipole2_weight_gl = np.asarray(
        cext_state.get("dipole2_weight_gl", np.zeros_like(q_weighted_gl)),
        dtype=np.float32,
    )
    segment_vectors_state = np.asarray(
        cext_state.get(
            "segment_vectors", np.zeros((gl_points_si.shape[0], 3), dtype=np.float32)
        ),
        dtype=np.float32,
    )
    seg_cap_gl = np.asarray(cext_state["seg_cap_gl"], dtype=np.float32)
    diffusivity_si = float(cext_state["diffusivity_si"])
    window_factor = float(cext_state.get("window_factor", _state.CEXT_WINDOW_FACTOR))
    max_nearby = min(_state.NEAREST_TISSUE_VESSELS, len(starts))
    result = np.zeros(points.shape[0], dtype=float)

    if _resolve_tissue_accel_mode() == "gpu" and _state._cp is not None:
        keep_mask, gpu_result = _compute_tissue_samples_greens_from_cext_state_gpu(
            points,
            starts,
            ends,
            radii,
            cext_state,
            tissue_cache=tissue_cache,
        )
        if _state.SOLVER_TIMING_DETAILS:
            print(
                "    Cext tissue diagnostics: "
                f"{_diagnostic_stats('c_iv_gl', np.asarray(cext_state['c_iv_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('c_ext_gl', np.asarray(cext_state['c_ext_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('c_iv_minus_c_ext', np.asarray(cext_state['c_iv_gl'], dtype=np.float32) - np.asarray(cext_state['c_ext_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('q_line_gl', np.asarray(cext_state['q_line_gl'], dtype=np.float32))} "
                f"{_diagnostic_stats('tissue', gpu_result)} "
                f"flux_check_ok={validation['ok']} flux_rel_l2={float(validation['rel_l2']):.3e}"
            )
        _state._LAST_TISSUE_TIMINGS.update(
            {
                "cext_flux_check_rel_l2": float(validation["rel_l2"]),
                "cext_flux_check_max_abs": float(validation["max_abs"]),
            }
        )
        return keep_mask, gpu_result

    if tissue_cache is not None and tissue_cache.get("dense_fused_cpu"):
        # The generic CPU tissue path can consume a geometry-only fused cache,
        # whereas the Cext source solver needs nearest-source arrays.  Upgrade
        # the shared dictionary in place so subsequent interactive/sweep cases
        # reuse the complete CPU cache instead of rebuilding it every solve.
        prepared = _prepare_tissue_geometry(
            points,
            starts,
            ends,
            radii,
            max_nearby=max_nearby,
        )
        tissue_cache.clear()
        tissue_cache.update(prepared)

    use_streaming = (
        tissue_cache is not None
        and (tissue_cache.get("streaming") or tissue_cache.get("gpu"))
    ) or (
        tissue_cache is None
        and int(points.shape[0]) >= int(_state.TISSUE_STREAMING_MIN_POINTS)
    )
    if use_streaming:
        if tissue_cache is not None and isinstance(tissue_cache.get("context"), dict):
            context = tissue_cache["context"]
            max_nearby = int(tissue_cache.get("max_nearby", max_nearby))
        else:
            context = _build_tissue_geometry_context(starts, ends, radii)
        _ensure_tissue_context_kdtree(context)
        valid_source_ids = np.flatnonzero(
            np.asarray(context["valid_mask"], dtype=bool)
        ).astype(np.int64, copy=False)
        lambda_valid = (
            np.asarray(lambda_iv_gl[valid_source_ids], dtype=np.float32)
            if valid_source_ids.size
            else np.empty((0, 0), dtype=np.float32)
        )
        influence_radius = (
            np.asarray(
                window_factor * np.max(lambda_valid, axis=1),
                dtype=context.get("cache_float", np.float32),
            )
            if lambda_valid.size
            else np.empty((0,), dtype=context.get("cache_float", np.float32))
        )
        chunk_size = _streaming_tissue_chunk_size(max_nearby)
        tasks = [
            (chunk_i, start_idx, min(start_idx + chunk_size, len(points)), None)
            for chunk_i, start_idx in enumerate(
                range(0, len(points), chunk_size), start=1
            )
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
            "prune_by_window": bool(_state.TISSUE_STREAMING_PRUNE_BY_WINDOW),
        }
        tasks = [
            (chunk_i, start_idx, end_idx, state)
            for chunk_i, start_idx, end_idx, _ in tasks
        ]
        keep_mask = np.zeros((points.shape[0],), dtype=bool)
        total_before_candidates = 0
        total_after_candidates = 0
        t_setup = perf_counter() - t0
        t0 = perf_counter()
        workers = max(int(_state.TISSUE_STREAMING_CHUNK_WORKERS), 1)
        if workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_compute_cext_streaming_tissue_chunk, task): task[0]
                    for task in tasks
                }
                for fut in as_completed(future_map):
                    _, start_idx, end_idx, chunk_keep, out, before_c, after_c = (
                        fut.result()
                    )
                    keep_mask[start_idx:end_idx] = chunk_keep
                    result[start_idx:end_idx] = out
                    total_before_candidates += int(before_c)
                    total_after_candidates += int(after_c)
        else:
            for task in tasks:
                _, start_idx, end_idx, chunk_keep, out, before_c, after_c = (
                    _compute_cext_streaming_tissue_chunk(task)
                )
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
        valid_source_ids = np.flatnonzero(
            np.asarray(tissue_cache["valid_mask"], dtype=bool)
        ).astype(np.int64, copy=False)
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
        ranges = [
            (i, min(i + _state.DISTANCE_CHUNK_SIZE, len(points)))
            for i in range(0, len(points), _state.DISTANCE_CHUNK_SIZE)
        ]
        t0 = perf_counter()
        if _state._HAVE_NUMBA and _state.TISSUE_USE_NUMBA:
            result = _cext_tissue_kernel_numba(
                np.asarray(points_si, dtype=np.float64),
                np.asarray(nearest_idx, dtype=np.int64),
                np.asarray(keep_mask, dtype=np.bool_),
                np.asarray(valid_source_ids, dtype=np.int64),
                np.asarray(gl_points_si, dtype=np.float32),
                np.asarray(segment_vectors_state, dtype=np.float32),
                np.asarray(lambda_iv_gl, dtype=np.float32),
                np.asarray(q_weighted_gl, dtype=np.float32),
                np.asarray(mono2_weight_gl, dtype=np.float32),
                np.asarray(dipole2_weight_gl, dtype=np.float32),
                np.asarray(seg_cap_gl, dtype=np.float32),
                float(diffusivity_si),
                float(window_factor),
            )
            cache_mode = "dense-numba"
        elif _state.TISSUE_PARALLEL_WORKERS > 1 and len(ranges) > 1:
            with ThreadPoolExecutor(
                max_workers=_state.TISSUE_PARALLEL_WORKERS
            ) as executor:
                for start_idx, out in executor.map(
                    lambda r: _process_cext_tissue_chunk(*r, worker_data),
                    ranges,
                ):
                    result[start_idx : start_idx + len(out)] = out
        else:
            for r in ranges:
                start_idx, out = _process_cext_tissue_chunk(*r, worker_data)
                result[start_idx : start_idx + len(out)] = out
        t_kernel = perf_counter() - t0
        if not (_state._HAVE_NUMBA and _state.TISSUE_USE_NUMBA):
            cache_mode = "dense"
        keep_k = int(nearest_idx.shape[1]) if nearest_idx.ndim == 2 else 0
        candidate_slots = (
            int(nearest_idx.shape[0] * nearest_idx.shape[1])
            if nearest_idx.ndim == 2
            else 0
        )

    if _state.SOLVER_TIMING_DETAILS:
        print(
            "  Tissue Greens solve: source_mode=cext_converged_q_flux "
            f"solver={cext_state.get('solver', 'unknown')} points={points.shape[0]} "
            f"cache={cache_mode} active_points={int(np.count_nonzero(keep_mask))} keep_k={keep_k} "
            f"candidate_slots={candidate_slots} "
            f"gl_order_tissue={gl_points_si.shape[1] if gl_points_si.ndim >= 3 else 0} "
            f"gl_order_cext={int(cext_state.get('cext_gl_order', gl_points_si.shape[1] if gl_points_si.ndim >= 3 else 0))} "
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

    _state._LAST_TISSUE_TIMINGS = {
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
    diffusivity: float = _state.SOLUTE_DIFFUSIVITY,
    vmax: float = _state.VMAX_MM,
    km: float = _state.K_M_MM,
    window_factor: float = _state.WINDOW_FACTOR,
    inlet_concentration: float | None = None,
    tissue_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate tissue oxygen from vessel line sources at requested points.

    Backend policy selects GPU, cached CPU, or streaming CPU evaluation. Every
    path applies the same physical window, consumption law, and inlet reference
    before returning the retained-point mask and concentrations.
    """

    # Mutable runtime state is centralized in configuration.solver_state.
    _state._LAST_TISSUE_TIMINGS = {}
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

    max_nearby = min(_state.NEAREST_TISSUE_VESSELS, len(starts))
    use_dense_fused_cpu = (
        _state._HAVE_NUMBA
        and _state.TISSUE_USE_NUMBA
        and max_nearby == len(starts)
        and (tissue_cache is None or tissue_cache.get("dense_fused_cpu"))
    )
    if use_dense_fused_cpu:
        try:
            return _compute_tissue_samples_greens_dense_cpu(
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
        except Exception:
            print(
                "WARNING: fused dense CPU tissue kernel failed; falling back to "
                "the candidate-cache path."
            )
            traceback.print_exc()
            tissue_cache = None

    if _state.TISSUE_STREAMING_ENABLED and (
        (tissue_cache is not None and tissue_cache.get("streaming"))
        or (
            tissue_cache is None
            and int(points.shape[0]) >= int(_state.TISSUE_STREAMING_MIN_POINTS)
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
    flows_si = flows[valid] * _state.CM3_TO_M3
    diffusivity_si = diffusivity * _state.CM2_TO_M2
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
    alpha_edge = (
        (2.0 * np.pi * radii_si / flow_mag)
        * (diffusivity_si / np.maximum(lam_edge, 1e-30))
        * ratio_edge
    )
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet()
    lam_ref = float(
        np.sqrt(
            diffusivity_si
            / max(vmax / max(km + float(inlet_concentration), 1e-30), 1e-30)
        )
    )

    flow_sign = np.sign(flows_si)
    result = np.zeros(points.shape[0], dtype=float)
    gl_nodes, gl_weights = _get_gl_nodes_weights(_state.GL_ORDER)
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

    ranges = [
        (i, min(i + _state.DISTANCE_CHUNK_SIZE, len(points)))
        for i in range(0, len(points), _state.DISTANCE_CHUNK_SIZE)
    ]
    used_numba = False
    t_kernel = 0.0
    if _state._HAVE_NUMBA and _state.TISSUE_USE_NUMBA and _state._K0_LUT.size:
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
                _state._KRATIO_XS,
                _state._K0_LUT,
            )
            t_kernel = perf_counter() - t0
            used_numba = True
        except Exception:
            print(
                "WARNING: Numba tissue kernel failed; falling back to non-numba path."
            )
            traceback.print_exc()
            used_numba = False
    if not used_numba and _state.TISSUE_PARALLEL_WORKERS > 1 and len(ranges) > 1:
        t0 = perf_counter()
        with ThreadPoolExecutor(max_workers=_state.TISSUE_PARALLEL_WORKERS) as executor:
            for start_idx, out in executor.map(
                lambda r: _process_tissue_chunk(*r, worker_data),
                ranges,
            ):
                result[start_idx : start_idx + len(out)] = out
        t_kernel = perf_counter() - t0
    elif not used_numba:
        t0 = perf_counter()
        for r in ranges:
            start_idx, out = _process_tissue_chunk(*r, worker_data)
            result[start_idx : start_idx + len(out)] = out
        t_kernel = perf_counter() - t0

    if _state.SOLVER_TIMING_DETAILS:
        active_points = int(np.count_nonzero(keep_mask))
        candidate_slots = (
            int(nearest_idx.shape[0] * nearest_idx.shape[1])
            if nearest_idx.ndim == 2
            else 0
        )
        print(
            "  Tissue Greens solve: "
            f"points={points.shape[0]} active_points={active_points} keep_k={nearest_idx.shape[1] if nearest_idx.ndim == 2 else 0} "
            f"candidate_slots={candidate_slots} gl_order={_state.GL_ORDER} numba={used_numba} "
            f"setup={_fmt_seconds(t_setup)} kernel={_fmt_seconds(t_kernel)} "
            f"total={_fmt_seconds(perf_counter() - t_total)}"
        )

    _state._LAST_TISSUE_TIMINGS = {
        "backend": "cpu",
        "t_tissue_geometry_s": 0.0,
        "t_tissue_oxygen_s": float(t_kernel),
        "t_tissue_total_s": float(perf_counter() - t_total),
    }

    return keep_mask, result


def _compute_tissue_samples_greens_dense_cpu(
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
    """Run the exact all-segment CPU path without candidate matrices."""
    t_total = perf_counter()
    t0 = perf_counter()
    context = (
        tissue_cache["context"]
        if tissue_cache is not None and tissue_cache.get("dense_fused_cpu")
        else _build_tissue_geometry_context(starts, ends, radii)
    )
    valid = np.asarray(context["valid_mask"], dtype=bool)
    points_si = np.asarray(points * _state.CM_TO_M, dtype=context["cache_float"])
    starts_si = context["starts_si"]
    segment_vectors = context["segment_vectors"]
    seg_len_sq = context["seg_len_sq"]
    seg_len = context["seg_len"]
    radii_si = np.maximum(np.nan_to_num(context["radii_si"], nan=0.0), 0.0)
    cin_pos = np.maximum(np.nan_to_num(np.asarray(cin)[valid], nan=0.0), 0.0)
    flows_si = np.nan_to_num(np.asarray(flows)[valid] * _state.CM3_TO_M3, nan=0.0)
    diffusivity_si = float(diffusivity * _state.CM2_TO_M2)
    denom = np.maximum(float(km) + cin_pos, 1.0e-30)
    lam_edge = np.sqrt(diffusivity_si / np.maximum(float(vmax) / denom, 1.0e-30))
    ratio_edge = _k_ratio(radii_si / np.maximum(lam_edge, 1.0e-30))
    alpha_edge = (
        (2.0 * np.pi * radii_si / np.maximum(np.abs(flows_si), 1.0e-30))
        * (diffusivity_si / np.maximum(lam_edge, 1.0e-30))
        * ratio_edge
    )
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet()
    lam_ref = float(
        np.sqrt(
            diffusivity_si
            / max(
                float(vmax) / max(float(km) + float(inlet_concentration), 1.0e-30),
                1.0e-30,
            )
        )
    )
    gl_nodes, gl_weights = _get_gl_nodes_weights(_state.GL_ORDER)
    t_setup = perf_counter() - t0
    t0 = perf_counter()
    keep_mask, result = _tissue_dense_kernel_numba(
        points_si,
        starts_si,
        segment_vectors,
        seg_len_sq,
        seg_len,
        radii_si,
        cin_pos,
        alpha_edge,
        np.sign(flows_si),
        float(diffusivity_si),
        float(km),
        float(vmax),
        float(window_factor),
        float(lam_ref),
        gl_nodes,
        gl_weights,
        _state._KRATIO_XS,
        _state._K0_LUT,
    )
    t_kernel = perf_counter() - t0
    total = perf_counter() - t_total
    if _state.SOLVER_TIMING_DETAILS:
        print(
            "  Tissue Greens solve: "
            f"points={points.shape[0]} active_points={int(np.count_nonzero(keep_mask))} "
            f"candidate_mode=dense_fused_cpu nseg={starts_si.shape[0]} "
            f"gl_order={_state.GL_ORDER} setup={_fmt_seconds(t_setup)} "
            f"kernel={_fmt_seconds(t_kernel)} total={_fmt_seconds(total)}"
        )
    _state._LAST_TISSUE_TIMINGS = {
        "backend": "cpu_dense_fused",
        "t_tissue_geometry_s": float(t_setup),
        "t_tissue_oxygen_s": float(t_kernel),
        "t_tissue_total_s": float(total),
        "t_tissue_kdtree_query_s": 0.0,
        "t_tissue_gpu_refine_s": 0.0,
        "t_tissue_gpu_transfer_s": 0.0,
    }
    return np.asarray(keep_mask, dtype=bool), np.asarray(result, dtype=float)


def estimate_bulk_tissue_concentration(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    vessel_sources: np.ndarray,
    *,
    extravascular_concentration: float = _state.EXTRAVASCULAR_CONCENTRATION,
    decay_length: float = _state.TISSUE_DECAY_LENGTH,
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
    for idx in range(0, len(points), _state.DISTANCE_CHUNK_SIZE):
        chunk = points[idx : idx + _state.DISTANCE_CHUNK_SIZE]
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
        concentration = (
            extravascular_concentration
            + (supplied - extravascular_concentration) * weight
        )
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
    extravascular_concentration: float = _state.EXTRAVASCULAR_CONCENTRATION,
    decay_length: float = _state.TISSUE_DECAY_LENGTH,
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
    max_nearby = min(_state.NEAREST_TISSUE_VESSELS, len(starts))
    for idx in range(0, len(points), _state.DISTANCE_CHUNK_SIZE):
        chunk = points[idx : idx + _state.DISTANCE_CHUNK_SIZE]
        if chunk.size == 0:
            continue
        diff = chunk[:, None, :] - starts[None, :, :]
        proj_raw = (
            np.sum(diff * segment_vectors[None, :, :], axis=2) / seg_len_sq[None, :]
        )
        proj = np.clip(proj_raw, 0.0, 1.0)
        closest = starts[None, :, :] + proj[:, :, None] * segment_vectors[None, :, :]
        distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
        nearest_idx = np.argpartition(distances, kth=max_nearby - 1, axis=1)[
            :, :max_nearby
        ]
        d_center = np.take_along_axis(distances, nearest_idx, axis=1)
        proj_sel = np.take_along_axis(proj_raw, nearest_idx, axis=1)
        radius_local = np.minimum(
            np.take_along_axis(radii[None, :], nearest_idx, axis=1),
            np.sqrt(np.take_along_axis(seg_len_sq[None, :], nearest_idx, axis=1)),
        )
        inside_any = np.any(
            (proj_sel >= 0.0) & (proj_sel <= 1.0) & (d_center <= radius_local), axis=1
        )
        keep_mask[idx : idx + len(chunk)] = ~inside_any
        d_wall = np.maximum(d_center - radius_local, 0.0)
        supplied = np.take_along_axis(vessel_sources[None, :], nearest_idx, axis=1)
        weight = np.clip(1.0 - d_wall / decay_length, 0.0, 1.0)
        conc_candidates = (
            extravascular_concentration
            + (supplied - extravascular_concentration) * weight
        )
        result[idx : idx + len(chunk)] = np.max(conc_candidates, axis=1)

    return keep_mask, result


__all__ = [
    "compute_tissue_samples_greens_from_cext_state",
    "compute_tissue_samples_greens",
    "estimate_bulk_tissue_concentration",
    "_compute_tissue_samples_linear",
]
