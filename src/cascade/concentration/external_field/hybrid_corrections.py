"""Finite-radius and self-interaction FFT corrections.

These corrections remove grid/self artifacts so the hybrid FFT field agrees
with finite-radius vessel Green's-function physics.
"""

from __future__ import annotations

import os
from time import perf_counter

import numpy as np

from cascade.accelerators.cuda import load_cuda_source
from cascade.configuration import solver_state as _state

from .direct import _ensure_cext_gpu_geometry_static
from .hybrid_deposit import (
    _cext_fft_deposit_computed_moment_grid_gpu,
    _cext_fft_deposit_moment_batch_gpu,
    _cext_fft_deposit_moment_grid_gpu,
)
from .hybrid_geometry import (
    _cext_hybrid_init_lambda_bins,
    _ensure_cext_hybrid_bg_gpu_static,
    _get_cext_fft_discrete_self_runtime_moment_kernel,
    _get_cext_fft_discrete_self_runtime_stencil_kernel,
)


def _sample_cext_hybrid_bg_gpu(*args, **kwargs):
    from .hybrid_solver import _sample_cext_hybrid_bg_gpu as implementation

    return implementation(*args, **kwargs)


def _cext_fft_o2_term_correction_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    state: dict,
    *,
    target_plan: dict | None = None,
) -> tuple[object | None, float]:
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
    if (
        bool(_state.CEXT_HYBRID_GPU_RUNTIME_WEIGHTS)
        and "mono2_weight_gl" in state
        and "dipole2_weight_gl" in state
    ):
        o2_node = (
            state["mono2_weight_gl"].ravel() + state["dipole2_weight_gl"].ravel()
        ).astype(_state._cp.float32, copy=False)
        if not bool(
            _state._cp.any(
                _state._cp.isfinite(o2_node) & (o2_node != _state._cp.float32(0.0))
            ).item()
        ):
            return None, 0.0
    else:
        mono = np.asarray(
            ext_state.get("mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])),
            dtype=np.float32,
        )
        dipole = np.asarray(
            ext_state.get(
                "dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])
            ),
            dtype=np.float32,
        )
        o2_weight = mono + dipole
        if not np.any(np.isfinite(o2_weight) & (o2_weight != 0.0)):
            return None, 0.0
        o2_node = _state._cp.asarray(o2_weight, dtype=_state._cp.float32).ravel()

    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gpu_static = _ensure_cext_gpu_geometry_static(context)
    grid_n = int(hybrid["grid_n"])
    spacing = float(hybrid["spacing"])
    diffusivity_si = float(context["diffusivity_si"])
    rhs_scale = np.float32(max((spacing**3) * diffusivity_si, 1.0e-30))
    edges, lambda_centers = _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    lambda_centers = np.asarray(lambda_centers, dtype=np.float32)
    static["lambda_bin_centers"] = _state._cp.asarray(
        lambda_centers, dtype=_state._cp.float32
    )

    kfreq = _state._cp.asarray(
        2.0 * np.pi * np.fft.fftfreq(grid_n, d=spacing), dtype=_state._cp.float32
    )
    kx = kfreq[:, None, None]
    ky = kfreq[None, :, None]
    kz = kfreq[None, None, :]
    symbols = (
        -static["k2"],
        kx * kx,
        ky * ky,
        kz * kz,
        _state._cp.float32(2.0) * kx * ky,
        _state._cp.float32(2.0) * kx * kz,
        _state._cp.float32(2.0) * ky * kz,
    )

    use_runtime_moments = bool(_state.CEXT_HYBRID_GPU_RUNTIME_MOMENTS) and bool(
        static.get("runtime_stencil", False)
    )
    tx = ty = tz = None
    moment_weights = ()
    if not use_runtime_moments:
        seg_vec = _state._cp.asarray(
            gpu_static["segment_vectors"], dtype=_state._cp.float32
        )
        seg_len = _state._cp.sqrt(_state._cp.sum(seg_vec * seg_vec, axis=1))
        seg_len = _state._cp.maximum(seg_len, _state._cp.float32(1.0e-30))
        t_hat = seg_vec / seg_len[:, None]
        node_seg_ids = static["node_seg_ids"]
        tx = t_hat[node_seg_ids, 0]
        ty = t_hat[node_seg_ids, 1]
        tz = t_hat[node_seg_ids, 2]
        moment_weights = (
            o2_node,
            o2_node * tx * tx,
            o2_node * ty * ty,
            o2_node * tz * tz,
            o2_node * tx * ty,
            o2_node * tx * tz,
            o2_node * ty * tz,
        )

    phi_corr = state.get("fft_o2_phi_corr_g")
    if phi_corr is None or tuple(phi_corr.shape) != tuple(
        state["active_mass_grids_g"].shape
    ):
        phi_corr = _state._cp.zeros_like(state["active_mass_grids_g"])
        state["fft_o2_phi_corr_g"] = phi_corr
    else:
        phi_corr.fill(_state._cp.float32(0.0))

    lambda_centers_g = static["lambda_bin_centers"]
    n_bins = int(phi_corr.shape[0])
    t0 = perf_counter()
    try:
        batched_limit = int(
            os.environ.get("SVV_FFT_O2_BATCHED_MAX_BYTES", str(1024**3))
        )
    except ValueError:
        batched_limit = 3 * 1024**3
    use_batched = int(phi_corr.size) * 8 <= max(batched_limit, 0)
    use_fused_ifft = bool(_state.CEXT_HYBRID_FFT_O2_FUSED_IFFT) and use_batched
    if use_fused_ifft:
        try:
            moment_batch = max(int(_state.CEXT_HYBRID_FFT_O2_MOMENT_BATCH), 1)
            lam = _state._cp.maximum(
                lambda_centers_g, _state._cp.float32(1.0e-8)
            ).reshape((n_bins, 1, 1, 1))
            denom_inv = _state._cp.reciprocal(
                static["k2"][None, :, :, :] + _state._cp.reciprocal(lam * lam)
            )
            corr_hat_sum = _state._cp.zeros(phi_corr.shape, dtype=_state._cp.complex64)
            if use_runtime_moments:
                for moment_idx, symbol in enumerate(symbols):
                    moment_grid = _cext_fft_deposit_computed_moment_grid_gpu(
                        context,
                        hybrid,
                        state,
                        o2_node,
                        np.asarray(edges, dtype=np.float32),
                        moment_idx=moment_idx,
                        synchronize=False,
                    )
                    rhs_hat = _state._cp.fft.fftn(moment_grid, axes=(1, 2, 3))
                    rhs_hat *= _state._cp.float32(1.0 / float(rhs_scale))
                    corr_hat_sum += (
                        rhs_hat
                        * _state._cp.asarray(symbol, dtype=_state._cp.float32)[
                            None, :, :, :
                        ]
                        * denom_inv
                    )
                    del rhs_hat
            elif moment_batch > 1:
                # Batch adjacent O(a^2) moment deposits so the stencil walk and
                # lambda-bin lookup are shared across multiple moment fields.
                for moment_start in range(0, len(symbols), moment_batch):
                    local_count = min(moment_batch, len(symbols) - moment_start)
                    moment_grid = _cext_fft_deposit_moment_batch_gpu(
                        context,
                        hybrid,
                        state,
                        o2_node,
                        tx,
                        ty,
                        tz,
                        np.asarray(edges, dtype=np.float32),
                        moment_start=moment_start,
                        moment_count=local_count,
                        synchronize=False,
                    )
                    rhs_hat = _state._cp.fft.fftn(moment_grid, axes=(2, 3, 4))
                    rhs_hat *= _state._cp.float32(1.0 / float(rhs_scale))
                    for local_idx in range(local_count):
                        symbol = symbols[moment_start + local_idx]
                        corr_hat_sum += (
                            rhs_hat[local_idx]
                            * _state._cp.asarray(symbol, dtype=_state._cp.float32)[
                                None, :, :, :
                            ]
                            * denom_inv
                        )
                    del rhs_hat
            else:
                for moment_weight, symbol in zip(moment_weights, symbols):
                    moment_grid = _cext_fft_deposit_moment_grid_gpu(
                        context,
                        hybrid,
                        state,
                        moment_weight,
                        np.asarray(edges, dtype=np.float32),
                        synchronize=False,
                    )
                    rhs_hat = _state._cp.fft.fftn(moment_grid, axes=(1, 2, 3))
                    rhs_hat *= _state._cp.float32(1.0 / float(rhs_scale))
                    corr_hat_sum += (
                        rhs_hat
                        * _state._cp.asarray(symbol, dtype=_state._cp.float32)[
                            None, :, :, :
                        ]
                        * denom_inv
                    )
                    del rhs_hat
            # The O(a^2) correction is linear in the seven second-order moment
            # fields, so summing their Fourier coefficients before the inverse
            # transform is algebraically equivalent to seven separate IFFTs.
            phi_corr += _state._cp.real(
                _state._cp.fft.ifftn(corr_hat_sum, axes=(1, 2, 3))
            ).astype(_state._cp.float32)
            del lam, denom_inv, corr_hat_sum
        except Exception:
            if str(
                os.environ.get("SVV_CEXT_HYBRID_FFT_O2_FUSED_STRICT", "false")
            ).strip().lower() in ("1", "true", "yes", "on"):
                raise
            use_fused_ifft = False
    if not use_fused_ifft:
        moment_iter = (
            enumerate(symbols) if use_runtime_moments else zip(moment_weights, symbols)
        )
        for moment_item, symbol in moment_iter:
            if use_runtime_moments:
                moment_grid = _cext_fft_deposit_computed_moment_grid_gpu(
                    context,
                    hybrid,
                    state,
                    o2_node,
                    np.asarray(edges, dtype=np.float32),
                    moment_idx=int(moment_item),
                )
            else:
                moment_grid = _cext_fft_deposit_moment_grid_gpu(
                    context,
                    hybrid,
                    state,
                    moment_item,
                    np.asarray(edges, dtype=np.float32),
                )
            if use_batched:
                rhs_hat = _state._cp.fft.fftn(moment_grid, axes=(1, 2, 3))
                rhs_hat *= _state._cp.float32(1.0 / float(rhs_scale))
                for bin_idx in range(n_bins):
                    lam = _state._cp.maximum(
                        lambda_centers_g[bin_idx], _state._cp.float32(1.0e-8)
                    )
                    denom = static["k2"] + _state._cp.reciprocal(lam * lam)
                    rhs_hat[bin_idx] *= symbol
                    rhs_hat[bin_idx] /= denom
                    del denom
                phi_corr += _state._cp.real(
                    _state._cp.fft.ifftn(rhs_hat, axes=(1, 2, 3))
                ).astype(_state._cp.float32)
                del rhs_hat
            else:
                for bin_idx in range(n_bins):
                    rhs = moment_grid[bin_idx] / rhs_scale
                    rhs_hat = _state._cp.fft.fftn(rhs, axes=(0, 1, 2))
                    lam = _state._cp.maximum(
                        lambda_centers_g[bin_idx], _state._cp.float32(1.0e-8)
                    )
                    denom = static["k2"] + _state._cp.reciprocal(lam * lam)
                    corr_hat = symbol * rhs_hat / denom
                    phi_corr[bin_idx] += _state._cp.real(
                        _state._cp.fft.ifftn(corr_hat, axes=(0, 1, 2))
                    ).astype(_state._cp.float32)
                    del rhs, rhs_hat, denom, corr_hat

    zero_mass = state.get("fft_o2_zero_mass_g")
    if zero_mass is None or tuple(zero_mass.shape) != tuple(phi_corr.shape):
        zero_mass = _state._cp.zeros_like(phi_corr)
        state["fft_o2_zero_mass_g"] = zero_mass
    corr_g, _ = _sample_cext_hybrid_bg_gpu(
        context,
        hybrid,
        phi_corr,
        zero_mass,
        target_plan=target_plan,
        runtime_state=state,
    )
    _state._cp.cuda.Stream.null.synchronize()
    return corr_g, perf_counter() - t0


def _get_cext_fft_discrete_self_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_FFT_DISCRETE_SELF_KERNEL is not None:
        return _state._CEXT_FFT_DISCRETE_SELF_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("cext_fft_discrete_self_kernel.cu")
    _state._CEXT_FFT_DISCRETE_SELF_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_fft_discrete_self_kernel"
    )
    return _state._CEXT_FFT_DISCRETE_SELF_KERNEL


def _get_cext_fft_discrete_self_fused_o2_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL is not None:
        return _state._CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("cext_fft_discrete_self_fused_o2_kernel.cu")
    _state._CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_fft_discrete_self_fused_o2_kernel"
    )
    return _state._CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL


def _cext_fft_response_grids_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    state: dict,
    *,
    symbol_g=None,
    cache_name: str | None = None,
):
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    _, lambda_centers = _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    lambda_centers = np.asarray(lambda_centers, dtype=np.float32)
    static["lambda_bin_centers"] = _state._cp.asarray(
        lambda_centers, dtype=_state._cp.float32
    )
    grid_shape = tuple(state["active_mass_grids_g"].shape)
    key = (
        int(hybrid["grid_n"]),
        float(hybrid["spacing"]),
        float(context["diffusivity_si"]),
        tuple(float(x) for x in lambda_centers),
        bool(symbol_g is not None),
    )
    if cache_name:
        cached = state.get(cache_name)
        if (
            cached is not None
            and tuple(cached.shape) == grid_shape
            and state.get(cache_name + "_key") == key
        ):
            return cached
        response = _state._cp.empty(grid_shape, dtype=_state._cp.float32)
        state[cache_name] = response
        state[cache_name + "_key"] = key
    else:
        response = state.get("fft_symbol_response_g")
        if response is None or tuple(response.shape) != grid_shape:
            response = _state._cp.empty(grid_shape, dtype=_state._cp.float32)
            state["fft_symbol_response_g"] = response
    rhs_scale = _state._cp.float32(
        max((float(hybrid["spacing"]) ** 3) * float(context["diffusivity_si"]), 1.0e-30)
    )
    lambda_centers_g = static["lambda_bin_centers"]
    if bool(_state.CEXT_HYBRID_FFT_RESPONSE_BATCHED):
        try:
            lam = _state._cp.maximum(
                lambda_centers_g, _state._cp.float32(1.0e-8)
            ).reshape((-1, 1, 1, 1))
            denom = static["k2"][None, :, :, :] + _state._cp.reciprocal(lam * lam)
            if symbol_g is None:
                response_hat = _state._cp.reciprocal(rhs_scale * denom)
            else:
                response_hat = _state._cp.asarray(symbol_g, dtype=_state._cp.float32)[
                    None, :, :, :
                ] / (rhs_scale * denom)
            response[...] = _state._cp.real(
                _state._cp.fft.ifftn(response_hat, axes=(1, 2, 3))
            ).astype(_state._cp.float32)
            del lam, denom, response_hat
            _state._cp.cuda.Stream.null.synchronize()
            return response
        except Exception:
            if str(
                os.environ.get("SVV_CEXT_HYBRID_FFT_RESPONSE_BATCHED_STRICT", "false")
            ).strip().lower() in ("1", "true", "yes", "on"):
                raise
    for bin_idx in range(int(response.shape[0])):
        lam = _state._cp.maximum(lambda_centers_g[bin_idx], _state._cp.float32(1.0e-8))
        denom = static["k2"] + _state._cp.reciprocal(lam * lam)
        if symbol_g is None:
            response_hat = _state._cp.reciprocal(rhs_scale * denom)
        else:
            response_hat = symbol_g / (rhs_scale * denom)
        response[bin_idx] = _state._cp.real(
            _state._cp.fft.ifftn(response_hat, axes=(0, 1, 2))
        ).astype(_state._cp.float32)
        del denom, response_hat
    _state._cp.cuda.Stream.null.synchronize()
    return response


def _cext_fft_response_cache_name(
    state: dict, name: str, grid_shape: tuple[int, ...]
) -> str | None:
    cached = state.get(name)
    if cached is not None and tuple(cached.shape) == tuple(grid_shape):
        return name
    bytes_needed = (
        int(np.prod(np.asarray(grid_shape, dtype=np.int64)))
        * np.dtype(np.float32).itemsize
    )
    try:
        env_budget = os.environ.get("SVV_FFT_RESPONSE_CACHE_MAX_BYTES")
        if env_budget is not None:
            budget = int(env_budget)
        elif _state._cp is not None:
            free_bytes, _total_bytes = _state._cp.cuda.Device().mem_info
            desired = (
                bytes_needed * 8
            )  # base response plus seven finite-radius moment responses.
            budget = int(
                min(max(512 * 1024**2, desired), max(0, int(0.5 * free_bytes)))
            )
        else:
            budget = 512 * 1024**2
    except ValueError:
        budget = 512 * 1024**2
    if budget <= 0:
        return None
    used = int(state.get("_fft_response_cache_bytes", 0))
    if used + bytes_needed > budget:
        return None
    state["_fft_response_cache_bytes"] = used + bytes_needed
    return name


def _cext_fft_discrete_same_segment_contribution_gpu(
    context: dict,
    hybrid: dict,
    ext_state: dict,
    state: dict,
    *,
    target_plan: dict | None = None,
    include_finite: bool = False,
) -> tuple[object | None, float]:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_hybrid_bg_gpu_static(context, hybrid)
    gpu_static = _ensure_cext_gpu_geometry_static(context)
    edges, _ = _cext_hybrid_init_lambda_bins(hybrid, ext_state)
    static["lambda_bin_edges"] = _state._cp.asarray(np.asarray(edges, dtype=np.float32))
    gl_order = int(np.asarray(context["gl_points_si"]).shape[1])
    nseg = int(np.asarray(context["gl_points_si"]).shape[0])
    target_cfg = target_plan if isinstance(target_plan, dict) else None
    target_count = (
        int(target_cfg.get("target_count", nseg)) if target_cfg is not None else nseg
    )
    if target_count <= 0:
        return None, 0.0
    if target_cfg is None or bool(target_cfg.get("uses_static", False)):
        target_seg_ids_g = static["all_target_seg_ids"]
    else:
        if "gpu_target_seg_ids" not in target_cfg:
            target_cfg["gpu_target_seg_ids"] = _state._cp.asarray(
                np.asarray(
                    target_cfg.get("target_seg_ids", np.zeros((0,), dtype=np.int32)),
                    dtype=np.int32,
                )
            )
        target_seg_ids_g = target_cfg["gpu_target_seg_ids"]

    out_g = state.get("fft_discrete_self_g")
    if out_g is None or tuple(out_g.shape) != (target_count, gl_order):
        out_g = _state._cp.zeros((target_count, gl_order), dtype=_state._cp.float32)
        state["fft_discrete_self_g"] = out_g
    else:
        out_g.fill(_state._cp.float32(0.0))

    target_sampling = (
        str(_state.CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING).strip().lower()
    )
    if target_sampling in ("matched", "assignment", "source"):
        assignment_mode = (
            1 if str(hybrid.get("assignment", "tsc")).strip().lower() == "tsc" else 0
        )
    elif target_sampling == "tsc":
        assignment_mode = 1
    else:
        assignment_mode = 0

    def accumulate(source_weight_g, response_g) -> None:
        total = int(target_count * gl_order)
        threads = 128
        blocks = (total + threads - 1) // threads
        if bool(static.get("runtime_stencil", False)):
            kernel = _get_cext_fft_discrete_self_runtime_stencil_kernel()
            kernel(
                (blocks,),
                (threads,),
                (
                    target_seg_ids_g,
                    gpu_static["gl_points_si"].ravel(),
                    _state._cp.asarray(
                        source_weight_g, dtype=_state._cp.float32
                    ).ravel(),
                    state["lambda_iv_gl"].ravel(),
                    state["active_source_mask_g"],
                    static["lambda_bin_edges"],
                    _state._cp.asarray(response_g, dtype=_state._cp.float32).ravel(),
                    np.int32(max(int(np.asarray(edges).size - 1), 1)),
                    np.int32(int(hybrid["grid_n"])),
                    np.int32(int(static["grid_cells"])),
                    np.int32(gl_order),
                    np.int32(target_count),
                    np.float32(float(hybrid["origin"][0])),
                    np.float32(float(hybrid["origin"][1])),
                    np.float32(float(hybrid["origin"][2])),
                    np.float32(float(hybrid["spacing"])),
                    np.int32(assignment_mode),
                    np.int32(int(static.get("assignment_mode", 1))),
                    out_g.ravel(),
                ),
            )
        else:
            kernel = _get_cext_fft_discrete_self_kernel()
            kernel(
                (blocks,),
                (threads,),
                (
                    target_seg_ids_g,
                    gpu_static["gl_points_si"].ravel(),
                    _state._cp.asarray(
                        source_weight_g, dtype=_state._cp.float32
                    ).ravel(),
                    state["lambda_iv_gl"].ravel(),
                    state["active_source_mask_g"],
                    static["stencil_flat_idx"].ravel(),
                    static["stencil_weight"].ravel(),
                    static["lambda_bin_edges"],
                    _state._cp.asarray(response_g, dtype=_state._cp.float32).ravel(),
                    np.int32(max(int(np.asarray(edges).size - 1), 1)),
                    np.int32(int(hybrid["grid_n"])),
                    np.int32(int(static["grid_cells"])),
                    np.int32(int(static["stencil_n"])),
                    np.int32(gl_order),
                    np.int32(target_count),
                    np.float32(float(hybrid["origin"][0])),
                    np.float32(float(hybrid["origin"][1])),
                    np.float32(float(hybrid["origin"][2])),
                    np.float32(float(hybrid["spacing"])),
                    np.int32(assignment_mode),
                    out_g.ravel(),
                ),
            )

    def accumulate_runtime_moment(o2_weight_g, response_g, moment_idx: int) -> None:
        if not bool(static.get("runtime_stencil", False)):
            raise RuntimeError(
                "runtime moment self-subtraction requires runtime stencil mode"
            )
        total = int(target_count * gl_order)
        threads = 128
        blocks = (total + threads - 1) // threads
        kernel = _get_cext_fft_discrete_self_runtime_moment_kernel()
        kernel(
            (blocks,),
            (threads,),
            (
                target_seg_ids_g,
                gpu_static["gl_points_si"].ravel(),
                _state._cp.asarray(o2_weight_g, dtype=_state._cp.float32).ravel(),
                gpu_static["segment_vectors"].ravel(),
                state["lambda_iv_gl"].ravel(),
                state["active_source_mask_g"],
                static["lambda_bin_edges"],
                _state._cp.asarray(response_g, dtype=_state._cp.float32).ravel(),
                np.int32(int(moment_idx)),
                np.int32(max(int(np.asarray(edges).size - 1), 1)),
                np.int32(int(hybrid["grid_n"])),
                np.int32(int(static["grid_cells"])),
                np.int32(gl_order),
                np.int32(target_count),
                np.float32(float(hybrid["origin"][0])),
                np.float32(float(hybrid["origin"][1])),
                np.float32(float(hybrid["origin"][2])),
                np.float32(float(hybrid["spacing"])),
                np.int32(assignment_mode),
                np.int32(int(static.get("assignment_mode", 1))),
                out_g.ravel(),
            ),
        )

    t0 = perf_counter()
    base_response_g = _cext_fft_response_grids_gpu(
        context,
        hybrid,
        ext_state,
        state,
        cache_name="fft_base_response_g",
    )

    has_o2 = False
    o2_node = None
    symbols = ()
    moment_weights = ()
    tx = ty = tz = None
    use_runtime_moments = bool(_state.CEXT_HYBRID_GPU_RUNTIME_MOMENTS) and bool(
        static.get("runtime_stencil", False)
    )
    if include_finite:
        if (
            bool(_state.CEXT_HYBRID_GPU_RUNTIME_WEIGHTS)
            and "mono2_weight_gl" in state
            and "dipole2_weight_gl" in state
        ):
            o2_node = (
                state["mono2_weight_gl"].ravel() + state["dipole2_weight_gl"].ravel()
            ).astype(_state._cp.float32, copy=False)
            has_o2 = bool(
                _state._cp.any(
                    _state._cp.isfinite(o2_node) & (o2_node != _state._cp.float32(0.0))
                ).item()
            )
        else:
            mono = np.asarray(
                ext_state.get(
                    "mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])
                ),
                dtype=np.float32,
            )
            dipole = np.asarray(
                ext_state.get(
                    "dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])
                ),
                dtype=np.float32,
            )
            o2_weight = mono + dipole
            has_o2 = bool(np.any(np.isfinite(o2_weight) & (o2_weight != 0.0)))
            o2_node = _state._cp.asarray(o2_weight, dtype=_state._cp.float32).ravel()
        if has_o2:
            grid_n = int(hybrid["grid_n"])
            spacing = float(hybrid["spacing"])
            kfreq = _state._cp.asarray(
                2.0 * np.pi * np.fft.fftfreq(grid_n, d=spacing),
                dtype=_state._cp.float32,
            )
            kx = kfreq[:, None, None]
            ky = kfreq[None, :, None]
            kz = kfreq[None, None, :]
            symbols = (
                -static["k2"],
                kx * kx,
                ky * ky,
                kz * kz,
                _state._cp.float32(2.0) * kx * ky,
                _state._cp.float32(2.0) * kx * kz,
                _state._cp.float32(2.0) * ky * kz,
            )
            if not use_runtime_moments:
                seg_vec = _state._cp.asarray(
                    gpu_static["segment_vectors"], dtype=_state._cp.float32
                )
                seg_len = _state._cp.sqrt(_state._cp.sum(seg_vec * seg_vec, axis=1))
                seg_len = _state._cp.maximum(seg_len, _state._cp.float32(1.0e-30))
                t_hat = seg_vec / seg_len[:, None]
                node_seg_ids = static["node_seg_ids"]
                tx = t_hat[node_seg_ids, 0]
                ty = t_hat[node_seg_ids, 1]
                tz = t_hat[node_seg_ids, 2]
                moment_weights = (
                    o2_node,
                    o2_node * tx * tx,
                    o2_node * ty * ty,
                    o2_node * tz * tz,
                    o2_node * tx * ty,
                    o2_node * tx * tz,
                    o2_node * ty * tz,
                )

    fused_self_done = False
    if (
        has_o2
        and bool(_state.CEXT_HYBRID_FFT_SELF_SUB_FUSED)
        and not use_runtime_moments
        # This kernel reads stored per-source stencils. Runtime-stencil mode
        # supplies one-element placeholders, not those arrays.
        and not bool(static.get("runtime_stencil", False))
    ):
        try:
            grid_shape = tuple(state["active_mass_grids_g"].shape)
            finite_responses = []
            for moment_idx, symbol in enumerate(symbols):
                cache_name = _cext_fft_response_cache_name(
                    state, f"fft_finite_self_response_g_{moment_idx}", grid_shape
                )
                if cache_name is None:
                    break
                response_g = _cext_fft_response_grids_gpu(
                    context,
                    hybrid,
                    ext_state,
                    state,
                    symbol_g=symbol,
                    cache_name=cache_name,
                )
                finite_responses.append(response_g)
            if len(finite_responses) == 7:
                kernel = _get_cext_fft_discrete_self_fused_o2_kernel()
                total = int(target_count * gl_order)
                threads = 128
                blocks = (total + threads - 1) // threads
                kernel(
                    (blocks,),
                    (threads,),
                    (
                        target_seg_ids_g,
                        gpu_static["gl_points_si"].ravel(),
                        state["q_weighted_gl"].ravel(),
                        o2_node.ravel(),
                        tx.ravel(),
                        ty.ravel(),
                        tz.ravel(),
                        state["lambda_iv_gl"].ravel(),
                        state["active_source_mask_g"],
                        static["stencil_flat_idx"].ravel(),
                        static["stencil_weight"].ravel(),
                        static["lambda_bin_edges"],
                        _state._cp.asarray(
                            base_response_g, dtype=_state._cp.float32
                        ).ravel(),
                        _state._cp.asarray(
                            finite_responses[0], dtype=_state._cp.float32
                        ).ravel(),
                        _state._cp.asarray(
                            finite_responses[1], dtype=_state._cp.float32
                        ).ravel(),
                        _state._cp.asarray(
                            finite_responses[2], dtype=_state._cp.float32
                        ).ravel(),
                        _state._cp.asarray(
                            finite_responses[3], dtype=_state._cp.float32
                        ).ravel(),
                        _state._cp.asarray(
                            finite_responses[4], dtype=_state._cp.float32
                        ).ravel(),
                        _state._cp.asarray(
                            finite_responses[5], dtype=_state._cp.float32
                        ).ravel(),
                        _state._cp.asarray(
                            finite_responses[6], dtype=_state._cp.float32
                        ).ravel(),
                        np.int32(max(int(np.asarray(edges).size - 1), 1)),
                        np.int32(int(hybrid["grid_n"])),
                        np.int32(int(static["grid_cells"])),
                        np.int32(int(static["stencil_n"])),
                        np.int32(gl_order),
                        np.int32(target_count),
                        np.float32(float(hybrid["origin"][0])),
                        np.float32(float(hybrid["origin"][1])),
                        np.float32(float(hybrid["origin"][2])),
                        np.float32(float(hybrid["spacing"])),
                        np.int32(assignment_mode),
                        out_g.ravel(),
                    ),
                )
                fused_self_done = True
            elif str(
                os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_FUSED_STRICT", "false")
            ).strip().lower() in ("1", "true", "yes", "on"):
                raise RuntimeError(
                    "Not enough response-cache budget for fused same-segment subtraction."
                )
        except Exception:
            if str(
                os.environ.get("SVV_CEXT_HYBRID_FFT_SELF_SUB_FUSED_STRICT", "false")
            ).strip().lower() in ("1", "true", "yes", "on"):
                raise
            fused_self_done = False

    if not fused_self_done:
        accumulate(state["q_weighted_gl"], base_response_g)
        if has_o2:
            grid_shape = tuple(state["active_mass_grids_g"].shape)
            if use_runtime_moments:
                for moment_idx, symbol in enumerate(symbols):
                    response_g = _cext_fft_response_grids_gpu(
                        context,
                        hybrid,
                        ext_state,
                        state,
                        symbol_g=symbol,
                        cache_name=_cext_fft_response_cache_name(
                            state,
                            f"fft_finite_self_response_g_{moment_idx}",
                            grid_shape,
                        ),
                    )
                    accumulate_runtime_moment(o2_node, response_g, moment_idx)
            else:
                for moment_idx, (moment_weight, symbol) in enumerate(
                    zip(moment_weights, symbols)
                ):
                    response_g = _cext_fft_response_grids_gpu(
                        context,
                        hybrid,
                        ext_state,
                        state,
                        symbol_g=symbol,
                        cache_name=_cext_fft_response_cache_name(
                            state,
                            f"fft_finite_self_response_g_{moment_idx}",
                            grid_shape,
                        ),
                    )
                    accumulate(moment_weight, response_g)

    _state._cp.cuda.Stream.null.synchronize()
    return out_g, perf_counter() - t0


__all__ = [
    "_cext_fft_discrete_same_segment_contribution_gpu",
    "_cext_fft_o2_term_correction_gpu",
    "_cext_fft_response_cache_name",
    "_cext_fft_response_grids_gpu",
    "_get_cext_fft_discrete_self_fused_o2_kernel",
    "_get_cext_fft_discrete_self_kernel",
]
