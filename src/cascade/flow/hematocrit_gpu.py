"""Conservative GPU RBC mixing and binary Pries phase separation."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import numpy as np

from cascade.accelerators.cuda import load_cuda_source
from cascade.configuration import solver_state as state

from .graph_schedule import graph_schedule

LAST_GPU_HEMATOCRIT_TIMINGS = {}


def segment_viscosity_gpu(radii, hd, mu_base, *, pries=False, return_device=False):
    """Vectorized CUDA viscosity and hematocrit corrections, no host edge loop."""
    cp = state._cp
    d = 2.0 * cp.asarray(radii, dtype=cp.float64) * 1e4
    h = cp.clip(
        cp.asarray(hd, dtype=cp.float64), state.HEMATOCRIT_MIN, state.HEMATOCRIT_MAX
    )
    if pries:
        dc = cp.maximum(d * state.PRIES_SECOMB_MCV_CORR, 1e-9)
        geom = (dc / cp.maximum(dc - state.PRIES_SECOMB_OPTW_UM, 1e-9)) ** 2
        inv = 1.0 / (
            1.0 + 10.0**state.PRIES_SECOMB_CPAR_3 * dc**state.PRIES_SECOMB_CPAR_4
        )
        c = (state.PRIES_SECOMB_CPAR_1 + cp.exp(state.PRIES_SECOMB_CPAR_2 * dc)) * (
            -1.0 + inv
        ) + inv
        eta = (
            state.PRIES_SECOMB_VISCPAR_1 * cp.exp(state.PRIES_SECOMB_VISCPAR_2 * dc)
            + state.PRIES_SECOMB_VISCPAR_3
            + state.PRIES_SECOMB_VISCPAR_4
            * cp.exp(state.PRIES_SECOMB_VISCPAR_5 * dc**state.PRIES_SECOMB_VISCPAR_6)
        )
        den = (1.0 - 0.45) ** c - 1.0
        factor = cp.where(den != 0.0, ((1.0 - h) ** c - 1.0) / den, 0.0)
        mu = (
            (1.0 + (eta - 1.0) * factor * geom)
            * geom
            * state.PRIES_SECOMB_VPLAS_CP
            * 0.01
        )
        mu = cp.where(
            cp.isfinite(mu) & (mu > 0.0), mu, state.PRIES_SECOMB_VPLAS_CP * 0.01
        )
    else:
        a = 4.0 / (1.0 + cp.exp(-0.593 * (d - 6.74)))
        term = 110.0 * cp.exp(-1.424 * d) + 3.0 - 3.45 * cp.exp(-0.035 * d)
        den = cp.exp(0.45 * a) - 1.0
        factor = cp.where(den != 0.0, (cp.exp(h) - 1.0) / den, 0.0)
        mu = mu_base * (1.0 + factor * term)
    return mu if return_device else cp.asnumpy(mu)


@lru_cache(maxsize=1)
def _kernel():
    return state._cp.RawKernel(
        load_cuda_source("network_hematocrit_kernel.cu"), "network_hematocrit_kernel"
    )


def compute_network_hematocrit_gpu(
    up,
    down,
    flows,
    radii,
    *,
    hd_root=None,
    model="uniform_tube",
    context=None,
    return_device=False,
):
    """Mix discharge hematocrit by RBC flux and propagate along the flow DAG.

    Pries law applies to a single-parent binary bifurcation. For nodes with
    multiple incoming vessels and binary outgoing vessels this extension uses
    the incoming flow-weighted diameter. Higher-degree splits use proportional
    RBC partition; this explicit modeling choice preserves flux exactly.
    Directed circulating supplied flows are rejected rather than silently
    applying a tree algorithm to them.
    """
    cp = state._cp
    if cp is None:
        raise RuntimeError("GPU hematocrit requires CUDA/CuPy")
    started = perf_counter()
    up, down = (
        np.asarray(up, dtype=np.int64).copy(),
        np.asarray(down, dtype=np.int64).copy(),
    )
    q = np.abs(np.asarray(flows, dtype=float))
    flipped = np.asarray(flows) < 0.0
    up[flipped], down[flipped] = down[flipped], up[flipped]
    root = float(
        np.clip(
            state.HD_DISCHARGE if hd_root is None else hd_root,
            state.HEMATOCRIT_MIN,
            state.HEMATOCRIT_MAX,
        )
    )
    hd = cp.full(len(q), root, cp.float64)
    radius = cp.asarray(radii, dtype=cp.float64)
    if model == "pries_secomb" and len(q):
        # Magnitude changes preserve the schedule; direction/active-edge changes
        # require new CSR dependencies. Never reuse weights from an old flow.
        cache = context.get("gpu_hematocrit") if context is not None else None
        active_mask = q > 1e-30
        schedule_reused = (
            cache is not None
            and cache.get("device_id") == int(cp.cuda.Device().id)
            and all(
                np.array_equal(cache[key], value)
                for key, value in (
                    ("up_host", up),
                    ("down_host", down),
                    ("active_host", active_mask),
                )
            )
        )
        if not schedule_reused:
            schedule = graph_schedule(up, down, q)
            if schedule is None:
                raise ValueError(
                    "GPU hematocrit requires a pressure-directed DAG; circulating supplied flows are unsupported"
                )
            n = int(max(up.max(), down.max()) + 1)
            active = np.flatnonzero(q > 1e-30)
            arrays = []
            for ids in (down, up):
                edges = active[np.argsort(ids[active], kind="stable")]
                offsets = np.concatenate(
                    ([0], np.cumsum(np.bincount(ids[active], minlength=n)))
                )
                arrays.extend(
                    (
                        cp.asarray(offsets, dtype=cp.int32),
                        cp.asarray(edges, dtype=cp.int32),
                    )
                )
            nodes, offsets, _, _ = schedule
            qi = np.bincount(down, weights=q, minlength=n)
            qo = np.bincount(up, weights=q, minlength=n)
            source = np.where(qo - qi > 1e-8 * np.maximum(qo, 1e-30), qo - qi, 0.0)
            cache = {
                "device_id": int(cp.cuda.Device().id),
                "flows_host": np.asarray(flows).copy(),
                "up_host": up.copy(),
                "down_host": down.copy(),
                "active_host": active_mask.copy(),
                "q": cp.asarray(q),
                "nodes": cp.asarray(nodes),
                "offsets": offsets,
                "csr": arrays,
                "source": cp.asarray(source),
            }
            if context is not None:
                context["gpu_hematocrit"] = cache
        elif not np.array_equal(cache["flows_host"], flows):
            n = len(cache["source"])
            qi = np.bincount(down, weights=q, minlength=n)
            qo = np.bincount(up, weights=q, minlength=n)
            source = np.where(qo - qi > 1e-8 * np.maximum(qo, 1e-30), qo - qi, 0.0)
            cache["q"].set(q)
            cache["source"].set(source)
            cache["flows_host"] = np.asarray(flows).copy()
        diameter = 2.0 * radius * 1e4
        csr = cache["csr"]
        for level in range(len(cache["offsets"]) - 1):
            nodes = cache["nodes"][
                cache["offsets"][level] : cache["offsets"][level + 1]
            ]
            if len(nodes):
                _kernel()(
                    ((len(nodes) + 127) // 128,),
                    (128,),
                    (
                        np.int32(len(nodes)),
                        nodes,
                        *csr,
                        cache["q"],
                        diameter,
                        cache["source"],
                        hd,
                        np.float64(root),
                        np.float64(state.HEMATOCRIT_MIN),
                        np.float64(state.HEMATOCRIT_MAX),
                        np.float64(state.PRIES_SECOMB_BIFPAR_1),
                        np.float64(state.PRIES_SECOMB_BIFPAR_2),
                        np.float64(state.PRIES_SECOMB_BIFPAR_3),
                        np.int32(1),
                    ),
                )
    elif model != "uniform_tube":
        raise ValueError("model must be 'uniform_tube' or 'pries_secomb'")
    diameter = 2.0 * radius * 1e4
    tube = cp.where(
        (radius > 0.0) & cp.isfinite(radius),
        hd
        * (
            hd
            + (1.0 - hd)
            * (1.0 + 1.7 * cp.exp(-0.415 * diameter) - 0.6 * cp.exp(-0.011 * diameter))
        ),
        0.0,
    )
    cp.cuda.Stream.null.synchronize()
    LAST_GPU_HEMATOCRIT_TIMINGS.update(
        backend="gpu",
        total_s=perf_counter() - started,
        segments=len(q),
        model=model,
        schedule_reused=bool(model == "pries_secomb" and len(q) and schedule_reused),
    )
    return (hd, tube) if return_device else (cp.asnumpy(hd), cp.asnumpy(tube))
