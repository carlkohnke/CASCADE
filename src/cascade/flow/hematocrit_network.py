"""CPU counterpart of the conservative graph hematocrit CUDA sweep."""

from __future__ import annotations

import numpy as np

from cascade.configuration import solver_state as state

from .graph_schedule import graph_schedule
from .hematocrit import (
    _normalize_hematocrit_model,
    _tube_hematocrit_from_hd_radius,
    njit,
)


@njit(cache=True)
def _propagate(
    nodes,
    incoming,
    in_edges,
    outgoing,
    out_edges,
    q,
    diameter,
    source,
    hd,
    root,
    hmin,
    hmax,
    par1,
    par2,
    par3,
):
    for node in nodes:
        first, last = outgoing[node], outgoing[node + 1]
        qi, rbc, dp, qo = 0.0, 0.0, 0.0, 0.0
        for j in range(incoming[node], incoming[node + 1]):
            edge = in_edges[j]
            qi += q[edge]
            rbc += q[edge] * hd[edge]
            dp += q[edge] * diameter[edge]
        for j in range(first, last):
            qo += q[out_edges[j]]
        if qo <= 0:
            continue
        hp = (rbc + source[node] * root) / qo if qi > 0 else root
        if last - first == 2:
            left, right = out_edges[first], out_edges[first + 1]
            f = q[left] / qo
            parent = (
                dp / qi
                if qi > 0
                else np.sqrt(diameter[left] ** 2 + diameter[right] ** 2)
            )
            parent = max(parent, 1e-9)
            x0 = min(max(par1 * (1 - hp) / parent, 0.0), 0.49)
            if f <= x0:
                fe = 0.0
            elif f >= 1 - x0:
                fe = 1.0
            else:
                quotient = diameter[left] ** 2 / max(diameter[right] ** 2, 1e-30)
                a = par3 * (quotient - 1) / (quotient + 1) * (1 - hp) / parent
                b = 1 + par2 * (1 - hp) / parent
                z = min(max((f - x0) / (1 - 2 * x0), 1e-12), 1 - 1e-12)
                fe = 1 / (
                    1 + np.exp(-min(max(a + b * np.log(z / (1 - z)), -50.0), 50.0))
                )
            if hp > 0:
                low = max(hmin * f / hp, 1 - hmax * (1 - f) / hp)
                high = min(hmax * f / hp, 1 - hmin * (1 - f) / hp)
                fe = min(max(fe, low), high)
            hd[left], hd[right] = hp * fe / f, hp * (1 - fe) / (1 - f)
        else:
            for j in range(first, last):
                hd[out_edges[j]] = hp
    return hd


def compute_network_hematocrit_cpu(
    up, down, flows, radii, *, hd_root=None, model=None, context=None
):
    """Use the same merger, binary Pries, and higher-degree rules as CUDA."""
    up, down = (
        np.asarray(up, dtype=np.int64).copy(),
        np.asarray(down, dtype=np.int64).copy(),
    )
    flows = np.asarray(flows, dtype=float)
    q = abs(flows)
    flip = flows < 0
    up[flip], down[flip] = down[flip], up[flip]
    root = float(
        np.clip(
            state.HD_DISCHARGE if hd_root is None else hd_root,
            state.HEMATOCRIT_MIN,
            state.HEMATOCRIT_MAX,
        )
    )
    hd = np.full(len(q), root)
    if _normalize_hematocrit_model(model) == "pries_secomb" and len(q):
        active = q > 1e-30
        cache = context.get("cpu_hematocrit") if context is not None else None
        if cache is None or not all(
            np.array_equal(cache[key], value)
            for key, value in (("up", up), ("down", down), ("active", active))
        ):
            schedule = graph_schedule(up, down, q)
            if schedule is None:
                raise ValueError(
                    "Network hematocrit requires a pressure-directed DAG; circulating supplied flows are unsupported"
                )
            n = int(max(up.max(), down.max()) + 1)
            ids = np.flatnonzero(active)
            csr = []
            for nodes in (down, up):
                edges = ids[np.argsort(nodes[ids], kind="stable")]
                offsets = np.concatenate(
                    ([0], np.cumsum(np.bincount(nodes[ids], minlength=n)))
                )
                csr.extend((offsets, edges))
            cache = dict(
                up=up, down=down, active=active.copy(), nodes=schedule[0], csr=csr, n=n
            )
            if context is not None:
                context["cpu_hematocrit"] = cache
        qi = np.bincount(down, weights=q, minlength=cache["n"])
        qo = np.bincount(up, weights=q, minlength=cache["n"])
        source = np.where(qo - qi > 1e-8 * np.maximum(qo, 1e-30), qo - qi, 0.0)
        hd = _propagate(
            cache["nodes"],
            *cache["csr"],
            q,
            2 * np.asarray(radii) * 1e4,
            source,
            hd,
            root,
            state.HEMATOCRIT_MIN,
            state.HEMATOCRIT_MAX,
            state.PRIES_SECOMB_BIFPAR_1,
            state.PRIES_SECOMB_BIFPAR_2,
            state.PRIES_SECOMB_BIFPAR_3,
        )
    return hd, _tube_hematocrit_from_hd_radius(radii, hd)
