"""General-network intravascular concentration transport.

This module owns the node-mixing and nonlinear segment-decay solve for
arbitrary directed vessel networks. Tree-specialized transport lives in
``topdown.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cascade.configuration import _legacy_state as _state
from cascade.flow.rheology import tube_hematocrit
from cascade.flow.topology import _build_node_indices

from .greens import (
    _blood_greens_decay_factor,
    _greens_decay_factor,
    segment_O2_capacity_from_HT,
)

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _splinalg
except ImportError:  # pragma: no cover - dense fallback remains available
    _sp = None
    _splinalg = None


def solve_network_concentrations(
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    flows: np.ndarray,
    inlet_nodes: Sequence[int],
    outlet_nodes: Optional[Sequence[int]],
    inlet_concentration: float,
    *,
    fluid: str,
    prox_ids: np.ndarray | None = None,
    dist_ids: np.ndarray | None = None,
    diffusivity: float = _state.SOLUTE_DIFFUSIVITY,
    vmax: float = _state.VMAX_MM,
    km: float = _state.K_M_MM,
    max_iter: int = 100,
    tol: float = 1e-3,
    omega: float = _state.OMEGA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[float]]]:
    if prox_ids is None or dist_ids is None:
        geom = np.zeros((starts.shape[0], 6), dtype=float)
        geom[:, 0:3] = starts
        geom[:, 3:6] = ends
        prox_ids, dist_ids, _ = _build_node_indices(geom)
    else:
        prox_ids = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
        dist_ids = np.asarray(dist_ids, dtype=np.int64).reshape(-1)

    q = np.abs(flows)
    up = prox_ids.copy()
    down = dist_ids.copy()
    flip = flows < 0
    up[flip] = dist_ids[flip]
    down[flip] = prox_ids[flip]

    num_nodes = int(max(up.max(), down.max()) + 1) if up.size else 0
    if num_nodes == 0:
        empty_hist: Dict[str, List[float]] = {"iter": [], "max_delta": [], "M_in": [], "M_out": [], "M_drop": [], "MB_resid": []}
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), empty_hist

    valid_edges = q > 0.0
    outgoing: List[List[int]] = [[] for _ in range(num_nodes)]
    incoming: List[List[int]] = [[] for _ in range(num_nodes)]
    for i in range(q.size):
        if not valid_edges[i]:
            continue
        outgoing[up[i]].append(i)
        incoming[down[i]].append(i)

    inlet_edges = np.array(
        [edge for node in inlet_nodes for edge in outgoing[int(node)]],
        dtype=int,
    )
    if outlet_nodes:
        sink_nodes = [int(n) for n in outlet_nodes]
    else:
        sink_nodes = [n for n in range(num_nodes) if incoming[n] and not outgoing[n]]
    if sink_nodes:
        sink_edges = np.concatenate([np.array(incoming[n], dtype=int) for n in sink_nodes])
    else:
        sink_edges = np.array([], dtype=int)

    sum_out = np.zeros(num_nodes, dtype=float)
    sum_in = np.zeros(num_nodes, dtype=float)
    for i in range(q.size):
        if not valid_edges[i]:
            continue
        sum_out[up[i]] += q[i]
        sum_in[down[i]] += q[i]

    dirichlet = {int(n): float(inlet_concentration) for n in inlet_nodes}
    C = np.full(num_nodes, float(inlet_concentration), dtype=float)
    for node, value in dirichlet.items():
        C[node] = value

    fluid_mode = (fluid or _state.ACTIVE_FLUID).lower()
    Chb_max = np.zeros_like(radii)
    if fluid_mode == "blood":
        for i, r in enumerate(radii):
            HT = tube_hematocrit(r, hd=_state.HD_DISCHARGE)
            Chb_max[i] = segment_O2_capacity_from_HT(HT)

    history: Dict[str, List[float]] = {
        "iter": [],
        "max_delta": [],
        "M_in": [],
        "M_out": [],
        "M_drop": [],
        "MB_resid": [],
    }

    for it in range(1, max_iter + 1):
        decay = np.ones_like(q)
        for i in range(q.size):
            if not valid_edges[i]:
                continue
            cin = C[up[i]]
            if fluid_mode == "blood":
                decay[i] = _blood_greens_decay_factor(
                    q[i] * _state.CM3_TO_M3,
                    radii[i] * _state.CM_TO_M,
                    lengths[i] * _state.CM_TO_M,
                    diffusivity * _state.CM2_TO_M2,
                    vmax,
                    km,
                    cin,
                    Chb_max[i],
                )
            else:
                decay[i] = _greens_decay_factor(
                    q[i] * _state.CM3_TO_M3,
                    radii[i] * _state.CM_TO_M,
                    lengths[i] * _state.CM_TO_M,
                    diffusivity * _state.CM2_TO_M2,
                    vmax,
                    km,
                    cin,
                )

        if _state._HAVE_SCIPY_SPARSE:
            rows: List[int] = []
            cols: List[int] = []
            data: List[float] = []
            for i in range(q.size):
                if not valid_edges[i]:
                    continue
                rows.append(int(down[i]))
                cols.append(int(up[i]))
                data.append(float(-q[i] * decay[i]))

            sink_set = set(sink_nodes)
            for n in range(num_nodes):
                if n in sink_set:
                    diag = sum_in[n] if sum_in[n] > 0.0 else 1.0
                elif sum_out[n] > 0.0:
                    diag = sum_out[n]
                elif sum_in[n] > 0.0:
                    diag = sum_in[n]
                else:
                    diag = 1.0
                rows.append(n)
                cols.append(n)
                data.append(float(diag))

            A = _sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tolil()
            b = np.zeros(num_nodes, dtype=float)
            for node, value in dirichlet.items():
                col = np.asarray(A[:, node].todense()).ravel()
                b -= col * value
                A[:, node] = 0.0
                A[node, :] = 0.0
                A[node, node] = 1.0
                b[node] = value
            C_new = _splinalg.spsolve(A.tocsr(), b)
        else:
            A = np.zeros((num_nodes, num_nodes), dtype=float)
            for i in range(q.size):
                if not valid_edges[i]:
                    continue
                A[down[i], up[i]] -= q[i] * decay[i]

            sink_set = set(sink_nodes)
            for n in range(num_nodes):
                if n in sink_set:
                    if sum_in[n] > 0.0:
                        A[n, n] += sum_in[n]
                    else:
                        A[n, n] += 1.0
                elif sum_out[n] > 0.0:
                    A[n, n] += sum_out[n]
                elif sum_in[n] > 0.0:
                    A[n, n] += sum_in[n]
                else:
                    A[n, n] += 1.0

            b = np.zeros(num_nodes, dtype=float)
            for node, value in dirichlet.items():
                b -= A[:, node] * value
                A[:, node] = 0.0
                A[node, :] = 0.0
                A[node, node] = 1.0
                b[node] = value

            C_new = np.linalg.solve(A, b)
        dC = C_new - C
        delta = float(np.nanmax(np.abs(dC)))

        cin = C[up]
        cout = cin * decay
        M_in = float(np.sum(q[inlet_edges] * cin[inlet_edges])) if inlet_edges.size else 0.0
        M_out = float(np.sum(q[sink_edges] * cout[sink_edges])) if sink_edges.size else 0.0
        M_drop = float(np.sum(q * (cin - cout)))
        MB_resid = (M_in - M_out) - M_drop

        history["iter"].append(it)
        history["max_delta"].append(delta)
        history["M_in"].append(M_in)
        history["M_out"].append(M_out)
        history["M_drop"].append(M_drop)
        history["MB_resid"].append(MB_resid)
        C = C + omega * dC
        if delta < tol:
            break

    decay = np.ones_like(q)
    for i in range(q.size):
        if not valid_edges[i]:
            continue
        cin_edge = C[up[i]]
        if fluid_mode == "blood":
            decay[i] = _blood_greens_decay_factor(
                q[i] * _state.CM3_TO_M3,
                radii[i] * _state.CM_TO_M,
                lengths[i] * _state.CM_TO_M,
                diffusivity * _state.CM2_TO_M2,
                vmax,
                km,
                cin_edge,
                Chb_max[i],
            )
        else:
            decay[i] = _greens_decay_factor(
                q[i] * _state.CM3_TO_M3,
                radii[i] * _state.CM_TO_M,
                lengths[i] * _state.CM_TO_M,
                diffusivity * _state.CM2_TO_M2,
                vmax,
                km,
                cin_edge,
            )
    cin = C[up]
    cout = cin * decay
    return cin, cout, C, history


__all__ = ["solve_network_concentrations"]
