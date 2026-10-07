"""General-network intravascular concentration transport.

This module owns the node-mixing and nonlinear segment-decay solve for
arbitrary directed vessel networks. Tree-specialized transport lives in
``topdown.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.flow.topology import _build_node_indices

from .greens import (
    _blood_greens_decay_factor,
    _greens_decay_factor,
    severinghaus_dSdP,
)
from .oxygen_transport import (
    junction_flux_residual,
    nodal_capacity,
    oxygen_content,
    total_content_enabled,
    transport_capacity,
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
    outlet_nodes: Sequence[int] | None,
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
    discharge_hematocrit: np.ndarray | None = None,
    accel: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Solve concentration transport on an arbitrary directed vessel graph.

    Flow signs determine edge direction. Incoming solute fluxes mix at each
    junction before segment-level wall loss is applied. Fixed-point iteration
    continues to the requested tolerance while recording mass-balance history.
    """

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
        empty_hist: dict[str, list[float]] = {
            "iter": [],
            "max_delta": [],
            "M_in": [],
            "M_out": [],
            "M_drop": [],
            "MB_resid": [],
            "total_oxygen_junction_rel_resid": [],
        }
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), empty_hist

    fluid_mode = (fluid or _state.ACTIVE_FLUID).lower()
    Chb_max = np.zeros_like(radii)
    if fluid_mode == "blood":
        HD = (np.full_like(radii, float(_state.HD_DISCHARGE))
              if discharge_hematocrit is None else np.asarray(discharge_hematocrit,dtype=float))
        if HD.shape != np.asarray(radii).shape:
            raise ValueError("discharge_hematocrit must have one value per edge")
        from cascade.flow.hematocrit import _tube_hematocrit_from_hd_radius
        Chb_max = transport_capacity(HD, _tube_hematocrit_from_hd_radius(radii,HD))
    from .network_gpu import resolve_network_accel, solve_network_greens_gpu
    if resolve_network_accel(accel) == 'gpu':
        return solve_network_greens_gpu(up, down, q, np.asarray(radii), np.asarray(lengths),
                                       Chb_max, inlet_nodes, outlet_nodes, inlet_concentration,
                                       fluid_mode, diffusivity, vmax, km, max_iter, tol, omega)

    valid_edges = q > 0.0
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
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
        sink_edges = np.concatenate(
            [np.array(incoming[n], dtype=int) for n in sink_nodes]
        )
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

    conserve_total = total_content_enabled(fluid_mode)
    node_capacity = nodal_capacity(up, down, q, Chb_max, num_nodes)

    history: dict[str, list[float]] = {
        "iter": [],
        "max_delta": [],
        "M_in": [],
        "M_out": [],
        "M_drop": [],
        "MB_resid": [],
        "total_oxygen_junction_rel_resid": [],
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

        node_flow = np.where(sum_out > 0.0, sum_out, sum_in)
        node_flow[np.asarray(sink_nodes, dtype=int)] = sum_in[
            np.asarray(sink_nodes, dtype=int)
        ]
        node_flow = np.where(node_flow > 0.0, node_flow, 1.0)
        edge_weight = q.copy()
        diagonal_weight = node_flow.copy()
        rhs = np.zeros(num_nodes)
        if conserve_total:
            cout_old = C[up] * decay
            edge_B = 1.0 + Chb_max / _state.ALPHA_MMHG * severinghaus_dSdP(
                cout_old / _state.ALPHA_MMHG
            )
            node_B = 1.0 + node_capacity / _state.ALPHA_MMHG * severinghaus_dSdP(
                C / _state.ALPHA_MMHG
            )
            edge_d = oxygen_content(cout_old, Chb_max) - edge_B * cout_old
            node_d = oxygen_content(C, node_capacity) - node_B * C
            edge_weight *= edge_B
            diagonal_weight *= node_B
            np.add.at(rhs, down, q * edge_d)
            rhs -= node_flow * node_d

        if _state._HAVE_SCIPY_SPARSE:
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            for i in range(q.size):
                if not valid_edges[i]:
                    continue
                rows.append(int(down[i]))
                cols.append(int(up[i]))
                data.append(float(-edge_weight[i] * decay[i]))

            for n in range(num_nodes):
                rows.append(n)
                cols.append(n)
                data.append(float(diagonal_weight[n]))

            A = _sp.coo_matrix(
                (data, (rows, cols)), shape=(num_nodes, num_nodes)
            ).tolil()
            b = rhs.copy()
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
                A[down[i], up[i]] -= edge_weight[i] * decay[i]

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

            if conserve_total:
                A[np.diag_indices(num_nodes)] += diagonal_weight - node_flow
            b = rhs.copy()
            for node, value in dirichlet.items():
                b -= A[:, node] * value
                A[:, node] = 0.0
                A[node, :] = 0.0
                A[node, node] = 1.0
                b[node] = value

            C_new = np.linalg.solve(A, b)
        dC = C_new - C
        if conserve_total and np.any(C_new < 0.0):
            falling = dC < 0.0
            step = min(1.0, 0.9 * float(np.min(C[falling] / -dC[falling])))
            dC *= step
        delta = float(np.nanmax(np.abs(dC)))

        cin = C[up]
        cout = cin * decay
        content_in = oxygen_content(cin, Chb_max)
        content_out = oxygen_content(cout, Chb_max)
        if conserve_total:
            cin_balance, cout_balance = content_in, content_out
        else:
            cin_balance, cout_balance = cin, cout
        M_in = (
            float(np.sum(q[inlet_edges] * cin_balance[inlet_edges]))
            if inlet_edges.size
            else 0.0
        )
        M_out = (
            float(np.sum(q[sink_edges] * cout_balance[sink_edges]))
            if sink_edges.size
            else 0.0
        )
        returning = np.isin(down,list(inlet_nodes))
        M_in -= float(np.sum(q[returning]*cout_balance[returning]))
        leaving_sink = np.isin(up,sink_nodes)
        M_out -= float(np.sum(q[leaving_sink]*cin_balance[leaving_sink]))
        M_drop = float(np.sum(q * (cin_balance - cout_balance)))
        MB_resid = (M_in - M_out) - M_drop

        history["iter"].append(it)
        history["max_delta"].append(delta)
        history["M_in"].append(M_in)
        history["M_out"].append(M_out)
        history["M_drop"].append(M_drop)
        history["MB_resid"].append(MB_resid)
        junction_residual = junction_flux_residual(
            up, down, q, cin, cout, Chb_max, num_nodes, boundary_nodes=inlet_nodes
        )
        history["total_oxygen_junction_rel_resid"].append(junction_residual)
        C = C + omega * dC
        if delta < tol and (not conserve_total or junction_residual < tol):
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
    if history["total_oxygen_junction_rel_resid"]:
        history["total_oxygen_junction_rel_resid"][-1] = junction_flux_residual(
            up, down, q, cin, cout, Chb_max, num_nodes, boundary_nodes=inlet_nodes
        )
    return cin, cout, C, history


__all__ = ["solve_network_concentrations"]
