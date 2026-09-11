"""Sparse and tree-specialized Kirchhoff pressure-flow solvers.

The numerical implementations retain legacy configuration defaults for older
callers, while the public ``solve_flow`` API supplies solver and boundary-mode
choices explicitly.
"""

from __future__ import annotations

from time import perf_counter
from typing import Sequence, Tuple

import numpy as np

from cascade.configuration import _legacy_state as _state

from .linear_system import (
    _fixed_terminal_flows_numba,
    _kirchhoff_tree_mixed_numba,
    _kirchhoff_tree_neumann_numba,
    _pressures_from_tree_flows_numba,
    _solve_kirchhoff_sparse,
    solve_kirchhoff_dirichlet,
)
from .topology import _build_node_indices, _normalize_kirchhoff_bc_mode


def _kirchhoff_rhs_anchor_and_root(
    num_nodes: int,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
) -> tuple[np.ndarray, int, int]:
    rhs = np.zeros(num_nodes, dtype=float)
    inlet_arr = np.asarray(list(inlet_nodes), dtype=np.int64).reshape(-1)
    outlet_arr = np.asarray(list(outlet_nodes), dtype=np.int64).reshape(-1)

    if inlet_arr.size:
        inlet_share = float(inlet_flow_cm3_s) / float(inlet_arr.size)
        rhs[inlet_arr] += inlet_share
        root_node = int(inlet_arr[0])
    else:
        root_node = 0
    if outlet_arr.size:
        outlet_share = float(inlet_flow_cm3_s) / float(outlet_arr.size)
        rhs[outlet_arr] -= outlet_share

    anchor = max(num_nodes - 1, 0)
    if (
        (inlet_arr.size and bool(np.any(inlet_arr == anchor)))
        or (outlet_arr.size and bool(np.any(outlet_arr == anchor)))
    ):
        anchor = max(num_nodes - 2, 0)
    return rhs, root_node, anchor


def _kirchhoff_residual_norms(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    flows: np.ndarray,
    rhs: np.ndarray,
    num_nodes: int,
) -> tuple[float, float]:
    balance = np.bincount(prox_ids, weights=flows, minlength=num_nodes).astype(float, copy=False)
    balance -= np.bincount(dist_ids, weights=flows, minlength=num_nodes).astype(float, copy=False)
    resid = balance - rhs
    abs_norm = float(np.linalg.norm(resid))
    rhs_norm = float(np.linalg.norm(rhs))
    rel_norm = abs_norm / rhs_norm if rhs_norm > 0.0 else abs_norm
    return abs_norm, rel_norm


def solve_kirchhoff_tree(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
    *,
    num_nodes: int | None = None,
    boundary_condition: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not _state._HAVE_NUMBA:
        raise RuntimeError("Tree Kirchhoff solver requires numba; use a sparse Kirchhoff solver instead.")

    prox_ids = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
    dist_ids = np.asarray(dist_ids, dtype=np.int64).reshape(-1)
    resistances = np.asarray(resistances, dtype=float).reshape(-1)
    num_edges = int(prox_ids.size)
    if num_edges == 0:
        raise ValueError("No segments provided for Kirchhoff solve.")
    if num_nodes is None:
        num_nodes = int(max(int(prox_ids.max()), int(dist_ids.max())) + 1)
    if num_nodes <= 0:
        raise ValueError("No nodes provided for Kirchhoff solve.")

    inlet_arr = np.asarray(list(inlet_nodes), dtype=np.int64).reshape(-1)
    root_node = int(inlet_arr[0]) if inlet_arr.size else 0
    outlet_arr = np.asarray(list(outlet_nodes), dtype=np.int64).reshape(-1)
    outlet_arr = outlet_arr[(outlet_arr >= 0) & (outlet_arr < int(num_nodes))]
    bc_mode = _normalize_kirchhoff_bc_mode(boundary_condition)
    t0 = perf_counter()
    rhs_diag = np.zeros(int(num_nodes), dtype=float)
    if bc_mode == "legacy_equal_terminal_flow":
        rhs, root_node, anchor = _kirchhoff_rhs_anchor_and_root(
            int(num_nodes),
            inlet_nodes,
            float(inlet_flow_cm3_s),
            outlet_arr,
        )
        rhs_diag = rhs
        pressures, flows, n_flow_nodes, n_pressure_nodes = _kirchhoff_tree_neumann_numba(
            prox_ids,
            dist_ids,
            np.maximum(resistances, 1e-30),
            rhs,
            int(root_node),
            int(anchor),
            int(num_nodes),
        )
        solver_used = "tree_neumann"
    else:
        if outlet_arr.size == 0:
            raise ValueError("At least one terminal pressure node is required for the mixed tree Kirchhoff solve.")
        pressures, flows, n_flow_nodes, n_pressure_nodes = _kirchhoff_tree_mixed_numba(
            prox_ids,
            dist_ids,
            np.maximum(resistances, 1e-30),
            outlet_arr,
            int(root_node),
            float(inlet_flow_cm3_s),
            int(num_nodes),
        )
        solver_used = "tree_mixed"
        if inlet_arr.size:
            inlet_share = float(inlet_flow_cm3_s) / float(inlet_arr.size)
            rhs_diag[inlet_arr] = inlet_share
    solve_s = perf_counter() - t0

    if n_flow_nodes != int(num_nodes) or n_pressure_nodes != int(num_nodes):
        raise RuntimeError(
            "Tree Kirchhoff solver could not traverse all nodes; graph may be disconnected or not tree-like. "
            f"flow_nodes={n_flow_nodes}/{num_nodes} pressure_nodes={n_pressure_nodes}/{num_nodes}"
        )
    if not np.all(np.isfinite(pressures)) or not np.all(np.isfinite(flows)):
        raise RuntimeError("Tree Kirchhoff solver produced non-finite pressures or flows.")

    if _state.KIRCHHOFF_DIAGNOSTICS:
        balance = np.bincount(prox_ids, weights=flows, minlength=int(num_nodes)).astype(float, copy=False)
        balance -= np.bincount(dist_ids, weights=flows, minlength=int(num_nodes)).astype(float, copy=False)
        if bc_mode == "legacy_equal_terminal_flow":
            check_mask = np.ones(int(num_nodes), dtype=bool)
        else:
            check_mask = np.ones(int(num_nodes), dtype=bool)
            check_mask[outlet_arr] = False
        resid = balance[check_mask] - rhs_diag[check_mask]
        rhs_norm = float(np.linalg.norm(rhs_diag[check_mask]))
        rel_resid = float(np.linalg.norm(resid) / rhs_norm) if rhs_norm > 0.0 else float(np.linalg.norm(resid))
        print(
            f"Kirchhoff diagnostics: solver_used={solver_used} "
            f"bc={bc_mode} solve_time={solve_s:.3f}s true_rel_resid={rel_resid:.3e}"
        )

    return pressures, flows, prox_ids, dist_ids, np.empty((0, 3), dtype=float)


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float).reshape(-1) - np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(np.asarray(b, dtype=float).reshape(-1)))
    num = float(np.linalg.norm(diff))
    return num / denom if denom > 0.0 else num


def solve_kirchhoff(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
    *,
    num_nodes: int | None = None,
    solver_mode: str | None = None,
    boundary_condition: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    solver_mode = str(solver_mode or _state.KIRCHHOFF_SOLVER).strip().lower()
    if solver_mode in ("tree", "tree_neumann", "tree-current-bc"):
        t_tree = perf_counter()
        tree_result = solve_kirchhoff_tree(
            prox_ids,
            dist_ids,
            resistances,
            inlet_nodes,
            inlet_flow_cm3_s,
            outlet_nodes,
            num_nodes=num_nodes,
            boundary_condition=boundary_condition,
        )
        t_tree = perf_counter() - t_tree
        if _state.KIRCHHOFF_VALIDATE_TREE:
            t_sparse = perf_counter()
            sparse_result = _solve_kirchhoff_sparse(
                prox_ids,
                dist_ids,
                resistances,
                inlet_nodes,
                inlet_flow_cm3_s,
                outlet_nodes,
                num_nodes=num_nodes,
                sparse_solver=_state.KIRCHHOFF_VALIDATE_SPARSE_SOLVER,
                boundary_condition=boundary_condition,
            )
            t_sparse = perf_counter() - t_sparse
            p_tree, q_tree = tree_result[0], tree_result[1]
            p_sparse, q_sparse = sparse_result[0], sparse_result[1]
            pressure_rel_l2 = _relative_l2(p_tree, p_sparse)
            flow_rel_l2 = _relative_l2(q_tree, q_sparse)
            pressure_max_abs = float(np.max(np.abs(p_tree - p_sparse))) if p_tree.size else 0.0
            flow_max_abs = float(np.max(np.abs(q_tree - q_sparse))) if q_tree.size else 0.0
            speedup = t_sparse / max(t_tree, 1e-30)
            print(
                "Kirchhoff tree validation: "
                f"tree={t_tree:.3f}s sparse={t_sparse:.3f}s speedup={speedup:.2f}x "
                f"flow_rel_l2={flow_rel_l2:.3e} flow_max_abs={flow_max_abs:.3e} "
                f"pressure_rel_l2={pressure_rel_l2:.3e} pressure_max_abs={pressure_max_abs:.3e}"
            )
        return tree_result

    return _solve_kirchhoff_sparse(
        prox_ids,
        dist_ids,
        resistances,
        inlet_nodes,
        inlet_flow_cm3_s,
        outlet_nodes,
        num_nodes=num_nodes,
        sparse_solver=solver_mode,
        boundary_condition=boundary_condition,
    )




__all__ = ['_solve_kirchhoff_sparse', '_fixed_terminal_flows_numba', '_pressures_from_tree_flows_numba', '_kirchhoff_tree_neumann_numba', '_kirchhoff_tree_mixed_numba', '_kirchhoff_rhs_anchor_and_root', '_kirchhoff_residual_norms', 'solve_kirchhoff_tree', '_relative_l2', 'solve_kirchhoff', 'solve_kirchhoff_dirichlet']
