"""Sparse linear-system backend for Kirchhoff pressure-flow solves.

This module owns matrix construction, conditioning diagnostics, and iterative
or direct sparse solver dispatch. Tree topology orchestration remains in
:mod:`cascade.flow.kirchhoff`.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence, Tuple

import numpy as np

from cascade.configuration import solver_state as _state

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _splinalg
except ImportError:  # pragma: no cover - dependency validation reports this earlier
    _sp = None
    _splinalg = None

try:
    from numba import njit
except ImportError:  # pragma: no cover - dependency validation reports this earlier

    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate


from .topology import _build_node_indices, _normalize_kirchhoff_bc_mode


def _diagnostic_stats(name: str, values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return f"{name}=empty"
    return (
        f"{name}[min={float(np.min(finite)):.3e}, "
        f"median={float(np.median(finite)):.3e}, max={float(np.max(finite)):.3e}]"
    )


def _solve_kirchhoff_sparse(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_flow_cm3_s: float,
    outlet_nodes: Sequence[int],
    *,
    num_nodes: int | None = None,
    sparse_solver: str | None = None,
    boundary_condition: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve a general vascular Kirchhoff system with sparse linear algebra.

    The system supports prescribed outlet pressure or equal terminal-flow
    constraints. Direct and iterative solver choices share the same assembled
    conductance matrix and residual checks.
    """

    prox_ids = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
    dist_ids = np.asarray(dist_ids, dtype=np.int64).reshape(-1)
    num_edges = int(prox_ids.size)
    if num_edges == 0:
        raise ValueError("No segments provided for Kirchhoff solve.")

    if num_nodes is None:
        num_nodes = int(max(int(prox_ids.max()), int(dist_ids.max())) + 1)
    if num_nodes <= 0:
        raise ValueError("No nodes provided for Kirchhoff solve.")

    edge_conductance = 1.0 / np.maximum(
        np.asarray(resistances, dtype=float).reshape(-1), 1e-30
    )

    rhs = np.zeros(num_nodes, dtype=float)
    if inlet_nodes:
        inlet_share = inlet_flow_cm3_s / max(len(inlet_nodes), 1)
        for node in inlet_nodes:
            rhs[int(node)] += inlet_share
    outlet_arr = np.asarray(list(outlet_nodes), dtype=np.int64).reshape(-1)
    outlet_arr = outlet_arr[(outlet_arr >= 0) & (outlet_arr < num_nodes)]
    bc_mode = _normalize_kirchhoff_bc_mode(boundary_condition)
    dirichlet_mask = np.zeros(num_nodes, dtype=bool)
    anchor = -1
    if bc_mode == "legacy_equal_terminal_flow":
        if outlet_arr.size:
            outlet_share = inlet_flow_cm3_s / max(int(outlet_arr.size), 1)
            rhs[outlet_arr] -= outlet_share
        anchor = num_nodes - 1
        inlet_set = {int(node) for node in inlet_nodes}
        outlet_set = {int(node) for node in outlet_arr.tolist()}
        if (anchor in inlet_set) or (anchor in outlet_set):
            anchor = max(num_nodes - 2, 0)
        dirichlet_mask[int(anchor)] = True
    else:
        if outlet_arr.size == 0:
            raise ValueError(
                "At least one terminal pressure node is required for the mixed Kirchhoff solve."
            )
        dirichlet_mask[outlet_arr] = True

    if _state._HAVE_SCIPY_SPARSE:
        # Build graph Laplacian directly from edges (avoid B @ G @ B.T intermediates).
        u = prox_ids
        v = dist_ids
        g = edge_conductance
        rows = np.concatenate([u, v, u, v])
        cols = np.concatenate([u, v, v, u])
        data = np.concatenate([g, g, -g, -g])
        laplacian = _sp.coo_matrix(
            (data, (rows, cols)), shape=(num_nodes, num_nodes)
        ).tocsr()

        mask = np.ones(num_nodes, dtype=bool)
        mask[dirichlet_mask] = False
        A = laplacian[mask][:, mask].tocsr()
        b = rhs[mask]

        solver = str(sparse_solver or _state.KIRCHHOFF_SPARSE_SOLVER).strip().lower()
        use_cg = solver == "cg" or (
            solver == "auto" and A.shape[0] >= int(_state.KIRCHHOFF_CG_MIN_NODES)
        )
        use_gmres_ilu = solver == "gmres_ilu"
        gmres_failed_reason: str | None = None
        cg_failed_reason: str | None = None
        gmres_ilu_build_s = 0.0
        gmres_solve_s = 0.0
        gmres_iters = 0
        gmres_cycles = 0
        gmres_final_resid = float("nan")
        solver_used = "none"

        if _state.KIRCHHOFF_DIAGNOSTICS:
            print(
                f"Kirchhoff diagnostics: n={A.shape[0]} nnz={int(A.nnz)} "
                f"{_diagnostic_stats('R', resistances)} "
                f"{_diagnostic_stats('G', edge_conductance)} "
                f"{_diagnostic_stats('diag(A)', A.diagonal())}"
            )

        x = None
        if use_gmres_ilu:
            try:
                gmres_systems: list[
                    tuple[str, _sp.csr_matrix, np.ndarray, np.ndarray | None]
                ] = []
                if _state.KIRCHHOFF_GMRES_EQUILIBRATE:
                    diag_a = np.abs(A.diagonal()).astype(float, copy=False)
                    diag_pos = diag_a[diag_a > 0.0]
                    if diag_pos.size:
                        diag_floor = max(
                            float(np.median(diag_pos))
                            * float(_state.KIRCHHOFF_GMRES_EQ_DIAG_FLOOR_REL),
                            1e-30,
                        )
                    else:
                        diag_floor = 1e-30
                    diag_safe = np.maximum(diag_a, diag_floor)
                    equil_scale = np.divide(
                        1.0,
                        np.sqrt(diag_safe),
                        out=np.zeros_like(diag_safe),
                        where=diag_safe > 0.0,
                    )
                    if not np.all(np.isfinite(equil_scale)) or np.any(
                        equil_scale <= 0.0
                    ):
                        raise RuntimeError(
                            "computed non-finite diagonal equilibration scale"
                        )
                    S = _sp.diags(equil_scale)
                    A_eq = (S @ A @ S).tocsr()
                    b_eq = equil_scale * b
                    gmres_systems.append(("equilibrated", A_eq, b_eq, equil_scale))
                    if _state.KIRCHHOFF_DIAGNOSTICS:
                        print(
                            f"Kirchhoff diagnostics: gmres_equilibrate=True "
                            f"diag_floor={diag_floor:.3e} "
                            f"{_diagnostic_stats('eq_scale', equil_scale)} "
                            f"{_diagnostic_stats('diag(A_eq)', A_eq.diagonal())}"
                        )
                if (
                    not gmres_systems
                    or _state.KIRCHHOFF_GMRES_RETRY_UNSCALED_IF_EQ_FAIL
                ):
                    gmres_systems.append(("unscaled", A, b, None))

                gmres_failed_reason = "gmres_ilu did not run"
                for system_name, A_gmres, b_gmres, equil_scale in gmres_systems:
                    diag_ref_arr = np.abs(A_gmres.diagonal()).astype(float, copy=False)
                    diag_ref_pos = diag_ref_arr[diag_ref_arr > 0.0]
                    diag_ref = (
                        float(np.median(diag_ref_pos)) if diag_ref_pos.size else 1.0
                    )
                    if not np.isfinite(diag_ref) or diag_ref <= 0.0:
                        diag_ref = 1.0

                    ilu = None
                    attempt_reason: str | None = None
                    used_permc = "unknown"
                    used_shift = 0.0
                    for permc_spec in _state.KIRCHHOFF_ILU_PERMC_SPECS:
                        for shift_rel in _state.KIRCHHOFF_ILU_SHIFT_RELS:
                            shift_abs = float(max(shift_rel, 0.0)) * diag_ref
                            if shift_abs > 0.0:
                                A_ilu = A_gmres + (
                                    _sp.identity(A_gmres.shape[0], format="csr")
                                    * shift_abs
                                )
                            else:
                                A_ilu = A_gmres
                            t_ilu = perf_counter()
                            try:
                                ilu_try = _splinalg.spilu(
                                    A_ilu.tocsc(),
                                    drop_tol=float(_state.KIRCHHOFF_ILU_DROP_TOL),
                                    fill_factor=float(_state.KIRCHHOFF_ILU_FILL_FACTOR),
                                    permc_spec=str(permc_spec),
                                )
                                gmres_ilu_build_s = perf_counter() - t_ilu
                                ilu = ilu_try
                                used_permc = str(permc_spec)
                                used_shift = float(shift_abs)
                                print(
                                    f"Kirchhoff gmres_ilu: ilu_build={gmres_ilu_build_s:.3f}s "
                                    f"(starting GMRES) n={A.shape[0]} system={system_name} "
                                    f"permc={used_permc} shift={used_shift:.3e}"
                                )
                                if _state.KIRCHHOFF_DIAGNOSTICS:
                                    ilu_nnz = int(ilu.L.nnz + ilu.U.nnz)
                                    fill_ratio = float(
                                        ilu_nnz / max(int(A_gmres.nnz), 1)
                                    )
                                    print(
                                        f"Kirchhoff diagnostics: ilu_nnz={ilu_nnz} "
                                        f"fill_ratio={fill_ratio:.3f}"
                                    )
                                break
                            except Exception as exc:
                                attempt_reason = f"{type(exc).__name__}: {exc}"
                                if _state.KIRCHHOFF_DIAGNOSTICS:
                                    print(
                                        f"Kirchhoff diagnostics: ILU attempt failed "
                                        f"system={system_name} permc={permc_spec} "
                                        f"shift={shift_abs:.3e} reason={attempt_reason}"
                                    )
                        if ilu is not None:
                            break

                    if ilu is None:
                        gmres_failed_reason = (
                            f"ILU failed for system={system_name} "
                            f"after {len(_state.KIRCHHOFF_ILU_PERMC_SPECS) * len(_state.KIRCHHOFF_ILU_SHIFT_RELS)} attempts; "
                            f"last_error={attempt_reason}"
                        )
                        continue

                    M = _splinalg.LinearOperator(
                        A_gmres.shape, matvec=ilu.solve, dtype=float
                    )
                    gmres_resids: list[float] = []

                    def _gmres_callback(value: object, residuals=gmres_resids) -> None:
                        try:
                            residuals.append(float(value))
                        except Exception:
                            pass

                    t_gmres = perf_counter()
                    y, info = _splinalg.gmres(
                        A_gmres,
                        b_gmres,
                        M=M,
                        rtol=float(_state.KIRCHHOFF_GMRES_RTOL),
                        atol=0.0,
                        maxiter=int(_state.KIRCHHOFF_GMRES_MAXITER),
                        restart=int(_state.KIRCHHOFF_GMRES_RESTART),
                        callback=_gmres_callback,
                        callback_type="pr_norm",
                    )
                    gmres_solve_s = perf_counter() - t_gmres
                    gmres_iters = len(gmres_resids)
                    gmres_cycles = (
                        int(
                            math.ceil(
                                gmres_iters
                                / max(int(_state.KIRCHHOFF_GMRES_RESTART), 1)
                            )
                        )
                        if gmres_iters
                        else 0
                    )
                    if gmres_resids:
                        gmres_final_resid = float(gmres_resids[-1])
                    if info != 0 or y is None:
                        gmres_failed_reason = (
                            f"gmres returned info={info} "
                            f"on system={system_name} permc={used_permc} shift={used_shift:.3e}"
                        )
                        x = None
                        continue

                    y_arr = np.asarray(y, dtype=float).reshape(-1)
                    if equil_scale is not None:
                        x = equil_scale * y_arr
                    else:
                        x = y_arr
                    if not np.all(np.isfinite(x)):
                        gmres_failed_reason = (
                            f"gmres returned non-finite solution "
                            f"on system={system_name} permc={used_permc} shift={used_shift:.3e}"
                        )
                        x = None
                        continue

                    solver_used = "gmres_ilu"
                    break
            except Exception as exc:
                gmres_failed_reason = f"{type(exc).__name__}: {exc}"
                x = None
            status = "success" if x is not None else "failed"
            resid_text = (
                f"{gmres_final_resid:.3e}" if np.isfinite(gmres_final_resid) else "nan"
            )
            print(
                f"Kirchhoff gmres_ilu timings: gmres_solve={gmres_solve_s:.3f}s "
                f"iter={gmres_iters} cycles={gmres_cycles} final_pr_norm={resid_text} "
                f"status={status} n={A.shape[0]}"
            )

        if x is None and use_cg:
            try:
                diag = A.diagonal()
                inv_diag = np.divide(
                    1.0, diag, out=np.zeros_like(diag), where=diag != 0.0
                )
                M = _splinalg.LinearOperator(
                    A.shape, matvec=lambda z: inv_diag * z, dtype=float
                )
                x, info = _splinalg.cg(
                    A,
                    b,
                    M=M,
                    rtol=float(_state.KIRCHHOFF_CG_RTOL),
                    atol=0.0,
                    maxiter=int(_state.KIRCHHOFF_CG_MAXITER),
                )
                if info != 0 or x is None or not np.all(np.isfinite(x)):
                    cg_failed_reason = f"cg returned info={info}"
                    x = None
                else:
                    solver_used = "cg"
            except Exception as exc:
                cg_failed_reason = f"{type(exc).__name__}: {exc}"
                x = None

        if x is None:
            # Make solver fallback explicit in CLI output for large runs.
            if use_gmres_ilu and gmres_failed_reason is not None:
                print(
                    f"Kirchhoff fallback: gmres_ilu failed ({gmres_failed_reason}); "
                    f"using sparse LU (spsolve). n={A.shape[0]}"
                )
            elif use_cg and cg_failed_reason is not None:
                print(
                    f"Kirchhoff fallback: cg failed ({cg_failed_reason}); "
                    f"using sparse LU (spsolve). n={A.shape[0]}"
                )
            try:
                x = _splinalg.spsolve(A, b)
                solver_used = "spsolve"
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Failed to solve sparse Kirchhoff system. Check BCs and connectivity."
                ) from exc

        if _state.KIRCHHOFF_DIAGNOSTICS and x is not None:
            x_arr = np.asarray(x, dtype=float).reshape(-1)
            r = b - A.dot(x_arr)
            bnorm = float(np.linalg.norm(b))
            if bnorm > 0.0:
                rel_true_resid = float(np.linalg.norm(r) / bnorm)
            else:
                rel_true_resid = float(np.linalg.norm(r))
            print(
                f"Kirchhoff diagnostics: solver_used={solver_used} "
                f"bc={bc_mode} true_rel_resid={rel_true_resid:.3e}"
            )

        pressures = np.zeros(num_nodes, dtype=float)
        pressures[mask] = np.asarray(x, dtype=float).reshape(-1)
    else:
        # Dense fallback (only feasible for small problems).
        laplacian = np.zeros((num_nodes, num_nodes), dtype=float)
        for i in range(num_edges):
            ui = int(prox_ids[i])
            vi = int(dist_ids[i])
            gi = float(edge_conductance[i])
            laplacian[ui, ui] += gi
            laplacian[vi, vi] += gi
            laplacian[ui, vi] -= gi
            laplacian[vi, ui] -= gi

        mask = np.ones(num_nodes, dtype=bool)
        mask[dirichlet_mask] = False
        A = laplacian[np.ix_(mask, mask)]
        b = rhs[mask]
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Failed to solve Kirchhoff system. Check BCs and connectivity."
            ) from exc
        pressures = np.zeros(num_nodes, dtype=float)
        pressures[mask] = x

    flows = edge_conductance * (pressures[prox_ids] - pressures[dist_ids])
    return pressures, flows, prox_ids, dist_ids, np.empty((0, 3), dtype=float)


if _state._HAVE_NUMBA:

    @njit(cache=True)
    def _fixed_terminal_flows_numba(
        order: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        inlet_flow_cm3_s: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        nseg = int(left_child.shape[0])
        downstream_terms = np.zeros((nseg,), dtype=np.int64)
        for oi in range(order.shape[0] - 1, -1, -1):
            seg = int(order[oi])
            if seg < 0 or seg >= nseg:
                continue
            left = int(left_child[seg])
            right = int(right_child[seg])
            has_left = left >= 0 and left < nseg
            has_right = right >= 0 and right < nseg
            if not has_left and not has_right:
                downstream_terms[seg] = 1
            else:
                total = 0
                if has_left:
                    total += int(downstream_terms[left])
                if has_right:
                    total += int(downstream_terms[right])
                downstream_terms[seg] = total
        total_terms = 0
        for seg in range(nseg):
            left = int(left_child[seg])
            right = int(right_child[seg])
            if not (left >= 0 and left < nseg) and not (right >= 0 and right < nseg):
                total_terms += 1
        if total_terms <= 0:
            total_terms = 1
        terminal_flow = float(inlet_flow_cm3_s) / float(total_terms)
        flows = np.empty((nseg,), dtype=np.float64)
        for seg in range(nseg):
            flows[seg] = float(downstream_terms[seg]) * terminal_flow
        return flows, downstream_terms, total_terms

    @njit(cache=True)
    def _pressures_from_tree_flows_numba(
        order: np.ndarray,
        prox_ids: np.ndarray,
        dist_ids: np.ndarray,
        flows: np.ndarray,
        resistances: np.ndarray,
        root_node: int,
        root_pressure: float,
        num_nodes: int,
    ) -> tuple[np.ndarray, int]:
        pressures = np.empty((num_nodes,), dtype=np.float64)
        for node_i in range(num_nodes):
            pressures[node_i] = np.nan
        if root_node >= 0 and root_node < num_nodes:
            pressures[root_node] = root_pressure
        assigned_edges = 0
        for oi in range(order.shape[0]):
            seg = int(order[oi])
            if seg < 0 or seg >= flows.shape[0]:
                continue
            u = int(prox_ids[seg])
            v = int(dist_ids[seg])
            q = float(flows[seg])
            r = float(resistances[seg])
            if u >= 0 and u < num_nodes and v >= 0 and v < num_nodes:
                pu_known = np.isfinite(pressures[u])
                pv_known = np.isfinite(pressures[v])
                if pu_known and not pv_known:
                    pressures[v] = pressures[u] - q * r
                    assigned_edges += 1
                elif pv_known and not pu_known:
                    pressures[u] = pressures[v] + q * r
                    assigned_edges += 1
                elif pu_known and pv_known:
                    assigned_edges += 1
        return pressures, assigned_edges

    @njit(cache=True)
    def _kirchhoff_tree_neumann_numba(
        prox_ids: np.ndarray,
        dist_ids: np.ndarray,
        resistances: np.ndarray,
        rhs: np.ndarray,
        root_node: int,
        anchor_node: int,
        num_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        num_edges = prox_ids.shape[0]

        degree = np.zeros(num_nodes, dtype=np.int64)
        for edge_i in range(num_edges):
            degree[int(prox_ids[edge_i])] += 1
            degree[int(dist_ids[edge_i])] += 1

        offsets = np.empty(num_nodes + 1, dtype=np.int64)
        total = 0
        offsets[0] = 0
        for node_i in range(num_nodes):
            total += degree[node_i]
            offsets[node_i + 1] = total

        cursor = offsets[:-1].copy()
        adj_to = np.empty(total, dtype=np.int64)
        adj_edge = np.empty(total, dtype=np.int64)
        for edge_i in range(num_edges):
            u = int(prox_ids[edge_i])
            v = int(dist_ids[edge_i])
            pos = cursor[u]
            adj_to[pos] = v
            adj_edge[pos] = edge_i
            cursor[u] = pos + 1
            pos = cursor[v]
            adj_to[pos] = u
            adj_edge[pos] = edge_i
            cursor[v] = pos + 1

        parent_node = np.full(num_nodes, -2, dtype=np.int64)
        parent_edge = np.full(num_nodes, -1, dtype=np.int64)
        order = np.empty(num_nodes, dtype=np.int64)
        stack = np.empty(num_nodes, dtype=np.int64)

        top = 1
        stack[0] = root_node
        parent_node[root_node] = -1
        n_order = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            order[n_order] = node
            n_order += 1
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != -2:
                    continue
                parent_node[nbr] = node
                parent_edge[nbr] = int(adj_edge[pos])
                stack[top] = nbr
                top += 1

        subtree_rhs = rhs.copy()
        flows = np.zeros(num_edges, dtype=np.float64)
        for order_i in range(n_order - 1, 0, -1):
            node = int(order[order_i])
            edge_i = int(parent_edge[node])
            parent = int(parent_node[node])
            sub_rhs = subtree_rhs[node]
            q_parent_to_node = -sub_rhs
            if int(prox_ids[edge_i]) == parent and int(dist_ids[edge_i]) == node:
                flows[edge_i] = q_parent_to_node
            else:
                flows[edge_i] = -q_parent_to_node
            subtree_rhs[parent] += sub_rhs

        pressures = np.empty(num_nodes, dtype=np.float64)
        visited = np.zeros(num_nodes, dtype=np.uint8)
        for node_i in range(num_nodes):
            pressures[node_i] = np.nan

        top = 1
        stack[0] = anchor_node
        visited[anchor_node] = 1
        pressures[anchor_node] = 0.0
        n_pressure = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            n_pressure += 1
            p_node = pressures[node]
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if visited[nbr] != 0:
                    continue
                edge_i = int(adj_edge[pos])
                if int(prox_ids[edge_i]) == node and int(dist_ids[edge_i]) == nbr:
                    pressures[nbr] = p_node - flows[edge_i] * resistances[edge_i]
                else:
                    pressures[nbr] = p_node + flows[edge_i] * resistances[edge_i]
                visited[nbr] = 1
                stack[top] = nbr
                top += 1

        return pressures, flows, n_order, n_pressure

    @njit(cache=True)
    def _kirchhoff_tree_mixed_numba(
        prox_ids: np.ndarray,
        dist_ids: np.ndarray,
        resistances: np.ndarray,
        outlet_nodes: np.ndarray,
        root_node: int,
        inlet_flow_cm3_s: float,
        num_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        num_edges = prox_ids.shape[0]

        degree = np.zeros(num_nodes, dtype=np.int64)
        for edge_i in range(num_edges):
            degree[int(prox_ids[edge_i])] += 1
            degree[int(dist_ids[edge_i])] += 1

        offsets = np.empty(num_nodes + 1, dtype=np.int64)
        total = 0
        offsets[0] = 0
        for node_i in range(num_nodes):
            total += degree[node_i]
            offsets[node_i + 1] = total

        cursor = offsets[:-1].copy()
        adj_to = np.empty(total, dtype=np.int64)
        adj_edge = np.empty(total, dtype=np.int64)
        for edge_i in range(num_edges):
            u = int(prox_ids[edge_i])
            v = int(dist_ids[edge_i])
            pos = cursor[u]
            adj_to[pos] = v
            adj_edge[pos] = edge_i
            cursor[u] = pos + 1
            pos = cursor[v]
            adj_to[pos] = u
            adj_edge[pos] = edge_i
            cursor[v] = pos + 1

        outlet_mask = np.zeros(num_nodes, dtype=np.uint8)
        for i in range(outlet_nodes.shape[0]):
            node = int(outlet_nodes[i])
            if 0 <= node < num_nodes:
                outlet_mask[node] = 1

        parent_node = np.full(num_nodes, -2, dtype=np.int64)
        parent_edge = np.full(num_nodes, -1, dtype=np.int64)
        order = np.empty(num_nodes, dtype=np.int64)
        stack = np.empty(num_nodes, dtype=np.int64)

        top = 1
        stack[0] = root_node
        parent_node[root_node] = -1
        n_order = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            order[n_order] = node
            n_order += 1
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != -2:
                    continue
                parent_node[nbr] = node
                parent_edge[nbr] = int(adj_edge[pos])
                stack[top] = nbr
                top += 1

        eq_g = np.zeros(num_nodes, dtype=np.float64)
        branch_g = np.zeros(num_edges, dtype=np.float64)
        for order_i in range(n_order - 1, -1, -1):
            node = int(order[order_i])
            if outlet_mask[node] != 0:
                eq_g[node] = 0.0
                continue
            gsum = 0.0
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != node:
                    continue
                edge_i = int(adj_edge[pos])
                r_edge = resistances[edge_i]
                if r_edge < 1.0e-300:
                    r_edge = 1.0e-300
                if outlet_mask[nbr] != 0:
                    bg = 1.0 / r_edge
                else:
                    child_g = eq_g[nbr]
                    if child_g > 0.0:
                        bg = 1.0 / (r_edge + 1.0 / child_g)
                    else:
                        bg = 0.0
                branch_g[edge_i] = bg
                gsum += bg
            eq_g[node] = gsum

        pressures = np.empty(num_nodes, dtype=np.float64)
        flows = np.zeros(num_edges, dtype=np.float64)
        visited = np.zeros(num_nodes, dtype=np.uint8)
        for node_i in range(num_nodes):
            pressures[node_i] = np.nan

        root_g = eq_g[root_node]
        if root_g <= 0.0:
            return pressures, flows, n_order, 0
        pressures[root_node] = inlet_flow_cm3_s / root_g
        visited[root_node] = 1
        top = 1
        stack[0] = root_node
        n_pressure = 0
        while top > 0:
            top -= 1
            node = int(stack[top])
            n_pressure += 1
            p_node = pressures[node]
            for pos in range(offsets[node], offsets[node + 1]):
                nbr = int(adj_to[pos])
                if parent_node[nbr] != node:
                    continue
                edge_i = int(adj_edge[pos])
                q_parent_to_child = branch_g[edge_i] * p_node
                if int(prox_ids[edge_i]) == node and int(dist_ids[edge_i]) == nbr:
                    flows[edge_i] = q_parent_to_child
                else:
                    flows[edge_i] = -q_parent_to_child
                child_p = p_node - q_parent_to_child * resistances[edge_i]
                if outlet_mask[nbr] != 0:
                    child_p = 0.0
                pressures[nbr] = child_p
                visited[nbr] = 1
                stack[top] = nbr
                top += 1

        return pressures, flows, n_order, n_pressure


def solve_kirchhoff_dirichlet(
    starts: np.ndarray,
    ends: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_pressure: float,
    outlet_nodes: Sequence[int],
    outlet_flow_per_node: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geom = np.zeros((starts.shape[0], 6), dtype=float)
    geom[:, 0:3] = starts
    geom[:, 3:6] = ends
    prox_ids, dist_ids, nodes = _build_node_indices(geom)

    num_edges = starts.shape[0]
    num_nodes = int(max(prox_ids.max(), dist_ids.max()) + 1) if num_edges > 0 else 0
    if num_nodes == 0:
        raise ValueError("No segments provided for Kirchhoff solve.")
    if not inlet_nodes:
        raise ValueError("No inlet nodes provided for Dirichlet solve.")

    edge_conductance = np.zeros(num_edges, dtype=float)
    edge_conductance[:] = 1.0 / np.maximum(resistances, 1e-30)

    rhs = np.zeros(num_nodes, dtype=float)
    if outlet_nodes:
        sink_share = float(outlet_flow_per_node)
        for node in outlet_nodes:
            rhs[int(node)] -= sink_share

    if _state._HAVE_SCIPY_SPARSE:
        rows = np.concatenate([prox_ids, dist_ids])
        cols = np.concatenate([np.arange(num_edges), np.arange(num_edges)])
        data = np.concatenate([np.ones(num_edges), -np.ones(num_edges)])
        B = _sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_edges)).tocsr()
        G = _sp.diags(edge_conductance, 0, shape=(num_edges, num_edges), format="csr")
        laplacian = (B @ G @ B.T).tolil()

        for node in inlet_nodes:
            prescribed_pressure = float(inlet_pressure)
            column = np.asarray(laplacian[:, node].toarray(), dtype=float).reshape(-1)
            rhs -= column * prescribed_pressure
            laplacian[:, node] = 0.0
            laplacian[node, :] = 0.0
            laplacian[node, node] = 1.0
            rhs[node] = prescribed_pressure

        try:
            pressures = _splinalg.spsolve(laplacian.tocsr(), rhs)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to solve sparse Kirchhoff system. Check BCs and connectivity."
            ) from exc
    else:
        B = np.zeros((num_nodes, num_edges), dtype=float)
        for idx in range(num_edges):
            B[prox_ids[idx], idx] = 1.0
            B[dist_ids[idx], idx] = -1.0

        BG = B * edge_conductance
        laplacian = BG @ B.T

        for node in inlet_nodes:
            prescribed_pressure = float(inlet_pressure)
            rhs -= laplacian[:, node] * prescribed_pressure
            laplacian[:, node] = 0.0
            laplacian[node, :] = 0.0
            laplacian[node, node] = 1.0
            rhs[node] = prescribed_pressure

        try:
            pressures = np.linalg.solve(laplacian, rhs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Failed to solve Kirchhoff system. Check BCs and connectivity."
            ) from exc

    flows = edge_conductance * (pressures[prox_ids] - pressures[dist_ids])
    return pressures, flows, prox_ids, dist_ids, nodes


def _solve_pressure_dirichlet(
    prox_ids: np.ndarray,
    dist_ids: np.ndarray,
    resistances: np.ndarray,
    inlet_nodes: Sequence[int],
    inlet_pressure: float,
    outlet_nodes: Sequence[int],
    outlet_pressure: float,
    *,
    num_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Impose inlet/outlet pressures and solve all unconstrained node pressures."""

    prox_ids = np.asarray(prox_ids, dtype=np.int64).reshape(-1)
    dist_ids = np.asarray(dist_ids, dtype=np.int64).reshape(-1)
    resistances = np.asarray(resistances, dtype=float).reshape(-1)
    if not (prox_ids.size == dist_ids.size == resistances.size) or prox_ids.size == 0:
        raise ValueError("Pressure topology and resistance arrays must be nonempty and equal length.")
    if num_nodes is None:
        num_nodes = int(max(int(prox_ids.max()), int(dist_ids.max())) + 1)
    num_nodes = int(num_nodes)

    inlet_arr = np.asarray(list(inlet_nodes), dtype=np.int64).reshape(-1)
    outlet_arr = np.asarray(list(outlet_nodes), dtype=np.int64).reshape(-1)
    boundary_nodes = np.concatenate([inlet_arr, outlet_arr])
    boundary_values = np.concatenate(
        [
            np.full(inlet_arr.size, float(inlet_pressure), dtype=float),
            np.full(outlet_arr.size, float(outlet_pressure), dtype=float),
        ]
    )
    pressures = np.empty(num_nodes, dtype=float)
    pressures.fill(np.nan)
    pressures[boundary_nodes] = boundary_values
    boundary_mask = np.zeros(num_nodes, dtype=bool)
    boundary_mask[boundary_nodes] = True
    free = np.flatnonzero(~boundary_mask)
    conductance = 1.0 / resistances

    if _state._HAVE_SCIPY_SPARSE:
        rows = np.concatenate([prox_ids, dist_ids, prox_ids, dist_ids])
        cols = np.concatenate([prox_ids, dist_ids, dist_ids, prox_ids])
        data = np.concatenate([conductance, conductance, -conductance, -conductance])
        laplacian = _sp.coo_matrix(
            (data, (rows, cols)), shape=(num_nodes, num_nodes)
        ).tocsr()
        if free.size:
            system = laplacian[free][:, free].tocsr()
            rhs = -(laplacian[free][:, boundary_nodes] @ boundary_values)
            try:
                pressures[free] = _splinalg.spsolve(system, rhs)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Failed to solve pressure-pressure Kirchhoff system. "
                    "Check boundary nodes and connectivity."
                ) from exc
    else:
        laplacian = np.zeros((num_nodes, num_nodes), dtype=float)
        np.add.at(laplacian, (prox_ids, prox_ids), conductance)
        np.add.at(laplacian, (dist_ids, dist_ids), conductance)
        np.add.at(laplacian, (prox_ids, dist_ids), -conductance)
        np.add.at(laplacian, (dist_ids, prox_ids), -conductance)
        if free.size:
            system = laplacian[np.ix_(free, free)]
            rhs = -(laplacian[np.ix_(free, boundary_nodes)] @ boundary_values)
            try:
                pressures[free] = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "Failed to solve pressure-pressure Kirchhoff system. "
                    "Check boundary nodes and connectivity."
                ) from exc

    if not np.all(np.isfinite(pressures)):
        raise RuntimeError(
            "Pressure-pressure Kirchhoff solve is singular. Every connected "
            "component must contain an inlet or outlet pressure boundary."
        )
    flows = conductance * (pressures[prox_ids] - pressures[dist_ids])
    return pressures, flows
