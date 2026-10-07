"""Device-resident assembly and pressure updates for blood feedback."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse as sp

from cascade.configuration import solver_state as state

from .gpu import GPUAMGSolver, LAST_GPU_FLOW_TIMINGS
from .topology import _normalize_kirchhoff_bc_mode


_CONTEXT = {}


def solve_device_pressure(
    up, down, resistance, inlets, outlets, inlet_flow, *, pressure_boundary=None
):
    cp = state._cp
    from cupyx.scipy import sparse as sparse_gpu

    up, down = np.asarray(up, dtype=np.int64), np.asarray(down, dtype=np.int64)
    n = int(max(up.max(), down.max()) + 1)
    boundary = np.zeros(n, dtype=bool)
    prescribed = cp.zeros(n, dtype=cp.float64)
    rhs = cp.zeros(n, dtype=cp.float64)
    if pressure_boundary is not None:
        boundary[list(inlets) + list(outlets)] = True
        prescribed[list(inlets)] = pressure_boundary[0]
        prescribed[list(outlets)] = pressure_boundary[1]
    else:
        rhs[list(inlets)] = inlet_flow / len(inlets)
        mode = _normalize_kirchhoff_bc_mode(state.KIRCHHOFF_BC_MODE)
        if mode == "legacy_equal_terminal_flow":
            rhs[list(outlets)] -= inlet_flow / len(outlets)
            anchor = n - 1
            if anchor in inlets or anchor in outlets:
                anchor = max(n - 2, 0)
            boundary[anchor] = True
        else:
            boundary[list(outlets)] = True
    free = ~boundary
    context = _CONTEXT
    reused = (
        bool(context)
        and all(
            np.array_equal(context[key], value)
            for key, value in (("up", up), ("down", down), ("free", free))
        )
        and context["device"] == int(cp.cuda.Device().id)
    )
    started = perf_counter()
    g = 1.0 / cp.maximum(cp.asarray(resistance, dtype=cp.float64), 1e-30)
    if not np.any(free):
        flows = g * (prescribed[cp.asarray(up)] - prescribed[cp.asarray(down)])
        LAST_GPU_FLOW_TIMINGS.update(
            assembly_s=perf_counter() - started,
            setup_s=0.0,
            solve_s=0.0,
            iterations=0,
            backend="gpu-amg",
            flow_balance_relative_residual=0.0,
        )
        return prescribed, flows
    if not reused:
        context.clear()
        free_ids = np.full(n, -1, dtype=np.int64)
        free_ids[free] = np.arange(free.sum())
        rows = free_ids[np.r_[up, down, up, down]]
        cols = free_ids[np.r_[up, down, down, up]]
        valid = (rows >= 0) & (cols >= 0)
        edge_ids = np.tile(np.arange(len(up)), 4)[valid]
        signs = np.repeat([1.0, 1.0, -1.0, -1.0], len(up))[valid]
        rows, cols = rows[valid], cols[valid]
        pattern = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(int(free.sum()),) * 2
        ).tocsr()
        pattern_rows = np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr))
        positions = np.searchsorted(
            pattern_rows * pattern.shape[0] + pattern.indices,
            rows * pattern.shape[0] + cols,
        )
        context.update(
            up=up.copy(),
            down=down.copy(),
            free=free.copy(),
            device=int(cp.cuda.Device().id),
            up_g=cp.asarray(up),
            down_g=cp.asarray(down),
            free_g=cp.asarray(free),
            positions=cp.asarray(positions),
            edge_ids=cp.asarray(edge_ids),
            signs=cp.asarray(signs),
            indptr=cp.asarray(pattern.indptr),
            indices=cp.asarray(pattern.indices),
            pattern=pattern,
        )
        # Initial CPU hierarchy preparation also supplies a deterministic disk-cache key.
        pattern.data = np.bincount(
            positions, weights=signs * cp.asnumpy(g)[edge_ids], minlength=pattern.nnz
        )
        context["solver"] = GPUAMGSolver(pattern)
    changed = reused and not bool(cp.array_equal(context["conductance"], g))
    if not reused or changed:
        data = cp.bincount(
            context["positions"],
            weights=context["signs"] * g[context["edge_ids"]],
            minlength=context["pattern"].nnz,
        )
        context["matrix"] = sparse_gpu.csr_matrix(
            (data, context["indices"], context["indptr"]),
            shape=context["pattern"].shape,
        )
        context["conductance"] = g.copy()
    matrix = context["matrix"]
    up_g, down_g, free_g = context["up_g"], context["down_g"], context["free_g"]
    if pressure_boundary is not None:
        rhs += cp.bincount(up_g, weights=g * prescribed[down_g], minlength=n)
        rhs += cp.bincount(down_g, weights=g * prescribed[up_g], minlength=n)
    LAST_GPU_FLOW_TIMINGS.update(
        assembly_s=perf_counter() - started,
        assembly_topology_reused=reused,
        hierarchy_reused=reused,
        hierarchy_values_updated=changed,
    )
    solver = context["solver"]
    if changed:
        solver.update(matrix)
    elif reused:
        solver.last_setup_seconds = 0.0
    pressure = prescribed
    if np.any(free):
        pressure[free_g] = solver.solve(rhs[free_g], return_device=True)
    flows = g * (pressure[up_g] - pressure[down_g])
    balance = cp.bincount(up_g, weights=flows, minlength=n) - cp.bincount(
        down_g, weights=flows, minlength=n
    )
    target = cp.zeros_like(rhs) if pressure_boundary is not None else rhs
    norm = cp.maximum(cp.linalg.norm(rhs[free_g]), 1e-30)
    rel = float((cp.linalg.norm((balance - target)[free_g]) / norm).item())
    LAST_GPU_FLOW_TIMINGS["flow_balance_relative_residual"] = rel
    if not np.isfinite(rel) or rel > 1e-8:
        raise RuntimeError(f"GPU device flow balance failed: residual={rel:.3e}")
    return pressure, flows
