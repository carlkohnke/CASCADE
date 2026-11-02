"""CUDA graph replays must only touch memory owned alongside the graph."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.configuration import solver_state as state


def cuda():
    from cascade.concentration.vessel.network_gpu import resolve_network_accel

    if resolve_network_accel("auto") != "gpu":
        pytest.skip("CUDA/CuPy unavailable")


def released_pool_sentinels():
    """Return the pool to the driver, then refill it with known values."""
    cp = state._cp
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    # Sizes span the pool bins of small reduction and temporary scratch.
    sentinels = [cp.full(size, 7.0) for size in (64, 512, 1536, 4096) * 500]
    cp.cuda.Device().synchronize()
    return sentinels


def assert_untouched(sentinels):
    cp = state._cp
    cp.cuda.Device().synchronize()
    assert sum(int(cp.count_nonzero(s != 7.0)) for s in sentinels) == 0


def test_graph_capture_refuses_pool_allocations():
    cuda()
    from cascade.accelerators.graph_capture import capture_graph

    cp = state._cp
    a = cp.arange(100_000, dtype=cp.float64)
    out = cp.empty((), cp.float64)
    cp.sum(a, out=out)
    assert capture_graph(lambda: cp.sum(a, out=out)) is None
    graph = capture_graph(lambda: cp.multiply(a, 2.0, out=a))
    assert graph is not None
    graph.launch()
    np.testing.assert_array_equal(cp.asnumpy(a), 2.0 * np.arange(100_000))


@pytest.mark.parametrize("captured_krylov", [True, False])
def test_gpu_cached_pressure_graphs_survive_pool_release(captured_krylov):
    cuda()
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    from cascade.flow.gpu import GPUAMGSolver

    rng = np.random.default_rng(23)
    size = 4096
    g = 10.0 ** rng.uniform(-2, 2, size)
    a = sp.diags(
        (-g[1:], g + np.roll(g, -1) + 0.01, -g[1:]), (-1, 0, 1), shape=(size, size)
    ).tocsr()
    a.setdiag(np.asarray(abs(a).sum(axis=1)).ravel() - a.diagonal() + 0.01)
    rhs = rng.normal(size=size)
    expected = spla.spsolve(a, rhs)
    solver = GPUAMGSolver(a, capture_krylov=captured_krylov)
    np.testing.assert_allclose(
        solver.solve(rhs, max_iterations=999), expected, rtol=1e-7, atol=1e-8
    )
    graph = solver.krylov_graph if captured_krylov else solver.cycle_graph
    assert graph is not None and graph is not False
    sentinels = released_pool_sentinels()
    # Cold start so the cached graph is replayed for many iterations.
    solver.previous_solution = None
    result = solver.solve(rhs, max_iterations=999)
    assert (solver.krylov_graph if captured_krylov else solver.cycle_graph) is graph
    assert_untouched(sentinels)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("closure", ["wellmixed", "graetz"])
def test_gpu_cached_oxygen_sweep_survives_pool_release(monkeypatch, closure):
    cuda()
    from cascade.concentration.vessel.network_gpu import solve_network_ext_gpu

    monkeypatch.setattr(state, "LUMEN_WALL_CLOSURE", closure)
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    ctx = {
        "network_topology": {
            "prox_ids": np.array([0, 1, 2, 3, 3]),
            "dist_ids": np.array([2, 2, 3, 4, 5]),
            "inlet_nodes": [0, 1],
        },
        "flows_si": np.array([1.0, 1.0, 2.0, 1.5, 0.5]) * 1e-12,
        "radii_si": np.full(5, 5e-6),
        "lengths_si": np.array([2e-4, 1e-4, 3e-4, 2e-4, 1e-4]),
        "gl_t": np.linspace(0.05, 0.95, 5),
        "diffusivity_si": 2.41e-9,
    }
    kw = {
        "inlet_concentration": 0.14,
        "vmax": 0.04,
        "km": 0.0069,
        "chb_max": np.full(5, 9.0),
        "fluid_mode": "blood",
    }
    ext = {"c_ext_gl": np.full((5, 5), 0.01, np.float32)}
    expected = solve_network_ext_gpu(ctx, ext, **kw)
    graph = ctx["network_gpu_graph"]["sweep_graph"]
    assert state._LAST_CEXT_FROZEN_STEP_TIMINGS["network_captured_sweep"]
    sentinels = released_pool_sentinels()
    actual = solve_network_ext_gpu(ctx, ext, **kw)
    assert ctx["network_gpu_graph"]["sweep_graph"] is graph
    assert_untouched(sentinels)
    for a, b in zip(actual[:3], expected[:3]):
        np.testing.assert_array_equal(a, b)
