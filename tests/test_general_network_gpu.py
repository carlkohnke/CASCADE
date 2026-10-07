"""CUDA regression checks on graphs with mergers, splits, and anatomical loops."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.configuration import solver_state as state
from cascade.flow.graph_schedule import graph_schedule


def test_graph_schedule_detects_cycles_and_handles_mergers():
    assert graph_schedule(np.array([0, 1, 2]), np.array([1, 2, 0]), np.ones(3)) is None
    result = graph_schedule(
        np.array([0, 0, 1, 2, 3]), np.array([1, 2, 3, 3, 4]), np.ones(5)
    )
    nodes, offsets, edges, edge_offsets = result
    assert offsets.tolist() == [0, 1, 3, 4, 5]
    assert nodes.tolist() == [0, 1, 2, 3, 4]
    assert len(edges) == edge_offsets[-1] == 5


def test_gpu_supplied_circulation_uses_converged_fixed_point(monkeypatch):
    cuda()
    from cascade.concentration.vessel.network import solve_network_concentrations

    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", "discharge")
    kw = {
        "starts": np.zeros((4, 3)),
        "ends": np.ones((4, 3)),
        "radii": np.full(4, 0.001),
        "lengths": np.array([0.02, 0.05, 0.03, 0.04]),
        "flows": np.array([1.0, 2.0, 1.0, 1.0]) * 1e-4,
        "inlet_nodes": [0],
        "outlet_nodes": [3],
        "inlet_concentration": 0.14,
        "fluid": "blood",
        "prox_ids": np.array([0, 1, 2, 2]),
        "dist_ids": np.array([1, 2, 1, 3]),
        "discharge_hematocrit": np.full(4, 0.42),
        "max_iter": 500,
        "tol": 1e-9,
    }
    a = solve_network_concentrations(**kw, accel="cpu")
    b = solve_network_concentrations(**kw, accel="gpu")
    np.testing.assert_allclose(b[1], a[1], rtol=1e-6, atol=1e-9)
    assert state._LAST_NETWORK_GPU_TIMINGS["schedule"] == "cyclic-fixed-point"
    assert state._LAST_NETWORK_GPU_TIMINGS["converged"]
    assert len(b[3]["M_in"]) == len(b[3]["iter"])
    assert b[3]["total_oxygen_junction_rel_resid"][-1] < 1e-8


def cuda():
    from cascade.concentration.vessel.network_gpu import resolve_network_accel

    if resolve_network_accel("auto") != "gpu":
        pytest.skip("CUDA/CuPy unavailable")


def test_network_two_hop_exclusions_match_python_reference():
    from cascade.concentration.external_field.state import (
        _build_network_local_exclusion_lists,
        _build_network_local_exclusion_lists_python,
    )

    rng = np.random.default_rng(42)
    up, down = rng.integers(0, 30, (2, 200))
    # Include parallel edges, self-loops and a high-degree hub.
    up[:80] = 0
    for limit in (0, 5, 32):
        a = _build_network_local_exclusion_lists(up, down, max_local=limit)
        b = _build_network_local_exclusion_lists_python(up, down, max_local=limit)
        for actual, expected in zip(a, b):
            np.testing.assert_array_equal(actual, expected)


def test_gpu_hematocrit_cache_refreshes_weights_and_dependencies():
    cuda()
    from cascade.flow.hematocrit_gpu import compute_network_hematocrit_gpu

    up, down = np.array([0, 0, 1, 2, 3, 3]), np.array([1, 2, 3, 3, 4, 5])
    radius = np.full(6, 0.0005)
    context = {}
    q = np.array([0.7, 0.3, 0.7, 0.3, 0.4, 0.6])
    compute_network_hematocrit_gpu(
        up, down, q, radius, model="pries_secomb", context=context
    )
    schedule = context["gpu_hematocrit"]
    q = np.array([0.6, 0.4, 0.6, 0.4, 0.2, 0.8])
    actual = compute_network_hematocrit_gpu(
        up, down, q, radius, model="pries_secomb", context=context
    )
    expected = compute_network_hematocrit_gpu(up, down, q, radius, model="pries_secomb")
    assert context["gpu_hematocrit"] is schedule
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-13)
    actual = compute_network_hematocrit_gpu(
        up, down, -q, radius, model="pries_secomb", context=context
    )
    expected = compute_network_hematocrit_gpu(
        up, down, -q, radius, model="pries_secomb"
    )
    assert context["gpu_hematocrit"] is not schedule
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-13)


def test_gpu_cached_assembly_handles_parallel_edges_and_boundary_changes():
    cuda()
    from cascade.flow.kirchhoff import solve_kirchhoff

    up, down = np.array([0, 0, 0, 1, 2]), np.array([1, 1, 2, 3, 3])
    for resistance, bc, outlets in (
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "terminal_pressure", [3]),
        (np.array([2.0, 1.0, 5.0, 3.0, 4.0]), "terminal_pressure", [3]),
        (np.array([2.0, 1.0, 5.0, 3.0, 4.0]), "terminal_pressure", [2, 3]),
        (np.array([2.0, 1.0, 5.0, 3.0, 4.0]), "legacy_equal_terminal_flow", [3]),
    ):
        args = (up, down, resistance, [0], 0.015, outlets)
        actual = solve_kirchhoff(*args, solver_mode="gpu_amg", boundary_condition=bc)
        expected = solve_kirchhoff(*args, solver_mode="spsolve", boundary_condition=bc)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-8, atol=1e-13)


@pytest.mark.parametrize("lambda_source", ["lambda_t", "lambda_if"])
def test_gpu_complete_source_handoff_matches_cpu(monkeypatch, lambda_source):
    cuda()
    from cascade.concentration.external_field.direct import _ensure_cext_gpu_static
    from cascade.concentration.external_field.geometry import (
        _build_cext_geometry_context,
    )
    from cascade.concentration.external_field.state import (
        _build_cext_iteration_cache,
        _build_cext_iteration_cache_gpu,
        validate_cext_tissue_flux_consistency,
    )

    monkeypatch.setattr(state, "GL_ORDER_CEXT", 5)
    monkeypatch.setattr(state, "CEXT_LAMBDA_SOURCE", lambda_source)
    up, down = np.array([0, 1, 2]), np.array([2, 2, 3])
    starts = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]])
    ends = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    context = _build_cext_geometry_context(
        None,
        np.array([1.0, 1.0, 2.0]) * 1e-4,
        starts,
        ends,
        np.full(3, 0.001),
        np.linalg.norm(ends - starts, axis=1),
        inlet_concentration=0.14,
        diffusivity=2.41e-5,
        vmax=0.04,
        km=0.0069,
        network_topology={"prox_ids": up, "dist_ids": down, "inlet_nodes": [0, 1]},
        build_candidate_index=False,
    )
    bulk = np.linspace(0.0001, 0.14, 15).reshape(3, 5).astype(np.float32)
    inputs = {
        "c_iv_gl": bulk,
        "c_wall_gl": bulk * 0.9,
        "c_ext_gl": bulk * 0.2,
        "vmax": 0.04,
        "km": 0.0069,
    }
    expected = dict(inputs)
    _build_cext_iteration_cache(context, expected)
    cp = state._cp
    runtime = {
        name: cp.empty((3, 5), cp.float32)
        for name in (
            "q_weighted_gl",
            "mono2_weight_gl",
            "dipole2_weight_gl",
            "lambda_iv_gl",
            "c_ext_gl",
        )
    }
    runtime["seg_cap_gl"] = cp.empty(3, cp.float32)
    _ensure_cext_gpu_static(context)
    actual = dict(inputs)
    _build_cext_iteration_cache_gpu(context, actual, runtime, materialize=True)
    for name in (
        "lambda_iv_gl",
        "lambda_if_gl",
        "k_if_gl",
        "q_line_gl",
        "q_weighted_gl",
        "mono2_weight_gl",
        "dipole2_weight_gl",
        "seg_cap_gl",
    ):
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-6, atol=1e-18)
    assert validate_cext_tissue_flux_consistency(actual)["ok"]
    # Reset only the numerical CPU field while retaining the runtime workspace.
    # Intermediate updates must not consume the previous solve's GPU field.
    runtime["c_ext_gl"].fill(0.13)
    actual["c_ext_gl"] = np.zeros_like(bulk)
    reference = dict(actual)
    _build_cext_iteration_cache(context, reference)
    _build_cext_iteration_cache_gpu(context, actual, runtime)
    for name in ("q_weighted_gl", "lambda_iv_gl", "dipole2_weight_gl"):
        np.testing.assert_allclose(
            cp.asnumpy(runtime[name]), reference[name], rtol=3e-6, atol=1e-18
        )


def test_gpu_captured_oxygen_updates_all_physical_inputs(monkeypatch):
    cuda()
    from cascade.concentration.external_field.coupling_steps import (
        _run_network_ext_frozen_step,
    )
    from cascade.concentration.vessel.network_gpu import solve_network_ext_gpu

    monkeypatch.setattr(state, "LUMEN_WALL_CLOSURE", "wellmixed")
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    ctx = {
        "network_topology": {
            "prox_ids": np.array([0, 1, 2]),
            "dist_ids": np.array([2, 2, 3]),
            "inlet_nodes": [0, 1],
        },
        "flows_si": np.array([1.0, 1.0, 2.0]) * 1e-9,
        "radii_si": np.full(3, 1e-5),
        "lengths_si": np.array([0.001, 0.01, 0.0001]),
        "gl_t": np.linspace(0.05, 0.95, 5),
        "diffusivity_si": 2.41e-9,
    }
    ext = {"c_ext_gl": np.zeros((3, 5), np.float32)}
    cached_graph = None
    for index in range(3):
        kw = {
            "inlet_concentration": 0.14 - index * 0.01,
            "vmax": 0.04,
            "km": 0.0069,
            "chb_max": np.array([8.0, 10.0, 9.0]) + index,
            "fluid_mode": "blood",
        }
        if index == 2:
            ctx["flows_si"] *= 1.3
        ext["c_ext_gl"][:] = index * 0.001
        actual = solve_network_ext_gpu(ctx, ext, **kw)
        if cached_graph is None:
            cached_graph = ctx["network_gpu_graph"]
        else:
            assert ctx["network_gpu_graph"] is cached_graph
        expected = _run_network_ext_frozen_step(
            ctx, {"c_ext_gl": ext["c_ext_gl"]}, **kw
        )
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_allclose(a, b, rtol=2e-4, atol=3e-6)


def test_gpu_captured_cg_handles_nonmultiple_iteration_limit_and_matrix_update():
    cuda()
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    from cascade.flow.gpu import GPUAMGSolver

    # Multiple AMG levels, an irregular coefficient field, and nontrivial RHS.
    rng = np.random.default_rng(18)
    g = 10.0 ** rng.uniform(-2, 2, 256)
    a = sp.diags(
        (-g[1:], g + np.roll(g, -1) + 0.01, -g[1:]), (-1, 0, 1), shape=(256, 256)
    ).tocsr()
    # Strict diagonal dominance guarantees SPD for this independent fixture.
    a.setdiag(np.asarray(abs(a).sum(axis=1)).ravel() - a.diagonal() + 0.01)
    solver = GPUAMGSolver(a)
    rhs = rng.normal(size=256)
    result = solver.solve(rhs, max_iterations=999)
    np.testing.assert_allclose(result, spla.spsolve(a, rhs), rtol=1e-7, atol=1e-8)
    assert solver.krylov_graph is not None
    a = a + sp.diags(np.full(256, 0.1))
    solver.update(a)
    assert solver.krylov_graph is None
    result = solver.solve(rhs, max_iterations=999)
    np.testing.assert_allclose(result, spla.spsolve(a, rhs), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("bc", ["legacy_equal_terminal_flow", "terminal_pressure"])
def test_gpu_pressure_matches_cpu_on_loopy_graph(bc):
    cuda()
    from cascade.flow.kirchhoff import solve_kirchhoff

    up, down = np.array([0, 0, 1, 2, 1]), np.array([1, 2, 3, 3, 2])
    args = (up, down, np.array([1.0, 2.0, 3.0, 4.0, 2.0]), [0], 0.015, [3])
    cpu = solve_kirchhoff(*args, solver_mode="spsolve", boundary_condition=bc)
    gpu = solve_kirchhoff(*args, solver_mode="gpu_amg", boundary_condition=bc)
    np.testing.assert_allclose(gpu[1], cpu[1], rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("fluid", ["blood", "water"])
@pytest.mark.parametrize("balance", ["dissolved", "total_content"])
@pytest.mark.parametrize("reverse", [False, True])
def test_gpu_standalone_matches_cpu(monkeypatch, fluid, balance, reverse):
    cuda()
    from cascade.concentration.vessel.network import solve_network_concentrations

    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", balance)
    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", "discharge")
    up, down = np.array([0, 0, 1, 2, 3]), np.array([1, 2, 3, 3, 4])
    q = np.array([1.0, 1.0, 1.0, 1.0, 2.0]) * 1e-4
    hd = np.array([0.3, 0.5, 0.3, 0.5, 0.4])
    if reverse:
        up, down, q = down, up, -q
    kw = {
        "starts": np.zeros((5, 3)),
        "ends": np.ones((5, 3)),
        "radii": np.full(5, 0.001),
        "lengths": np.array([0.03, 0.2, 0.02, 0.01, 0.04]),
        "flows": q,
        "inlet_nodes": [0],
        "outlet_nodes": [4],
        "inlet_concentration": 0.14,
        "prox_ids": up,
        "dist_ids": down,
        "fluid": fluid,
        "discharge_hematocrit": hd,
        "tol": 1e-10,
        "max_iter": 500,
    }
    cpu = solve_network_concentrations(**kw, accel="cpu")
    gpu = solve_network_concentrations(**kw, accel="gpu")
    for a, b in zip(cpu[:3], gpu[:3]):
        np.testing.assert_allclose(a, b, rtol=2e-7, atol=2e-9)
    if balance == "total_content":
        assert gpu[3]["total_oxygen_junction_rel_resid"][-1] < 1e-10


@pytest.mark.parametrize("closure", ["wellmixed", "graetz"])
def test_gpu_coupled_merger_matches_cpu(monkeypatch, closure):
    cuda()
    from cascade.concentration.external_field.coupling_steps import (
        _run_network_ext_frozen_step,
    )
    from cascade.concentration.vessel.network_gpu import solve_network_ext_gpu

    monkeypatch.setattr(state, "LUMEN_WALL_CLOSURE", closure)
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    ctx = {
        "network_topology": {
            "prox_ids": np.array([0, 1, 2]),
            "dist_ids": np.array([2, 2, 3]),
            "inlet_nodes": [0, 1],
        },
        "flows_si": np.array([1e-9, 1e-9, 2e-9]),
        "radii_si": np.full(3, 1e-5),
        "lengths_si": np.array([0.01, 0.0001, 0.001]),
        "gl_t": np.array([0.5]),
        "diffusivity_si": 2.41e-9,
    }
    kw = {
        "inlet_concentration": 0.14,
        "vmax": 0.04,
        "km": 0.0069,
        "chb_max": np.full(3, 9.0),
        "fluid_mode": "blood",
    }
    a = _run_network_ext_frozen_step(ctx, {"c_ext_gl": np.zeros((3, 1))}, **kw)
    b = solve_network_ext_gpu(ctx, {"c_ext_gl": np.zeros((3, 1))}, **kw)
    np.testing.assert_allclose(b[0], a[0], rtol=1e-4, atol=3e-6)
    np.testing.assert_allclose(b[1], a[1], rtol=1e-4, atol=3e-6)
    assert b[3] == "gpu-network"


def test_gpu_hematocrit_conserves_rbc_at_merge_split(monkeypatch):
    cuda()
    from cascade.flow.hematocrit_gpu import compute_network_hematocrit_gpu

    monkeypatch.setattr(state, "HEMATOCRIT_MIN", 0.0)
    up, down = np.array([0, 0, 1, 2, 3, 3]), np.array([1, 2, 3, 3, 4, 5])
    q = np.array([0.7, 0.3, 0.7, 0.3, 0.4, 0.6])
    hd, ht = compute_network_hematocrit_gpu(
        up, down, q, np.full(6, 0.0005), model="pries_secomb"
    )
    flux = np.bincount(up, weights=q * hd, minlength=6) - np.bincount(
        down, weights=q * hd, minlength=6
    )
    np.testing.assert_allclose(flux[1:4], 0.0, atol=1e-14)
    assert np.all(np.isfinite(ht))


def test_gpu_pries_binary_matches_cpu_phase_law():
    cuda()
    from cascade.flow.hematocrit import _phase_fraction_pries_numba
    from cascade.flow.hematocrit_gpu import compute_network_hematocrit_gpu

    q = np.array([1.0, 0.7, 0.3])
    r = np.array([0.001, 0.0007, 0.0005])
    hd, _ = compute_network_hematocrit_gpu(
        np.array([0, 1, 1]), np.array([1, 2, 3]), q, r, model="pries_secomb"
    )
    fraction = _phase_fraction_pries_numba(
        0.7,
        20.0,
        14.0,
        10.0,
        state.HD_DISCHARGE,
        state.HEMATOCRIT_MIN,
        state.HEMATOCRIT_MAX,
    )
    np.testing.assert_allclose(
        hd[1:],
        state.HD_DISCHARGE * np.array([fraction / 0.7, (1 - fraction) / 0.3]),
        rtol=1e-12,
    )


def test_gpu_viscosity_matches_cpu():
    cuda()
    from cascade.flow.hematocrit_gpu import segment_viscosity_gpu
    from cascade.flow.rheology import eta_rel_pries, pries_secomb_viscor_cgs

    r = np.geomspace(0.0002, 0.01, 128)
    h = np.linspace(0.1, 0.8, 128)
    for pries, expected in [
        (False, 0.012 * eta_rel_pries(2 * r * 1e4, h)),
        (True, pries_secomb_viscor_cgs(2 * r * 1e4, h)),
    ]:
        np.testing.assert_allclose(
            segment_viscosity_gpu(r, h, 0.012, pries=pries), expected, rtol=1e-12
        )


def test_gpu_amg_refreshes_resistance_values():
    cuda()
    from cascade.flow.gpu import LAST_GPU_FLOW_TIMINGS
    from cascade.flow.kirchhoff import solve_kirchhoff

    up, down = np.array([0, 0, 1, 2, 1]), np.array([1, 2, 3, 3, 2])
    for resistance in (np.ones(5), np.array([2.0, 3.0, 1.0, 7.0, 4.0])):
        args = (up, down, resistance, [0], 0.015, [3])
        cpu = solve_kirchhoff(
            *args, solver_mode="spsolve", boundary_condition="terminal_pressure"
        )
        gpu = solve_kirchhoff(
            *args, solver_mode="gpu_amg", boundary_condition="terminal_pressure"
        )
        np.testing.assert_allclose(gpu[1], cpu[1], rtol=1e-9, atol=1e-13)
    assert LAST_GPU_FLOW_TIMINGS["hierarchy_values_updated"]


def test_gpu_pressure_pressure_boundary():
    cuda()
    from cascade.flow.gpu import solve_pressure_dirichlet_gpu
    from cascade.flow.linear_system import _solve_pressure_dirichlet

    args = (
        np.array([0, 0, 1, 2, 1]),
        np.array([1, 2, 3, 3, 2]),
        np.array([2.0, 3.0, 1.0, 7.0, 4.0]),
        [0],
        100.0,
        [3],
        20.0,
    )
    a, b = _solve_pressure_dirichlet(*args), solve_pressure_dirichlet_gpu(*args)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-9)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-9)


def test_gpu_pressure_one_free_node_exact_solution():
    cuda()
    from cascade.flow.kirchhoff import solve_kirchhoff

    p, q, _, _, _ = solve_kirchhoff(
        np.array([0]),
        np.array([1]),
        np.array([2.0]),
        [0],
        0.015,
        [1],
        solver_mode="gpu_amg",
        boundary_condition="terminal_pressure",
    )
    np.testing.assert_allclose(p, [0.03, 0.0], rtol=1e-12)
    np.testing.assert_allclose(q, [0.015], rtol=1e-12)


def test_gpu_blood_outer_iteration_reports_conservative_final_state(monkeypatch):
    cuda()
    from cascade.flow.network_blood_gpu import solve_network_blood_gpu

    monkeypatch.setattr(state, "HEMATOCRIT_MODEL", "pries_secomb")
    monkeypatch.setattr(state, "HEMATOCRIT_FLOW_ITERATIONS", 3)
    monkeypatch.setattr(state, "KIRCHHOFF_BC_MODE", "terminal_pressure")
    up, down = np.array([0, 0, 1, 2, 1]), np.array([1, 2, 3, 3, 2])
    _p, q, hd, ht, timing = solve_network_blood_gpu(
        up,
        down,
        np.array([0.0008, 0.0004, 0.0006, 0.0007, 0.0003]),
        np.full(5, 0.01),
        [0],
        [3],
        0.015,
        0.012,
    )
    balance = np.bincount(up, weights=q * hd, minlength=4) - np.bincount(
        down, weights=q * hd, minlength=4
    )
    np.testing.assert_allclose(balance[1:3], 0.0, atol=1e-11)
    assert timing["backend"] == "gpu-network"
    assert np.all(np.isfinite(ht))


def test_gpu_near_zero_content_inversion_avoids_cancellation(monkeypatch):
    cuda()
    from cascade.concentration.vessel.network_gpu import _mix, gpu_graph_arrays
    from cascade.concentration.vessel.oxygen_transport import oxygen_content

    cp = state._cp
    graph = gpu_graph_arrays(
        np.array([0, 1, 2]),
        np.array([2, 2, 3]),
        np.array([1.0, 1.0, 2.0]),
        np.array([4.0, 8.0, 6.0]),
        [0, 1],
        dtype=np.float64,
    )
    current = cp.full(4, 0.14, dtype=cp.float64)
    cout = cp.asarray([1e-22, 1e-19, 0.14], dtype=cp.float64)
    _mix(
        graph,
        cout,
        current,
        current,
        cp.empty(4),
        cp.empty(4),
        0.14,
        1.0,
        True,
        double=True,
        nodes=cp.asarray([2], dtype=cp.int32),
    )
    value = float(current[2].item())
    target = float(
        np.sum(oxygen_content(np.array([1e-22, 1e-19]), np.array([4.0, 8.0]))) / 2.0
    )
    np.testing.assert_allclose(oxygen_content(value, 6.0), target, rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("accel", ["cpu", "gpu"])
def test_lattice_entry_point_uses_blood_coupling_and_retained_tissue_sources(
    monkeypatch,
    accel,
):
    if accel == "gpu":
        cuda()
    else:
        monkeypatch.setattr(state, "_cp", None)
    from types import SimpleNamespace

    from cascade.concentration.external_field.state import (
        validate_cext_tissue_flux_consistency,
    )
    from cascade.configuration.bridge import load_runtime_module
    from cascade.simulation.simple import simple_details, solve_simple_network
    from cascade.vessels.simple import SimpleNetwork, _simple_tree_data

    for name, value in {
        "NETWORK_TRANSPORT_ACCEL": accel,
        "KIRCHHOFF_SOLVER": "gpu_amg" if accel == "gpu" else "spsolve",
        "KIRCHHOFF_BC_MODE": "terminal_pressure",
        "HEMATOCRIT_MODEL": "pries_secomb",
        "HEMATOCRIT_FLOW_ITERATIONS": 2,
        "CEXT_ACCEL_MODE": accel,
        "CEXT_FROZEN_ACCEL_MODE": accel,
        "TISSUE_ACCEL_MODE": accel,
        "CEXT_VESS_COUPLING_MAX_ITER": 2,
        "CEXT_WINDOW_FACTOR": 6.0,
        "LUMEN_WALL_CLOSURE": "wellmixed",
        "SOLVER_TIMING_DETAILS": False,
        "JUNCTION_OXYGEN_BALANCE": "total_content",
        "BLOOD_CONVECTIVE_HEMATOCRIT": "discharge",
    }.items():
        monkeypatch.setattr(state, name, value)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.015, 0.0],
            [0.03, -0.015, 0.0],
            [0.06, 0.0, 0.0],
            [0.09, 0.0, 0.0],
        ]
    )
    up, down = np.array([0, 0, 1, 2, 3]), np.array([1, 2, 3, 3, 4])
    starts, ends = positions[up], positions[down]
    radius = np.array([0.001, 0.0008, 0.001, 0.0008, 0.001])
    lengths = np.linalg.norm(ends - starts, axis=1)
    network = SimpleNetwork(
        starts=starts,
        ends=ends,
        radii=radius,
        lengths=lengths,
        flows=np.zeros(5),
        cin=np.zeros(5),
        cout=np.zeros(5),
        inlet_nodes=[0],
        outlet_nodes=[4],
        prox_ids=up,
        dist_ids=down,
        data=_simple_tree_data(starts, ends, radius, lengths, np.zeros(5)),
        parameters=SimpleNamespace(fluid="blood"),
        segment_count=5,
        n_terminals=1,
        mode="lattice",
        decay_starts=starts,
        decay_ends=ends,
        tissue_starts=starts,
        tissue_ends=ends,
        metadata={
            "flow_ul_min": 6.0,
            "concentration_inlet": 0.14,
            "solve_channels_separately": False,
        },
    )
    config = SimpleNamespace(
        network=SimpleNamespace(simple={}),
        simulation=SimpleNamespace(fluid="blood", concentration_solver="network_ext"),
        domain=SimpleNamespace(side_length=1.0),
    )
    runtime = load_runtime_module()
    solve_simple_network(network, runtime, config)
    assert network.metadata["hemodynamics_timing"]["backend"] == f"{accel}-network"
    assert network.metadata["concentration_timings"]["cext_frozen_backend"] == (
        "gpu-network" if accel == "gpu" else "cpu-network-numba"
    )
    assert network.cext_source_state["solver"] == "network_ext"
    assert validate_cext_tissue_flux_consistency(network.cext_source_state)["ok"]
    summary, details = simple_details(
        network, np.array([[0.03, 0.0, 0.01], [0.02, 0.0, 0.005]]), runtime, config
    )
    assert summary["concentration_solver"] == "network_ext"
    assert np.all(np.isfinite(details["tissue_values"]))
    if accel == "gpu":
        assert state._LAST_TISSUE_TIMINGS["backend"] == "cext_gpu_cell_list"


@pytest.mark.parametrize("reverse", [False, True])
def test_graph_hematocrit_cpu_gpu_parity_at_merge_and_split(reverse):
    cuda()
    from cascade.flow.hematocrit_gpu import compute_network_hematocrit_gpu
    from cascade.flow.hematocrit_network import compute_network_hematocrit_cpu

    up = np.array([0, 0, 1, 2, 3, 3])
    down = np.array([1, 2, 3, 3, 4, 5])
    q = np.array([0.7, 0.3, 0.7, 0.3, 0.4, 0.6])
    if reverse:
        up, down, q = down, up, -q
    radii = np.array([0.0007, 0.0003, 0.0007, 0.0003, 0.0004, 0.0006])
    for multiplier in (1.0, 1.3):
        cpu = compute_network_hematocrit_cpu(
            up, down, q * multiplier, radii, model="pries_secomb"
        )
        gpu = compute_network_hematocrit_gpu(
            up, down, q * multiplier, radii, model="pries_secomb"
        )
        for actual, reference in zip(cpu, gpu):
            np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("pressure_boundary", [None, (100.0, 20.0)])
def test_blood_feedback_cpu_gpu_parity(monkeypatch, pressure_boundary):
    cuda()
    from cascade.flow.network_blood import solve_network_blood

    monkeypatch.setattr(state, "HEMATOCRIT_MODEL", "pries_secomb")
    monkeypatch.setattr(state, "HEMATOCRIT_FLOW_ITERATIONS", 3)
    monkeypatch.setattr(state, "KIRCHHOFF_BC_MODE", "terminal_pressure")
    up, down = np.array([0, 0, 1, 2, 3]), np.array([1, 2, 3, 3, 4])
    radii = np.array([0.0008, 0.0004, 0.0008, 0.0004, 0.0007])
    lengths = np.full(5, 0.01)
    kw = dict(pressure_boundary=pressure_boundary)
    cpu = solve_network_blood(
        up,
        down,
        radii,
        lengths,
        [0],
        [4],
        1e-6,
        0.012,
        pressure_solver="spsolve",
        accel="cpu",
        **kw,
    )
    gpu = solve_network_blood(
        up,
        down,
        radii,
        lengths,
        [0],
        [4],
        1e-6,
        0.012,
        pressure_solver="gpu_amg",
        accel="gpu",
        **kw,
    )
    for actual, reference in zip(gpu[:4], cpu[:4]):
        np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-13)


@pytest.mark.parametrize("convection", ["discharge", "tube"])
def test_independent_blood_channels_cpu_gpu_parity(monkeypatch, convection):
    cuda()
    from cascade.configuration.bridge import load_runtime_module
    from cascade.simulation.simple import _solve_independent_channels

    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", convection)
    runtime = load_runtime_module()
    inputs = (
        runtime,
        np.zeros((3, 3)),
        np.array([0.0005, 0.0006, 0.0007]),
        np.array([0.002, 0.005, 0.01]),
        1e-7,
        0.14,
        "blood",
    )
    kw = {"diffusivity": 2.41e-5, "vmax": 0.04, "km": 0.0069}
    monkeypatch.setattr(state, "NETWORK_TRANSPORT_ACCEL", "cpu")
    cpu = _solve_independent_channels(*inputs, **kw)
    monkeypatch.setattr(state, "NETWORK_TRANSPORT_ACCEL", "gpu")
    gpu = _solve_independent_channels(*inputs, **kw)
    for actual, reference in zip(gpu, cpu):
        np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-10)
