"""Network oxygen transport convergence reporting and GPU fallback warnings."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.concentration.external_field import coupling_steps
from cascade.concentration.external_field.coupling_steps import (
    _run_network_ext_frozen_step,
    _run_topdown_ext_frozen_step,
)
from cascade.concentration.solve_report import (
    concentration_solve_report,
    solve_report_details,
)
from cascade.configuration import solver_state as state

TRANSPORT = {"vmax": 0.04, "km": 0.0069, "fluid_mode": "plasma"}


@pytest.fixture(autouse=True)
def wellmixed_dissolved(monkeypatch):
    monkeypatch.setattr(state, "LUMEN_WALL_CLOSURE", "wellmixed")
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "dissolved")


def chain(n, *, return_edge=False):
    """Return an ``n``-segment chain; ``return_edge`` closes it into a cycle.

    The return edge feeds the fixed inlet node, so the cyclic graph has the same
    exact answer as the open chain while forcing the fixed-point path.
    """
    prox, dist = np.arange(n), np.arange(1, n + 1)
    if return_edge:
        prox, dist = np.append(prox, n), np.append(dist, 0)
    m = prox.size
    context = {
        "network_topology": {"prox_ids": prox, "dist_ids": dist, "inlet_nodes": [0]},
        "flows_si": np.full(m, 1e-10),
        "radii_si": np.full(m, 5e-6),
        "lengths_si": np.full(m, 4e-5),
        "gl_t": np.array([0.5]),
        "diffusivity_si": 2.41e-9,
    }
    return context, {"c_ext_gl": np.zeros((m, 1))}


def solve(context, ext_state, inlet=0.14):
    nseg = len(context["flows_si"])
    return _run_network_ext_frozen_step(
        context,
        ext_state,
        inlet_concentration=inlet,
        chb_max=np.zeros(nseg),
        **TRANSPORT,
    )


def sequential_outlets(n):
    """March the chain one segment at a time; each step is a single exact edge."""
    values, concentration = [], 0.14
    for _ in range(n):
        concentration = float(solve(*chain(1), inlet=concentration)[1][0])
        values.append(concentration)
    return np.asarray(values)


@pytest.fixture(scope="module")
def exact_1000():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(state, "LUMEN_WALL_CLOSURE", "wellmixed")
        patch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "dissolved")
        return sequential_outlets(1000)


@pytest.mark.parametrize("compiled", [True, False])
def test_long_chain_matches_exact_answer(monkeypatch, exact_1000, compiled):
    # The former 100-iteration Jacobi cap left this outlet 819% too high.
    monkeypatch.setattr(state, "CONC_USE_NUMBA", compiled)
    cout = solve(*chain(1000))[1]
    timings = state._LAST_CEXT_FROZEN_STEP_TIMINGS
    assert timings["network_schedule"] == "dag"
    assert timings["network_converged"]
    np.testing.assert_allclose(cout, exact_1000, rtol=1e-5)


def test_long_cyclic_chain_iterates_to_tolerance(exact_1000, capsys):
    cout = solve(*chain(1000, return_edge=True))[1][:1000]
    timings = state._LAST_CEXT_FROZEN_STEP_TIMINGS
    assert timings["network_schedule"] == "cyclic-fixed-point"
    assert timings["network_converged"]
    assert timings["network_iterations"] > 1000
    np.testing.assert_allclose(cout, exact_1000, rtol=1e-3)
    assert "former 100-iteration limit" in capsys.readouterr().out


def test_dag_sweep_matches_converged_fixed_point(monkeypatch):
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    context = {
        "network_topology": {
            "prox_ids": np.array([0, 0, 1, 2, 3, 3]),
            "dist_ids": np.array([1, 2, 3, 3, 4, 5]),
            "inlet_nodes": [0],
        },
        "flows_si": np.array([0.7, 0.3, 0.7, 0.3, 0.4, 0.6]) * 1e-10,
        "radii_si": np.full(6, 5e-6),
        "lengths_si": np.array([4e-5, 9e-5, 2e-5, 3e-5, 5e-5, 6e-5]),
        "gl_t": np.array([0.2, 0.5, 0.8]),
        "diffusivity_si": 2.41e-9,
    }
    ext = {"c_ext_gl": np.full((6, 3), 0.01)}
    kw = {"inlet_concentration": 0.14, "vmax": 0.04, "km": 0.0069}
    kw.update(chb_max=np.full(6, 9.0), fluid_mode="blood")
    exact = _run_network_ext_frozen_step(dict(context), dict(ext), **kw)
    monkeypatch.setattr(coupling_steps, "graph_schedule", lambda *args: None)
    monkeypatch.setattr(coupling_steps, "NETWORK_TRANSPORT_TOL", 1e-12)
    fixed_point = _run_network_ext_frozen_step(dict(context), dict(ext), **kw)
    assert (
        state._LAST_CEXT_FROZEN_STEP_TIMINGS["network_schedule"] == "cyclic-fixed-point"
    )
    for a, b in zip(exact[:3], fixed_point[:3]):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def test_slow_convergence_warns_once_per_solve(capsys):
    context, ext = chain(200, return_edge=True)
    with concentration_solve_report() as report:
        for _ in range(3):
            cin, *_ = solve(context, ext)
            ext["cin_seg"] = cin
    output = capsys.readouterr().out
    assert output.count("Warning: CPU network oxygen transport needed") == 1
    assert "tolerance 1e-06" in output
    details = solve_report_details(report)
    assert details["cext_network_calls"] == 3
    assert details["cext_network_iterations_max"] > 100
    assert details["cext_network_converged"]
    # The warm-started repeats converge quickly but do not reset the record.
    assert state._LAST_CEXT_FROZEN_STEP_TIMINGS["network_iterations"] < 100


@pytest.mark.parametrize("compiled", [True, False])
def test_non_converging_cycle_warns_instead_of_hanging(monkeypatch, capsys, compiled):
    monkeypatch.setattr(state, "CONC_USE_NUMBA", compiled)
    monkeypatch.setattr(coupling_steps, "network_iteration_limit", lambda nnode: 5)
    _cin, cout, *_ = solve(*chain(50, return_edge=True))
    output = capsys.readouterr().out
    assert "CPU network oxygen transport did not converge" in output
    assert "safety limit of 5 iterations" in output
    assert "tolerance 1e-06" in output
    timings = state._LAST_CEXT_FROZEN_STEP_TIMINGS
    assert timings["network_iterations"] == 5
    assert not timings["network_converged"]
    assert timings["network_max_delta"] >= 1e-6
    assert np.all(np.isfinite(cout))


def cuda():
    from cascade.concentration.vessel.network_gpu import resolve_network_accel

    if resolve_network_accel("auto") != "gpu":
        pytest.skip("CUDA/CuPy unavailable")


def solve_gpu(context, ext_state):
    from cascade.concentration.vessel.network_gpu import solve_network_ext_gpu

    nseg = len(context["flows_si"])
    return solve_network_ext_gpu(
        context,
        ext_state,
        inlet_concentration=0.14,
        chb_max=np.zeros(nseg),
        **TRANSPORT,
    )


def test_gpu_cyclic_chain_iterates_past_former_cap(capsys):
    cuda()
    cpu = solve(*chain(150, return_edge=True))[1]
    capsys.readouterr()
    gpu = solve_gpu(*chain(150, return_edge=True))[1]
    timings = state._LAST_CEXT_FROZEN_STEP_TIMINGS
    assert timings["network_schedule"] == "cyclic-fixed-point"
    assert timings["network_converged"] and timings["network_iterations"] > 100
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-7)
    assert "GPU network oxygen transport needed" in capsys.readouterr().out


def test_gpu_non_converging_cycle_warns(monkeypatch, capsys):
    cuda()
    from cascade.concentration.vessel import network_gpu

    monkeypatch.setattr(network_gpu, "network_iteration_limit", lambda nnode: 5)
    solve_gpu(*chain(50, return_edge=True))
    assert "GPU network oxygen transport did not converge" in capsys.readouterr().out
    timings = state._LAST_CEXT_FROZEN_STEP_TIMINGS
    assert timings["network_iterations"] == 5 and not timings["network_converged"]


def tree_context():
    from cascade.concentration.external_field.backend import (
        _build_topdown_level_slices,
    )

    parents = np.array([-1, 0, 0], dtype=np.int32)
    level_order, level_offsets, _ = _build_topdown_level_slices(
        np.array([0, 1, 2]), parents
    )
    return {
        "level_order": level_order,
        "level_offsets": level_offsets,
        "parents": parents,
        "flows_si": np.array([1.0, 0.6, 0.4]) * 1e-10,
        "radii_si": np.full(3, 5e-6),
        "lengths_si": np.array([4e-5, 3e-5, 5e-5]),
        "gl_t": np.array([0.2, 0.5, 0.8]),
        "diffusivity_si": 2.41e-9,
    }


def run_tree(backend):
    return _run_topdown_ext_frozen_step(
        tree_context(),
        {"c_ext_gl": np.zeros((3, 3), np.float32)},
        inlet_concentration=0.14,
        chb_max=np.zeros(3),
        frozen_backend=backend,
        **TRANSPORT,
    )


def test_auto_gpu_failure_warns_and_finishes_on_cpu(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated CUDA launch failure")

    monkeypatch.setattr(state, "CEXT_FROZEN_ACCEL_MODE", "auto")
    monkeypatch.setattr(coupling_steps, "_solve_topdown_ext_frozen_gpu", fail)
    expected = run_tree("cpu")
    with concentration_solve_report() as report:
        actual = run_tree("gpu")
        run_tree("gpu")
    output = capsys.readouterr().out
    assert output.count("Warning: GPU frozen top-down transport failed") == 1
    assert "RuntimeError: simulated CUDA launch failure" in output
    assert "falling back to CPU" in output
    assert actual[3] == "cpu"
    for a, b in zip(actual[:3], expected[:3]):
        np.testing.assert_array_equal(a, b)
    assert (
        "simulated CUDA launch failure"
        in (state._LAST_CEXT_FROZEN_STEP_TIMINGS["gpu_fallback"])
    )
    assert solve_report_details(report)["cext_gpu_fallbacks"] == (
        "frozen top-down transport: RuntimeError: simulated CUDA launch failure"
    )


def test_explicit_gpu_mode_still_raises(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated CUDA launch failure")

    monkeypatch.setattr(state, "CEXT_FROZEN_ACCEL_MODE", "gpu")
    monkeypatch.setattr(coupling_steps, "_solve_topdown_ext_frozen_gpu", fail)
    with pytest.raises(RuntimeError, match="simulated CUDA launch failure"):
        run_tree("gpu")


def test_graetz_cpu_error_chains_original_gpu_failure(monkeypatch, capsys):
    original = MemoryError("simulated device allocation failure")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(state, "CEXT_FROZEN_ACCEL_MODE", "auto")
    monkeypatch.setattr(state, "LUMEN_WALL_CLOSURE", "graetz")
    monkeypatch.setattr(coupling_steps, "_solve_topdown_ext_frozen_graetz_gpu", fail)
    with pytest.raises(RuntimeError, match="requires the GPU") as raised:
        run_tree("gpu")
    assert raised.value.__cause__ is original
    assert "MemoryError: simulated device allocation failure" in str(raised.value)
    assert "Warning: GPU frozen top-down transport failed" in capsys.readouterr().out
