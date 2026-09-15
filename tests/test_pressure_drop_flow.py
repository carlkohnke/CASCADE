import numpy as np
import pytest

from cascade.configuration import solver_state
from cascade.flow import PressureDropProblem, solve_pressure_drop
from cascade.simulation.network import run_tree_simulation


@pytest.fixture(autouse=True)
def _quiet_kirchhoff_diagnostics(monkeypatch):
    monkeypatch.setattr(solver_state, "KIRCHHOFF_DIAGNOSTICS", False)


def test_single_resistor_obeys_ohms_law_with_absolute_pressure_offset():
    result = solve_pressure_drop(
        PressureDropProblem(
            proximal_nodes=np.array([0]),
            distal_nodes=np.array([1]),
            resistances=np.array([4.0]),
            inlet_nodes=[0],
            outlet_nodes=[1],
            outlet_pressure=2.0,
            pressure_drop=8.0,
            solver="spsolve",
        )
    )

    np.testing.assert_allclose(result.pressures, [10.0, 2.0])
    np.testing.assert_allclose(result.flows_cm3_s, [2.0])
    assert result.inlet_flow_cm3_s == pytest.approx(2.0)
    assert result.outlet_flow_cm3_s == pytest.approx(2.0)


def test_two_series_resistors_use_sum_of_resistances_tree_path():
    result = solve_pressure_drop(
        PressureDropProblem(
            proximal_nodes=np.array([0, 1]),
            distal_nodes=np.array([1, 2]),
            resistances=np.array([2.0, 6.0]),
            inlet_nodes=[0],
            outlet_nodes=[2],
            outlet_pressure=3.0,
            pressure_drop=8.0,
            solver="tree",
        )
    )

    # R_eq = 8, dP = 8, Q = 1; the intermediate drop is Q * R_1 = 2.
    np.testing.assert_allclose(result.pressures, [11.0, 9.0, 3.0])
    np.testing.assert_allclose(result.flows_cm3_s, [1.0, 1.0])
    assert result.inlet_flow_cm3_s == pytest.approx(1.0)


def test_symmetric_branch_splits_flow_and_conserves_at_junction():
    result = solve_pressure_drop(
        PressureDropProblem(
            proximal_nodes=np.array([0, 1, 1]),
            distal_nodes=np.array([1, 2, 3]),
            resistances=np.array([2.0, 4.0, 4.0]),
            inlet_nodes=[0],
            outlet_nodes=[2, 3],
            outlet_pressure=0.0,
            pressure_drop=8.0,
            solver="tree",
        )
    )

    # The two distal resistors are 2 in parallel, in series with the root 2.
    # R_eq = 4, Q_in = 2, and the two branches each carry 1.
    np.testing.assert_allclose(result.pressures, [8.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(result.flows_cm3_s, [2.0, 1.0, 1.0])
    np.testing.assert_allclose(result.node_net_outflows_cm3_s, [2.0, 0.0, -1.0, -1.0])
    assert result.inlet_flow_cm3_s == pytest.approx(2.0)
    assert result.outlet_flow_cm3_s == pytest.approx(2.0)


def test_multiple_pressure_inlets_use_general_dirichlet_solve():
    result = solve_pressure_drop(
        PressureDropProblem(
            proximal_nodes=np.array([0, 2]),
            distal_nodes=np.array([1, 3]),
            resistances=np.array([4.0, 2.0]),
            inlet_nodes=[0, 2],
            outlet_nodes=[1, 3],
            outlet_pressure=2.0,
            pressure_drop=4.0,
            solver="tree",
        )
    )

    np.testing.assert_allclose(result.flows_cm3_s, [1.0, 2.0])
    assert result.inlet_flow_cm3_s == pytest.approx(3.0)
    assert result.outlet_flow_cm3_s == pytest.approx(3.0)


def test_pressure_drop_must_be_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        PressureDropProblem(
            proximal_nodes=np.array([0]),
            distal_nodes=np.array([1]),
            resistances=np.array([1.0]),
            inlet_nodes=[0],
            outlet_nodes=[1],
            outlet_pressure=2.0,
            pressure_drop=0.0,
        )


def test_tree_simulation_pressure_mode_ignores_placeholder_inlet_flow(monkeypatch):
    from cascade.runtime import tissuesim as ts

    monkeypatch.setattr(solver_state, "KIRCHHOFF_BC_MODE", "pressure_pressure")
    monkeypatch.setattr(solver_state, "KIRCHHOFF_SOLVER", "tree")
    monkeypatch.setattr(solver_state, "HEMATOCRIT_FLOW_ITERATIONS", 0)
    monkeypatch.setattr(solver_state, "COMPUTE_AVG_DISTANCE_TO_CHANNEL", False)

    domain = ts.build_domain(1.0, random_seed=42)
    tree = ts.Tree(data_dtype=np.float64, index_dtype=np.int64, preallocation_step=4)
    tree.set_domain(domain)
    tree.parameters.root_pressure = 7000.0
    tree.parameters.terminal_pressure = 6000.0
    tree.set_root(
        np.asarray([0.49, -0.49, -0.49]),
        np.asarray([-0.49, 0.49, 0.49]),
    )

    summary_a, details_a = run_tree_simulation(
        tree,
        np.empty((0, 3)),
        1,
        side_length=1.0,
        fluid="water",
        inlet_flow_cm3_s=1.0e-12,
        concentration_solver="topdown",
        return_details=True,
    )
    summary_b, details_b = run_tree_simulation(
        tree,
        np.empty((0, 3)),
        1,
        side_length=1.0,
        fluid="water",
        inlet_flow_cm3_s=10.0,
        concentration_solver="topdown",
        return_details=True,
    )

    assert summary_a["pressure_in_root"] == pytest.approx(7000.0)
    assert summary_a["pressure_out_terminals"] == pytest.approx(6000.0)
    assert summary_a["pressure_drop"] == pytest.approx(1000.0)
    assert summary_a["inlet_flow_ul_per_min"] > 0.0
    assert summary_b["inlet_flow_ul_per_min"] == pytest.approx(
        summary_a["inlet_flow_ul_per_min"]
    )
    np.testing.assert_allclose(details_b["flows"], details_a["flows"])
    np.testing.assert_allclose(details_b["pressures"], details_a["pressures"])
