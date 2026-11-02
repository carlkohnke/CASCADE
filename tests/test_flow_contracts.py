from __future__ import annotations

import numpy as np
import pytest

from cascade.configuration import solver_state
from cascade.flow import (
    FlowProblem,
    PressureDropProblem,
    solve_flow,
    solve_pressure_drop,
)


@pytest.fixture(autouse=True)
def _quiet_kirchhoff_diagnostics(monkeypatch):
    monkeypatch.setattr(solver_state, "KIRCHHOFF_DIAGNOSTICS", False)


def _one_edge(**overrides):
    values = {
        "proximal_nodes": [0],
        "distal_nodes": [1],
        "resistances": [2.0],
        "inlet_nodes": [0],
        "outlet_nodes": [1],
        "inlet_flow_cm3_s": 1.0,
        "node_count": 2,
        "solver": "spsolve",
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("proximal_nodes", [0.5], "finite integers"),
        ("inlet_nodes", [0.5], "finite integers"),
        ("outlet_nodes", [1, 1], "duplicate"),
        ("node_count", 2.5, "finite integers"),
        ("proximal_nodes", [2**63], "signed 64-bit"),
    ],
)
def test_flow_problem_rejects_ambiguous_integer_inputs(field, value, message):
    with pytest.raises(ValueError, match=message):
        FlowProblem(**_one_edge(**{field: value}))


def test_flow_problem_rejects_overlapping_boundaries():
    with pytest.raises(ValueError, match="must not overlap"):
        FlowProblem(**_one_edge(outlet_nodes=[0, 1]))


def test_pressure_problem_rejects_duplicate_boundaries():
    with pytest.raises(ValueError, match="duplicate"):
        PressureDropProblem(
            proximal_nodes=[0],
            distal_nodes=[1],
            resistances=[1.0],
            inlet_nodes=[0],
            outlet_nodes=[1, 1],
            pressure_drop=1.0,
        )


def test_sparse_solver_rejects_disconnected_unanchored_component():
    problem = FlowProblem(
        proximal_nodes=[0, 2],
        distal_nodes=[1, 3],
        resistances=[1.0, 1.0],
        inlet_nodes=[0],
        outlet_nodes=[1],
        inlet_flow_cm3_s=1.0,
        node_count=4,
        solver="spsolve",
        boundary_condition="terminal_pressure",
    )
    with pytest.raises(RuntimeError, match="connectivity|non-finite|solve"):
        solve_flow(problem)


def test_explicit_tree_solver_rejects_multiple_inlets():
    problem = FlowProblem(
        proximal_nodes=[0, 1],
        distal_nodes=[1, 2],
        resistances=[1.0, 1.0],
        inlet_nodes=[0, 1],
        outlet_nodes=[2],
        inlet_flow_cm3_s=1.0,
        node_count=3,
        solver="tree",
        boundary_condition="terminal_pressure",
    )
    with pytest.raises(ValueError, match="exactly one inlet"):
        solve_flow(problem)


def _looped_problem(solver, boundary_condition="terminal_pressure"):
    # Segments 1-2-4 and 1-3-4 form one loop between nodes 1 and 4.
    return FlowProblem(
        proximal_nodes=[0, 1, 1, 2, 3, 4],
        distal_nodes=[1, 2, 3, 4, 4, 5],
        resistances=[1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        inlet_nodes=[0],
        outlet_nodes=[5],
        inlet_flow_cm3_s=1.0,
        node_count=6,
        solver=solver,
        boundary_condition=boundary_condition,
    )


@pytest.mark.parametrize(
    "boundary_condition", ["terminal_pressure", "legacy_equal_terminal_flow"]
)
def test_explicit_tree_solver_rejects_looped_networks(boundary_condition):
    with pytest.raises(ValueError, match="loop-free"):
        solve_flow(_looped_problem("tree", boundary_condition))


def test_tree_pressure_drop_rejects_looped_networks():
    problem = PressureDropProblem(
        proximal_nodes=[0, 1, 1, 2, 3, 4],
        distal_nodes=[1, 2, 3, 4, 4, 5],
        resistances=[1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        inlet_nodes=[0],
        outlet_nodes=[5],
        outlet_pressure=0.0,
        pressure_drop=1.0,
        solver="tree",
    )
    with pytest.raises(ValueError, match="loop-free"):
        solve_pressure_drop(problem)


def test_auto_solver_splits_flow_around_loop():
    result = solve_flow(_looped_problem("auto"))
    np.testing.assert_allclose(
        result.flows_cm3_s, [1.0, 0.6, 0.4, 0.6, 0.4, 1.0], rtol=1e-10
    )


def test_auto_solver_handles_multiple_inlets_with_sparse_backend():
    result = solve_flow(
        FlowProblem(
            proximal_nodes=[0, 1],
            distal_nodes=[1, 2],
            resistances=[1.0, 1.0],
            inlet_nodes=[0, 1],
            outlet_nodes=[2],
            inlet_flow_cm3_s=1.0,
            node_count=3,
            solver="auto",
            boundary_condition="terminal_pressure",
        )
    )
    np.testing.assert_allclose(result.pressures, [1.5, 1.0, 0.0])
    np.testing.assert_allclose(result.flows_cm3_s, [0.5, 1.0])
