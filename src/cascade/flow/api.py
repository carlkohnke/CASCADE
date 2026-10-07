"""Stable public entry points for pressure and flow calculations."""

from __future__ import annotations

import numpy as np

from cascade.flow.contracts import (
    FlowProblem,
    FlowResult,
    PressureDropProblem,
    PressureDropResult,
)

from .kirchhoff import solve_kirchhoff
from .linear_system import _solve_pressure_dirichlet


def solve_flow(problem: FlowProblem) -> FlowResult:
    """Solve a validated network problem using the configured Kirchhoff backend."""
    pressures, flows, proximal, distal, _legacy_nodes = solve_kirchhoff(
        problem.proximal_nodes,
        problem.distal_nodes,
        problem.resistances,
        problem.inlet_nodes,
        float(problem.inlet_flow_cm3_s),
        problem.outlet_nodes,
        num_nodes=problem.node_count,
        solver_mode=problem.solver,
        boundary_condition=problem.boundary_condition,
    )
    return FlowResult(
        pressures=pressures,
        flows_cm3_s=flows,
        proximal_nodes=proximal,
        distal_nodes=distal,
    )


def solve_pressure_drop(problem: PressureDropProblem) -> PressureDropResult:
    """Directly impose inlet/outlet pressures and return the emergent flow.

    A single-inlet tree uses one unit-flow solve followed by exact linear
    scaling. General graphs use one reduced Dirichlet linear-system solve.
    Neither path searches over candidate inlet flows.
    """

    use_tree_scaling = problem.solver == "tree" and len(problem.inlet_nodes) == 1
    if use_tree_scaling:
        reference_pressures, reference_flows, proximal, distal, _ = solve_kirchhoff(
            problem.proximal_nodes,
            problem.distal_nodes,
            problem.resistances,
            problem.inlet_nodes,
            1.0,
            problem.outlet_nodes,
            num_nodes=problem.node_count,
            solver_mode="tree",
            boundary_condition="pressure_pressure",
        )
        reference_outlet = float(
            np.mean(
                reference_pressures[np.asarray(problem.outlet_nodes, dtype=np.int64)]
            )
        )
        reference_drop = (
            float(reference_pressures[problem.inlet_nodes[0]]) - reference_outlet
        )
        if not np.isfinite(reference_drop) or reference_drop <= 0.0:
            raise RuntimeError(
                "Unit-flow tree solve did not produce a positive finite pressure drop."
            )
        scale = problem.pressure_drop / reference_drop
        pressures = problem.outlet_pressure + scale * (
            reference_pressures - reference_outlet
        )
        flows = scale * reference_flows
    else:
        implementation = _solve_pressure_dirichlet
        if problem.solver in {"gpu", "gpu_amg", "auto"}:
            from cascade.concentration.vessel.network_gpu import resolve_network_accel

            if problem.solver != "auto" or resolve_network_accel() == "gpu":
                from .gpu import solve_pressure_dirichlet_gpu

                implementation = solve_pressure_dirichlet_gpu
        pressures, flows = implementation(
            problem.proximal_nodes,
            problem.distal_nodes,
            problem.resistances,
            problem.inlet_nodes,
            problem.inlet_pressure,
            problem.outlet_nodes,
            problem.outlet_pressure,
            num_nodes=problem.node_count,
        )
        proximal = problem.proximal_nodes
        distal = problem.distal_nodes

    node_net_outflow = np.bincount(
        proximal, weights=flows, minlength=int(problem.node_count)
    ).astype(float, copy=False)
    node_net_outflow -= np.bincount(
        distal, weights=flows, minlength=int(problem.node_count)
    ).astype(float, copy=False)
    inlet_flow = float(
        np.sum(node_net_outflow[np.asarray(problem.inlet_nodes, dtype=np.int64)])
    )
    outlet_flow = float(
        -np.sum(node_net_outflow[np.asarray(problem.outlet_nodes, dtype=np.int64)])
    )
    from cascade.configuration import solver_state as _state

    if _state.KIRCHHOFF_DIAGNOSTICS:
        implementation = (
            "tree_unit_flow_scaling" if use_tree_scaling else "reduced_dirichlet"
        )
        print(
            "Pressure boundary solve: mode=pressure_pressure "
            f"implementation={implementation} "
            f"solved_inlet_flow_ul_min={inlet_flow * 60000.0:.12g}"
        )
    scale = max(abs(inlet_flow), abs(outlet_flow), 1.0)
    if abs(inlet_flow - outlet_flow) > 1.0e-10 * scale:
        raise RuntimeError(
            "Pressure-pressure Kirchhoff solution failed global flow conservation."
        )
    return PressureDropResult(
        pressures=np.asarray(pressures, dtype=float),
        flows_cm3_s=np.asarray(flows, dtype=float),
        proximal_nodes=np.asarray(proximal, dtype=np.int64),
        distal_nodes=np.asarray(distal, dtype=np.int64),
        inlet_flow_cm3_s=inlet_flow,
        outlet_flow_cm3_s=outlet_flow,
        node_net_outflows_cm3_s=node_net_outflow,
    )


__all__ = [
    "FlowProblem",
    "FlowResult",
    "PressureDropProblem",
    "PressureDropResult",
    "solve_flow",
    "solve_pressure_drop",
]
