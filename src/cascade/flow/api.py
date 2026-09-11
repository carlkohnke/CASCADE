"""Stable public entry points for pressure and flow calculations."""

from __future__ import annotations

from cascade.core import FlowProblem, FlowResult

from .kirchhoff import solve_kirchhoff


def solve_flow(problem: FlowProblem) -> FlowResult:
    """Solve a validated network problem using the configured Kirchhoff backend."""
    pressures, flows, proximal, distal, nodes = solve_kirchhoff(
        problem.proximal_nodes,
        problem.distal_nodes,
        problem.resistances,
        problem.inlet_nodes,
        float(problem.inlet_flow_cm3_s),
        problem.outlet_nodes,
        num_nodes=problem.node_count,
    )
    return FlowResult(
        pressures=pressures,
        flows_cm3_s=flows,
        proximal_nodes=proximal,
        distal_nodes=distal,
        node_coordinates_cm=nodes,
    )


__all__ = ["FlowProblem", "FlowResult", "solve_flow"]
