"""Tree-specialized vessel concentration orchestration."""

from __future__ import annotations

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.concentration.properties import get_concentration_inlet
from cascade.concentration.vessel.dispatch import _solve_channel_concentrations
from cascade.flow.tree import recompute_tree_flows


def solve_tree_greens(
    tree: _state.Tree,
    inlet_flow_cm3_s: float,
    *,
    fluid: str,
    inlet_concentration: float | None = None,
    diffusivity: float = _state.SOLUTE_DIFFUSIVITY,
    vmax: float = _state.VMAX_MM,
    km: float = _state.K_M_MM,
    omega: float = _state.OMEGA,
    concentration_solver: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float],
]:
    """Recompute tree flow and solve its intravascular concentration field."""
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet(fluid)
    (
        flows,
        pressures,
        inlet_nodes,
        outlet_nodes,
        starts,
        ends,
        radii,
        lengths,
        proximal_ids,
        distal_ids,
    ) = recompute_tree_flows(tree, inlet_flow_cm3_s, fluid=fluid)

    cin, cout = _solve_channel_concentrations(
        tree,
        flows,
        inlet_nodes,
        outlet_nodes,
        starts,
        ends,
        radii,
        lengths,
        prox_ids=proximal_ids,
        dist_ids=distal_ids,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=vmax,
        km=km,
        fluid=fluid,
        solver=concentration_solver,
    )
    pressure_in = (
        float(pressures[inlet_nodes[0]])
        if pressures.size and inlet_nodes
        else float("nan")
    )
    pressure_out = (
        float(np.mean(pressures[outlet_nodes]))
        if pressures.size and outlet_nodes
        else float("nan")
    )
    return (
        starts,
        ends,
        radii,
        lengths,
        flows,
        cin,
        cout,
        (pressure_in, pressure_out),
    )


__all__ = ["solve_tree_greens"]
