"""Stable public entry point for intravascular concentration transport."""

from __future__ import annotations

from cascade.core import VesselConcentrationProblem, VesselConcentrationResult


def solve_vessel_concentration(
    problem: VesselConcentrationProblem,
) -> VesselConcentrationResult:
    """Solve vessel concentrations without exposing legacy tuple/state conventions."""
    from cascade.runtime import tissuesim

    inlet, outlet = tissuesim._solve_channel_concentrations(
        problem.tree,
        problem.flows_cm3_s,
        problem.inlet_nodes,
        problem.outlet_nodes,
        problem.segment_starts_cm,
        problem.segment_ends_cm,
        problem.segment_radii_cm,
        problem.segment_lengths_cm,
        prox_ids=problem.proximal_nodes,
        dist_ids=problem.distal_nodes,
        inlet_concentration=float(problem.inlet_concentration),
        diffusivity=float(problem.diffusivity_cm2_s),
        vmax=float(problem.vmax),
        km=float(problem.km),
        fluid=str(problem.fluid),
        solver=str(problem.solver),
    )
    return VesselConcentrationResult(
        inlet=inlet,
        outlet=outlet,
        external_field=tissuesim._LAST_CEXT_SOURCE_STATE,
        diagnostics=dict(tissuesim._LAST_CONCENTRATION_TIMINGS or {}),
    )


__all__ = [
    "VesselConcentrationProblem",
    "VesselConcentrationResult",
    "solve_vessel_concentration",
]
