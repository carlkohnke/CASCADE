"""Blood-flow, pressure, rheology, and hematocrit solvers."""

from __future__ import annotations

from .api import FlowProblem, FlowResult, solve_flow
from .boundary_conditions import allocate_inlet_flows, tree_root_flow_cm3_s
from .hematocrit import compute_tree_hematocrit
from .kirchhoff import solve_kirchhoff, solve_kirchhoff_dirichlet, solve_kirchhoff_tree
from .rheology import (
    apply_fahraeus_lindqvist_resistance,
    compute_segment_viscosity,
    segment_viscosity_from_radius,
    segment_viscosity_from_radius_hd,
    tube_hematocrit,
)

__all__ = [
    "FlowProblem",
    "FlowResult",
    "allocate_inlet_flows",
    "apply_fahraeus_lindqvist_resistance",
    "compute_segment_viscosity",
    "compute_tree_hematocrit",
    "segment_viscosity_from_radius",
    "segment_viscosity_from_radius_hd",
    "solve_flow",
    "solve_kirchhoff",
    "solve_kirchhoff_dirichlet",
    "solve_kirchhoff_tree",
    "tube_hematocrit",
    "tree_root_flow_cm3_s",
]
