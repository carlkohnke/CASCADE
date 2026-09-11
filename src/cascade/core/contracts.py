"""Typed data exchanged between the scientific solver layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _vector(value: Any, name: str, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if not np.all(np.isfinite(array)) and np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class FlowProblem:
    """Network resistance problem with prescribed total inlet flow."""

    proximal_nodes: np.ndarray
    distal_nodes: np.ndarray
    resistances: np.ndarray
    inlet_nodes: Sequence[int]
    outlet_nodes: Sequence[int]
    inlet_flow_cm3_s: float
    node_count: int | None = None
    solver: str = "tree"
    boundary_condition: str = "legacy_equal_terminal_flow"

    def __post_init__(self) -> None:
        proximal = _vector(self.proximal_nodes, "proximal_nodes", dtype=np.int64)
        distal = _vector(self.distal_nodes, "distal_nodes", dtype=np.int64)
        resistance = _vector(self.resistances, "resistances", dtype=float)
        if not (proximal.size == distal.size == resistance.size):
            raise ValueError("Flow topology and resistance arrays must have equal lengths.")
        if proximal.size == 0:
            raise ValueError("A flow problem requires at least one segment.")
        if np.any(proximal < 0) or np.any(distal < 0):
            raise ValueError("Flow node indices must be nonnegative.")
        if np.any(resistance <= 0.0):
            raise ValueError("Flow resistances must be strictly positive.")
        inlet_flow = float(self.inlet_flow_cm3_s)
        if not np.isfinite(inlet_flow) or inlet_flow <= 0.0:
            raise ValueError("inlet_flow_cm3_s must be finite and strictly positive.")
        inlet_nodes = tuple(int(value) for value in self.inlet_nodes)
        outlet_nodes = tuple(int(value) for value in self.outlet_nodes)
        if not inlet_nodes:
            raise ValueError("A flow problem requires at least one inlet node.")
        if not outlet_nodes:
            raise ValueError("A flow problem requires at least one outlet node.")
        inferred_node_count = int(max(int(proximal.max()), int(distal.max())) + 1)
        node_count = inferred_node_count if self.node_count is None else int(self.node_count)
        if node_count < inferred_node_count:
            raise ValueError(
                f"node_count={node_count} does not include topology node {inferred_node_count - 1}."
            )
        if any(node < 0 or node >= node_count for node in inlet_nodes + outlet_nodes):
            raise ValueError("Flow boundary node indices must lie within node_count.")
        solver = str(self.solver).strip().lower().replace("-", "_")
        solver_aliases = {"tree_neumann": "tree", "tree_current_bc": "tree"}
        solver = solver_aliases.get(solver, solver)
        if solver not in {"tree", "auto", "cg", "gmres_ilu", "spsolve"}:
            raise ValueError(
                "solver must be one of 'tree', 'auto', 'cg', 'gmres_ilu', or 'spsolve'."
            )
        boundary_condition = str(self.boundary_condition).strip().lower().replace("-", "_")
        if boundary_condition not in {"legacy_equal_terminal_flow", "terminal_pressure"}:
            raise ValueError(
                "boundary_condition must be 'legacy_equal_terminal_flow' or 'terminal_pressure'."
            )
        object.__setattr__(self, "proximal_nodes", proximal)
        object.__setattr__(self, "distal_nodes", distal)
        object.__setattr__(self, "resistances", resistance)
        object.__setattr__(self, "inlet_nodes", inlet_nodes)
        object.__setattr__(self, "outlet_nodes", outlet_nodes)
        object.__setattr__(self, "inlet_flow_cm3_s", inlet_flow)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "boundary_condition", boundary_condition)


@dataclass(frozen=True)
class FlowResult:
    """Pressures and signed segment flows returned by the flow subsystem."""

    pressures: np.ndarray
    flows_cm3_s: np.ndarray
    proximal_nodes: np.ndarray
    distal_nodes: np.ndarray


@dataclass(frozen=True)
class VesselConcentrationProblem:
    """Inputs for top-down or general-network intravascular transport."""

    flows_cm3_s: np.ndarray
    segment_starts_cm: np.ndarray
    segment_ends_cm: np.ndarray
    segment_radii_cm: np.ndarray
    segment_lengths_cm: np.ndarray
    inlet_nodes: Sequence[int]
    outlet_nodes: Sequence[int] | None
    inlet_concentration: float
    diffusivity_cm2_s: float
    vmax: float
    km: float
    fluid: str
    solver: str
    tree: Any | None = None
    proximal_nodes: np.ndarray | None = None
    distal_nodes: np.ndarray | None = None

    def __post_init__(self) -> None:
        starts = np.asarray(self.segment_starts_cm, dtype=float)
        ends = np.asarray(self.segment_ends_cm, dtype=float)
        if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
            raise ValueError("Segment start/end arrays must have shape (n_segments, 3).")
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
            raise ValueError("Segment start/end arrays must contain only finite values.")
        flows = _vector(self.flows_cm3_s, "flows_cm3_s", dtype=float)
        radii = _vector(self.segment_radii_cm, "segment_radii_cm", dtype=float)
        lengths = _vector(self.segment_lengths_cm, "segment_lengths_cm", dtype=float)
        if not (starts.shape[0] == flows.size == radii.size == lengths.size):
            raise ValueError("All vessel concentration segment arrays must have equal lengths.")
        if np.any(radii <= 0.0) or np.any(lengths <= 0.0):
            raise ValueError("Segment radii and lengths must be strictly positive.")
        proximal = None if self.proximal_nodes is None else _vector(
            self.proximal_nodes, "proximal_nodes", dtype=np.int64
        )
        distal = None if self.distal_nodes is None else _vector(
            self.distal_nodes, "distal_nodes", dtype=np.int64
        )
        if (proximal is None) != (distal is None):
            raise ValueError("proximal_nodes and distal_nodes must be supplied together.")
        if proximal is not None and (proximal.size != flows.size or distal.size != flows.size):
            raise ValueError("Concentration topology arrays must match the segment count.")
        if proximal is not None and (np.any(proximal < 0) or np.any(distal < 0)):
            raise ValueError("Concentration topology node indices must be nonnegative.")
        inlet_concentration = float(self.inlet_concentration)
        diffusivity = float(self.diffusivity_cm2_s)
        vmax = float(self.vmax)
        km = float(self.km)
        if not np.isfinite(inlet_concentration):
            raise ValueError("inlet_concentration must be finite.")
        if not np.isfinite(diffusivity) or diffusivity <= 0.0:
            raise ValueError("diffusivity_cm2_s must be finite and strictly positive.")
        if not np.isfinite(vmax) or vmax < 0.0:
            raise ValueError("vmax must be finite and nonnegative.")
        if not np.isfinite(km) or km <= 0.0:
            raise ValueError("km must be finite and strictly positive.")
        fluid = str(self.fluid).strip().lower()
        if fluid not in {"blood", "water", "media", "cell media"}:
            raise ValueError("fluid must be 'blood', 'water', 'media', or 'cell media'.")
        solver = str(self.solver).strip().lower()
        supported_solvers = {
            "topdown",
            "network",
            "network_ext",
            "topdown_ext",
            "topdown_ext_hybrid_bg",
            "network_ext_hybrid_bg",
            "topdown_ext_treecode",
        }
        if solver not in supported_solvers:
            raise ValueError(f"Unsupported vessel concentration solver: {self.solver!r}.")
        object.__setattr__(self, "flows_cm3_s", flows)
        object.__setattr__(self, "segment_starts_cm", starts)
        object.__setattr__(self, "segment_ends_cm", ends)
        object.__setattr__(self, "segment_radii_cm", radii)
        object.__setattr__(self, "segment_lengths_cm", lengths)
        object.__setattr__(self, "inlet_concentration", inlet_concentration)
        object.__setattr__(self, "diffusivity_cm2_s", diffusivity)
        object.__setattr__(self, "vmax", vmax)
        object.__setattr__(self, "km", km)
        object.__setattr__(self, "fluid", fluid)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "inlet_nodes", tuple(int(value) for value in self.inlet_nodes))
        if self.outlet_nodes is not None:
            object.__setattr__(self, "outlet_nodes", tuple(int(value) for value in self.outlet_nodes))
        object.__setattr__(self, "proximal_nodes", proximal)
        object.__setattr__(self, "distal_nodes", distal)


@dataclass(frozen=True)
class VesselConcentrationResult:
    """Segment inlet/outlet concentrations and optional coupled-field state."""

    inlet: np.ndarray
    outlet: np.ndarray
    external_field: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TissueOxygenProblem:
    """Inputs required for Green's Function Method tissue sampling."""

    points_cm: np.ndarray
    segment_starts_cm: np.ndarray
    segment_ends_cm: np.ndarray
    segment_radii_cm: np.ndarray
    vessel_inlet_concentration: np.ndarray
    segment_flows_cm3_s: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_cm, dtype=float)
        starts = np.asarray(self.segment_starts_cm, dtype=float)
        ends = np.asarray(self.segment_ends_cm, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_cm must have shape (n_points, 3).")
        if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
            raise ValueError("Segment start/end arrays must have shape (n_segments, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_cm must contain only finite values.")
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
            raise ValueError("Segment start/end arrays must contain only finite values.")
        radii = _vector(self.segment_radii_cm, "segment_radii_cm", dtype=float)
        inlet = _vector(self.vessel_inlet_concentration, "vessel_inlet_concentration", dtype=float)
        flows = _vector(self.segment_flows_cm3_s, "segment_flows_cm3_s", dtype=float)
        if not (starts.shape[0] == radii.size == inlet.size == flows.size):
            raise ValueError("All tissue problem segment arrays must have equal lengths.")
        if np.any(radii <= 0.0):
            raise ValueError("Segment radii must be strictly positive.")
        object.__setattr__(self, "points_cm", points)
        object.__setattr__(self, "segment_starts_cm", starts)
        object.__setattr__(self, "segment_ends_cm", ends)
        object.__setattr__(self, "segment_radii_cm", radii)
        object.__setattr__(self, "vessel_inlet_concentration", inlet)
        object.__setattr__(self, "segment_flows_cm3_s", flows)


@dataclass(frozen=True)
class TissueOxygenResult:
    """Tissue concentrations with the mask of points retained by the GFM solve."""

    retained_points: np.ndarray
    concentration: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
