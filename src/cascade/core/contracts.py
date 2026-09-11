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
        if not np.isfinite(float(self.inlet_flow_cm3_s)):
            raise ValueError("inlet_flow_cm3_s must be finite.")
        object.__setattr__(self, "proximal_nodes", proximal)
        object.__setattr__(self, "distal_nodes", distal)
        object.__setattr__(self, "resistances", resistance)
        object.__setattr__(self, "inlet_nodes", tuple(int(value) for value in self.inlet_nodes))
        object.__setattr__(self, "outlet_nodes", tuple(int(value) for value in self.outlet_nodes))


@dataclass(frozen=True)
class FlowResult:
    """Pressures and signed segment flows returned by the flow subsystem."""

    pressures: np.ndarray
    flows_cm3_s: np.ndarray
    proximal_nodes: np.ndarray
    distal_nodes: np.ndarray
    node_coordinates_cm: np.ndarray


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
        object.__setattr__(self, "flows_cm3_s", flows)
        object.__setattr__(self, "segment_starts_cm", starts)
        object.__setattr__(self, "segment_ends_cm", ends)
        object.__setattr__(self, "segment_radii_cm", radii)
        object.__setattr__(self, "segment_lengths_cm", lengths)
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
class ExternalFieldResult:
    """Converged Cext arrays and diagnostics independent of export formatting."""

    state: Mapping[str, Any]
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
        radii = _vector(self.segment_radii_cm, "segment_radii_cm", dtype=float)
        inlet = _vector(self.vessel_inlet_concentration, "vessel_inlet_concentration", dtype=float)
        flows = _vector(self.segment_flows_cm3_s, "segment_flows_cm3_s", dtype=float)
        if not (starts.shape[0] == radii.size == inlet.size == flows.size):
            raise ValueError("All tissue problem segment arrays must have equal lengths.")
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
