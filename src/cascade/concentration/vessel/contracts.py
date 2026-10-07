"""Validated input and result contracts for vessel transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _vector(value: Any, name: str, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


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
            raise ValueError(
                "Segment start/end arrays must have shape (n_segments, 3)."
            )
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
            raise ValueError(
                "Segment start/end arrays must contain only finite values."
            )
        flows = _vector(self.flows_cm3_s, "flows_cm3_s", dtype=float)
        radii = _vector(self.segment_radii_cm, "segment_radii_cm", dtype=float)
        lengths = _vector(self.segment_lengths_cm, "segment_lengths_cm", dtype=float)
        if not (starts.shape[0] == flows.size == radii.size == lengths.size):
            raise ValueError(
                "All vessel concentration segment arrays must have equal lengths."
            )
        if np.any(radii <= 0.0) or np.any(lengths <= 0.0):
            raise ValueError("Segment radii and lengths must be strictly positive.")
        proximal = (
            None
            if self.proximal_nodes is None
            else _vector(self.proximal_nodes, "proximal_nodes", dtype=np.int64)
        )
        distal = (
            None
            if self.distal_nodes is None
            else _vector(self.distal_nodes, "distal_nodes", dtype=np.int64)
        )
        if (proximal is None) != (distal is None):
            raise ValueError(
                "proximal_nodes and distal_nodes must be supplied together."
            )
        if proximal is not None and (
            proximal.size != flows.size or distal.size != flows.size
        ):
            raise ValueError(
                "Concentration topology arrays must match the segment count."
            )
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
            raise ValueError(
                "fluid must be 'blood', 'water', 'media', or 'cell media'."
            )
        solver = str(self.solver).strip().lower()
        supported_solvers = {
            "topdown",
            "network",
            "network_ext",
            "topdown_ext",
            "topdown_ext_hybrid_bg",
            "network_ext_hybrid_bg",
            "topdown_ext_treecode",
            "network_ext_treecode",
        }
        if solver not in supported_solvers:
            raise ValueError(
                f"Unsupported vessel concentration solver: {self.solver!r}."
            )
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
        object.__setattr__(self, "inlet_nodes", tuple(int(v) for v in self.inlet_nodes))
        if self.outlet_nodes is not None:
            object.__setattr__(
                self, "outlet_nodes", tuple(int(v) for v in self.outlet_nodes)
            )
        object.__setattr__(self, "proximal_nodes", proximal)
        object.__setattr__(self, "distal_nodes", distal)


@dataclass(frozen=True)
class VesselConcentrationResult:
    """Segment inlet/outlet concentrations and optional coupled-field state."""

    inlet: np.ndarray
    outlet: np.ndarray
    external_field: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["VesselConcentrationProblem", "VesselConcentrationResult"]
