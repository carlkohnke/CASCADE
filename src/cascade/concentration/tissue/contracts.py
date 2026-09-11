"""Validated input and result contracts for tissue oxygen solving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


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
            raise ValueError(
                "Segment start/end arrays must have shape (n_segments, 3)."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("points_cm must contain only finite values.")
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
            raise ValueError(
                "Segment start/end arrays must contain only finite values."
            )
        radii = _vector(self.segment_radii_cm, "segment_radii_cm")
        inlet = _vector(self.vessel_inlet_concentration, "vessel_inlet_concentration")
        flows = _vector(self.segment_flows_cm3_s, "segment_flows_cm3_s")
        if not (starts.shape[0] == radii.size == inlet.size == flows.size):
            raise ValueError(
                "All tissue problem segment arrays must have equal lengths."
            )
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
    """Tissue concentrations with the mask retained by the GFM solve."""

    retained_points: np.ndarray
    concentration: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["TissueOxygenProblem", "TissueOxygenResult"]
