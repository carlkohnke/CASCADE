"""Stable public entry point for Green's Function Method tissue oxygen."""

from __future__ import annotations

import math

import numpy as np

from cascade.configuration import _legacy_state as _state
from cascade.core import TissueOxygenProblem, TissueOxygenResult

from .greens import compute_tissue_samples_greens


def solve_tissue_oxygen(
    problem: TissueOxygenProblem,
    *,
    diffusivity_cm2_s: float,
    vmax: float,
    km: float,
    window_factor: float,
    inlet_concentration: float | None = None,
    tissue_cache: dict | None = None,
) -> TissueOxygenResult:
    """Evaluate tissue oxygen without exposing legacy tuple conventions."""
    if not math.isfinite(float(diffusivity_cm2_s)) or diffusivity_cm2_s <= 0.0:
        raise ValueError("diffusivity_cm2_s must be finite and strictly positive.")
    if not math.isfinite(float(vmax)) or vmax < 0.0:
        raise ValueError("vmax must be finite and nonnegative.")
    if not math.isfinite(float(km)) or km <= 0.0:
        raise ValueError("km must be finite and strictly positive.")
    if not math.isfinite(float(window_factor)) or window_factor <= 0.0:
        raise ValueError("window_factor must be finite and strictly positive.")
    if inlet_concentration is None and problem.vessel_inlet_concentration.size:
        inlet_concentration = float(np.max(problem.vessel_inlet_concentration))
    retained, concentration = compute_tissue_samples_greens(
        problem.points_cm,
        problem.segment_starts_cm,
        problem.segment_ends_cm,
        problem.segment_radii_cm,
        problem.vessel_inlet_concentration,
        problem.segment_flows_cm3_s,
        diffusivity=float(diffusivity_cm2_s),
        vmax=float(vmax),
        km=float(km),
        window_factor=float(window_factor),
        inlet_concentration=inlet_concentration,
        tissue_cache=tissue_cache,
    )
    return TissueOxygenResult(
        retained_points=retained,
        concentration=concentration,
        diagnostics=dict(_state._LAST_TISSUE_TIMINGS or {}),
    )


__all__ = ["TissueOxygenProblem", "TissueOxygenResult", "solve_tissue_oxygen"]
