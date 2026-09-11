"""Stable public entry point for Green's Function Method tissue oxygen."""

from __future__ import annotations

from cascade.core import TissueOxygenProblem, TissueOxygenResult


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
    from cascade.runtime import tissuesim

    retained, concentration = tissuesim.compute_tissue_samples_greens(
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
        diagnostics=dict(tissuesim._LAST_TISSUE_TIMINGS or {}),
    )


__all__ = ["TissueOxygenProblem", "TissueOxygenResult", "solve_tissue_oxygen"]
