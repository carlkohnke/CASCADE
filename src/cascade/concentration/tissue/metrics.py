"""Tissue concentration and viability metrics.

This module composes the public flow and concentration implementations without
depending on import order or implicit namespace injection.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.concentration.properties import get_concentration_inlet
from cascade.concentration.tissue.greens import compute_tissue_samples_greens
from cascade.concentration.vessel.tree import solve_tree_greens
from cascade.flow.hematocrit import _tree_exact_connectivity


def compute_concentration_profiles(
    tree,
    *,
    inlet_concentration: float | None = None,
    extravascular_concentration: float = _state.EXTRAVASCULAR_CONCENTRATION,
    diffusivity: float = _state.SOLUTE_DIFFUSIVITY,
    fluid: str | None = None,
    concentration_solver: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _ = extravascular_concentration
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet(fluid)
    q_inlet_cm3_s = float(_state.QIN_TARGET) * 1e-3 / 60.0
    _, _, _, _, _, cin, cout, _ = solve_tree_greens(
        tree,
        q_inlet_cm3_s,
        fluid=fluid or _state.ACTIVE_FLUID,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=_state.VMAX_MM,
        km=_state.K_M_MM,
        omega=_state.OMEGA,
        concentration_solver=concentration_solver,
    )
    return cin, cout


def compute_concentration_metrics(
    tree,
    sample_points: np.ndarray,
    *,
    inlet_concentration: float | None = None,
    extravascular_concentration: float = _state.EXTRAVASCULAR_CONCENTRATION,
    diffusivity: float = _state.SOLUTE_DIFFUSIVITY,
    # tissue_lengthscale: float = TISSUE_DECAY_LENGTH,
    fluid: str | None = None,
    concentration_solver: str | None = None,
) -> tuple[float, float, Dict[str, float]]:
    if inlet_concentration is None:
        inlet_concentration = get_concentration_inlet(fluid)
    q_inlet_cm3_s = float(_state.QIN_TARGET) * 1e-3 / 60.0
    starts, ends, radii, lengths, flows, cin, cout, _ = solve_tree_greens(
        tree,
        q_inlet_cm3_s,
        fluid=fluid or _state.ACTIVE_FLUID,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        vmax=_state.VMAX_MM,
        km=_state.K_M_MM,
        omega=_state.OMEGA,
        concentration_solver=concentration_solver,
    )

    vessel_map = getattr(tree, "vessel_map", {}) or {}
    terminals = (
        [
            idx
            for idx in range(cout.size)
            if not vessel_map.get(idx, {}).get("downstream")
        ]
        if vessel_map
        else []
    )
    if not terminals:
        data = np.asarray(tree.data[: cout.size], dtype=float)
        conn = _tree_exact_connectivity(tree, data)
        terminals = list(np.where((conn[:, 0] < 0) & (conn[:, 1] < 0))[0])
    target_concs = cout[terminals] if terminals else cout
    finite = target_concs[np.isfinite(target_concs)]
    c_lq = float(np.percentile(finite, 25.0)) if finite.size else float("nan")

    mask, tissue_conc = compute_tissue_samples_greens(
        sample_points,
        starts,
        ends,
        radii,
        cin,
        flows,
        diffusivity=_state.SOLUTE_DIFFUSIVITY,
        vmax=_state.VMAX_MM,
        km=_state.K_M_MM,
        window_factor=_state.WINDOW_FACTOR,
        inlet_concentration=inlet_concentration,
    )
    tissue_vals = tissue_conc[mask]
    tissue_avg = float(np.nanmean(tissue_vals)) if tissue_vals.size else float("nan")

    if (
        np.isfinite(_state.CONC_MAX_FOR_NORMALIZATION)
        and _state.CONC_MAX_FOR_NORMALIZATION != 0.0
    ):
        ratio_lq = (
            c_lq / _state.CONC_MAX_FOR_NORMALIZATION
            if np.isfinite(c_lq)
            else float("nan")
        )
        ratio_tiss = (
            tissue_avg / _state.CONC_MAX_FOR_NORMALIZATION
            if np.isfinite(tissue_avg)
            else float("nan")
        )
        fractions = {}
        if tissue_vals.size:
            normalized = tissue_vals / _state.CONC_MAX_FOR_NORMALIZATION
            for threshold, key in [
                (0.50, "FracAbove50pct"),
                (0.25, "FracAbove25pct"),
                (0.10, "FracAbove10pct"),
                (0.05, "FracAbove5pct"),
                (0.01, "FracAbove1pct"),
            ]:
                fractions[key] = float(np.mean(normalized >= threshold))
        else:
            fractions = {
                k: float("nan")
                for k in [
                    "FracAbove50pct",
                    "FracAbove25pct",
                    "FracAbove10pct",
                    "FracAbove5pct",
                    "FracAbove1pct",
                ]
            }
    else:
        ratio_lq = float("nan")
        ratio_tiss = float("nan")
        fractions = {
            k: float("nan")
            for k in [
                "FracAbove50pct",
                "FracAbove25pct",
                "FracAbove10pct",
                "FracAbove5pct",
                "FracAbove1pct",
            ]
        }
    return ratio_lq, ratio_tiss, fractions


__all__ = ["compute_concentration_profiles", "compute_concentration_metrics"]
