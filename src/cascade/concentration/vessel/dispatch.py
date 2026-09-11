"""Vessel concentration backend selection."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.concentration.external_field.diagnostics import (
    _default_cext_timing_details,
)
from cascade.concentration.external_field.hybrid_coupled_solver import (
    _solve_channel_concentrations_topdown_ext_hybrid_bg,
)
from cascade.concentration.external_field.topdown_solver import (
    _solve_channel_concentrations_topdown_ext,
)
from cascade.concentration.external_field.treecode_solver import (
    _solve_channel_concentrations_topdown_ext_treecode,
)
from cascade.flow.topology import _build_node_indices

from .network import solve_network_concentrations
from .topdown import _solve_channel_concentrations_topdown


def _resolve_concentration_solver(value: str | None) -> str:
    if value is None:
        value = _state.CONCENTRATION_SOLVER
    mode = str(value).strip().lower()
    if mode in (
        "topdown",
        "network",
        "network_ext",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "network_ext_hybrid_bg",
        "topdown_ext_treecode",
    ):
        return mode
    raise ValueError(
        "concentration solver must be 'topdown', 'network', 'network_ext', 'topdown_ext', 'topdown_ext_hybrid_bg', 'network_ext_hybrid_bg', or 'topdown_ext_treecode'."
    )


def _solve_channel_concentrations(
    tree: _state.Tree,
    flows: np.ndarray,
    inlet_nodes: Sequence[int],
    outlet_nodes: Optional[Sequence[int]],
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    prox_ids: np.ndarray | None = None,
    dist_ids: np.ndarray | None = None,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    fluid: str,
    solver: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mode = _resolve_concentration_solver(solver)
    # Mutable runtime state is centralized in configuration.solver_state.
    _state._LAST_CONCENTRATION_TIMINGS = _default_cext_timing_details()
    _state._LAST_CEXT_SOURCE_STATE = None
    if mode == "topdown":
        return _solve_channel_concentrations_topdown(
            tree,
            flows,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    if mode == "topdown_ext":
        return _solve_channel_concentrations_topdown_ext(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    if mode == "network_ext":
        if prox_ids is None or dist_ids is None:
            geometry = np.zeros((starts.shape[0], 6), dtype=float)
            geometry[:, 0:3] = starts
            geometry[:, 3:6] = ends
            prox_ids, dist_ids, _ = _build_node_indices(geometry)
        return _solve_channel_concentrations_topdown_ext(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
            network_topology={
                "prox_ids": np.asarray(prox_ids, dtype=np.int64),
                "dist_ids": np.asarray(dist_ids, dtype=np.int64),
                "inlet_nodes": tuple(int(node) for node in inlet_nodes),
                "outlet_nodes": None
                if outlet_nodes is None
                else tuple(int(node) for node in outlet_nodes),
            },
        )
    if mode == "topdown_ext_hybrid_bg":
        return _solve_channel_concentrations_topdown_ext_hybrid_bg(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    if mode == "network_ext_hybrid_bg":
        if prox_ids is None or dist_ids is None:
            geometry = np.zeros((starts.shape[0], 6), dtype=float)
            geometry[:, 0:3] = starts
            geometry[:, 3:6] = ends
            prox_ids, dist_ids, _ = _build_node_indices(geometry)
        return _solve_channel_concentrations_topdown_ext_hybrid_bg(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
            network_topology={
                "prox_ids": np.asarray(prox_ids, dtype=np.int64),
                "dist_ids": np.asarray(dist_ids, dtype=np.int64),
                "inlet_nodes": tuple(int(node) for node in inlet_nodes),
                "outlet_nodes": None
                if outlet_nodes is None
                else tuple(int(node) for node in outlet_nodes),
            },
        )
    if mode == "topdown_ext_treecode":
        return _solve_channel_concentrations_topdown_ext_treecode(
            tree,
            flows,
            starts,
            ends,
            radii,
            lengths,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            fluid=fluid,
        )
    cin, cout, _, _ = solve_network_concentrations(
        starts,
        ends,
        radii,
        lengths,
        flows,
        inlet_nodes,
        outlet_nodes,
        inlet_concentration,
        fluid=fluid,
        prox_ids=prox_ids,
        dist_ids=dist_ids,
        diffusivity=diffusivity,
        vmax=vmax,
        km=km,
    )
    return cin, cout


__all__ = ["_resolve_concentration_solver", "_solve_channel_concentrations"]
